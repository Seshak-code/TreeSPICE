#pragma once
#include "../netlist/circuit.h"
#include "../math/linalg.h"
#include <vector>
#include <cmath>
#include <memory>

/*
 * integration_method.h
 * Strategy Pattern for Numerical Integration Methods.
 * Supports Trapezoidal, Gear 1 (Backward Euler), and Gear 2 (variable-step BDF2) with predictors.
 */


enum class IntegrationType {
    TRAPEZOIDAL,
    GEAR_1_EULER,
    GEAR_2
};

// Abstract Base Class for Integration Strategies.

class IIntegrationMethod {
public:
    virtual ~IIntegrationMethod() = default;

    virtual void stampCapacitor(
        int nodeI, int nodeJ, double C, double timeStep,
        const std::vector<double>& v_hist, const std::vector<double>& i_hist,
        MatrixConstructor& mat, std::vector<double>& rhs) = 0;

    virtual void updateHistory(
        std::vector<double>& v_hist, std::vector<double>& i_hist,
        double v_new, double i_new_calculated) = 0;

    virtual void prepareStep(double dt) {}

    virtual double predict(const std::vector<double>& v_hist) = 0;
        
    virtual IntegrationType getType() const = 0;

    // The last accepted step: variable-step Gear 2 weighs the older history by it (0: none yet, then Gear 2 is
    // backward Euler, as ngspice's order 1 start).
    double hPrev = 0.0;

    // dq/dt at the new point from q there, its last two accepted values q1, q2 and the last dq/dt i1, over the
    // step h: trapezoidal 2/h (q - q1) - i1; backward Euler (q - q1)/h; Gear 2 the slope at the new point of the
    // parabola through the three points (ngspice NIcomCof, order 2, at variable step). Affine in q with slope
    // coef(h), so a linear element's companion source is rate(0, ...).
    template <class T> T rate(const T &q, double q1, double q2, double i1, double h) const {
        switch (getType()) {
        case IntegrationType::TRAPEZOIDAL: return (2.0 / h) * (q - q1) - i1;
        case IntegrationType::GEAR_2:
            if (hPrev > 0.0) return coef(h) * q - (h + hPrev) / (h * hPrev) * q1 + h / (hPrev * (h + hPrev)) * q2;
            [[fallthrough]];
        default: return (q - q1) / h;
        }
    }
    double coef(double h) const {
        switch (getType()) {
        case IntegrationType::TRAPEZOIDAL: return 2.0 / h;
        case IntegrationType::GEAR_2: return hPrev > 0.0 ? 1.0 / h + 1.0 / (h + hPrev) : 1.0 / h;
        default: return 1.0 / h;
        }
    }
};

// ============================================================================
// CONCRETE STRATEGIES
// ============================================================================

class TrapezoidalIntegration : public IIntegrationMethod {
public:
    IntegrationType getType() const override { return IntegrationType::TRAPEZOIDAL; }

    void stampCapacitor(int nodeI, int nodeJ, double C, double timeStep,
                        const std::vector<double>& v_hist, const std::vector<double>& i_hist,
                        MatrixConstructor& mat, std::vector<double>& rhs) override {
        if(v_hist.empty()) return;
        double G_eq = (2.0 * C) / timeStep;
        double v_prev = v_hist[0];
        double i_prev = i_hist[0];
        double I_eq = -G_eq * v_prev - i_prev;

        auto add = [&](int r, int c, double val) { if (r != 0 && c != 0) mat.add(r-1, c-1, val); };
        add(nodeI, nodeI, G_eq); add(nodeJ, nodeJ, G_eq);
        add(nodeI, nodeJ, -G_eq); add(nodeJ, nodeI, -G_eq);
        if (nodeI != 0) rhs[nodeI - 1] -= I_eq;
        if (nodeJ != 0) rhs[nodeJ - 1] += I_eq;
    }

    void updateHistory(std::vector<double>& v_hist, std::vector<double>& i_hist,
                        double v_new, double i_new) override {
        for(size_t i = v_hist.size()-1; i > 0; --i) {
            v_hist[i] = v_hist[i-1];
            i_hist[i] = i_hist[i-1];
        }
        v_hist[0] = v_new;
        i_hist[0] = i_new;
    }

    double predict(const std::vector<double>& v_hist) override {
        if (v_hist.size() < 2) return v_hist.empty() ? 0.0 : v_hist[0];
        return 2.0 * v_hist[0] - v_hist[1];
    }
};

class Gear1Integration : public IIntegrationMethod {
public:
    IntegrationType getType() const override { return IntegrationType::GEAR_1_EULER; }

    void stampCapacitor(int nodeI, int nodeJ, double C, double timeStep,
                        const std::vector<double>& v_hist, const std::vector<double>& i_hist,
                        MatrixConstructor& mat, std::vector<double>& rhs) override {
        if(v_hist.empty()) return;
        double G_eq = C / timeStep;
        double v_prev = v_hist[0];
        double I_eq = -G_eq * v_prev;

        auto add = [&](int r, int c, double val) { if (r != 0 && c != 0) mat.add(r-1, c-1, val); };
        add(nodeI, nodeI, G_eq); add(nodeJ, nodeJ, G_eq);
        add(nodeI, nodeJ, -G_eq); add(nodeJ, nodeI, -G_eq);
        if (nodeI != 0) rhs[nodeI - 1] -= I_eq;
        if (nodeJ != 0) rhs[nodeJ - 1] += I_eq;
    }

    void updateHistory(std::vector<double>& v_hist, std::vector<double>& i_hist,
                        double v_new, double i_new) override {
        for(size_t i = v_hist.size()-1; i > 0; --i) {
            v_hist[i] = v_hist[i-1];
            i_hist[i] = i_hist[i-1];
        }
        v_hist[0] = v_new;
        i_hist[0] = i_new;
    }

    double predict(const std::vector<double>& v_hist) override {
        return v_hist.empty() ? 0.0 : v_hist[0];
    }
};

class Gear2Integration : public IIntegrationMethod {
public:
    IntegrationType getType() const override { return IntegrationType::GEAR_2; }

    void stampCapacitor(int nodeI, int nodeJ, double C, double timeStep,
                        const std::vector<double>& v_hist, const std::vector<double>& i_hist,
                        MatrixConstructor& mat, std::vector<double>& rhs) override {
        if (v_hist.size() < 2) {
             // Fallback to Gear 1 if history is sparse
             Gear1Integration fallback;
             fallback.stampCapacitor(nodeI, nodeJ, C, timeStep, v_hist, i_hist, mat, rhs);
             return;
        }

        double G_eq = C * coef(timeStep);
        double I_eq_source = -C * rate(0.0, v_hist[0], v_hist[1], 0.0, timeStep);

        auto add = [&](int r, int c, double val) { if (r != 0 && c != 0) mat.add(r-1, c-1, val); };
        add(nodeI, nodeI, G_eq); add(nodeJ, nodeJ, G_eq);
        add(nodeI, nodeJ, -G_eq); add(nodeJ, nodeI, -G_eq);
        if (nodeI != 0) rhs[nodeI - 1] += I_eq_source;
        if (nodeJ != 0) rhs[nodeJ - 1] -= I_eq_source;
    }

    void updateHistory(std::vector<double>& v_hist, std::vector<double>& i_hist,
                        double v_new, double i_new) override {
        for(size_t i = v_hist.size()-1; i > 0; --i) {
            v_hist[i] = v_hist[i-1];
            i_hist[i] = i_hist[i-1];
        }
        v_hist[0] = v_new;
        i_hist[0] = i_new;
    }

    double predict(const std::vector<double>& v_hist) override {
        if (v_hist.size() < 3) return v_hist.empty() ? 0.0 : v_hist[0];
        return 2.5 * v_hist[0] - 2.0 * v_hist[1] + 0.5 * v_hist[2];
    }
};

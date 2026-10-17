#pragma once

#include <cmath>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <deque>
#include <iterator>
#include <functional>
#include "../netlist/circuit.h"
#include "../tensors/physics_tensors.h"
#include "../math/integration_method.h"

/**
 * lte_controller.h — Local Truncation Error Controller & Method Arbiter (Gap 5)
 *
 * Implements SPICE3-compatible adaptive timestep control based on Local
 * Truncation Error (LTE) estimation for reactive elements (capacitors,
 * inductors).
 *
 * LTE Estimation (3rd-order finite-difference):
 *   For Trapezoidal:  LTE ≈ (h³/12) · d³v/dt³ ≈ (1/12)(v_n - 2·v_{n-1} + v_{n-2})
 *   For Gear2:        LTE ≈ (2h³/9) · d³v/dt³
 *
 * Timestep Control:
 *   dt_new = dt · (tol / maxLTE)^(1/(p+1))
 *   where p = method order (2 for Trap, 2 for Gear2)
 *
 * Method Arbiter:
 *   Detects trapezoidal ringing (sign oscillation in dv/dt for 3+ consecutive
 *   steps) and switches to Gear2 (BDF2) which is A-stable and ring-free.
 *   Returns to Trapezoidal after ringing subsides (4 consecutive non-ringing
 *   steps) for better accuracy on oscillatory circuits.
 *
 * Reference: SPICE3 tran.c: TRTOL, CHGTOL timestep control
 */

// Forward declare IntegrationType from the integrator header
enum class IntegrationType;

struct LTEResult {
    double maxLTE;               // worst-case LTE across all reactive elements
    double suggestedDt;          // LTE-limited timestep suggestion
    int worstElementIndex;       // index of element with largest LTE
    bool worstIsCapacitor;       // true = capacitor, false = inductor
    bool requiresRejection;      // true if LTE > tolerance (timestep too large)
    bool deviceRejects;          // ... by a diode / BJT branch, which the post-corner hold does not cover
    double safetyFactor;         // actual ratio tol/maxLTE (< 1 means reject)
};

class LTEController {
public:
    explicit LTEController(double tolerance = 1e-4, double trtol = 7.0)
        : tolerance_(tolerance), trtol_(trtol) {}

    /**
     * Estimate the Local Truncation Error for all reactive elements.
     *
     * Uses the 3-step difference formula to approximate the 3rd derivative:
     *   d³v/dt³ ≈ (v_n - 3·v_{n-1} + 3·v_{n-2} - v_{n-3}) / dt³
     * Simplified for 2-step history (available when history depth ≥ 3):
     *   LTE_est ≈ |v_n - 2·v_{n-1} + v_{n-2}| / coeff
     * where coeff depends on the integration method.
     *
     * @param netlist    Circuit netlist with capacitor/inductor state history
     * @param v_curr     Current converged node voltages
     * @param dt         Current timestep
     * @param method     Current integration method
     * @return LTEResult with worst-case LTE and timestep suggestion
     */
    LTEResult estimateLTE(
        const TensorNetlist& netlist,
        const std::vector<double>& v_curr,
        double dt,
        IntegrationType method,
        // Extra charge branches the netlist scan cannot see (diode/BJT charges on
        // internal nodes): extra(visit) calls visit(v_now, {v[t-1], v[t-2], v[t-3]}, C) with the
        // branch's capacitance (< 0: unknown, the voltage error counts in full).
        const std::function<void(const std::function<void(double, const std::array<double, 3>&, double)>&)>& extra = nullptr) const
    {
        LTEResult result{};
        result.maxLTE = 0.0;
        result.suggestedDt = dt;
        result.worstElementIndex = -1;
        result.worstIsCapacitor = true;
        result.requiresRejection = false;
        result.deviceRejects = false;
        result.safetyFactor = 1.0;

        if (dt <= 0.0) return result;

        // LTE from divided differences over the ACTUAL step sizes. The old
        // (v_n - 2v_{n-1} + v_{n-2}) form assumed equal steps; right after the
        // tstep/100 cold-start sweep it degenerated to a first difference
        // (~dV), rejected every step, and hung the transient.
        //   BE:    LTE = h^2/2  * |v2|,  v2 ~ 2 f[t0,t1,t2]
        //   Trap:  LTE = h^3/12 * |v3|,  v3 ~ 6 f[t0,t1,t2,t3]
        //   Gear2: LTE = 2h^3/9 * |v3|
        const bool firstOrder = (method == IntegrationType::GEAR_1_EULER);
        const size_t needSteps = firstOrder ? 1 : 2;
        if (stepHistory_.size() < needSteps) return result;  // not enough history yet

        const double t0 = 0.0, t1 = -dt;
        const double t2 = t1 - stepHistory_[0];
        const double t3 = (stepHistory_.size() > 1) ? t2 - stepHistory_[1] : t2;

        // y0 = value now, h[0..2] = previous accepted values (most recent first)
        auto lteOf = [&](double y0, const auto& h) -> double {
            if (std::size(h) < needSteps + 1) return 0.0;
            double d01 = (y0 - h[0]) / (t0 - t1);
            double d12 = (h[0] - h[1]) / (t1 - t2);
            double d012 = (d01 - d12) / (t0 - t2);
            if (firstOrder) return dt * dt * std::abs(d012);
            double d23 = (h[1] - h[2]) / (t2 - t3);
            double d123 = (d12 - d23) / (t1 - t3);
            double d0123 = (d012 - d123) / (t0 - t3);
            double coeff = (method == IntegrationType::GEAR_2) ? (4.0 / 3.0) : 0.5;
            return coeff * dt * dt * dt * std::abs(d0123);
        };

        auto getV = [&](int nodeIdx) -> double {
            if (nodeIdx <= 0 || nodeIdx > (int)v_curr.size()) return 0.0;
            return v_curr[nodeIdx - 1];
        };

        // Capacitors, inductors and MOSFET charge branches of the global block AND every subcircuit
        // instance (instance devices were skipped, so hierarchical designs ran
        // with no LTE step control). Inductors use branch voltage: the solver
        // does not write back inductor current history, so hist.i is unusable.
        auto scan = [&](const TensorBlock& blk, const BlockState& st, auto mapNode) {
            for (size_t i = 0; i < blk.capacitors.size() && i < st.capacitorState.size(); ++i) {
                const auto& cap = blk.capacitors[i];
                double lte_v = lteOf(getV(mapNode(cap.nodePlate1)) - getV(mapNode(cap.nodePlate2)),
                                     st.capacitorState[i].v);
                if (lte_v > result.maxLTE) {
                    result.maxLTE = lte_v;
                    result.worstElementIndex = static_cast<int>(i);
                    result.worstIsCapacitor = true;
                }
            }
            for (size_t i = 0; i < blk.inductors.size() && i < st.inductorState.size(); ++i) {
                const auto& ind = blk.inductors[i];
                double lte_v = lteOf(getV(mapNode(ind.nodeCoil1)) - getV(mapNode(ind.nodeCoil2)),
                                     st.inductorState[i].v);
                if (lte_v > result.maxLTE) {
                    result.maxLTE = lte_v;
                    result.worstElementIndex = static_cast<int>(i);
                    result.worstIsCapacitor = false;
                }
            }
            for (size_t i = 0; i < blk.mosfets.size() && i < st.mosfetState.size(); ++i) {
                const auto& m = blk.mosfets[i];
                const auto p = mos1Params(m);
                if (!p.caps) continue;  // no charge, no history
                const auto& vh = st.mosfetState[i].v;
                const auto br = mos1_branches(getV(mapNode(m.drain)), getV(mapNode(m.gate)),
                                              getV(mapNode(m.source)), getV(mapNode(m.body)), p.polarity);
                for (int k = 0; k < 5; ++k) {
                    const double hist[3] = {vh[0][k], vh[1][k], vh[2][k]};
                    const double lte_v = lteOf(br[k], hist);
                    if (lte_v > result.maxLTE) {
                        result.maxLTE = lte_v;
                        result.worstElementIndex = static_cast<int>(i);
                        result.worstIsCapacitor = true;
                    }
                }
            }
        };
        scan(netlist.globalBlock, netlist.globalState, [](int n) { return n; });
        for (size_t k = 0; k < netlist.instances.size() && k < netlist.instanceStates.size(); ++k) {
            const auto& inst = netlist.instances[k];
            auto it = netlist.blockDefinitions.find(inst.blockName);
            if (it == netlist.blockDefinitions.end()) continue;
            scan(it->second, netlist.instanceStates[k], [&](int n) {
                return (n <= 0 || n > (int)inst.nodeMapping.size()) ? 0 : inst.nodeMapping[n - 1];
            });
        }
        if (extra)
            extra([&](double y0, const std::array<double, 3>& hist, double cap) {
                // ngspice truncates on charge with a floor (CHGTOL): a branch whose charge error at the
                // voltage tolerance stays under it cannot limit the step. A comparator's fF junction
                // capacitances, switching volts in a step, otherwise rejected it down to femtoseconds.
                // ponytail: a voltage LTE scaled by the branch's charge sensitivity, not ngspice's CKTterr
                // (divided differences of the charge against max(volttol, chargetol)); .options chgtol is
                // not read. A direct CKTterr port is on mac/conv-cktterr: it moved the diode snap-off and
                // MOS1 edges (qa_d_rev_rec, qa3_*), needing ngspice's order and breakpoint control as well.
                const double lte_v = lteOf(y0, hist) * (cap >= 0.0 ? std::min(1.0, cap * trtol_ * tolerance_ / CHGTOL) : 1.0);
                result.deviceRejects |= cap >= 0.0 && lte_v > trtol_ * tolerance_;
                if (lte_v > result.maxLTE) {
                    result.maxLTE = lte_v;
                    result.worstElementIndex = -1;
                    result.worstIsCapacitor = true;
                }
            });
        // Compute safety factor and timestep suggestion
        double effective_tol = trtol_ * tolerance_;
        if (result.maxLTE > 0.0) {
            result.safetyFactor = effective_tol / result.maxLTE;
            result.requiresRejection = (result.safetyFactor < 1.0);

            // Optimal timestep: dt_new = dt * (tol/LTE)^(1/(p+1))
            // p = 1 for BE (exponent 1/2), 2 for Trap/Gear2 (exponent 1/3)
            // Apply safety margin of 0.8 to avoid re-rejection
            double ratio = std::pow(result.safetyFactor, firstOrder ? 0.5 : 1.0 / 3.0);
            ratio = std::min(ratio, 2.0);   // cap growth at 2× per step
            ratio = std::max(ratio, 0.1);   // prevent collapse to zero
            result.suggestedDt = dt * ratio * 0.8;
        }

        return result;
    }

    /**
     * Suggest a new timestep based on LTE estimate.
     * Applies SPICE3-style safety margins and growth limits.
     */
    double suggestTimestep(double currentDt, double maxLTE,
                           double tolerance, IntegrationType method) const
    {
        if (maxLTE <= 0.0) return currentDt * 1.5;  // no LTE info → grow cautiously

        double effective_tol = trtol_ * tolerance;
        double ratio = effective_tol / maxLTE;

        // p+1 = 3 for both Trapezoidal and Gear2 (both are 2nd-order)
        double scale = std::pow(ratio, 1.0 / 3.0);

        // SPICE3-style limits:
        //   - Never grow more than 2× per step
        //   - Never shrink below 0.125× per step
        //   - Apply 0.8 safety factor
        scale = std::clamp(scale * 0.8, 0.125, 2.0);

        return currentDt * scale;
    }

    /**
     * Reset controller state (call on topology change).
     */
    /** Record an ACCEPTED step size (call once per history update). */
    void recordStep(double h) {
        stepHistory_.push_front(h);
        if (stepHistory_.size() > 3) stepHistory_.pop_back();
    }

    void reset() {
        stepHistory_.clear();
    }

    // Configuration
    void setTolerance(double tol) { tolerance_ = tol; }
    void setTRTOL(double trtol) { trtol_ = trtol; }
    double getTolerance() const { return tolerance_; }

private:
    static constexpr double CHGTOL = 1e-14;  // C
    double tolerance_;     // absolute LTE tolerance (default 1e-4)
    double trtol_;         // SPICE TRTOL multiplier (default 7.0)

    std::deque<double> stepHistory_;  // last accepted step sizes, most recent first
};

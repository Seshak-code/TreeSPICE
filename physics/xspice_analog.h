#pragma once
// XSPICE analog code models without state, as ngspice 47's cfunc.mod files compute them (icm/analog,
// icm/xtradev). Each output is a behavioural source (Behavioural::cm) over the instance's inputs; its
// Jacobian comes from Dual, the exact derivative of the value ngspice computes (ngspice's own partials
// only steer its Newton iteration).
#include <cmath>
#include <string>
#include <vector>

namespace xspice {

enum class AKind { GAIN, SUMMER, MULT, DIVIDE, LIMIT, CLIMIT, PWL, ASWITCH };

struct AnalogModel {
    AKind kind = AKind::GAIN;
    // gain, limit, climit: in_offset, gain, out_offset; summer / mult: per input in_offset, in_gain
    double inOffset = 0.0, gain = 1.0, outOffset = 0.0, outGain = 1.0;
    std::vector<double> inOffsets, inGains;
    // divide
    double numOffset = 0.0, numGain = 1.0, denOffset = 0.0, denGain = 1.0, denLower = 1e-10, denDomain = 1e-16;
    // limit / climit: limits, smoothing range (absolute after `fraction`), climit's deltas
    double lo = 0.0, hi = 1.0, range = 1e-6, upperDelta = 0.0, lowerDelta = 0.0;
    bool fraction = false;
    // pwl: the breakpoints with ngspice's extra pair at either end, smoothing domain
    std::vector<double> x, y;
    double domain = 0.01;
    // aswitch
    double cntlOff = 0.0, cntlOn = 1.0, rOff = 1e12, rOn = 1.0;
    bool logR = true, limitR = false;
};

// cm_smooth_corner: the parabola through the corner (xc, yc) of slopes s0 and s1 over xc - d .. xc + d
template <typename T>
inline T smoothCorner(const T &x, double xc, double yc, double d, double s0, double s1) {
    const double xu = xc + d, yu = yc + s1 * d, a = (s1 - s0) / 4.0 / d, b = s1 - 2.0 * a * xu, c = yu - a * xu * xu - b * xu;
    return a * x * x + b * x + c;
}

// The output for inputs `in` (port order; aswitch: control, then the voltage across its own port)
template <typename T>
T analogOut(const AnalogModel &m, const std::vector<T> &in) {
    switch (m.kind) {
    case AKind::GAIN: return m.outOffset + m.gain * (in[0] + m.inOffset);
    case AKind::SUMMER: {
        T s(0.0);
        for (size_t i = 0; i < in.size(); ++i) s = s + m.inGains[i] * (in[i] + m.inOffsets[i]);
        return s * m.outGain + m.outOffset;
    }
    case AKind::MULT: {
        T p(1.0);
        double g = 1.0;
        for (size_t i = 0; i < in.size(); ++i) { p = p * (in[i] + m.inOffsets[i]); g *= m.inGains[i]; }
        return p * g * m.outGain + m.outOffset;
    }
    case AKind::DIVIDE: {
        const T num = (in[0] + m.numOffset) * m.numGain, den = (in[1] + m.denOffset) * m.denGain;
        const double dom = m.fraction ? m.denDomain * m.denLower : m.denDomain;
        const double up = m.denLower + dom, low = m.denLower - dom;
        T d = den;
        if (den < up && den >= 0.0) d = den > low ? smoothCorner(den, m.denLower, m.denLower, dom, 0.0, 1.0) : T(m.denLower);
        else if (den > -up && den < 0.0) d = den < -low ? smoothCorner(den, -m.denLower, -m.denLower, dom, 0.0, 1.0) : T(-m.denLower);
        return num / d * m.outGain + m.outOffset;
    }
    case AKind::LIMIT: case AKind::CLIMIT: {
        // climit: the limits follow the control inputs (cm_climit_fcn)
        const T lo = m.kind == AKind::CLIMIT ? in[2] + m.lowerDelta : T(m.lo), hi = m.kind == AKind::CLIMIT ? in[1] - m.upperDelta : T(m.hi);
        const T r = m.fraction ? m.range * (hi - lo) : T(m.range);
        const T out = m.gain * (m.inOffset + in[0]);
        if (out < lo + r) {
            if (out > lo - r) {  // the parabola of cm_smooth_corner at the lower corner, its centre on the limit
                const T a = (1.0 / 4.0) / r, xu = lo + r, b = 1.0 - 2.0 * a * xu, c = xu - a * xu * xu - b * xu;
                return a * out * out + b * out + c;
            }
            return lo;
        }
        if (out > hi - r) {
            if (out < hi + r) {
                const T a = (-1.0 / 4.0) / r, xu = hi + r, b = -2.0 * a * xu, c = hi - a * xu * xu - b * xu;
                return a * out * out + b * out + c;
            }
            return hi;
        }
        return out;
    }
    case AKind::PWL: {
        const auto &x = m.x, &y = m.y;
        const size_t n = x.size();
        const T &v = in[0];
        if (v <= (x[0] + x[1]) / 2.0) return y[0] + (v - x[0]) * ((y[1] - y[0]) / (x[1] - x[0]));
        if (v >= (x[n - 2] + x[n - 1]) / 2.0) return y[n - 1] + (v - x[n - 1]) * ((y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]));
        for (size_t i = 1; i + 1 < n; ++i) {
            if (!(v < (x[i] + x[i + 1]) / 2.0)) continue;
            const double lseg = x[i] - x[i - 1], useg = x[i + 1] - x[i];
            const double dom = m.fraction ? m.domain * std::min(lseg, useg) : m.domain;
            const double sl = (y[i] - y[i - 1]) / lseg, su = (y[i + 1] - y[i]) / useg;
            if (v < x[i] - dom) return y[i] + (v - x[i]) * sl;
            if (v < x[i] + dom) return smoothCorner(v, x[i], y[i], dom, sl, su);
            return y[i] + (v - x[i]) * su;
        }
        return y[n - 1];  // unreachable: the last midpoint test above
    }
    case AKind::ASWITCH: {
        using std::exp;
        const double ron = std::max(m.rOn, 1e-3);
        T r;
        if (m.logR) r = ron * exp(std::log(m.rOff / ron) / (m.cntlOn - m.cntlOff) * (m.cntlOn - in[0]));
        else r = in[0] * ((ron - m.rOff) / (m.cntlOn - m.cntlOff)) + (m.rOff * m.cntlOn - ron * m.cntlOff) / (m.cntlOn - m.cntlOff);
        if (m.limitR) { if (r < ron) r = T(ron); if (r > m.rOff) r = T(m.rOff); }
        else if (r <= 1e-9) r = T(1e-9);
        return in[1] / r;
    }
    }
    return T(0.0);
}

}  // namespace xspice

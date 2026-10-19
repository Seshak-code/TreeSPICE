#pragma once
// XSPICE analog code models without state, as ngspice 47's cfunc.mod files compute them (icm/analog,
// icm/xtradev). Each output is a behavioural source (Behavioural::cm) over the instance's inputs; its
// Jacobian comes from Dual, the exact derivative of the value ngspice computes (ngspice's own partials
// only steer its Newton iteration).
#include <cmath>
#include <complex>
#include <string>
#include <type_traits>
#include <vector>

namespace xspice {

enum class AKind { GAIN, SUMMER, MULT, DIVIDE, LIMIT, CLIMIT, PWL, ASWITCH, INT, D_DT, S_XFER, SLEW, HYST, SINE, TRIANGLE, SQUARE };

// A code model's state at the last accepted time point (int: its limited output; d_dt: its input; s_xfer:
// its integrator chain, lowest first; slew: input and output; hyst: its leg, -1 before the first evaluation)
struct AnalogState { double out = 0.0, in = 0.0; std::vector<double> v; int leg = -1; };
inline bool oscillator(AKind k) { return k == AKind::SINE || k == AKind::TRIANGLE || k == AKind::SQUARE; }
inline bool stateful(AKind k) {
    return k == AKind::INT || k == AKind::D_DT || k == AKind::S_XFER || k == AKind::SLEW || k == AKind::HYST || oscillator(k);
}
inline bool acFunction(AKind k) { return k == AKind::INT || k == AKind::D_DT || k == AKind::S_XFER; }  // AC gain not the DC slope

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
    // int: its output at t = 0
    double outIc = 0.0;
    // slew: rise / fall slopes [V/s]; hyst: input breakpoints, hysteresis, output limits, smoothing domain
    double riseSlope = 1e9, fallSlope = 1e9, inLow = 0.0, inHigh = 1.0, hyst = 0.1;
    // sine / triangle / square: frequency against control (pwl), output levels, duty cycle, square's edges
    std::vector<double> cntl{0.0, 1.0}, freq{1e3, 2e3};
    double outLow = -1.0, outHigh = 1.0, duty = 0.5, tRise = 1e-9, tFall = 1e-9;
    // s_xfer: numerator and denominator in ascending powers of s (frequency-scaled, the top denominator
    // coefficient divided out into gain), the integrators' initial values (lowest first)
    std::vector<double> num, den, intIc;
};

// s_xfer's controller-canonical chain one step on (ngspice cm_s_xfer): the top integrator takes the input less
// the fed-back accepted states, each lower one integrates the one above by backward Euler over h; at the
// operating point (h = 0) the integrators hold their state (cm_analog_integrate does nothing in DC)
template <typename T>
std::vector<T> sxferChain(const AnalogModel &m, const T &in, const AnalogState &st, double h) {
    const size_t n = m.den.size();
    std::vector<T> s(n);
    for (size_t i = 0; i + 1 < n; ++i) s[i] = T(st.v[i]);
    T top = m.gain * (in + m.inOffset);
    for (size_t i = 0; i + 1 < n; ++i) top = top - st.v[i] * m.den[i];
    s[n - 1] = top;
    if (h > 0.0)
        for (size_t i = n - 1; i > 0; --i) s[i - 1] = st.v[i - 1] + h * s[i];
    return s;
}

// cm_smooth_corner: the parabola through the corner (xc, yc) of slopes s0 and s1 over xc - d .. xc + d
template <typename T>
inline T smoothCorner(const T &x, double xc, double yc, double d, double s0, double s1) {
    const double xu = xc + d, yu = yc + s1 * d, a = (s1 - s0) / 4.0 / d, b = s1 - 2.0 * a * xu, c = yu - a * xu * xu - b * xu;
    return a * x * x + b * x + c;
}

// int / d_dt: ngspice's limit with cm_smooth_corner's corners at lo / hi (range r) on the raw output
template <typename T>
inline T smoothLimits(const T &out, double lo, double hi, double r) {
    if (out < lo - r) return T(lo);
    if (out < lo + r) return smoothCorner(out, lo, lo, r, 0.0, 1.0);
    if (out > hi + r) return T(hi);
    if (out > hi - r) return smoothCorner(out, hi, hi, r, 1.0, 0.0);
    return out;
}

// The controlled oscillators (ngspice cm_sine, cm_triangle, cm_square) one step on, to time t from t - h
// (h = 0: the operating point). State: phase (cycles), the corner times t1..t4 of the current cycle (triangle:
// t1, t2 and the ramp's start), then the breakpoints this step asks the next to land on. The output does not
// depend on the control within a step (ngspice's partial is 0 too).
struct OscStep { double out; std::vector<double> st; };
inline OscStep oscStep(const AnalogModel &m, double cntl, const AnalogState &s, double t, double h) {
    const auto &x = m.cntl, &y = m.freq;
    const size_t n = x.size();
    double f = 0.0;
    if (cntl <= x[0]) { f = y[0] + (cntl - x[0]) * (y[1] - y[0]) / (x[1] - x[0]); if (f <= 0.0) f = 1e-16; }
    else if (cntl >= x[n - 1]) f = y[n - 1] + (cntl - x[n - 1]) * (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
    else for (size_t i = 0; i + 1 < n; ++i) if (cntl < x[i + 1] && cntl >= x[i]) f = (cntl - x[i]) / (x[i + 1] - x[i]) * (y[i + 1] - y[i]) + y[i];
    const double lo = m.outLow, hi = m.outHigh;
    if (!(h > 0.0)) {  // DC: phase 0, the corners unset (-1; the triangle's ramp from 0), the first breakpoints ahead
        if (m.kind == AKind::SINE) return {(hi + lo) / 2.0, {0.0, -1, -1, -1, -1, -1, -1, -1, -1}};
        if (m.kind == AKind::TRIANGLE) return {lo, {0.0, -1, -1, 0.0, -1, m.duty / f, 1.0 / f, -1, -1}};
        const double t1 = (1.0 - m.duty) / f;
        return {lo, {0.0, -1, -1, -1, -1, t1, t1 + m.tRise, -1, -1}};
    }
    const double t0 = t - h, phase = s.v[0] + f * h;
    double dphase = s.v[0] - (int)s.v[0];
    std::vector<double> st(9, -1.0);
    st[0] = phase;
    if (m.kind == AKind::SINE) return {(hi - lo) / 2.0 * std::sin(phase * 2.0 * M_PI) + (hi + lo) / 2.0, st};
    double time1 = s.v[1], time2 = s.v[2], time3 = s.v[3], time4 = s.v[4], out;
    if (m.kind == AKind::TRIANGLE) {
        double tStart = s.v[3];
        if (time1 <= t && t <= time2) {
            time2 = t0 + (1 - dphase) / f;
            if (t < time2) st[5] = time2;
            tStart = time2;
            out = hi - (t - time1) / (time2 - time1) * (hi - lo);
        } else {
            if (dphase > m.duty) dphase -= 1.0;
            time1 = t0 + (m.duty - dphase) / f;
            time2 = t0 + (1 - dphase) / f;
            if (t < time1 || t0 == 0.0) st[5] = time1;
            st[6] = time2;
            out = lo + (t - tStart) / (time1 - tStart) * (hi - lo);
        }
        st[1] = time1; st[2] = time2; st[3] = tStart;
        return {out, st};
    }
    if (time1 <= t && t <= time2) {
        time3 = t0 + (1 - dphase) / f; time4 = time3 + m.tFall;
        if (t < time2) st[5] = time2;
        st[6] = time3; st[7] = time4;
        out = lo + (t - time1) / (time2 - time1) * (hi - lo);
    } else if (time2 <= t && t <= time3) {
        time3 = t0 + (1.0 - dphase) / f; time4 = time3 + m.tFall;
        if (t < time3) st[5] = time3;
        st[6] = time4;
        out = hi;
    } else if (time3 <= t && t <= time4) {
        if (dphase > 1 - m.duty) dphase -= 1.0;
        time1 = t0 + ((1 - m.duty) - dphase) / f; time2 = time1 + m.tRise;
        if (t < time4) st[5] = time4;
        st[6] = time1; st[7] = time2;
        out = hi + (t - time3) / (time4 - time3) * (lo - hi);
    } else {
        if (dphase > 1 - m.duty) dphase -= 1.0;
        time1 = t0 + ((1 - m.duty) - dphase) / f; time2 = time1 + m.tRise;
        if (t < time1 || t0 == 0.0) st[5] = time1;
        st[6] = time2;
        out = lo;
    }
    st[1] = time1; st[2] = time2; st[3] = time3; st[4] = time4;
    return {out, st};
}

// hyst (ngspice cm_hyst): the leg the input is on, rising (0) or falling (1), and the output on it; a leg ends
// past its far corner. The leg comes from the last accepted point, or before the first one from the input.
template <typename T>
inline int hystLeg(const AnalogModel &m, const T &in, const AnalogState &st) {
    if (st.leg >= 0) return st.leg;
    const double dom = m.fraction ? m.domain * (m.inHigh - m.inLow) : m.domain;
    return in < m.inHigh + m.hyst + dom ? 0 : 1;
}
template <typename T>
std::pair<T, int> hystOut(const AnalogModel &m, const T &in, int leg) {
    const double slope = (m.hi - m.lo) / (m.inHigh - m.inLow), d = m.fraction ? m.domain * (m.inHigh - m.inLow) : m.domain;
    const double rl = m.inLow + m.hyst, rz = m.inHigh + m.hyst, fl = m.inHigh - m.hyst, fz = m.inLow - m.hyst;
    if (leg == 0) {
        if (in <= rl - d) return {T(m.lo), 0};
        if (in <= rl + d) return {smoothCorner(in, rl, m.lo, d, 0.0, slope), 0};
        if (in <= rz - d) return {(in - rl) * slope + m.lo, 0};
        if (in <= rz + d) return {smoothCorner(in, rz, m.hi, d, slope, 0.0), 0};
        return {T(m.hi), 1};
    }
    if (in >= fl + d) return {T(m.hi), 1};
    if (in >= fl - d) return {smoothCorner(in, fl, m.hi, d, slope, 0.0), 1};
    if (in >= fz + d) return {(in - fz) * slope + m.lo, 1};
    if (in >= fz - d) return {smoothCorner(in, fz, m.lo, d, 0.0, slope), 1};
    return {T(m.lo), 0};
}

// The output for inputs `in` (port order; aswitch: control, then the voltage across its own port). The
// stateful models read the last accepted state st and step h (0: the operating point, t = 0): int integrates
// by backward Euler, as XSPICE's cm_analog_integrate does under TRAP, d_dt differentiates over the step.
template <typename T>
T analogOut(const AnalogModel &m, const std::vector<T> &in, const AnalogState *st = nullptr, double h = 0.0, double t = 0.0) {
    switch (m.kind) {
    case AKind::SINE: case AKind::TRIANGLE: case AKind::SQUARE: {
        auto value = [](const T &v) { if constexpr (std::is_same_v<T, double>) return v; else return v.val; };
        return T(oscStep(m, value(in[0]), st ? *st : AnalogState{}, t, h).out);
    }
    case AKind::INT: {
        const T out = h > 0.0 && st ? st->out + h * m.gain * (in[0] + m.inOffset) : T(m.outIc);
        return smoothLimits(out, m.lo, m.hi, m.range);
    }
    case AKind::D_DT: {
        const T out = h > 0.0 && st ? m.gain * (in[0] - st->in) / h + m.outOffset : T(0.0);
        return smoothLimits(out, m.lo, m.hi, m.range);
    }
    case AKind::SLEW: {  // ngspice cm_slew: the output follows the input at most at the slew rates
        if (!(h > 0.0) || !st) return in[0];
        const T slope = (in[0] - st->in) / h;
        if (slope >= 0.0) {
            const double slew = st->out + m.riseSlope * h;
            if (in[0] < st->out - m.fallSlope * h) return T(st->out - m.fallSlope * h);
            if (slope > m.riseSlope || (slope < m.riseSlope && slew <= in[0])) return T(slew);
            return in[0];
        }
        const double slew = st->out - m.fallSlope * h;
        if (in[0] > st->out + m.riseSlope * h) return T(st->out + m.riseSlope * h);
        if (-slope > m.fallSlope || (-slope < m.fallSlope && slew > in[0])) return T(slew);
        return in[0];
    }
    case AKind::HYST: return hystOut(m, in[0], st ? hystLeg(m, in[0], *st) : 0).first;
    case AKind::S_XFER: {
        const auto s = sxferChain(m, in[0], *st, h);
        T out(0.0);
        for (size_t i = 0; i < m.num.size(); ++i) out = out + s[i] * m.num[i];
        return out;
    }
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

// The small-signal gain of a stateful model at angular frequency w (ngspice's AC_GAIN) beyond the slope its
// operating-point Jacobian already gives: int gain / jw, d_dt gain jw, s_xfer gain N(jw)/D(jw)
inline std::complex<double> analogAcGain(const AnalogModel &m, double w) {
    if (m.kind == AKind::S_XFER) {  // gain N(jw) / D(jw)
        const std::complex<double> s(0.0, w);
        std::complex<double> nn(0.0), dd(0.0), p(1.0);
        for (size_t i = 0; i < m.den.size(); ++i, p *= s) { if (i < m.num.size()) nn += m.num[i] * p; dd += m.den[i] * p; }
        // less the slope the operating point's Jacobian already holds (a numerator of the denominator's order)
        return m.gain * nn / dd - (m.num.size() == m.den.size() ? m.gain * m.num.back() : 0.0);
    }
    return m.kind == AKind::INT ? std::complex<double>(0.0, -m.gain / w) : std::complex<double>(0.0, m.gain * w);
}

}  // namespace xspice

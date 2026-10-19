#pragma once
// XSPICE analog code models without state, as ngspice 47's cfunc.mod files compute them (icm/analog,
// icm/xtradev). Each output is a behavioural source (Behavioural::cm) over the instance's inputs; its
// Jacobian comes from Dual, the exact derivative of the value ngspice computes (ngspice's own partials
// only steer its Newton iteration).
#include <algorithm>
#include <cmath>
#include <complex>
#include <string>
#include <type_traits>
#include <vector>

namespace xspice {

enum class AKind { GAIN, SUMMER, MULT, DIVIDE, LIMIT, CLIMIT, PWL, ASWITCH, INT, D_DT, S_XFER, SLEW, HYST, SINE, TRIANGLE, SQUARE, ZENER, MIPWL, OTA, ILIMIT, ONESHOT, POT, CAPIC, INDIC, SIDIODE, PSWITCH, MEMRISTOR, FILESRC, PWLTS, XFER };

// A code model's state at the last accepted time point (int: its limited output; d_dt: its input; s_xfer:
// its integrator chain, lowest first; slew: input and output; hyst: its leg, -1 before the first evaluation)
struct AnalogState { double out = 0.0, in = 0.0; std::vector<double> v; int leg = -1; };
inline bool oscillator(AKind k) { return k == AKind::SINE || k == AKind::TRIANGLE || k == AKind::SQUARE; }
inline bool stateful(AKind k) {
    return k == AKind::INT || k == AKind::D_DT || k == AKind::S_XFER || k == AKind::SLEW || k == AKind::HYST || k == AKind::ONESHOT ||
           k == AKind::CAPIC || k == AKind::INDIC || k == AKind::MEMRISTOR || oscillator(k);
}
inline bool acFunction(AKind k) {  // AC gain not the DC slope
    return k == AKind::INT || k == AKind::D_DT || k == AKind::S_XFER || k == AKind::CAPIC || k == AKind::INDIC || k == AKind::XFER;
}

struct AnalogModel {
    AKind kind = AKind::GAIN;
    int out = 0;  // which of the instance's outputs this source is (ota: 0 the input port, 1 the output port)
    // gain, limit, climit: in_offset, gain, out_offset; summer / mult: per input in_offset, in_gain
    double inOffset = 0.0, gain = 1.0, outOffset = 0.0, outGain = 1.0;
    std::vector<double> inOffsets, inGains;
    // divide
    double numOffset = 0.0, numGain = 1.0, denOffset = 0.0, denGain = 1.0, denLower = 1e-10, denDomain = 1e-16;
    // limit / climit: limits, smoothing range (absolute after `fraction`), climit's deltas
    double lo = 0.0, hi = 1.0, range = 1e-6, upperDelta = 0.0, lowerDelta = 0.0;
    bool fraction = false;
    // pwl, pwlts (its input is time): the breakpoints with ngspice's extra pair at either end, smoothing domain;
    // pwlts' transient breakpoints (each corner and either side of its smoothing)
    std::vector<double> x, y, bkpts;
    double domain = 0.01;
    // aswitch, pswitch (+ its control port's resistance rCntl) (ota: gain = gm, rOff = rout, rOn = rin)
    double rCntl = 1e12;
    double cntlOff = 0.0, cntlOn = 1.0, rOff = 1e12, rOn = 1.0;
    bool logR = true, limitR = false;
    // int: its output at t = 0
    double outIc = 0.0;
    // slew: rise / fall slopes [V/s]; hyst: input breakpoints, hysteresis, output limits, smoothing domain
    double riseSlope = 1e9, fallSlope = 1e9, inLow = 0.0, inHigh = 1.0, hyst = 0.1;
    // sine / triangle / square: frequency against control (pwl), output levels, duty cycle, square's edges
    std::vector<double> cntl{0.0, 1.0}, freq{1e3, 2e3};
    double outLow = -1.0, outHigh = 1.0, duty = 0.5, tRise = 1e-9, tFall = 1e-9;
    // ilimit: r_out_source / sink, i_limit_source / sink, their smoothing ranges, r_out_domain (v_pwr_range: range);
    // a NULL supply port
    double rSource = 1.0, rSink = 1.0, iSource = 10e-3, iSink = 10e-3, iSourceRange = 1e-9, iSinkRange = 1e-9, rDomain = 1e-9;
    bool posNull = false, negNull = false;
    // oneshot (x, y: cntl_array, pw_array; outLow / outHigh, tRise / tFall): clk_trig, its edge, the delays, retrig
    double trig = 0.5, rDelay = 1e-9, fDelay = 1e-9;
    bool posEdge = true, retrig = false, clearNull = false, cntlNull = false;
    // xfer: x its angular frequencies (increasing), sx its complex gains there
    std::vector<std::complex<double>> sx;
    // filesource: x the times, y the values row by row (width columns); fraction: amplstep
    int width = 1;
    // memristor: rmin, rmax, alpha, beta, vt (rinit: outIc)
    double rMin = 10.0, rMax = 1e4, mAlpha = 0.0, mBeta = 1.0, mVt = 0.0;
    // potentiometer: its r0-wiper and wiper-r1 resistances (rOn, rOff)
    // sidiode: ron, roff, rrev, vfwd, vrev, ilimit, revilimit, epsilon, revepsilon
    double sRon = 1.0, sRoff = 1.0, sRrev = 0.0, vFwd = 0.0, vRev = 1e30, iLim = 1e30, revILim = 1e30, eps = 0.0, revEps = 0.0;
    // multi_input_pwl: x, y; its gate (and, nand, or, nor)
    int gate = 0;
    // zener
    double vBreak = 0.0, iBreak = 2e-2, rBreak = 1.0, iRev = 1e-6, iSat = 1e-12, nFwd = 1.0;
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

// ngspice cm_oneshot at an evaluation (t, the step h reaching it; h = 0: the operating point, which resets it and
// makes the first transient step's previous clock 0). State: t1 .. t4 (delay, rise, width, fall ends), set, state,
// locked, the output and the clock at the last accepted point. ngspice compares the old output to out_low / out_high
// with a signed "< 1e-20", kept as is: the retrigger branch sees any output at or below out_high.
struct OneshotStep { double out; std::vector<double> st; };
inline OneshotStep oneshotStep(const AnalogModel &m, double clk, double cntl, double clear, const AnalogState &s, double t, double h) {
    const double low = m.outLow, hi = m.outHigh;
    if (h <= 0.0 || s.v.size() != 9) return {low, {-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, low, 0.0}};
    double t1 = s.v[0], t2 = s.v[1], t3 = s.v[2], t4 = s.v[3], clock = s.v[8], out;
    bool set = s.v[4] != 0.0, state = s.v[5] != 0.0, locked = s.v[6] != 0.0;
    const double outOld = s.v[7], tr = std::max(m.tRise, 1e-12), tf = std::max(m.tFall, 1e-12);
    if (!m.clearNull && clear > m.trig) {
        t1 = t2 = t3 = t4 = -1.0;
        set = locked = state = false;
        out = low;
    } else {
        const double c = m.cntlNull ? 0.0 : cntl;
        const auto &x = m.x, &y = m.y;
        const size_t n = x.size();
        double pw = 0.0;
        if (c <= x[0]) pw = std::max(0.0, y[0] + (c - x[0]) * (y[1] - y[0]) / (x[1] - x[0]));
        else if (c >= x[n - 1]) pw = y[n - 1] + (c - x[n - 1]) * (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
        else for (size_t i = 0; i + 1 < n; ++i) if (c < x[i + 1] && c >= x[i]) pw = (c - x[i]) / (x[i + 1] - x[i]) * (y[i + 1] - y[i]) + y[i];
        const bool up = clk > clock && clk > m.trig, down = clk < clock && clk < m.trig;
        if (!set) { if (m.posEdge ? up : down) state = set = true; }
        else if (m.posEdge ? down : up) set = false;
        if (state && outOld - low < 1e-20 && !locked) {
            t1 = t + m.rDelay; t2 = t1 + tr; t3 = t2 + pw + m.fDelay; t4 = t3 + tf;
            if (!m.retrig) locked = true;
            state = false;
        } else if (state && outOld - hi < 1e-20 && !locked) {  // retriggered: the pulse ends pw after this edge
            t3 = t + pw + m.rDelay + m.fDelay + tr; t4 = t3 + tf;
            state = false;
        }
        if (state && locked) state = false;
        if (t < t1) out = low;
        else if (t < t2) out = low + (t - t1) / (t2 - t1) * (hi - low);
        else if (t < t3) out = hi;
        else if (t < t4) out = hi + (t - t3) / (t4 - t3) * (low - hi);
        else { out = low; if (!m.retrig) locked = false; }
        clock = clk;
    }
    return {out, {t1, t2, t3, t4, double(set), double(state), double(locked), out, clock}};
}

// ngspice cm_memristor's resistance after a step h from r (its state, the last accepted point): it integrates
// f1(v) = beta v + (alpha - beta)(|v + vt| - |v - vt|) / 2 by backward Euler while inside [rmin, rmax] (ngspice
// tests the bound on its last iterate; here on the accepted r); at the operating point it holds
template <typename T>
inline T memristorR(const AnalogModel &m, const T &v, double r, double h) {
    if (!(h > 0.0)) return T(r);
    using std::abs;
    const T g = m.mBeta * v + 0.5 * (m.mAlpha - m.mBeta) * (abs(v + m.mVt) - abs(v - m.mVt));
    const bool on = (g > 0.0 && r < m.rMax) || (g < 0.0 && r > m.rMin);
    return on ? r + h * g : T(r);
}

// A value carrying a given slope against x (a model whose own partial ngspice uses as its AC gain)
template <typename T>
inline T withSlope(const T &x, double value, double slope) {
    if constexpr (std::is_same_v<T, double>) return value;
    else { T r(value); r.grad = slope * x.grad; return r; }
}

// ngspice cm_sidiode: off / on / reverse conductances, optional quadratic overlaps (epsilon) and tanh current limits;
// returns {current, ngspice's partial} (in the tanh regions the partial is ngspice's g (1 - tanh^2), not exact)
inline std::pair<double, double> sidiode(const AnalogModel &m, double v) {
    const double Vrev = -m.vRev, Vfwd = m.vFwd, Ilim = m.iLim, Revilim = -m.revILim;
    const double goff = 1.0 / m.sRoff, gon = 1.0 / m.sRon, grev = m.sRrev == 0.0 ? gon : 1.0 / m.sRrev;
    const double Va = Vrev - m.revEps, Vb = Vrev, Vc = Vfwd, Vd = Vfwd + m.eps;
    const bool revepsi = m.revEps > 0.0, epsi = m.eps > 0.0, ili = Ilim < 1e29, revili = Revilim > -1e29;
    const double a1 = revepsi ? (goff - grev) / m.revEps : 0.0, a2 = grev / Revilim, b1 = epsi ? (gon - goff) / m.eps : 0.0, b2 = gon / Ilim;
    if (v < Va) {
        if (revili) {
            const double t = std::tanh(a2 * (v - Va)), ia = goff * Va + 0.5 * (Va - Vb) * (Va - Vb) * a1;
            return {(Revilim - ia) * t + ia, grev * (1.0 - t * t)};
        }
        return {Vrev * goff + (v - Vrev) * grev, grev};
    }
    if (revepsi && v < Vb) return {0.5 * (v - Vb) * (v - Vb) * a1 + v * goff, (v - Vb) * a1 + goff};
    if (v >= Vb && v < Vc) return {v * goff, goff};
    if (epsi && v >= Vc && v < Vd) return {0.5 * (v - Vc) * (v - Vc) * b1 + v * goff, (v - Vc) * b1 + goff};
    if (ili) {
        const double t = std::tanh(b2 * (v - Vd)), id = goff * Vd + 0.5 * (Vd - Vc) * (Vd - Vc) * b1;
        return {(Ilim - id) * t + id, gon * (1.0 - t * t)};
    }
    return {Vfwd * goff + (v - Vfwd) * gon, gon};
}

// cm_smooth_discontinuity: y_lower below x_lower, y_upper above x_upper, two parabolas meeting at the centre
template <typename T>
inline T smoothDiscontinuity(const T &x, double xl, double yl, double xu, double yu) {
    const double xc = (xu + xl) / 2.0, yc = (yu + yl) / 2.0, cs = 2.0 * (yu - yl) / (xu - xl);
    if (x < xl) return T(yl);
    if (x < xc) { const double a = cs / (xu - xl), b = cs - 2.0 * a * xc, c = yc - a * xc * xc - b * xc; return a * x * x + b * x + c; }
    if (x < xu) { const double a = -cs / (xu - xl), b = -2.0 * a * xu, c = yu - a * xu * xu - b * xu; return a * x * x + b * x + c; }
    return T(yu);
}

// cm_climit_fcn's value: out limited to [lo, hi], cm_smooth_corner's parabolas over r about each limit
template <typename T>
inline T climitFcn(const T &out, const T &lo, const T &hi, const T &r) {
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
    case AKind::ZENER: {  // ngspice cm_zener: forward diode, a linear region, the reverse breakdown, a leakage
        using std::exp; using std::log;
        const T &v = in[0];
        const double vt = 0.026, nvt = m.nFwd * vt, v12 = nvt * std::log(nvt / 10.0), k = 1.0 / m.iBreak / m.rBreak;
        const double v23 = -m.vBreak + std::log(10.0 / m.iSat / m.rBreak) / k;
        T i;
        if (v23 < v12) {  // three segments
            const double i0 = 1e-6, v0 = -m.vBreak + 1.0 / k * std::log(m.iBreak / i0);
            const double ord = m.iSat * (std::exp(v12 / nvt) - 1.0), a = m.iSat / 10.0, b = ord - a * v12;
            const double c = a * v23 + b + i0 * std::exp(-k * (v23 - v0));
            if (v >= v12) i = m.iSat * (exp(v / nvt) - 1.0);
            else if (v >= v23) i = a * v + b;
            else i = -i0 * exp(-k * (v - v0)) + c;
        } else {          // two segments
            const double i0 = m.iBreak / (std::exp(k * m.vBreak) - 1.0), s1 = m.iSat / nvt, s2 = i0 * k;
            if (v >= 0.0) { i = m.iSat * (exp(v / nvt) - 1.0); if (s2 - s1 > 0.0) i = i + (s2 - s1) * v; }
            else { i = -i0 * (exp(-k * v) - 1.0); if (s1 - s2 > 0.0) i = i + (s1 - s2) * v; }
        }
        return i + m.iRev / m.vBreak * v;
    }
    case AKind::OTA:  // ngspice cm_ota: the input port draws v_in / rin, the output port gm v_in + v_out / rout
        return m.out == 0 ? in[0] / m.rOn : m.gain * in[0] + in[1] / m.rOff;
    case AKind::ONESHOT: {  // ngspice's partials are 0 (its AC gain too): the value alone
        auto value = [](const T &v) { if constexpr (std::is_same_v<T, double>) return v; else return v.val; };
        return T(oneshotStep(m, value(in[0]), value(in[1]), value(in[2]), st ? *st : AnalogState{}, t, h).out);
    }
    case AKind::CAPIC: case AKind::INDIC:
        // ngspice cm_capacitor (an hd port: its voltage integrates its current / c) / cm_inductor (gd: its current
        // integrates its voltage / l), by cm_analog_integrate's backward Euler; at the operating point they hold ic
        // (the inductor a current source, not a short)
        return h > 0.0 && st ? st->out + h * in[0] / m.gain : T(m.outIc);
    case AKind::XFER: return m.sx[0].real() * in[0];  // ngspice cm_xfer: DC is the first entry's real part
    case AKind::FILESRC: {
        // ngspice cm_filesource at time t, output m.out: between rows linear (amplstep: the earlier row's, up to and
        // including the later row's time); before the first row the second row's value; past the last row 0 (ngspice
        // reads past its data there, zero in practice). No inputs, no partials: AC 0.
        const auto &x = m.x;
        const size_t n = x.size(), w = m.width, j = m.out;
        if (t > x[n - 1]) return T(0.0);
        size_t k = 0;
        while (k + 2 < n && t > x[k + 1]) ++k;
        const double y0 = m.y[k * w + j], y1 = m.y[(k + 1) * w + j];
        if (t < x[k]) return T(y1);
        if (m.fraction) return T(y0);
        const double a = (x[k + 1] - t) / (x[k + 1] - x[k]);
        return T(a * y0 + (1.0 - a) * y1);
    }
    case AKind::MEMRISTOR: return in[0] / memristorR(m, in[0], st ? st->out : m.outIc, h);
    case AKind::PSWITCH: {
        // ngspice cm_pswitch: refs control, out; outputs the control port's current (in / r_cntl_in), the switch's
        // (v / r). r_on is at least 1e-3, r_off at most 1e12; log: a cubic in the normalised control between
        // ln(r_off) and ln(r_on) (r_on at least); linear: r interpolates, at least 1e-9. Partials automatic (exact).
        if (m.out == 0) return in[0] / m.rCntl;
        const double ron = std::max(m.rOn, 1e-3), roff = std::min(m.rOff, 1e12);
        double on = m.cntlOn, off = m.cntlOff;
        if (std::abs(on - off) < 1e-12) { on += 0.001; off -= 0.001; }
        const T &c = in[0];
        T r;
        if (m.logR) {
            const double L = std::log(ron / roff), lm = std::log(std::sqrt(ron * roff)), d = on > off ? 1.0 : -1.0;
            const double c1 = 1.5 * L / d, c3 = 2.0 * L / (d * d * d);
            const bool rising = on > off;
            if (rising ? c > on : c < on) r = T(ron);
            else if (rising ? c < off : c > off) r = T(roff);
            else {
                const T im = (rising ? (c - off) : (on - c)) / (on - off) - 0.5;
                using std::exp;
                r = exp(lm + c1 * im - c3 * im * im * im);
                if (r < ron) r = T(ron);
            }
        } else {
            const double d = on - off, k = (ron - roff) / d;
            if (d >= 0 ? c < off : c > off) r = T(roff);
            else if (d >= 0 ? c > on : c < on) r = T(ron);
            else r = c * k + (roff * on - ron * off) / d;
            if (r <= 1e-9) r = T(1e-9);
        }
        return in[1] / r;
    }
    case AKind::SIDIODE: {
        auto value = [](const T &v) { if constexpr (std::is_same_v<T, double>) return v; else return v.val; };
        const auto [i, g] = sidiode(m, value(in[0]));
        return withSlope(in[0], i, g);
    }
    case AKind::POT:  // ngspice cm_potentiometer: refs r0, wiper, r1; outputs r0, wiper, r1 (each port's current in)
        if (m.out == 0) return (in[0] - in[1]) / m.rOn;
        if (m.out == 2) return (in[2] - in[1]) / m.rOff;
        return (in[1] - in[0]) / m.rOn + (in[1] - in[2]) / m.rOff;
    case AKind::MIPWL: {
        // ngspice cm_multi_input_pwl: the smallest input (and, nand) or largest (or, nor) through the x-y table,
        // nand / nor read it backwards with the slope never set (a staircase). No partials: ngspice's Jacobian,
        // hence its AC gain, is 0, so the value is returned without its derivative.
        auto value = [](const T &v) { if constexpr (std::is_same_v<T, double>) return v; else return v.val; };
        double c = value(in[0]);
        for (const auto &v : in) c = m.gate < 2 ? std::min(c, value(v)) : std::max(c, value(v));
        const size_t n = m.x.size();
        const bool rev = m.gate == 1 || m.gate == 3;
        if (c <= m.x[0]) return T(rev ? m.y[n - 1] : m.y[0]);
        if (c >= m.x[n - 1]) return T(rev ? m.y[0] : m.y[n - 1]);
        for (size_t i = 1; i < n; ++i)
            if (c > m.x[i - 1] && c <= m.x[i]) {
                if (rev) return T(m.y[n - i - 1]);
                const double s = (m.y[i] - m.y[i - 1]) / (m.x[i] - m.x[i - 1]);
                return T(s * c + (m.y[i] - s * m.x[i]));
            }
        return T(0.0);
    }
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
        return climitFcn(m.gain * (m.inOffset + in[0]), lo, hi, m.fraction ? m.range * (hi - lo) : T(m.range));
    }
    case AKind::ILIMIT: {
        // ngspice cm_ilimit: refs in, pos_pwr, neg_pwr, out (a NULL supply reads +-1e6 V and draws nothing). The output
        // heads for the input limited between the supplies through r_out, its current limited; the supplies source /
        // sink it, smoothed near 0. As ngspice, the upper corner's smoothing is i_sink_range, and the supply current's
        // smoothing near 0 ends with a step of its range at twice it.
        const T pos = m.posNull ? T(1e6) : in[1], neg = m.negNull ? T(-1e6) : in[2], &vout = in[3];
        const T veq = climitFcn(m.gain * (m.inOffset + in[0]), neg, pos, T(m.range));
        const T rout = m.rSource == m.rSink ? T(m.rSource) : smoothDiscontinuity(veq - vout, -m.rDomain, m.rSink, m.rDomain, m.rSource);
        T iout = (veq - vout) / rout, ipos(0.0), ineg(0.0);
        if (iout < 0.0) {
            if (iout < -m.iSink + m.iSinkRange) {
                iout = iout < -m.iSink - m.iSinkRange ? T(-m.iSink) : smoothCorner(iout, -m.iSink, -m.iSink, m.iSinkRange, 0.0, 1.0);
                ineg = -iout;
            } else ineg = iout > -2.0 * m.iSinkRange ? -smoothCorner(iout, -m.iSinkRange, 0.0, m.iSinkRange, 1.0, 0.0) : -iout;
        } else {
            if (iout > m.iSource - m.iSourceRange) {
                iout = iout > m.iSource + m.iSourceRange ? T(m.iSource) : smoothCorner(iout, m.iSource, m.iSource, m.iSinkRange, 1.0, 0.0);
                ipos = -iout;
            } else ipos = iout < 2.0 * m.iSourceRange ? -smoothCorner(iout, m.iSourceRange, 0.0, m.iSourceRange, 0.0, 1.0) : -iout;
        }
        return m.out == 0 ? -ipos : m.out == 1 ? -ineg : -iout;
    }
    case AKind::PWL: case AKind::PWLTS: {
        const auto &x = m.x, &y = m.y;
        const size_t n = x.size();
        const T v = m.kind == AKind::PWLTS ? T(t) : in[0];
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
    if (m.kind == AKind::XFER) {  // the table interpolated linearly in w (held past its ends), less the DC slope
        const auto &f = m.x;
        const size_t n = f.size();
        std::complex<double> g;
        if (w <= f[0]) g = m.sx[0];
        else if (w >= f[n - 1]) g = m.sx[n - 1];
        else {
            size_t i = 1;
            while (!(f[i] > w)) ++i;
            g = m.sx[i - 1] + (w - f[i - 1]) / (f[i] - f[i - 1]) * (m.sx[i] - m.sx[i - 1]);
        }
        return g - m.sx[0].real();
    }
    // capacitoric: an impedance 1/(jwc) on its current; inductoric: an admittance 1/(jwl) on its voltage. ngspice 47
    // gives inductoric jwl, which is no inductor's (a shunt one then acts as a series one): a deliberate divergence
    if (m.kind == AKind::CAPIC || m.kind == AKind::INDIC) return std::complex<double>(0.0, -1.0 / (w * m.gain));
    return m.kind == AKind::INT ? std::complex<double>(0.0, -m.gain / w) : std::complex<double>(0.0, m.gain * w);
}

}  // namespace xspice

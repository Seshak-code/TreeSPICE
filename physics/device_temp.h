#pragma once
#include <cmath>
#include <type_traits>
#include "physics_constants.h"

// Temperature laws shared by the diode, BJT and MOS level 1 (ngspice diotemp / bjttemp / mos1temp).
// The parser applies them once per circuit temperature, so the device models run at their final values; a
// self-heating diode runs the diode's (diodeAt) at its junction temperature, T a Dual there.
namespace devtemp {

constexpr double REFTEMP = 300.15;  // ngspice: 27 C

template <class T> inline T vt(T t) { return PhysicsConstants::K_BOLTZMANN * t / PhysicsConstants::Q_ELECTRON; }

// Band gap [eV]: silicon's, or a diode's TLEV = 2 law EG - GAP1 T^2 / (T + GAP2) (ngspice diotemp)
struct Gap { double eg = 1.16, gap1 = 7.02e-4, gap2 = 1108.0; };
template <class T> inline T egfet(T t, const Gap &g = {}) { return g.eg - g.gap1 * t * t / (t + g.gap2); }

// Junction potential shift (ngspice pbfact)
template <class T> inline T pbfact(T t, const Gap &g = {}) {
    using std::log;
    const double k = PhysicsConstants::K_BOLTZMANN;
    const T arg = -egfet(t, g) / (2 * k * t) + 1.1150877 / (k * (REFTEMP + REFTEMP));
    return -2.0 * vt(t) * (1.5 * log(t / REFTEMP) + PhysicsConstants::Q_ELECTRON * arg);
}

// 1 + c1 dt + c2 dt^2: the TC1/TC2 style coefficient law
template <class T> inline T poly(double c1, double c2, T dt) { return 1.0 + c1 * dt + c2 * dt * dt; }

// Saturation current factor exp((T/Tn - 1) EG/(N Vt) + XTI/N ln(T/Tn)) (diode; the BJT passes N = 1)
template <class T> inline T isFactor(T t, double Tn, double eg, double xti, double n) {
    using std::exp; using std::log;
    return exp((t / Tn - 1.0) * eg / (n * vt(t)) + xti / n * log(t / Tn));
}

// A junction's potential pot (at Tn) moved to T, and the factor its zero-bias capacitance gets, for grading
// mj (ngspice's pbo / gma law, the same in diotemp, bjttemp and mos1temp).
template <class T> inline T junctionAt(T &pot, T mj, double Tn, T t, const Gap &g = {}) {
    const double fact1 = Tn / REFTEMP;
    const T fact2 = t / REFTEMP;
    const T pbo = (pot - pbfact(Tn, g)) / fact1, gmaold = (pot - pbo) / pbo;
    pot = pbfact(t, g) + fact2 * pbo;
    const T gmanew = (pot - pbo) / pbo;
    return (1.0 + mj * (4e-4 * (t - REFTEMP) - gmanew)) / (1.0 + mj * (4e-4 * (Tn - REFTEMP) - gmaold));
}

// A diode model's temperature law (ngspice diotemp, probed on ngspice 47): its values at TNOM and coefficients
struct DiodeLaw {
    double tn = 300.15, eg = 1.11, xti = 3.0, keg = 1.0, xtitun = 3.0;
    int tlev = 0, tlevc = 0;
    Gap gap;                                         // TLEV 2's band gap (else silicon's)
    double tm1 = 0, tm2 = 0, ttt1 = 0, ttt2 = 0, trs1 = 0, trs2 = 0, tcv = 0, cta = 0, tpb = 0, ctp = 0, tphp = 0;
    double n = 1, ns = 1, nr = 2, ntun = 30, mjsw = 0.33, fc = 0.5;
    bool bvGiven = false;
    double is = 1e-14, m = 0.5, cjo = 0, vj = 1, cjsw = 0, php = 1, jsw = 0, isr = 0, jtun = 0, jtunsw = 0;  // at TNOM
    double tt = 0, rs = 0, rsw = 0, bv = 0;
};
// The temperature-dependent values at T
template <class P> struct DiodeAt { P vt, is, m, cjo, vj, cjsw, php, jsw, isr, jtun, jtunsw, tt, rs, rsw, bv; };

template <class T> DiodeAt<T> diodeAt(const DiodeLaw &l, T t) {
    using std::exp; using std::log;
    DiodeAt<T> a;
    const T dt = t - l.tn;
    a.vt = vt(t);
    a.is = T(l.is); a.cjo = T(l.cjo); a.vj = T(l.vj); a.cjsw = T(l.cjsw); a.php = T(l.php);
    a.jsw = T(l.jsw); a.isr = T(l.isr); a.jtun = T(l.jtun); a.jtunsw = T(l.jtunsw);
    a.m = l.m * poly(l.tm1, l.tm2, dt);
    if (a.m > 0.9) a.m = T(0.9);
    // a card at TNOM stays as given (the double path: the laws' rounding at dt = 0); a Dual carries the slopes there
    if (!std::is_same_v<T, double> || t != l.tn) {
        // saturation-current law for emission coefficient n, band gap scale keg and exponent x
        auto law = [&](double n, double keg, double x) -> T {
            if (l.tlev != 2) return isFactor(t, l.tn, keg * l.eg, x, n);
            return exp(keg * egfet(l.tn, l.gap) / (n * vt(l.tn)) - keg * egfet(t, l.gap) / (n * vt(t)) + x / n * log(t / l.tn));
        };
        a.is = a.is * law(l.n, 1.0, l.xti);
        if (l.tlevc == 1) {
            a.cjo = a.cjo * (1.0 + l.cta * dt); a.vj = a.vj - l.tpb * dt;
            a.cjsw = a.cjsw * (1.0 + l.ctp * dt); a.php = a.php - l.tphp * dt;
        } else {
            a.cjo = a.cjo * junctionAt(a.vj, a.m, l.tn, t, l.gap);
            a.cjsw = a.cjsw * junctionAt(a.php, T(l.mjsw), l.tn, t, l.gap);
        }
        a.jsw = a.jsw * law(l.ns, 1.0, l.xti);
        a.isr = a.isr * law(l.nr, 1.0, l.xti);  // probed: the NR law
        const T ft = law(l.ntun, l.keg, l.xtitun);
        a.jtun = a.jtun * ft; a.jtunsw = a.jtunsw * ft;
    }
    if (l.fc * a.vj > 1.0) a.vj = T(1.0 / l.fc);  // diotemp: the depletion knee stays below 1 V
    a.tt = l.tt * poly(l.ttt1, l.ttt2, dt);
    const T fr = poly(l.trs1, l.trs2, dt);
    a.rs = l.rs * fr; a.rsw = l.rsw * fr;
    a.bv = T(l.bv);
    if (l.bvGiven) a.bv = l.tlev != 0 ? l.bv * (1.0 - l.tcv * dt) : l.bv - l.tcv * dt;
    return a;
}

} // namespace devtemp

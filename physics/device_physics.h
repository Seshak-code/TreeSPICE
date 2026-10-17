#pragma once
#include "gui_boundary_guard.h"
#include "physics_constants.h"
#include <cmath>
#include <algorithm>
#include <array>

/*
 * device_physics.h - Pure Constitutive Equations for Semiconductor Devices
 *
 * This file contains "kernel-ready" physics functions. They are:
 * - Stateless (no side effects)
 * - Inline (for CPU performance / GPU transpilation)
 * - Bit-accurate (no approximations unless specified)
 * - Templated (for Automatic Differentiation / Dual Numbers)
 *
 * These functions can be called from:
 * 1. The MNA stamping logic (to linearize devices for NR)
 * 2. The KCL Residual Auditor (to verify physical conservation)
 * 3. Future GPU compute shaders (WGSL/GLSL)
 */

// ============================================================================
// DIODE (PN JUNCTION) PHYSICS
// ============================================================================

template <typename T>
struct DiodeParams 
{
    T Is;  // Saturation Current [A] at Tnom
    T N;   // Emission Coefficient (1.0 for ideal)
    T Vt;  // Thermal Voltage [V] (kT/q = 25.865 mV at 27 C)
    T Rs;   // Series Parasitic Resistance [Ω]
    T gmin = PhysicsConstants::JUNCTION_GMIN;  // GMIN across the junction (.options gmin)
};

/**
 * diode_current
 * Exact Shockley Equation: I = Is * (exp(Vd / (N*Vt)) - 1)
 * Exact Shockley Equation including Series Resistance iteration would require implicit solve.
 * Here we assume Rs is handled by external stamps (StampedResistor).
 */
template <typename T, typename P>
inline T diode_current(T vd, const DiodeParams<P>& p) 
{
    // Note: Implicit type promotion via operator overloads in dual.h
    using std::exp;
    return p.Is * (exp(vd / (p.N * p.Vt)) - 1.0);
}

/**
 * diode_model_pair
 * Returns {Current (I), Conductance (G)}
 * Handles linear continuation for convergence.
 */
template <typename T, typename P>
inline std::pair<T, T> diode_model_pair(T vd, const DiodeParams<P>& p, double max_arg = PhysicsConstants::JUNCTION_EXP_ARG_MAX) 
{
    // NASA Optimization: Keep params as scalars where possible
    auto n_vt = p.N * p.Vt;
    T arg = vd / n_vt;
    // ngspice GMIN: a 1e-12 S conductance in parallel with every PN junction —
    // never a node-to-ground leak (the solver's node gmin is dropped at
    // convergence; ngspice has none once converged). It lives here in the
    // junction model so the SoA kernels, the AoS stamps and both KCL audits
    // all carry it consistently.
    T GMIN = T(p.gmin);

    // Linear Continuation Region
    if (arg > T(max_arg)) {
        using std::exp;
        double exp_max_d = std::exp(max_arg);
        T exp_max = T(exp_max_d);
        
        T i_max = p.Is * (exp_max - T(1.0));
        T g_max = (p.Is / n_vt) * exp_max;
        
        T delta = vd - (n_vt * T(max_arg));
        return {i_max + delta * g_max + GMIN * vd, g_max + GMIN};
    }
    
    using std::exp;
    T exp_val = exp(arg);
    T i_val = p.Is * (exp_val - T(1.0)) + GMIN * vd;
    // Additive (g_base >= 0, so g >= GMIN without the old floor)
    T g_val = (p.Is / n_vt) * exp_val + GMIN;
    
    return {i_val, g_val};
}

/**
 * diode_current_clamped
 * Safe version for NR iteration. Clamps exponent to prevent inf/nan.
 */
template <typename T, typename P>
inline T diode_current_clamped(T vd, const DiodeParams<P>& p, double max_arg = PhysicsConstants::JUNCTION_EXP_ARG_MAX) 
{
    // Delegate to unified model
    return diode_model_pair(vd, p, max_arg).first;
}

/**
 * pnjlim - ngspice DEVpnjlim (devsup.c), exactly: a forward step past vcrit
 * larger than 2 vt is compressed logarithmically; a reverse step is limited to
 * 2*vold - 1 (or -vold - 1 from forward bias). The engine's earlier variant
 * skipped limiting above the exponent cap and capped reverse steps at 5 vt,
 * which let large BJT circuits (a 555 timer) diverge where ngspice converges.
 */
template <typename T>
inline T pnjlim(T vnew, T vold, T vt, T vcrit) {
    using std::abs; using std::log;
    if (vnew > vcrit && abs(vnew - vold) > vt + vt) {
        if (vold > T(0.0)) {
            const T arg = T(1.0) + (vnew - vold) / vt;
            return arg > T(0.0) ? vold + vt * log(arg) : vcrit;
        }
        return vt * log(vnew / vt);
    }
    if (vnew < T(0.0)) {
        const T arg = vold > T(0.0) ? -vold - T(1.0) : T(2.0) * vold - T(1.0);
        if (vnew < arg) return arg;
    }
    return vnew;
}

/**
 * compute_vcrit - Compute critical voltage for pnjlim
 * vcrit = Vt * ln(Vt / (sqrt(2) * Is))
 */
template <typename T>
inline T compute_vcrit(T vt, T Is) {
    using std::log;
    using std::sqrt;
    return vt * log(vt / (sqrt(T(2.0)) * Is));
}

// NOTE: Manual derivative functions (conductance) removed.
// Use Dual<double> with diode_current to get {I, G}.

// ============================================================================
// MOSFET (LEVEL 1 SHICHMAN-HODGES) PHYSICS
// ============================================================================

// SPICE level 1 (ngspice mos1load.c, DC): source/drain exchange when Vds < 0,
// body effect (GAMMA, PHI), channel-length modulation, and the bulk-drain /
// bulk-source junctions (diode_model_pair: same GMIN and exp cap as every
// junction). Returns the currents flowing INTO the device at drain, source and
// bulk (the gate draws none). The one MOSFET definition: stamps (via Dual),
// KCL audits, sensitivity and noise all evaluate this function.
template <typename P>
struct Mos1Params {
    P polarity = 1.0;   // +1 NMOS, -1 PMOS
    P beta = 2e-5;      // KP * W / (L - 2 LD) * M
    P vto = 0.0;        // VTO as on the card (PMOS negative)
    P lambda = 0.0, gamma = 0.0, phi = 0.6;
    P isat = 1e-14;     // bulk junction saturation current, IS * M
    P vt = PhysicsConstants::V_THERMAL_NOM;
    // Charge storage (mos1_charges), scaled by W, Leff and M like ngspice mos1temp
    bool caps = false;                       // any capacitance below is non-zero
    P cox = 0.0;                             // Meyer: COX * W * Leff * M (0 without TOX)
    P cgso = 0.0, cgdo = 0.0, cgbo = 0.0;    // overlap: CGSO*W*M, CGDO*W*M, CGBO*Leff*M
    P czbd = 0.0, czbdsw = 0.0;              // bulk-drain zero-bias bottom, sidewall
    P czbs = 0.0, czbssw = 0.0;              // bulk-source
    P mj = 0.5, mjsw = 0.5, pb = 0.8, fc = 0.5;
    P isats = -1.0;     // source junction saturation current; < 0: same as the drain's (isat)
    P gmin = PhysicsConstants::JUNCTION_GMIN;  // bulk junction GMIN
    // Levels 2 and 3 (ngspice mos2load / mos3load); phi, pb and vt are the instance-temperature values,
    // beta, cox and the oxide capacitance include M, Weff and Leff.
    int level = 1;
    P vbi = 0.0;        // tVbi as on the card (PMOS negative-side convention: polarity * vbi in the type domain)
    P leff = 1e-6, weff = 1e-6, coxf = 3.45e-4;  // Leff, Weff, oxide capacitance per area
    P mu = 0.0;         // low-field surface mobility [m^2/Vs] (level 2: the card's, level 3: at temperature)
    P nfs = 0.0, nsub = 0.0, xd = 0.0, xj = 0.0, ld = 0.0, delta = 0.0;  // NFS [1/cm^2], NSUB given, depletion width coefficient
    P ucrit = 1e4, uexp = 0.0, vmax = 0.0, neff = 1.0;                    // level 2
    P eta = 0.0, theta = 0.0, kappa = 0.2, alpha = 0.0;                   // level 3
};


// Channel of levels 2 and 3 in the type domain, source/drain already exchanged (lvgs, lvbs, lvds >= 0 as
// ngspice mos2load / mos3load): drain current without the junctions, and the von / vdsat the Meyer
// capacitances use. Only the values are ported; the Jacobian comes from Dual.
template <typename T> struct MosChannel { T id, von, vdsat; };

constexpr double EPSSIL = 11.7 * 8.854214871e-12;

template <typename T, typename P>
inline MosChannel<T> mos2_channel(T lvgs, T lvbs, T lvds, const Mos1Params<P>& p)
{
    using std::sqrt; using std::exp; using std::log; using std::atan; using std::cos; using std::fabs;
    auto mx = [](T a, double b) { return a > b ? a : T(b); };
    const P sphi = sqrt(p.phi), sphi3 = p.phi * sphi, L = p.leff;
    const bool nfs = p.nfs != P(0.0) && p.cox != P(0.0);
    const T phiMinVbs = p.phi - lvbs;
    T sarg, dsrgdb, barg, dbrgdb;
    if (lvbs <= P(0.0)) { sarg = sqrt(phiMinVbs); dsrgdb = -0.5 / sarg; }
    else { sarg = sphi / (1.0 + 0.5 * lvbs / p.phi); dsrgdb = -0.5 * sarg * (sarg / sphi3); }
    if (lvbs - lvds <= P(0.0)) { barg = sqrt(phiMinVbs + lvds); dbrgdb = -0.5 / barg; }
    else { barg = sphi / (1.0 + 0.5 * (lvbs - lvds) / p.phi); dbrgdb = -0.5 * barg * (barg / sphi3); }
    const P factor = 0.25 * M_PI * p.delta * EPSSIL / (p.coxf * p.weff), eta = 1.0 + factor;  // ngspice 47: the oxide capacitance here has no M
    const T vbin = p.polarity * p.vbi + factor * phiMinVbs;
    T gamasd = p.gamma, dgddvb = 0.0;
    if (p.gamma > P(0.0) || p.nsub > P(0.0)) {
        T argss = 0.0, argsd = 0.0, dbargs = 0.0, dbargd = 0.0, args = 1.0, argd = 1.0;
        if (p.xj > P(0.0)) {
            const P tmp = 2.0 / p.xj;
            args = sqrt(1.0 + p.xd * sarg * tmp); argd = sqrt(1.0 + p.xd * barg * tmp);
            argss = 0.5 * p.xj / L * (args - 1.0); argsd = 0.5 * p.xj / L * (argd - 1.0);
            dbargs = 0.5 / L * (p.xd * dsrgdb) / args; dbargd = 0.5 / L * (p.xd * dbrgdb) / argd;
        }
        gamasd = p.gamma * (1.0 - argss - argsd);
        dgddvb = -p.gamma * (dbargs + dbargd);
    }
    T von = vbin + gamasd * sarg, vdsat = 0.0, argg = 0.0;
    if (nfs) {
        const T cdonco = -(gamasd * dsrgdb + dgddvb * sarg) + factor;
        const T xn = 1.0 + PhysicsConstants::Q_ELECTRON * p.nfs * 1e4 / p.coxf + cdonco;
        const T tmp = p.vt * xn;
        von = von + tmp; argg = 1.0 / tmp;
    } else if (lvgs <= vbin) return {T(0.0), von, T(0.0)};
    const T vgst = lvgs - von;
    const T sarg3 = sarg * sarg * sarg, body0 = barg * barg * barg - sarg3;
    // effective mobility (UCRIT / UEXP)
    T ufact = 1.0;
    const P tcr = p.ucrit * 100.0 * EPSSIL / p.coxf;
    if (p.cox > P(0.0) && vgst > tcr) ufact = exp(p.uexp * log(tcr / vgst));
    const T ueff = p.mu * ufact;
    // saturation voltage, Grove-Frohman
    const T vgsx = nfs && von > lvgs ? von : lvgs;
    const T gammad = gamasd / eta;
    if (gammad > P(0.0)) {
        const T gammd2 = gammad * gammad, argv = (vgsx - vbin) / eta + phiMinVbs;
        if (argv > P(0.0)) vdsat = mx((vgsx - vbin) / eta + gammd2 * (1.0 - sqrt(1.0 + 4.0 * argv / gammd2)) / 2.0, 0.0);
    } else vdsat = mx((vgsx - vbin) / eta, 0.0);
    if (p.vmax > P(0.0)) {  // Baum's velocity saturation: the quartic's smallest valid root
        const T gammd2 = gammad * gammad, v1 = (vgsx - vbin) / eta + phiMinVbs, v2 = phiMinVbs;
        const T xv = p.vmax * L / ueff, a1 = gammad / 0.75, b1 = -2.0 * (v1 + xv), c1 = -2.0 * gammad * xv;
        const T d1 = 2.0 * v1 * (v2 + xv) - v2 * v2 - 4.0 / 3.0 * gammad * sarg3;
        const T a = -b1, b = a1 * c1 - 4.0 * d1, c = -d1 * (a1 * a1 - 4.0 * b1) - c1 * c1;
        const T r = -a * a / 3.0 + b, s = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c;
        const T p4 = s * s / 4.0 + r * r * r / 27.0, p0 = p4 < 0.0 ? -p4 : p4, p2 = sqrt(p0);
        T y3;
        if (p4 < 0.0) {
            const T ro = exp(log(sqrt(s * s / 4.0 + p0)) / 3.0), fi = atan(-2.0 * p2 / s);
            y3 = 2.0 * ro * cos(fi / 3.0) - a / 3.0;
        } else {
            const T t3 = -s / 2.0 + p2, t4 = -s / 2.0 - p2;
            y3 = exp(log(t3 < 0.0 ? -t3 : t3) / 3.0) + exp(log(t4 < 0.0 ? -t4 : t4) / 3.0) - a / 3.0;
        }
        const T a3 = sqrt(a1 * a1 / 4.0 - b1 + y3), b3 = sqrt(y3 * y3 / 4.0 - d1);
        static const double sig1[4] = {1.0, -1.0, 1.0, -1.0}, sig2[4] = {1.0, 1.0, -1.0, -1.0};
        T x4[8]; int iknt = 0, jknt = 0;
        for (int i = 0; i < 4; ++i) {
            const T aa = a1 / 2.0 + sig1[i] * a3, bb = y3 / 2.0 + sig2[i] * b3, delta4 = aa * aa / 4.0 - bb;
            if (delta4 < 0.0) continue;
            const T tmp = sqrt(delta4);
            x4[iknt++] = -aa / 2.0 + tmp;
            x4[iknt++] = -aa / 2.0 - tmp;
        }
        T xvalid = 0.0;
        for (int j = 0; j < iknt; ++j) {
            const T x = x4[j];
            if (x <= 0.0) continue;
            const T poly = x * x * x * x + a1 * x * x * x + b1 * x * x + c1 * x + d1;
            if ((poly < 0.0 ? -poly : poly) > 1.0e-6) continue;
            if (++jknt <= 1 || !(x > xvalid)) xvalid = x;
        }
        if (jknt > 0) vdsat = xvalid * xvalid - phiMinVbs;
    }
    // channel-length modulation
    T xlamda = p.lambda, bsarg = 0.0, bodys = 0.0;
    if (lvds != P(0.0)) {
        if (lvbs - vdsat <= P(0.0)) bsarg = sqrt(vdsat + phiMinVbs);
        else bsarg = sphi / (1.0 + 0.5 * (lvbs - vdsat) / p.phi);
        bodys = bsarg * bsarg * bsarg - sarg3;
        if (p.nsub != P(0.0) && !(xlamda > P(0.0))) {
            if (p.vmax <= P(0.0)) {
                const T argv = (lvds - vdsat) / 4.0, sargv = sqrt(1.0 + argv * argv);
                xlamda = p.xd / (L * lvds) * sqrt(argv + sargv);
            } else {
                const T xdv = p.xd / sqrt(p.neff), xlv = p.vmax * xdv / (2.0 * ueff);
                const T argv = mx(lvds - vdsat, 0.0);
                xlamda = xdv / (L * lvds) * (sqrt(xlv * xlv + argv) - xlv);
            }
        }
    }
    P xwb = p.xd * sqrt(p.pb);
    const P xld = L - xwb;
    T clfact = 1.0 - xlamda * lvds;
    T xleff = L * clfact;
    const T deltal = xlamda * lvds * L;
    if (p.nsub == P(0.0)) xwb = 0.25e-6;
    if (xleff < xwb) { xleff = xwb / (1.0 + (deltal - xld) / xwb); clfact = xleff / L; }
    const T beta1 = p.beta * ufact / clfact;
    T cdrain = 0.0;
    if (nfs && !(lvgs > von)) {  // subthreshold
        if (vdsat > P(0.0)) {
            const T vdson = vdsat < lvds ? vdsat : lvds, body = lvds > vdsat ? bodys : body0;
            cdrain = beta1 * ((von - vbin - eta * vdson * 0.5) * vdson - gamasd * body / 1.5) * exp(argg * (lvgs - von));
        }
    } else if (lvds <= vdsat) cdrain = beta1 * ((lvgs - vbin - eta * lvds / 2.0) * lvds - gamasd * body0 / 1.5);
    else cdrain = beta1 * ((lvgs - vbin - eta * vdsat / 2.0) * vdsat - gamasd * bodys / 1.5);
    return {cdrain, von, vdsat};
}

template <typename T, typename P>
inline MosChannel<T> mos3_channel(T lvgs, T lvbs, T lvds, const Mos1Params<P>& p)
{
    using std::sqrt; using std::exp;
    auto mx = [](T a, T b) { return a > b ? a : b; };
    const P L = p.leff, W = p.weff, narrow = p.delta * 0.5 * M_PI * EPSSIL / p.coxf, vt = p.vt;
    const P eta = p.eta * 8.15e-22 / (p.coxf * L * L * L);
    T phibs, sqphbs, dsqdvb;
    if (lvbs <= P(0.0)) { phibs = p.phi - lvbs; sqphbs = sqrt(phibs); dsqdvb = -0.5 / sqphbs; }
    else {
        const P sqphis = sqrt(p.phi), sqphs3 = p.phi * sqphis;
        sqphbs = sqphis / (1.0 + lvbs / (p.phi + p.phi)); phibs = sqphbs * sqphbs; dsqdvb = -phibs / (sqphs3 + sqphs3);
    }
    T fshort = 1.0;
    if (p.xj != P(0.0) && p.xd != P(0.0)) {
        const P xjonxl = p.xj / L, djonxj = p.ld / p.xj;
        const T wponxj = p.xd * sqphbs / p.xj, argc = wponxj / (1.0 + wponxj);
        const T arga = 0.0631353 + 0.8013292 * wponxj - 0.01110777 * wponxj * wponxj + djonxj;
        fshort = 1.0 - xjonxl * (arga * sqrt(1.0 - argc * argc) - djonxj);
    }
    const T gammas = p.gamma * fshort, fbodys = 0.5 * gammas / (sqphbs + sqphbs), fbody = fbodys + narrow / W;
    const T onfbdy = 1.0 / (1.0 + fbody), qbonco = gammas * sqphbs + narrow * phibs / W;
    const T vth = p.polarity * p.vbi - eta * lvds + qbonco;
    T von = vth, xn = 1.0;
    if (p.nfs != P(0.0)) {
        xn = 1.0 + PhysicsConstants::Q_ELECTRON * p.nfs * 1e4 / p.coxf + qbonco / (phibs + phibs);
        von = vth + vt * xn;
    } else if (lvgs <= von) return {T(0.0), von, T(0.0)};
    const T vgsx = mx(lvgs, von), onfg = 1.0 + p.theta * (vgsx - vth), fgate = 1.0 / onfg;
    const T us = p.mu * fgate;
    T vdsat = (vgsx - vth) * onfbdy, onvdsc = 0.0;
    if (p.vmax > P(0.0)) {
        const T vdsc = L * p.vmax / us, arga = (vgsx - vth) * onfbdy;
        onvdsc = 1.0 / vdsc;
        vdsat = arga + vdsc - sqrt(arga * arga + vdsc * vdsc);
    }
    const T vdsx = vdsat < lvds ? vdsat : lvds;
    if (!(vdsat > P(0.0))) return {T(0.0), von, vdsat};  // ngspice also returns 0 at vds = 0 but gives it a slope there; kept smooth instead
    T cdrain = p.beta * fgate * ((vgsx - vth - 0.5 * (1.0 + fbody) * vdsx) * vdsx), fdrain = 1.0;
    if (p.vmax > P(0.0)) { fdrain = 1.0 / (1.0 + vdsx * onvdsc); cdrain = fdrain * cdrain; }
    T delxl = 0.0;
    bool lengthMod = p.alpha != P(0.0);  // without NSUB delxl = 0 (and sqrt(0) would poison the Dual)
    if (lengthMod && lvds <= vdsat) {
        if (p.vmax > P(0.0)) lengthMod = false;
        else { T a = lvds / vdsat; a = a * a; delxl = sqrt(p.kappa * p.alpha * vdsat / 8.0) * a * a; }
    } else if (lengthMod && p.vmax <= P(0.0)) delxl = sqrt(p.kappa * p.alpha * (lvds - vdsat + vdsat / 8.0));
    else if (lengthMod) {
        const T gdsat = mx(T(1.0e-12), cdrain * (1.0 - fdrain) * onvdsc), emax = p.kappa * cdrain / L / gdsat;
        const T arga = 0.5 * emax * p.alpha, argb = sqrt(arga * arga + p.kappa * p.alpha * (lvds - vdsat));
        delxl = argb - arga;
    }
    if (lengthMod) {
        if (delxl > 0.5 * L) delxl = L - L * L / (4.0 * delxl);
        cdrain = cdrain / (1.0 - delxl / L);
    }
    if (lvgs < von) cdrain = cdrain * exp((lvgs - von) / (xn * vt));
    return {cdrain, von, vdsat};
}

// Threshold (type domain) at bulk bias vbx: VBS in normal mode, VBD reversed.
template <typename T, typename P>
inline T mos1_von(T vbx, const Mos1Params<P>& p)
{
    using std::sqrt;
    const P sphi = sqrt(p.phi);
    T sarg = vbx <= P(0.0) ? sqrt(p.phi - vbx) : sphi - vbx / (2.0 * sphi);
    if (sarg < P(0.0)) sarg = T(0.0);
    return p.polarity * p.vto + p.gamma * (sarg - sphi);
}

template <typename T> struct Mos1Currents { T d, s, b; };

template <typename T, typename P>
inline Mos1Currents<T> mos1_currents(T vd, T vg, T vs, T vb, const Mos1Params<P>& p)
{
    using std::sqrt;
    const P type = p.polarity;
    const T vgs = type * (vg - vs), vds = type * (vd - vs), vbs = type * (vb - vs);
    const T vbd = vbs - vds, vgd = vgs - vds;
    const DiodeParams<P> jd = {p.isat, P(1.0), p.vt, P(0.0), p.gmin}, js = {p.isats < P(0.0) ? p.isat : p.isats, P(1.0), p.vt, P(0.0), p.gmin};
    const T cbs = diode_model_pair(vbs, js).first;
    const T cbd = diode_model_pair(vbd, jd).first;

    const bool normal = vds >= P(0.0);          // mode: the higher terminal is the drain
    const T vgst = (normal ? vgs : vgd) - mos1_von(normal ? vbs : vbd, p);
    const T vdsm = normal ? vds : -vds;         // |Vds|
    T cdrain(0.0);
    if (p.level == 2) cdrain = mos2_channel(normal ? vgs : vgd, normal ? vbs : vbd, vdsm, p).id;
    else if (p.level == 3) cdrain = mos3_channel(normal ? vgs : vgd, normal ? vbs : vbd, vdsm, p).id;
    else if (vgst > P(0.0)) {
        const T betap = p.beta * (1.0 + p.lambda * vdsm);
        cdrain = vgst <= vdsm ? 0.5 * betap * vgst * vgst                // saturation
                              : betap * vdsm * (vgst - 0.5 * vdsm);      // linear
    }
    const T chan = normal ? cdrain : -cdrain;   // into the drain terminal
    return {type * (chan - cbd), type * (-chan - cbs), type * (cbs + cbd)};
}

// Meyer gate half-capacitances {gs, gd, gb} (ngspice DEVqmeyer). vgs/vgd are in
// the device's orientation (the caller swaps them, and the results, when reversed).
template <typename T, typename P>
inline std::array<T, 3> mos1_meyer(T vgs, T vgd, T von, T vdsat, P phi, P cox)
{
    const T vgst = vgs - von, vds = vgs - vgd;
    if (vdsat < 0.025) vdsat = T(0.025);  // MAGIC_VDS
    if (vgst <= -phi) return {T(0.0), T(0.0), T(cox / 2)};
    if (vgst <= -phi / 2) return {T(0.0), T(0.0), -vgst * cox / (2 * phi)};
    const T full = vgst <= P(0.0) ? vgst * cox / (1.5 * phi) + cox / 3 : T(cox / 3);
    const T gb = vgst <= P(0.0) ? -vgst * cox / (2 * phi) : T(0.0);
    if (vds >= vdsat) return {full, T(0.0), gb};
    const T vddif = 2.0 * vdsat - vds, vddif1 = vdsat - vds, vddif2 = vddif * vddif;
    return {full * (1.0 - vddif1 * vddif1 / vddif2), full * (1.0 - vdsat * vdsat / vddif2), gb};
}

// Depletion charge of a junction with zero-bias capacitance cz and grading m:
// C = cz (1 - v/pb)^-m below fc*pb, linearised above it (ngspice f2/f3/f4).
template <typename T, typename P>
inline T depletion_charge(T v, P cz, P m, P pb, P fc)
{
    using std::exp; using std::log;
    if (cz == P(0.0)) return T(0.0);
    const P v0 = fc * pb, a0 = 1.0 - fc;
    if (v < v0) return pb * cz * (1.0 - exp((1.0 - m) * log(1.0 - v / pb))) / (1.0 - m);
    const P c0 = cz * exp(-m * log(a0)), c1 = c0 * m / (pb * a0);
    const P q0 = pb * cz * (1.0 - exp((1.0 - m) * log(a0))) / (1.0 - m);
    const T dv = v - v0;
    return q0 + dv * (c0 + 0.5 * c1 * dv);
}

// Charges of the five capacitive branches gs, gd, gb, bd, bs (type domain,
// ngspice mos1load). The Meyer gate charges advance from the previous accepted
// step (vPrev, qPrev, halfPrev) by C*dv, C = half-capacitance now + then +
// overlap; the bulk junctions carry their depletion charge. half: the Meyer
// half-capacitances at v (the next step's halfPrev).
template <typename T> struct Mos1Charges { T v[5], q[5], half[3]; };

// Branch voltages gs, gd, gb, bd, bs (type domain) of the charges above.
template <typename T>
inline std::array<T, 5> mos1_branches(T vd, T vg, T vs, T vb, double type)
{
    return {type * (vg - vs), type * (vg - vd), type * (vg - vb), type * (vb - vd), type * (vb - vs)};
}

template <typename T>
inline Mos1Charges<T> mos1_charges(T vd, T vg, T vs, T vb, const Mos1Params<double>& p,
                                   const double* vPrev, const double* qPrev, const double* halfPrev)
{
    const double type = p.polarity;
    Mos1Charges<T> c;
    const auto br = mos1_branches(vd, vg, vs, vb, type);
    for (int k = 0; k < 5; ++k) c.v[k] = br[k];
    const T vgs = br[0], vgd = br[1], vbd = br[3], vbs = br[4];
    const bool normal = vgs >= vgd;  // vds >= 0
    T von = mos1_von(normal ? vbs : vbd, p);
    T vdsat = (normal ? vgs : vgd) - von;
    if (vdsat < 0.0) vdsat = T(0.0);
    if (p.level != 1) {
        const T vdsm = normal ? vgs - vgd : vgd - vgs, lg = normal ? vgs : vgd, lb = normal ? vbs : vbd;
        const auto ch = p.level == 2 ? mos2_channel(lg, lb, vdsm, p) : mos3_channel(lg, lb, vdsm, p);
        von = ch.von; vdsat = ch.vdsat;
    }
    const auto h = normal ? mos1_meyer(vgs, vgd, von, vdsat, p.phi, p.cox)
                          : mos1_meyer(vgd, vgs, von, vdsat, p.phi, p.cox);
    c.half[0] = normal ? h[0] : h[1];
    c.half[1] = normal ? h[1] : h[0];
    c.half[2] = h[2];
    const double ov[3] = {p.cgso, p.cgdo, p.cgbo};
    for (int k = 0; k < 3; ++k)
        c.q[k] = qPrev[k] + (c.half[k] + halfPrev[k] + ov[k]) * (c.v[k] - vPrev[k]);
    c.q[3] = depletion_charge(vbd, p.czbd, p.mj, p.pb, p.fc) + depletion_charge(vbd, p.czbdsw, p.mjsw, p.pb, p.fc);
    c.q[4] = depletion_charge(vbs, p.czbs, p.mj, p.pb, p.fc) + depletion_charge(vbs, p.czbssw, p.mjsw, p.pb, p.fc);
    return c;
}

// ============================================================================
// SPICE JUNCTION DEVICES: DIODE (level 1) AND GUMMEL-POON BJT
// ============================================================================

// Junction current of an exponential diode with saturation current is and
// emission voltage vte = N*Vt, as ngspice dioload/bjtload: exponential above
// -3 vte (linear continuation past JUNCTION_EXP_ARG_MAX, like every junction
// here), the cubic reverse tail below it. No GMIN (callers add it where ngspice does).
template <typename T, typename P>
inline T spice_junction(T v, P is, P vte)
{
    using std::exp;
    if (v >= -3.0 * vte) {
        const P amax = PhysicsConstants::JUNCTION_EXP_ARG_MAX;
        if (v > amax * vte) { const P e = exp(amax); return is * (e - 1.0) + (is / vte) * e * (v - amax * vte); }
        return is * (exp(v / vte) - 1.0);
    }
    T a = 3.0 * vte / (v * 2.718281828459045);  // CONSTe
    a = a * a * a;
    return -is * (1.0 + a);
}

// Diode, parameters resolved (AREA applied): ngspice dio level 1 without
// temperature dependence (the engine runs at TNOM).
template <typename P>
struct DiodeModel {
    P is = 1e-14, vte = PhysicsConstants::V_THERMAL_NOM;  // IS*AREA, N*Vt
    P rs = 0.0;                                            // RS/AREA (series, internal node)
    P cjo = 0.0, vj = 1.0, m = 0.5, fc = 0.5, tt = 0.0;    // CJO*AREA
    P isw = 0.0, vtesw = PhysicsConstants::V_THERMAL_NOM;  // sidewall: JSW*PJ, NS*Vt
    P cjsw = 0.0, mjsw = 0.33, vjsw = 1.0;                 // CJSW*PJ, MJSW, PHP
    P ikf = 0.0, ikr = 0.0;                                // knee currents (IKF*AREA, IKR*AREA); 0: none
    bool breakdown = false; P xbv = 0.0;                    // BV adjusted so I(-BV) = IBV (diotemp)
    P gmin = PhysicsConstants::JUNCTION_GMIN;
};

// Breakdown voltage xbv such that the reverse current at -BV is IBV (ngspice
// diotemp, NBV = N): returns BV unchanged when IBV is below the knee current.
inline double diode_breakdown_xbv(double bv, double ibv, double is, double vte, double reltol = 1e-3)
{
    double cbv = ibv;
    if (cbv < is * bv / vte) return bv;
    const double tol = reltol * cbv;
    double xbv = bv - vte * std::log(1.0 + cbv / is);
    for (int it = 0; it < 25; ++it) {
        xbv = bv - vte * std::log(cbv / is + 1.0 - xbv / vte);
        const double xcbv = is * (std::exp((bv - xbv) / vte) - 1.0 + xbv / vte);
        if (std::abs(xcbv - cbv) <= tol) break;
    }
    return xbv;
}

// Junction current (with GMIN across the junction) and charge (depletion +
// TT * diffusion) at junction voltage vd (ngspice dioload).
template <typename T, typename P>
inline std::pair<T, T> diode_iq(T vd, const DiodeModel<P>& p)
{
    using std::exp;
    T cdb;
    if (p.breakdown && vd < -p.xbv) cdb = -p.is * exp(-(p.xbv + vd) / p.vte);
    else {
        cdb = spice_junction(vd, p.is, p.vte);
        // high-injection knees (ngspice dioload): forward IKF, reverse IKR; the bottom current only
        using std::sqrt;
        if (p.ikf > P(0.0) && cdb > T(0.0)) cdb = cdb / (1.0 + sqrt(cdb / p.ikf));
        else if (p.ikr > P(0.0) && cdb < T(0.0)) cdb = cdb / (1.0 + sqrt(-cdb / p.ikr));
        if (p.isw != P(0.0)) cdb = cdb + spice_junction(vd, p.isw, p.vtesw);
    }
    const T q = depletion_charge(vd, p.cjo, p.m, p.vj, p.fc) + depletion_charge(vd, p.cjsw, p.mjsw, p.vjsw, p.fc) + p.tt * cdb;
    return {cdb + p.gmin * vd, q};
}

// Gummel-Poon BJT, parameters resolved (AREA applied; ngspice bjt level 1
// without temperature dependence, excess phase or bias-dependent RB).
template <typename P>
struct BjtModel {
    P gmin = PhysicsConstants::JUNCTION_GMIN;       // GMIN across the junctions
    P type = 1.0;                                  // +1 NPN, -1 PNP
    P is = 1e-16, bf = 100.0, br = 1.0, vtf = PhysicsConstants::V_THERMAL_NOM, vtr = PhysicsConstants::V_THERMAL_NOM;  // NF*Vt, NR*Vt
    P invVaf = 0.0, invVar = 0.0, oikf = 0.0, oikr = 0.0;     // 1/VAF, 1/VAR, 1/(IKF*AREA), 1/(IKR*AREA)
    P ise = 0.0, vte = 1.5 * PhysicsConstants::V_THERMAL_NOM; // ISE*AREA, NE*Vt
    P isc = 0.0, vtc = 2.0 * PhysicsConstants::V_THERMAL_NOM; // ISC*AREA, NC*Vt
    P rb = 0.0, rc = 0.0, re = 0.0;                            // per AREA; > 0: internal node
    bool varRb = false; P rbm = 0.0, xjrb = 0.0;               // RBM != RB: RB falls to RBM with the base current (IRB*AREA = xjrb)
    P nk = 0.5;                                                // high-current roll-off exponent
    P cje = 0.0, vje = 0.75, mje = 0.33;                        // *AREA
    P cjcInt = 0.0, cjcExt = 0.0, vjc = 0.75, mjc = 0.33;       // CJC*XCJC*AREA (B'-C'), rest (B-C')
    P cjs = 0.0, vjs = 0.75, mjs = 0.0, fc = 0.5;
    P iss = 0.0, vts = PhysicsConstants::V_THERMAL_NOM;       // substrate junction: ISS*AREA, NS*Vt
    P tf = 0.0, xtf = 0.0, ovtf = 0.0, xjtf = 0.0, tr = 0.0;   // ovtf = 1/(1.44 VTF), xjtf = ITF*AREA
};

// DC terminal currents (into C', B'), the substrate junction current cs (S -> C',
// ISS diode + GMIN, as ngspice 4 bjtload: it exists even with ISS = 0) and the
// four charges qbe (B'-E'), qbc (B'-C'), qbx (B-C'), qsub (S-C'), all in the
// type domain, at junction voltages vbe, vbc (internal), vbx (external base), vsub.
template <typename T> struct BjtEval { T cc, cb, cs, qb, q[4]; };

// Base resistance of a bias-dependent RB (ngspice bjtload): RBM + (RB - RBM)/qb, or with IRB the
// tan() law in the base current cb (qb is not used then).
template <typename T, typename P>
inline T bjt_rbb(T qb, T cb, const BjtModel<P>& p)
{
    using std::sqrt; using std::tan;
    const P rbpi = p.rb - p.rbm;
    if (p.xjrb == P(0.0)) return p.rbm + rbpi / qb;
    T arg1 = cb / p.xjrb;
    if (arg1 < T(1e-9)) arg1 = T(1e-9);
    const T z = (-1.0 + sqrt(1.0 + 14.59025 * arg1)) / (2.4317 * sqrt(arg1));
    const T t = tan(z);
    return p.rbm + 3.0 * rbpi * (t - z) / (z * t * t);
}

template <typename T, typename P>
inline BjtEval<T> bjt_gp(T vbe, T vbc, T vbx, T vsub, const BjtModel<P>& p)
{
    using std::exp; using std::log; using std::sqrt;
    const P gmin = p.gmin;
    const T cbe = spice_junction(vbe, p.is, p.vtf);
    const T cbc = spice_junction(vbc, p.is, p.vtr);
    const T cben = (p.ise != P(0.0) ? spice_junction(vbe, p.ise, p.vte) : T(0.0)) + gmin * vbe;
    const T cbcn = (p.isc != P(0.0) ? spice_junction(vbc, p.isc, p.vtc) : T(0.0)) + gmin * vbc;
    const T q1 = 1.0 / (1.0 - p.invVaf * vbc - p.invVar * vbe);
    T qb = q1;
    if (p.oikf != P(0.0) || p.oikr != P(0.0)) {
        T arg = 1.0 + 4.0 * (p.oikf * cbe + p.oikr * cbc);
        if (arg < P(0.0)) arg = T(0.0);
        qb = q1 * (1.0 + (p.nk == P(0.5) ? sqrt(arg) : pow(arg, p.nk))) / 2.0;
    }
    BjtEval<T> r;
    r.qb = qb;
    r.cc = (cbe - cbc) / qb - cbc / p.br - cbcn;
    r.cb = cbe / p.bf + cben + cbc / p.br + cbcn;
    r.cs = (p.iss != P(0.0) ? spice_junction(vsub, p.iss, p.vts) : T(0.0)) + gmin * vsub;

    T cbeq = cbe;  // transit-time current, XTF/VTF/ITF-modulated (ngspice bjtload)
    if (p.tf != P(0.0) && vbe > P(0.0)) {
        T argtf(0.0);
        if (p.xtf != P(0.0)) {
            argtf = T(p.xtf);
            if (p.ovtf != P(0.0)) argtf = argtf * exp(vbc * p.ovtf);
            if (p.xjtf != P(0.0)) { const T t = cbe / (cbe + p.xjtf); argtf = argtf * t * t; }
        }
        cbeq = cbe * (1.0 + argtf) / qb;
    }
    r.q[0] = p.tf * cbeq + depletion_charge(vbe, p.cje, p.mje, p.vje, p.fc);
    r.q[1] = p.tr * cbc + depletion_charge(vbc, p.cjcInt, p.mjc, p.vjc, p.fc);
    r.q[2] = depletion_charge(vbx, p.cjcExt, p.mjc, p.vjc, p.fc);
    if (p.cjs == P(0.0)) r.q[3] = T(0.0);
    else if (vsub <= P(0.0)) {
        const T arg = 1.0 - vsub / p.vjs;
        r.q[3] = p.vjs * p.cjs * (1.0 - arg * exp(-p.mjs * log(arg))) / (1.0 - p.mjs);
    } else r.q[3] = vsub * p.cjs * (1.0 + p.mjs * vsub / (2.0 * p.vjs));
    return r;
}

// ============================================================================
// JFET (ngspice level 1: Shichman-Hodges with B, gate junctions, depletion charges)
// ============================================================================

// Parameters resolved (AREA applied; RD / RS are series resistors added by the parser).
template <typename P>
struct JfetModel {
    P type = 1.0;                                      // +1 NJF, -1 PJF
    P beta = 1e-4, vto = -2.0, lambda = 0.0, b = 1.0;  // BETA*AREA, VTO, LAMBDA, B
    P is = 1e-14, vt = PhysicsConstants::V_THERMAL_NOM; // IS*AREA, N*Vt (gate junctions)
    P cgs = 0.0, cgd = 0.0, pb = 1.0, fc = 0.5;        // CGS*AREA, CGD*AREA, PB, FC
    P gmin = PhysicsConstants::JUNCTION_GMIN;
};

// Drain current of the normal mode (vds >= 0) at overdrive vgst (ngspice jfetload).
template <typename T, typename P>
inline T jfet_channel(T vgst, T vds, const JfetModel<P>& p)
{
    if (vgst <= P(0.0)) return T(0.0);
    const T betap = p.beta * (1.0 + p.lambda * vds);
    const P bfac = (1.0 - p.b) / (p.pb - p.vto);
    if (vgst >= vds) {  // linear
        const T apart = 2.0 * p.b + 3.0 * bfac * (vgst - vds);
        return betap * (vds * (vds * (bfac * vds - p.b) + vgst * apart));
    }
    return betap * (vgst * vgst * (p.b + bfac * vgst));  // saturation
}

// Currents in the type domain at gate junction voltages vgs, vgd: channel D->S, gate junctions G->S and
// G->D (each with GMIN, as every junction). The inverse mode (vds < 0) mirrors the normal one.
template <typename T> struct JfetEval { T id, igs, igd; };
template <typename T, typename P>
inline JfetEval<T> jfet_eval(T vgs, T vgd, const JfetModel<P>& p)
{
    const T vds = vgs - vgd;
    JfetEval<T> r;
    r.igs = spice_junction(vgs, p.is, p.vt) + p.gmin * vgs;
    r.igd = spice_junction(vgd, p.is, p.vt) + p.gmin * vgd;
    r.id = vds >= P(0.0) ? jfet_channel(vgs - p.vto, vds, p) : -jfet_channel(vgd - p.vto, -vds, p);
    return r;
}

// Depletion charges of the gate-source and gate-drain junctions (grading 0.5).
template <typename T, typename P>
inline std::array<T, 2> jfet_charges(T vgs, T vgd, const JfetModel<P>& p)
{
    return {depletion_charge(vgs, p.cgs, P(0.5), p.pb, p.fc), depletion_charge(vgd, p.cgd, P(0.5), p.pb, p.fc)};
}



// ============================================================================
// END OF DEVICE PHYSICS
// ============================================================================


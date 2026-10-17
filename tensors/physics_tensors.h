#pragma once
#include <vector>
#include <cstddef>
#include <map>
#include <unordered_map>
#include "../netlist/circuit.h"
#include "../physics/device_physics.h"
#include <array>

/*
 * physics_tensors.h - Structure-of-Arrays (SoA) Tensor Data Structures
 *
 * These structures enable batch-parallel processing of device physics:
 * - SIMD vectorization on CPU (AVX2/AVX-512)
 * - GPU compute shader dispatch (WebGPU/CUDA)
 * - Superposition decomposition (multi-source analysis)
 *
 * Each tensor stores device parameters in contiguous arrays for cache-optimal access.
 */

// Level-1 parameters of a MOSFET instance from its parser-resolved .model card
// (Mosfet::card). An unresolved card (the GUI's Verilog-A path) keeps the
// engine's 200u/0.7 V/0.02 NMOS and 100u/-0.7 V/0.02 PMOS by model name, no caps.
inline Mos1Params<double> mos1Params(const Mosfet& m, double gmin = PhysicsConstants::JUNCTION_GMIN) {
    const auto& c = m.card;
    Mos1Params<double> p;
    p.gmin = gmin;
    const bool pmos = c.resolved ? c.pmos
                                 : m.modelName.find("PMOS") != std::string::npos || m.modelName.find("pmos") != std::string::npos;
    p.polarity = pmos ? -1.0 : 1.0;
    const double M = m.extra.mult, leff = m.l - 2.0 * c.ld + c.xl, weff = m.w - 2.0 * c.wd + c.xw;
    if (!c.resolved) {
        p.vto = pmos ? -0.7 : 0.7;
        p.lambda = 0.02;
        p.beta = (pmos ? 100e-6 : 200e-6) * m.w / m.l * M;
        p.isat = 1e-14 * M;
        return p;
    }
    p.vto = c.vto; p.lambda = c.lambda; p.gamma = c.gamma; p.phi = c.phi;
    p.beta = c.kp * weff / leff * M;
    p.isat = c.is * M; p.vt = c.vt;
    if (c.js > 0.0 && m.geo.ad > 0.0 && m.geo.as > 0.0) { p.isat = c.js * m.geo.ad * M; p.isats = c.js * m.geo.as * M; }  // ngspice mos1temp
    p.cox = c.tox > 0.0 ? 3.9 * 8.854214871e-12 / c.tox * weff * leff * M : 0.0;
    p.cgso = c.cgso * weff * M; p.cgdo = c.cgdo * weff * M; p.cgbo = c.cgbo * leff * M;
    // CBD/CBS win over CJ * area (ngspice mos1temp)
    p.czbd = (c.cbd > 0.0 ? c.cbd : c.cj * m.geo.ad) * M; p.czbdsw = c.cjsw * m.geo.pd * M;
    p.czbs = (c.cbs > 0.0 ? c.cbs : c.cj * m.geo.as) * M; p.czbssw = c.cjsw * m.geo.ps * M;
    p.mj = c.mj; p.mjsw = c.mjsw; p.pb = c.pb; p.fc = c.fc;
    if (c.level > 1) {
        p.level = c.level; p.vbi = c.vbi; p.leff = leff; p.weff = weff; p.coxf = 3.9 * 8.854214871e-12 / c.tox; p.mu = c.mu;
        p.nfs = c.nfs; p.nsub = c.nsub; p.xd = c.xd; p.xj = c.xj; p.ld = c.ld; p.delta = c.delta;
        p.ucrit = c.ucrit; p.uexp = c.uexp; p.vmax = c.vmax; p.neff = c.neff;
        p.eta = c.eta; p.theta = c.theta; p.kappa = c.kappa; p.alpha = c.alpha;
    }
    p.caps = p.cox > 0.0 || p.cgso > 0.0 || p.cgdo > 0.0 || p.cgbo > 0.0 ||
             p.czbd > 0.0 || p.czbdsw > 0.0 || p.czbs > 0.0 || p.czbssw > 0.0;
    return p;
}


// Diode / Gummel-Poon parameters of an instance from its parser-resolved
// .model values (AREA applied as ngspice does: currents and capacitances
// scale, resistances divide). Devices built without a card keep the IS/N or
// IS/BF/BR they were constructed with and SPICE defaults for the rest.
// The parser holds them at the instance temperature (Vt in thermalVoltage_V_T_V, IS, betas, junctions).
inline DiodeModel<double> diodeModel(const Diode& d, double gmin = PhysicsConstants::JUNCTION_GMIN) {
    const auto& c = d.card;
    DiodeModel<double> p;
    p.gmin = gmin;
    const double vt = d.thermalVoltage_V_T_V, is = d.saturationCurrent_I_S_A;
    p.is = is * d.area;
    p.vte = d.emissionCoefficient_N * vt;
    p.rs = c.rs / d.area;
    p.cjo = c.cjo * d.area; p.vj = c.vj; p.m = c.m; p.fc = c.fc; p.tt = c.tt;
    p.isw = c.jsw * d.pj * d.mult; p.vtesw = c.ns * vt;
    p.cjsw = c.cjsw * d.pj * d.mult; p.mjsw = c.mjsw; p.vjsw = c.php;
    p.ikf = c.ikf * d.area; p.ikr = c.ikr * d.area;
    p.breakdown = c.bvGiven;
    if (p.breakdown) p.xbv = diode_breakdown_xbv(c.bv, c.ibv * d.area, p.is, p.vte);
    return p;
}

inline JfetModel<double> jfetModel(const JFET& j, double gmin = PhysicsConstants::JUNCTION_GMIN) {
    const auto& c = j.card;
    JfetModel<double> p;
    p.gmin = gmin;
    p.type = j.isNChannel ? 1.0 : -1.0;
    p.beta = j.beta * j.area; p.vto = j.Vto; p.lambda = j.lambda; p.b = c.b;
    p.is = c.is * j.area; p.vt = c.n * j.thermalVoltage_V_T_V;
    p.cgs = c.cgs * j.area; p.cgd = c.cgd * j.area; p.pb = c.pb; p.fc = c.fc;
    return p;
}

inline BjtModel<double> bjtModel(const BJT& q, double gmin = PhysicsConstants::JUNCTION_GMIN) {
    const auto& c = q.card;
    const double a = q.area;
    const double vt = q.thermalVoltage_V_T_V, is = q.saturationCurrent_I_S_A;
    BjtModel<double> p;
    p.gmin = gmin;
    p.type = q.isNPN ? 1.0 : -1.0;
    p.is = is * a; p.bf = q.betaF; p.br = q.betaR;
    p.vtf = c.nf * vt; p.vtr = c.nr * vt;
    p.invVaf = c.vaf != 0.0 ? 1.0 / c.vaf : 0.0; p.invVar = c.var != 0.0 ? 1.0 / c.var : 0.0;
    p.oikf = c.ikf != 0.0 ? 1.0 / (c.ikf * a) : 0.0; p.oikr = c.ikr != 0.0 ? 1.0 / (c.ikr * a) : 0.0;
    p.ise = c.ise * a; p.vte = c.ne * vt; p.isc = c.isc * a; p.vtc = c.nc * vt;
    p.rb = c.rb / a; p.rc = c.rc / a; p.re = c.re / a;
    p.varRb = c.rb > 0.0 && c.rbm != c.rb;
    p.rbm = c.rbm / a; p.xjrb = c.irb * a; p.nk = c.nk;
    p.cje = c.cje * a; p.vje = c.vje; p.mje = c.mje;
    p.cjcInt = c.cjc * c.xcjc * a; p.cjcExt = c.cjc * (1.0 - c.xcjc) * a; p.vjc = c.vjc; p.mjc = c.mjc;
    p.cjs = c.cjs * a; p.vjs = c.vjs; p.mjs = c.mjs; p.fc = c.fc;
    p.iss = c.iss * a; p.vts = c.ns * vt;
    p.tf = c.tf; p.xtf = c.xtf; p.ovtf = c.vtf != 0.0 ? 1.0 / (1.44 * c.vtf) : 0.0;
    p.xjtf = c.itf * a; p.tr = c.tr;
    return p;
}

// ============================================================================
// RESISTOR TENSOR (Linear - for completeness)
// ============================================================================

struct ResistorTensor {
    std::vector<int> nodes1;
    std::vector<int> nodes2;
    std::vector<double> R;  // Resistance [Ω]
    
    size_t size() const { return nodes1.size(); }
    
    void push(int n1, int n2, double r) {
        nodes1.push_back(n1);
        nodes2.push_back(n2);
        R.push_back(r);
    }
    
    void clear() {
        nodes1.clear(); nodes2.clear(); R.clear();
    }
};

// ============================================================================
// CAPACITOR TENSOR (for transient)
// ============================================================================

struct CapacitorTensor {
    std::vector<int> plates1;
    std::vector<int> plates2;
    std::vector<double> C;         // Capacitance [F]
    std::vector<double> v_prev;    // Voltage at t-1
    std::vector<double> v_prev2;   // Voltage at t-2 (Gear2)
    std::vector<double> i_prev;    // Current at t-1
    
    size_t size() const { return plates1.size(); }
    
    void push(int p1, int p2, double c) {
        plates1.push_back(p1);
        plates2.push_back(p2);
        C.push_back(c);
        v_prev.push_back(0.0);
        v_prev2.push_back(0.0);
        i_prev.push_back(0.0);
    }
    
    void clear() {
        plates1.clear(); plates2.clear(); C.clear();
        v_prev.clear(); v_prev2.clear(); i_prev.clear();
    }
};

// ============================================================================
// INDUCTOR TENSOR (for transient)
// ============================================================================

struct InductorTensor {
    std::vector<int> coil1;
    std::vector<int> coil2;
    std::vector<double> L;         // Inductance [H]
    std::vector<double> i_prev;    // Current at t-1
    std::vector<double> v_prev;    // Voltage at t-1
    
    size_t size() const { return coil1.size(); }
    
    void push(int c1, int c2, double l) {
        coil1.push_back(c1);
        coil2.push_back(c2);
        L.push_back(l);
        i_prev.push_back(0.0);
        v_prev.push_back(0.0);
    }
    
    void clear() {
        coil1.clear(); coil2.clear(); L.clear();
        i_prev.clear(); v_prev.clear();
    }
};

// ============================================================================
// SOURCE TENSORS
// ============================================================================

struct VoltageSourceTensor {
    std::vector<int> nodesPos;
    std::vector<int> nodesNeg;
    std::vector<double> voltages; // Nominal DC value
    
    size_t size() const { return nodesPos.size(); }
    
    void push(int np, int nn, double v) {
        nodesPos.push_back(np);
        nodesNeg.push_back(nn);
        voltages.push_back(v);
    }
    
    void clear() {
        nodesPos.clear(); nodesNeg.clear(); voltages.clear();
    }
};

struct CurrentSourceTensor {
    std::vector<int> nodesPos;
    std::vector<int> nodesNeg;
    std::vector<double> currents;
    
    size_t size() const { return nodesPos.size(); }
    
    void push(int np, int nn, double i) {
        nodesPos.push_back(np);
        nodesNeg.push_back(nn);
        currents.push_back(i);
    }
    
    void clear() {
        nodesPos.clear(); nodesNeg.clear(); currents.clear();
    }
};

// ============================================================================
// UNIFIED TENSOR BLOCK
// ============================================================================

/**
 * TensorizedBlock - Unified container for all device tensors.
 * This replaces the AoS `TensorBlock` for solver-side operations.
 */
struct TensorizedBlock {
    ResistorTensor resistors;
    CapacitorTensor capacitors;
    InductorTensor inductors;
    VoltageSourceTensor voltageSources;
    CurrentSourceTensor currentSources;
    
    void clear() {
        resistors.clear();
        capacitors.clear();
        inductors.clear();
        voltageSources.clear();
        currentSources.clear();
    }
};

// ============================================================================
// CONVERSION FUNCTIONS (AoS -> SoA)
// ============================================================================

// Forward declaration: TensorBlock is defined in circuit.h
struct TensorBlock;

/**
 * Convert a full TensorNetlist (Hierarchical) to a single TensorizedBlock (Flat SoA).
 * All hierarchy is flattened into global node indices.
 */
TensorizedBlock tensorizeNetlist(const class TensorNetlist& netlist);

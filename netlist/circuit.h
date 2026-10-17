#pragma once
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <map>
#include <set>
#include <cassert>
#include <iostream>
#include "../physics/physics_constants.h"
#include "graph_canonical.h"
#include "numparam.h"

namespace osdi { class Model; class Instance; }

/*
 * circuit.h
 * Circuit Netlist - Component definitions and hierarchical storage.
 *
 * Variable Naming Convention (EE Standard):
 *   - Resistor: nodeTerminal1/2
 *   - Capacitor: nodePlate1/2
 *   - Inductor: nodeCoil1/2
 *   - Diode: nodeAnode, nodeCathode
 *   - Mosfet: drain, gate, source, body
 *   - BJT: collector, base, emitter
 *   - Controlled Sources: nodeOutputPos/Neg, nodeControlPos/Neg
 */


// ============================================================================
// SPICE MODEL CARD
// ============================================================================

// Stores device model parameters parsed from .MODEL directives.
struct ModelCard {
    std::string name;        // e.g. "nfet_01v8", "DMOD"
    std::string type;        // "NMOS", "PMOS", "D", "NPN", "PNP", "NJFET", "PJFET"
    int level = 1;           // Model level (1 = Shichman-Hodges, 4 = BSIM4, etc.)
    std::map<std::string, double> params;  // Key-value: "KP"->200e-6, "VTO"->0.7, etc.

    double get(const std::string& key, double defaultVal) const {
        auto it = params.find(key);
        return (it != params.end()) ? it->second : defaultVal;
    }
};

// ============================================================================
// PASSIVE COMPONENTS
// ============================================================================

/*
 * Resistor (R)
 * Ohm's Law: V = I * R
 * MNA Stamp (Conductance Form):
 *   G[n1][n1] += 1/R, G[n2][n2] += 1/R
 *   G[n1][n2] -= 1/R, G[n2][n1] -= 1/R
 */
inline constexpr const char* TEMP_LIST_ERR = "a .temp list is supported for .op/.dc only";

// Instance temperature (ngspice TEMP in Celsius, DTEMP offset): TEMP wins, else DTEMP shifts the circuit's.
struct InstTemp {
    double temp = NAN, dtemp = 0.0;
    double kelvin(double circuitK) const { return std::isnan(temp) ? circuitK + dtemp : temp + 273.15; }
};

struct Resistor {
    std::string name;             // e.g. "R1"
    int nodeTerminal1;            // First terminal (node index)
    int nodeTerminal2;            // Second terminal (node index)
    double resistance_ohms;       // Resistance in Ohms [Ω]
    
    // Generic PDK Support
    std::string model_name;       // e.g., "poly_res"
    double width_m = 0.0;
    double length_m = 0.0;

    std::string expression;         // Symbolic override (e.g. "Rload", "2*Vbias/Iref")

    double tc1 = 0.0, tc2 = 0.0;    // TC1, TC2: applied by the parser's resolve step, with inst
    bool tc1Given = false, tc2Given = false;  // an instance TC overrides the model's
    double m = 1.0;                 // M=: parallel copies (applied by the parser's model resolution)
    double ac_ohms = NAN;           // AC=: the resistance .ac uses (NaN: resistance_ohms)
    InstTemp inst;
    double tcFactor = 1.0;          // resistance_ohms / nominal: .dc sweeps the nominal value
};

/*
 * Capacitor (C)
 * Physics: i_C = C * dv_C/dt
 * Trapezoidal Integration Model:
 *   G_eq = 2*C / dt
 *   I_eq = G_eq * v_prev + i_prev
 * The capacitor is replaced by a conductance in parallel with a history current source.
 */
/*
 * Switch (S voltage-controlled, W current-controlled): a resistor of RON or ROFF with ngspice's
 * hysteresis. The state is decided from the control value of the Newton iterate against the
 * last accepted state (BlockState::swPrev): on above VT+VH, off below VT-VH, else unchanged.
 */
struct Switch {
    std::string name;
    int nodePos = 0, nodeNeg = 0;
    int ctrlPos = 0, ctrlNeg = 0;   // S: control nodes
    std::string ctrlName;           // W: the V source whose current controls it
    int ctrlSource = -1;            // ... its index in the same block (resolveControlledSources)
    std::string modelName;
    double vt = 0.0, vh = 0.0, ron = 1.0, roff = 1e12;
    bool init = false;              // ON: the state before the first decision
};

struct Capacitor {
    std::string name;             // e.g. "C1"
    int nodePlate1;               // Positive plate (node index)
    int nodePlate2;               // Negative plate (node index)
    double capacitance_farads;    // Capacitance in Farads [F]
    
    // Generic PDK Support
    std::string model_name;       // e.g., "mim_cap"
    double width_m = 0.0;
    double length_m = 0.0;
    double m = 1.0;               // Multiplier

    std::string expression;         // Symbolic override for capacitance

    bool   icGiven = false;           // IC= (initial voltage), used only with UIC
    double ic_V    = 0.0;

    bool   tc1Given = false, tc2Given = false;  // instance TC1/TC2 override the model's
    double tc1 = 0.0, tc2 = 0.0;
    InstTemp inst;
};

/*
 * Inductor (L)
 * Physics: v_L = L * di_L/dt
 * Backward Euler Integration Model:
 *   R_eq = L / dt
 *   V_eq = R_eq * i_prev
 * Inductor adds an auxiliary current variable to the MNA system.
 */
struct Inductor {
    std::string name;             // e.g. "L1"
    int nodeCoil1;                // First coil terminal (node index)
    int nodeCoil2;                // Second coil terminal (node index)
    double inductance_henries;    // Inductance in Henries [H]

    // Generic PDK Support
    std::string model_name;       // e.g., "spiral_ind"
    double width_m = 0.0;         // Trace width
    double length_m = 0.0;        // Total path length
    double area_m2 = 0.0;         // Spiral area
    int turns = 0;                // Number of turns

    std::string expression;       // Symbolic override for inductance

    bool   icGiven = false;       // IC= (initial current), used only with UIC
    double ic_A    = 0.0;
};

/*
 * Mutual Inductor (K) - Transformer Coupling
 * Physics:
 *   v1 = L1 * di1/dt + M * di2/dt
 *   v2 = M * di1/dt + L2 * di2/dt
 * Where M = couplingCoefficient_k * sqrt(L1 * L2)
 */
struct MutualInductor {
    int inductor1_index;          // Index of first inductor in inductors vector
    int inductor2_index;          // Index of second inductor
    double couplingCoefficient_k; // Coupling factor (0.0 = none, 1.0 = ideal)
};

// ============================================================================
// INDEPENDENT SOURCES
// ============================================================================

/*
 * Independent Voltage Source (V)
 * Ideal: Forces v(+) - v(-) = V_dc
 * Adds an auxiliary branch current to MNA.
 */
struct VoltageSource {
    int nodePositive;
    int nodeNegative;
    double voltage_V = 0.0;  // DC value (parser: t=0 waveform value if no DC given)
    std::string type = "";   // "PULSE", "SINE", "DC", etc.

    // Waveform Parameters (SPICE-like)
    double ac_mag = 0.0;     // AC Magnitude (Small Signal)
    double ac_phase = 0.0;   // AC Phase
    
    // PULSE(v1 v2 td tr tf pw per)
    double pulse_v1 = 0.0;   // Initial value
    double pulse_v2 = 0.0;   // Pulsed value
    double pulse_td = 0.0;   // Delay time
    double pulse_tr = 0.0;   // Rise time
    double pulse_tf = 0.0;   // Fall time
    double pulse_pw = 0.0;   // Pulse width
    double pulse_per = 0.0;  // Period
    
    // SINE(vo va freq td theta)
    double sine_vo = 0.0;    // Offset (same as voltage_V for DC)
    double sine_va = 0.0;    // Amplitude
    double sine_freq = 0.0;  // Frequency [Hz]
    double sine_td = 0.0;    // Delay
    double sine_theta = 0.0; // Damping factor
    double sine_phase = 0.0; // Phase [deg] (Extension)

    // PWL(t1 v1 t2 v2 ...) [TD=delay] [R=repeat-from]: corner times and values
    std::vector<double> pwl_t, pwl_v;
    double pwl_td = 0.0, pwl_r = -1.0;  // r < 0: no repeat

    // EXP v1 v2 td1 tau1 td2 tau2 | SFFM vo va fc mdi fs | AM vo vmo vma fm fc td (ngspice order)
    std::vector<double> wf;

    std::string expression;  // Symbolic override for DC/main value
    std::string name;        // instance name (F/H cards reference it)
};

/*
 * Independent Current Source (I)
 * Ideal: Forces i = I_dc (from + to -)
 * Stamps directly into RHS vector.
 */
struct CurrentSource {
    int nodePositive;
    int nodeNegative;
    double current_A = 0.0;  // DC value (parser: t=0 waveform value if no DC given)
    std::string name;        // instance name (.dc sweeps reference it)
    std::string type = "";   // "PULSE", "SINE", "DC"

    // Waveform Parameters (Mirror of VoltageSource)
    double ac_mag = 0.0;
    double ac_phase = 0.0;

    double pulse_v1 = 0.0;
    double pulse_v2 = 0.0;
    double pulse_td = 0.0;
    double pulse_tr = 0.0;
    double pulse_tf = 0.0;
    double pulse_pw = 0.0;
    double pulse_per = 0.0;

    double sine_vo = 0.0;
    double sine_va = 0.0;
    double sine_freq = 0.0;
    double sine_td = 0.0;
    double sine_theta = 0.0;
    double sine_phase = 0.0;

    std::vector<double> pwl_t, pwl_v;
    double pwl_td = 0.0, pwl_r = -1.0;
    std::vector<double> wf;  // EXP / SFFM / AM arguments, as VoltageSource::wf

    std::string expression;  // Symbolic override for DC/main value
};

/*
 * Instantaneous value of an independent V or I source at time t (transient).
 * The single definition shared by the parser (t=0 DC value), both stamping
 * paths and the KCL audit, so they can never disagree on a source's value.
 * The transient lands on its corners via nextBreakpoint() below.
 */
// PWL time in the waveform's own frame: delay removed and, with R, folded back into
// (r, t_last] after the first pass (ngspice's repeat). A repeat boundary is a jump
// (v(t_last) -> v(r)); exactly on it ngspice holds v(t_last), so the fold maps there,
// within 1e-9 of the period (the breakpoint times carry rounding).
template <class Src>
inline double pwlLocalTime(const Src& s, double t) {
    double tt = t - s.pwl_td;
    const double last = s.pwl_t.back(), per = last - s.pwl_r;
    if (s.pwl_r >= 0.0 && tt > last && per > 0.0) {
        const double m = std::fmod(tt - s.pwl_r, per);
        tt = (m <= 1e-9 * per || m >= (1.0 - 1e-9) * per) ? last : s.pwl_r + m;
    }
    return tt;
}

template <class Src>
inline double waveformValue(const Src& s, double dcValue, double t) {
    if (s.type == "PWL") {
        const double tt = pwlLocalTime(s, t);
        const auto& T = s.pwl_t;
        if (tt <= T.front()) return s.pwl_v.front();
        if (tt >= T.back()) return s.pwl_v.back();
        // T[k-1] < tt <= T[k]: at a step (equal times) the instant holds the earlier value, as
        // ngspice; the later one would put the jump inside the step that lands on the corner
        const size_t k = std::lower_bound(T.begin(), T.end(), tt) - T.begin();
        return s.pwl_v[k - 1] + (s.pwl_v[k] - s.pwl_v[k - 1]) * (tt - T[k - 1]) / (T[k] - T[k - 1]);
    }
    if (s.type == "PULSE") {
        double tt = t - s.pulse_td;
        if (tt <= 0.0) return s.pulse_v1;
        tt = std::fmod(tt, s.pulse_per > 0.0 ? s.pulse_per : 1e99);
        if (tt < s.pulse_tr) return s.pulse_v1 + (s.pulse_v2 - s.pulse_v1) * (tt / s.pulse_tr);
        tt -= s.pulse_tr;
        if (tt < s.pulse_pw) return s.pulse_v2;
        tt -= s.pulse_pw;
        if (tt < s.pulse_tf) return s.pulse_v2 + (s.pulse_v1 - s.pulse_v2) * (tt / s.pulse_tf);
        return s.pulse_v1;
    }
    if (s.type == "EXP") {  // ngspice: the second rise starts at TD2, on top of the first
        const auto& w = s.wf;
        if (t <= w[2]) return w[0];
        const double a = w[0] + (w[1] - w[0]) * (1.0 - std::exp(-(t - w[2]) / w[3]));
        return t <= w[4] ? a : a + (w[0] - w[1]) * (1.0 - std::exp(-(t - w[4]) / w[5]));
    }
    if (s.type == "SFFM") {
        const auto& w = s.wf;
        const double kTwoPi = 6.283185307179586;
        return w[0] + w[1] * std::sin(kTwoPi * w[2] * t + w[3] * std::sin(kTwoPi * w[4] * t));
    }
    if (s.type == "AM") {
        const auto& w = s.wf;
        const double kTwoPi = 6.283185307179586, tt = t - w[5];  // 0 (not VO) until TD, as ngspice
        return tt <= 0.0 ? 0.0 : w[0] + std::sin(kTwoPi * w[4] * tt) * (w[1] + w[2] * std::sin(kTwoPi * w[3] * tt));
    }
    if (s.type == "SINE") {
        const double kPi = 3.141592653589793;
        double phase = s.sine_phase * (kPi / 180.0);
        double tt = t - s.sine_td;
        if (tt <= 0.0) return s.sine_vo + s.sine_va * std::sin(phase);
        return s.sine_vo + s.sine_va * std::exp(-tt * s.sine_theta) *
                               std::sin(2.0 * kPi * s.sine_freq * tt + phase);
    }
    return dcValue;
}

/*
 * First corner of waveformValue() strictly after t (PULSE edges, SIN start);
 * +inf if none. Corners are where the waveform's derivative jumps: a step that
 * straddles one integrates a kink as if it were smooth.
 */
template <class Src>
inline double nextBreakpoint(const Src& s, double t) {
    const double inf = std::numeric_limits<double>::infinity();
    if (s.type == "SINE") return t < s.sine_td ? s.sine_td : inf;
    if (s.type == "EXP") {  // the two rise starts
        double b = inf;
        for (double c : {s.wf[2], s.wf[4]}) if (c > t) b = std::min(b, c);
        return b;
    }
    if (s.type == "AM") return t < s.wf[5] ? s.wf[5] : inf;
    if (s.type == "PWL") {  // the next corner, in absolute time
        const double tt = pwlLocalTime(s, t), shift = t - tt;
        const auto it = std::upper_bound(s.pwl_t.begin(), s.pwl_t.end(), tt);
        if (it != s.pwl_t.end()) return *it + shift;
        if (s.pwl_r < 0.0) return inf;
        const auto r = std::upper_bound(s.pwl_t.begin(), s.pwl_t.end(), s.pwl_r);  // the repeat's first corner
        return r == s.pwl_t.end() ? inf : *r + shift + (s.pwl_t.back() - s.pwl_r);
    }
    if (s.type != "PULSE") return inf;
    const double c[4] = {0.0, s.pulse_tr, s.pulse_tr + s.pulse_pw, s.pulse_tr + s.pulse_pw + s.pulse_tf};
    const double per = s.pulse_per > 0.0 ? s.pulse_per : inf;
    // this period, then the next (k * per would be 0 * inf = NaN for a single pulse)
    double k = t < s.pulse_td ? 0.0 : std::floor((t - s.pulse_td) / per);
    for (int p = 0; p < 2; ++p, ++k) {
        const double base = s.pulse_td + (k == 0.0 ? 0.0 : k * per);
        for (double ci : c)
            if (base + ci > t) return base + ci;
    }
    return inf;
}

// ============================================================================
// DEPENDENT SOURCES (Bench + Block internal modes)
// ============================================================================

/*
 * VCVS - Voltage-Controlled Voltage Source
 * Vout = gain * Vcontrol
 * MNA: Adds branch equation with controlled coefficient.
 */
struct VCVS {
    int outPos, outNeg;
    int ctrlPos, ctrlNeg;
    double gain;
    std::string role;
    std::string name;          // instance name (i(name))
};

struct VCCS {
    int outPos, outNeg;
    int ctrlPos, ctrlNeg;
    double gm;
    std::string role;
};

struct CCVS {
    int outPos, outNeg;
    int ctrlPos, ctrlNeg;      // unused by the solver (legacy sense-node fields)
    double rm;
    std::string role;
    std::string name;          // instance name (i(name))
    int ctrlSource = -1;       // index of the controlling V source in the same block
    std::string ctrlName;      // its instance name (parser resolves ctrlSource from it)
};

struct CCCS {
    int outPos, outNeg;
    int ctrlPos, ctrlNeg;      // unused by the solver (legacy sense-node fields)
    double alpha;
    std::string role;
    int ctrlSource = -1;       // index of the controlling V source in the same block
    std::string ctrlName;      // its instance name (parser resolves ctrlSource from it)
};

/*
 * Behavioural source: B (V= / I=), E/G (VALUE, TABLE, POLY), F/H (POLY). The output (a voltage, which adds
 * a branch current, or a current from + through the source to -) is expr over node voltages and branch
 * currents; refs[k] binds expr->refs[k].
 */
struct Behavioural {
    struct Ref {
        int kind = 0;      // 0: V(a,b) over block-local nodes; I() of 1: V source a, 2: inductor a,
        int a = 0, b = 0;  // 3: voltage behavioural source a (counted among the voltage ones)
        std::string src;   // the source name I() gave, until the block is complete
    };
    std::string name;
    int outPos = 0, outNeg = 0;
    bool voltage = false;
    std::shared_ptr<const acutesim::numparam::Expr> expr;
    std::vector<Ref> refs;
};

// ============================================================================
// SYSTEM SIMULATION SOURCES (Phase 2.9)
// ============================================================================

/*
 * Power Rail (Vdd/Vcc)
 * Models a realistic power supply network (PDN).
 * Model: V_ideal -> [ESL] -> [ESR] -> Node_Chip -> [C_decap] -> GND
 */
;

/*
 * PDK Power Domain (Phase 1.3 Addition)
 * Represents a named voltage domain for PDK-aware design.
 * Multiple instances of the same domain share a common node.
 * Model: Ideal voltage source with optional internal resistance.
 */
struct PDKPowerDomain {
    std::string name;              // Instance name, e.g., "VDD_CORE1"
    std::string domain;            // Domain grouping, e.g., "CORE", "IO", "ANALOG"
    enum Type { VDD, VSS, VDDIO, VBB } type;
    
    int globalNode;                // Assigned node index in circuit matrix
    double nominalVoltage;         // 1.8V, 0V, 3.3V, -0.3V (for VBB)
    double maxCurrent;             // Maximum current capacity [A]
    double internalResistance;     // IR drop model: V = Vnom - I * Rint
    
    // Verilog-A Export Helper
    std::string getVerilogAModule() const {
        if (type == VSS) {
            return "// VSS connects to global ground (node 0)";
        }
        std::string code =
            "module " + name + "_supply(vout);\n"
            "    output vout;\n"
            "    electrical vout;\n"
            "    parameter real vnom = " + std::to_string(nominalVoltage) + ";\n"
            "    parameter real rint = " + std::to_string(internalResistance) + ";\n"
            "    analog begin\n"
            "        V(vout) <+ vnom - I(vout) * rint;\n"
            "    end\n"
            "endmodule\n";
        return code;
    }
};

/*
 * Signal Source (Waveform Generator)
 * Models complex signal inputs (Clock, Data, Noise).
 */
struct SignalSource {
    std::string name;              // "CLK", "DATA_IN"
    int nodePositive;
    int nodeNegative;
    
    enum WaveformType { DC, PULSE, SINE, PWL, NOISE } type;
    
    double amplitude_V;
    double frequency_Hz;
    double phase_deg;
    double offset_V;
    double duty_cycle;             // 0.0 to 1.0 (Pulse only)
    double rise_time_s;            // (Pulse only)
    double fall_time_s;            // (Pulse only)
    
    // For future PWL
    std::vector<std::pair<double, double>> points;
};

// ============================================================================
// DIGITAL VERIFICATION COMPONENTS (Phase 13)
// ============================================================================

enum class LogicState {
    LOW = 0,
    HIGH = 1,
    UNKNOWN = 2, // X
    IMPEDANCE = 3 // Z
};

struct BusProbe {
    std::string name;             // e.g., "DATA_BUS"
    int bit_width;                // 8, 16, 32
    std::vector<int> nodeIndices; // [LSB ... MSB]
    
    // Logic Level Config
    double v_th_low = 0.8;        // Below 0.8V = 0
    double v_th_high = 2.0;       // Above 2.0V = 1
    
    struct DigitalSample {
        double time;
        uint64_t value;
    };
    
};

/*
 * Simulation Environment
 * Global context for the simulation run.
 */
struct SimulationEnvironment {
    double ambient_temp_K = 300.15;   // 27°C
    int temp_count = 1;               // values in the netlist's .temp list: more than one is for .op/.dc only
    double temp_ramp_K_per_s = 0.0;   // For thermal cycling
    
    // Process Corners (Monte Carlo)
    enum Corner { TT, FF, SS, SF, FS } process = TT;
    double vdd_variation_sigma = 0.0; // σ for MC
    double global_voltage_scale = 1.0; // PVT Voltage Scale
    bool gmin_stepping = true;         // .options gminsteps=0 (ngspice): DC goes straight to source stepping

    // .options the solver honours (ngspice names and defaults). itl4 0: the engine's own transient limit (50).
    double reltol = 1e-3, abstol = 1e-12, vntol = 1e-6, trtol = 7.0, gmin = 1e-12, rshunt = 0.0;
    int itl1 = 100, itl2 = 50, itl4 = 0, maxord = 2;
    bool gear = false;                 // .options method=gear
    
    // Noise Profile
    bool thermal_noise_enabled = false;
    bool flicker_noise_enabled = false;
};




// ============================================================================
// NON-LINEAR SEMICONDUCTOR DEVICES
// ============================================================================

/*
 * Diode (D) - PN Junction
 * Shockley Equation:
 *   I_D = I_S * (exp(V_D / (N * V_T)) - 1)
 * Where:
 *   I_S = Saturation current (typ. 1e-14 A)
 *   N   = Emission coefficient (1.0 for ideal, 1.5-2 for real)
 *   V_T = Thermal voltage (kT/q = 25.865 mV at 27 C)
 *
 * Newton-Raphson Linearization:
 *   g_eq = dI/dV = (I_S / (N*V_T)) * exp(V_k / (N*V_T))
 *   I_eq = I(V_k) - g_eq * V_k
 */
struct Diode {
    std::string instanceName;         // e.g. "D1"
    int anode;                        // P-type side (+)
    int cathode;                      // N-type side (-)
    // Defaults match SPICE3 D-model defaults and addDiode(); left uninitialised
    // these were read as garbage by every parser-created diode (NaN results).
    double saturationCurrent_I_S_A = 1e-14;  // I_S [A]
    double emissionCoefficient_N = 1.0;      // N (dimensionless)
    double thermalVoltage_V_T_V = PhysicsConstants::V_THERMAL_NOM;   // V_T [V]

    // Thermal (Optional, default 300.15K)
    double temperature_K = 300.15;

    // Generic PDK Support
    std::string modelName = "diode_generic";
    double area_m2 = 1e-12;       // Junction Area
    double perimeter_m = 4e-6;    // Junction Perimeter

    // SPICE level-1 .model values beyond IS/N (ngspice dio), resolved by the
    // parser with SPICE defaults; area = the instance AREA factor.
    struct Card {
        double rs = 0.0, cjo = 0.0, vj = 1.0, m = 0.5, fc = 0.5, tt = 0.0;
        double jsw = 0.0, ns = 1.0, cjsw = 0.0, mjsw = 0.33, php = 1.0, ikf = 0.0, ikr = 0.0;  // sidewall, knees
        double kf = 0.0, af = 1.0;                                                            // flicker noise
        double bv = 0.0, ibv = 1e-3;
        bool bvGiven = false;
    } card;
    double area = 1.0;
    double pj = 0.0, mult = 1.0;  // PJ= (sidewall perimeter) and M= (the sidewall terms scale by M, not by AREA)
    InstTemp inst;  // card and saturation current are held at the instance temperature (resolved by the parser)
};

struct Mosfet {
    std::string instanceName;             // e.g., "M1"
    int drain;                        // D terminal
    int gate;                         // G terminal
    int source;                       // S terminal
    int body;                         // B (Bulk/Substrate) terminal
    
    // Standard SPICE / PDK Parameters
    std::string modelName;                // .model reference (e.g., "nfet_01v8")
    
    // Dimensions (Standard SPICE)
    double w = 1e-6;                      // Channel Width [m]
    double l = 1e-6;                      // Channel Length [m]
    


    // Layout Parasitics (Geometry)
    struct Geometry {
        double ad = 0.0; // Area Drain
        double as = 0.0; // Area Source
        double pd = 0.0; // Perimeter Drain
        double ps = 0.0; // Perimeter Source
        double nrd = 0.0; // Number of squares Drain
        double nrs = 0.0; // Number of squares Source
        double sa = 0.0, sb = 0.0, sd = 0.0;  // BSIM4 stress (LOD) distances
    } geo;
    // The instance parameters exactly as written (scaled, upper-case keys): a BSIM4
    // device receives only these, so the model's own defaults apply to the rest.
    std::map<std::string, double> given;

    // Advanced Params
    struct Params {
        int nf = 1;      // Number of fingers
        double mult = 1.0;  // Multiplier (M=): scales beta, the junctions and every capacitance
    } extra;

    // SPICE level-1 .model values, resolved by the parser (SPICE defaults for
    // missing keys). resolved = false (the GUI's Verilog-A path, no cards):
    // engine defaults chosen by model name, no capacitances.
    struct Card {
        bool resolved = false, pmos = false;
        double kp = 2e-5, vto = 0.0, lambda = 0.0, gamma = 0.0, phi = 0.6, is = 1e-14, ld = 0.0;
        double tox = 0.0;                                   // > 0: Meyer gate capacitance
        double cgso = 0.0, cgdo = 0.0, cgbo = 0.0;          // overlap [F/m]
        double cbd = 0.0, cbs = 0.0, cj = 0.0, cjsw = 0.0;  // bulk junctions [F, F/m^2, F/m]
        double mj = 0.5, mjsw = 0.33, pb = 0.8, fc = 0.5;
        double js = 0.0;                                    // bulk junction saturation current density [A/m^2]
        double kf = 0.0, af = 1.0;                          // flicker noise (AF is the frequency exponent in ngspice mos1noise)
        double vt = PhysicsConstants::V_THERMAL_NOM;        // bulk junctions' kT/q at the instance temperature
        // Levels 2 and 3 (0: level 1); vbi is tVbi, mu the surface mobility [m^2/Vs] the level's equations use
        int level = 1;
        double vbi = 0.0, mu = 0.0, nfs = 0.0, nsub = 0.0, xd = 0.0, xj = 0.0, delta = 0.0, wd = 0.0, xl = 0.0, xw = 0.0;
        double ucrit = 1e4, uexp = 0.0, vmax = 0.0, neff = 1.0, eta = 0.0, theta = 0.0, kappa = 0.2, alpha = 0.0;
    } card;
    InstTemp inst;  // card is held at the instance temperature (resolved by the parser)

    // Noise Parameters
    double KF = 0.0;     // Flicker noise coefficient
    double AF = 1.0;     // Flicker noise exponent
    double C_ox = 0.0;   // Gate oxide capacitance [F/m^2]

    double temperature_K = 300.15;
};

struct BJT {
    std::string instanceName;
    int nodeCollector = -1;                    // C terminal
    int base = -1;                             // B terminal
    int emitter = -1;                          // E terminal
    int substrate = 0;                         // S terminal (ground when not given, as SPICE)
    std::string modelName = "default";
    bool isNPN = true;                           // true=NPN, false=PNP
    
    // Model Parameters (Per-Instance for Phase 1, ideally per-model lookup)
    double betaF = 100.0;                         // Forward Beta
    double betaR = 1.0;                         // Reverse Beta
    double saturationCurrent_I_S_A = 1e-14;       // Is [A]
    double thermalVoltage_V_T_V = PhysicsConstants::V_THERMAL_NOM;          // Vt [V]

    // Gummel-Poon .model values beyond IS/BF/BR (ngspice bjt level 1), resolved
    // by the parser with SPICE defaults; area = the instance AREA factor.
    struct Card {
        double nf = 1.0, nr = 1.0, vaf = 0.0, var = 0.0, ikf = 0.0, ikr = 0.0;
        double ise = 0.0, ne = 1.5, isc = 0.0, nc = 2.0;
        double rb = 0.0, rc = 0.0, re = 0.0;
        double rbm = 0.0, irb = 0.0, nk = 0.5;                // RBM (= RB unless given), IRB, NK
        double kf = 0.0, af = 1.0;                            // flicker noise
        double cje = 0.0, vje = 0.75, mje = 0.33, cjc = 0.0, vjc = 0.75, mjc = 0.33, xcjc = 1.0;
        double cjs = 0.0, vjs = 0.75, mjs = 0.0, fc = 0.5;
        double iss = 0.0, ns = 1.0;                        // substrate junction
        bool lateral = false;  // SUBS = -1: substrate junction on the base, not the collector (PNP default)
        double tf = 0.0, xtf = 0.0, vtf = 0.0, itf = 0.0, tr = 0.0;
    } card;
    double area = 1.0;
    InstTemp inst;  // card, IS and betas are held at the instance temperature (resolved by the parser)

    double temperature_K = 300.15;
};

/*
 * OSDI device (N): a compiled Verilog-A model (OpenVAF -> .osdi, e.g. BSIM4,
 * BSIM-CMG), as ngspice: N<name> <terminals...> <model> [param value ...];
 * the .model card's type names the OSDI module and its values are the model
 * parameters. Runtime instances live in BlockState::osdiState.
 */
struct OsdiDevice {
    std::string instanceName;
    std::string modelName;
    std::vector<int> nodes;                                    // connected terminals, in module order
    std::vector<std::pair<std::string, double>> params;        // instance parameters
    std::shared_ptr<osdi::Model> model;                        // resolved by the parser
};

/*
 * JFET (J) - Junction Field Effect Transistor
 * Shichman-Hodges Model:
 *   Cutoff:     Vgs <= Vto (for N-ch) or Vgs >= -Vto (for P-ch)
 *   Linear:     Id = Beta * (2*(Vgs-Vto)*Vds - Vds^2) * (1 + lambda*Vds)
 *   Saturation: Id = Beta * (Vgs - Vto)^2 * (1 + lambda*Vds)
 * Where:
 *   Beta = IDSS / Vto^2 (typ. 1e-4 A/V^2)
 *   Vto  = Pinch-off voltage (typ. -2V for N-ch, +2V for P-ch)
 *   lambda = Channel-length modulation (typ. 0.01)
 */
struct JFET {
    std::string instanceName;             // e.g., "J1"
    int drain = -1;                       // D terminal
    int gate = -1;                        // G terminal
    int source = -1;                      // S terminal
    std::string modelName = "default";
    bool isNChannel = true;               // true=N-ch, false=P-ch
    double m = 1.0;                       // parallel multiplier (scales beta)
    
    // Model Parameters
    double beta = 1e-4;                   // Transconductance parameter [A/V^2]
    double Vto = -2.0;                    // Pinch-off voltage [V] (negative for N-ch)
    double lambda = 0.01;                 // Channel-length modulation [1/V]
    double thermalVoltage_V_T_V = PhysicsConstants::V_THERMAL_NOM; // gate junctions: Vt (N applied below)

    // ngspice level 1 .model values beyond BETA/VTO/LAMBDA, resolved by the parser; area = AREA * M
    // (scales BETA, IS, CGS, CGD; RD and RS become parser-added series resistors)
    struct Card {
        double is = 1e-14, n = 1.0, b = 1.0, cgs = 0.0, cgd = 0.0, pb = 1.0, fc = 0.5, kf = 0.0, af = 1.0;
    } card;
    double area = 1.0;

    double temperature_K = 300.15;
};

// ============================================================================
// INTERCONNECTS & TRANSMISSION LINES
// ============================================================================

/*
 * Lossless Transmission Line (T)
 * Telegrapher's Equations yield a time-delayed reflection model:
 *   v_1(t) + Z_0 * i_1(t) = v_2(t - T_d) + Z_0 * i_2(t - T_d)
 * Modeled as history-dependent current sources at each port.
 */
struct TransmissionLine {
    int nodePort1Pos, nodePort1Neg;       // Port 1 terminals
    int nodePort2Pos, nodePort2Neg;       // Port 2 terminals
    double characteristicImpedance_Z0_ohms; // Z_0 [Ω]
    double propagationDelay_Td_s;         // T_d [seconds]
    std::string name;
    bool icGiven = false;                 // IC=v1,i1,v2,i2 (with UIC): port voltages and currents into the line
    double ic[4] = {0, 0, 0, 0};
};

/*
 * Probe (Measurement Tool)
 * Does not affect physics (infinite impedance).
 * Used to request V(pos) - V(neg) for plotting/GUI.
 */
struct Probe {
    std::string name;
    int nodePositive;
    int nodeNegative;
};

// ============================================================================
// TENSOR HIERARCHY (Hierarchical Netlist Structure)
// ============================================================================

/*
 * TensorBlock - A reusable sub-circuit definition (e.g., Inverter, SRAM cell).
 * Components use LOCAL node indices (0 = local ground, 1..N = internal nodes).
 */

struct TensorBlock {
    std::string name;
    int numInternalNodes = 0;
    std::vector<std::string> pinNames;      // Metadata for Verilog-A port naming
    std::vector<int> portNodeIndices;         // Pin index -> internal node index (for subcircuit blocks)

    // Hierarchical interface pin metadata (populated during subcircuit resolution)
    struct InterfacePin {
        std::string name;        // Pin/PowerPort component name
        std::string netUUIDStr;  // Serialized NetUUID (from NetUUID::toString())
        int nodeIndex = 0;       // Solver node index inside this block
        bool isPower = false;    // true for PowerPort-type boundary pins
        std::string direction;   // "Input", "Output", "BiDir", "Power"
    };
    std::vector<InterfacePin> interfacePins;
    
    // Passive Components
    std::vector<Resistor> resistors;
    std::vector<Switch> switches;
    std::vector<Capacitor> capacitors;
    std::vector<Inductor> inductors;
    std::vector<MutualInductor> mutualInductors;
    
    // Independent Sources
    std::vector<VoltageSource> voltageSources;
    std::vector<CurrentSource> currentSources;

    // System Simulation Sources (Phase 2.9)
    std::vector<SignalSource> signalSources;
    
    // PDK Power Domains (Phase 1.3)
    std::vector<PDKPowerDomain> powerDomains;
    
    // Controlled Sources
    std::vector<VCVS> voltageControlledVoltageSources;
    std::vector<VCCS> voltageControlledCurrentSources;
    std::vector<CCVS> currentControlledVoltageSources;
    std::vector<CCCS> currentControlledCurrentSources;
    std::vector<Behavioural> behavioural;
    
    // Semiconductors
    std::vector<Diode> diodes;
    std::vector<Mosfet> mosfets;
    std::vector<BJT> bjts;
    std::vector<JFET> jfets;
    std::vector<OsdiDevice> osdiDevices;
    
    // Interconnects
    std::vector<TransmissionLine> transmissionLines;
    
    // Aux
    std::vector<Probe> probes;
    std::vector<BusProbe> busProbes;

    // --- Phase 1.5: Parasitic Extraction Support ---
    struct ParasiticResistor {
        std::string name;
        int node1, node2;
        double R;
    };
    struct ParasiticCapacitor {
        std::string name;
        int node1, node2;
        double C;
    };
    struct WireSegment {
        std::string layer;      // e.g. "M1", "M2"
        int node1, node2;
        double width_um;
        double length_um;
        double r_per_um = 0.0;
        double c_per_um = 0.0;
    };

    std::vector<ParasiticResistor> parasiticResistors;
    std::vector<ParasiticCapacitor> parasiticCapacitors;
    std::vector<WireSegment> wireSegments;

    bool isFrozen = false;       // Phase 1: Structural Immutability
    size_t topologyHash = 0;     // Phase 4: Symbolic Reuse Key

    // --- Helper Methods ---
    void freeze() {
        if (isFrozen) return;
        isFrozen = true;
        computeTopologyHash();
    }

    void computeTopologyHash() {
        // Phase 1.4: Canonical Graph Hashing
        // Uses orientation-invariant edges and BFS-based node relabeling.
        std::vector<GraphCanonical::CanonicalEdge> edges;
        
        // Type tags: 0=Resistor, 1=Capacitor, 2=Inductor, 3=Diode, 4=Mosfet, 5=BJT
        for (const auto& r : resistors) 
            edges.emplace_back(r.nodeTerminal1, r.nodeTerminal2, 0);
        for (const auto& c : capacitors) 
            edges.emplace_back(c.nodePlate1, c.nodePlate2, 1);
        for (const auto& l : inductors) 
            edges.emplace_back(l.nodeCoil1, l.nodeCoil2, 2);
        for (const auto& d : diodes) 
            edges.emplace_back(d.anode, d.cathode, 3);
        for (const auto& m : mosfets) {
            // MOSFET is a 4-terminal device; we represent as (D-S) and (G-S) edges
            edges.emplace_back(m.drain, m.source, 4);
            edges.emplace_back(m.gate, m.source, 4);
        }
        for (const auto& b : bjts) {
            // BJT: (C-E) and (B-E) edges
            edges.emplace_back(b.nodeCollector, b.emitter, 5);
            edges.emplace_back(b.base, b.emitter, 5);
            if (b.substrate != 0)  // substrate junction: C-S (vertical) or B-S (lateral)
                edges.emplace_back(b.card.lateral ? b.base : b.nodeCollector, b.substrate, 5);
        }
        for (const auto& o : osdiDevices)  // OSDI device: terminal 0 to each other terminal
            for (size_t k = 1; k < o.nodes.size(); ++k) edges.emplace_back(o.nodes[0], o.nodes[k], 6);
        
        // Ground is assumed to be node 0 for blocks
        topologyHash = GraphCanonical::computeTopologyHash(edges, 0);
    }

    void updateNodeCount(int node) {
        if (isFrozen) return; // Prevent node count updates on frozen blocks
        if (node > numInternalNodes) numInternalNodes = node;
    }

    void addResistor(const std::string& name, int n1, int n2, double R) {
        assert(n1 >= 0 && n2 >= 0 && "Node indices must be not negative");
        assert(R > 0 && "Resistance must be positive");
        if (isFrozen) return;
        resistors.push_back({name, n1, n2, R});
        updateNodeCount(n1); updateNodeCount(n2);
    }
    // Backward compatibility overload (generates name if missing, though typically netlist parser provides it)
    void addResistor(int n1, int n2, double R) {
        static int r_count = 0;
        addResistor("R_AUTO_" + std::to_string(++r_count), n1, n2, R);
    }
    void addCapacitor(const std::string& name, int n1, int n2, double C) {
        assert(n1 >= 0 && n2 >= 0 && "Node indices must be not negative");
        assert(C > 0 && "Capacitance must be positive");
        if (isFrozen) return;
        Capacitor cap = {name, n1, n2, C};
        capacitors.push_back(cap);
        updateNodeCount(n1); updateNodeCount(n2);
    }
    void addCapacitor(int n1, int n2, double C) {
        static int c_count = 0;
        addCapacitor("C_AUTO_" + std::to_string(++c_count), n1, n2, C);
    }
    void addInductor(const std::string& name, int n1, int n2, double L) {
        assert(n1 >= 0 && n2 >= 0 && "Node indices must be not negative");
        assert(L > 0 && "Inductance must be positive");
        if (isFrozen) return;
        Inductor ind = {name, n1, n2, L};
        ind.model_name = "ind_generic";

        inductors.push_back(ind);
        updateNodeCount(n1); updateNodeCount(n2);
    }
     void addInductor(int n1, int n2, double L) {
        static int l_count = 0;
        addInductor("L_AUTO_" + std::to_string(++l_count), n1, n2, L);
    }
    void addVoltageSource(int nPos, int nNeg, double V, std::string type = "") {
        assert(nPos >= 0 && nNeg >= 0 && "Node indices must be not negative");
        if (isFrozen) return;
        voltageSources.push_back({nPos, nNeg, V, type});
        updateNodeCount(nPos); updateNodeCount(nNeg);
    }
    void addCurrentSource(int nPos, int nNeg, double I) {
        if (isFrozen) return;
        currentSources.push_back({nPos, nNeg, I});
        updateNodeCount(nPos); updateNodeCount(nNeg);
    }
    
    // System Simulation Helpers
    
    void addSignalSource(const std::string& name, int nPos, int nNeg, 
                         SignalSource::WaveformType type, double amp, double freq) {
        if (isFrozen) return;
        SignalSource src;
        src.name = name;
        src.nodePositive = nPos;
        src.nodeNegative = nNeg;
        src.type = type;
        src.amplitude_V = amp;
        src.frequency_Hz = freq;
        // Defaults
        src.phase_deg = 0.0;
        src.offset_V = 0.0;
        src.duty_cycle = 0.5;
        
        signalSources.push_back(src);
        updateNodeCount(nPos); updateNodeCount(nNeg);
    }
    void addDiode(const std::string& name, int anode, int cathode, double Is, double N = 1.0, double Vt = PhysicsConstants::V_THERMAL_NOM) {
        if (isFrozen) return;
        Diode d;
        d.instanceName = name;
        d.anode = anode;
        d.cathode = cathode;
        d.saturationCurrent_I_S_A = Is;
        d.emissionCoefficient_N = N;
        d.thermalVoltage_V_T_V = Vt;
        d.modelName = "diode_generic"; // Default PDK model
        diodes.push_back(d);
        updateNodeCount(anode); updateNodeCount(cathode);
    }
    void addDiode(int anode, int cathode, double Is, double N = 1.0, double Vt = PhysicsConstants::V_THERMAL_NOM) {
        static int d_count = 0;
        addDiode("D_AUTO_" + std::to_string(++d_count), anode, cathode, Is, N, Vt);
    }
    void addMosfet(const std::string& name, int D, int G, int S, int B, 
                   const std::string& model, double W, double L) {
        if (isFrozen) return;
        Mosfet m;
        m.instanceName = name;
        m.drain = D; m.gate = G; m.source = S; m.body = B;
        m.modelName = model;
        m.w = W; m.l = L;
        
        mosfets.push_back(m);
        updateNodeCount(D); updateNodeCount(G); updateNodeCount(S); updateNodeCount(B);
    }
    void addBJT(const std::string& name, int C, int B, int E, 
                const std::string& model, bool isNPN = true,
                double betaF = 100.0, double betaR = 1.0, 
                double Is = 1e-14, double Vt = PhysicsConstants::V_THERMAL_NOM) {
        if (isFrozen) return;
        
        BJT newBjt;
        newBjt.instanceName = name;
        newBjt.nodeCollector = C;
        newBjt.base = B;
        newBjt.emitter = E;
        newBjt.modelName = model;
        newBjt.isNPN = isNPN;
        newBjt.betaF = betaF;
        newBjt.betaR = betaR;
        newBjt.saturationCurrent_I_S_A = Is;
        newBjt.thermalVoltage_V_T_V = Vt;
        
        // Defensive Fix: Prevent invalid Is
        if (newBjt.saturationCurrent_I_S_A <= 1e-20) {
             std::cerr << "[FIX] BJT " << name << " initialized with Is=" << newBjt.saturationCurrent_I_S_A 
                      << ". Forcing default 1e-14." << std::endl;
            newBjt.saturationCurrent_I_S_A = 1e-14;
        }

        bjts.push_back(newBjt);
        updateNodeCount(C); updateNodeCount(B); updateNodeCount(E);
    }
    void addJFET(const std::string& name, int D, int G, int S, 
                 const std::string& model, bool isNChannel = true,
                 double beta = 1e-4, double Vto = -2.0, double lambda = 0.01) {
        if (isFrozen) return;
        
        JFET newJfet;
        newJfet.instanceName = name;
        newJfet.drain = D;
        newJfet.gate = G;
        newJfet.source = S;
        newJfet.modelName = model;
        newJfet.isNChannel = isNChannel;
        newJfet.beta = beta;
        newJfet.Vto = Vto;
        newJfet.lambda = lambda;
        
        jfets.push_back(newJfet);
        updateNodeCount(D); updateNodeCount(G); updateNodeCount(S);
    }
    void addTransmissionLine(int p1p, int p1n, int p2p, int p2n, double Z0, double Td, const std::string& name = "") {
        if (isFrozen) return;
        transmissionLines.push_back({p1p, p1n, p2p, p2n, Z0, Td, name});
        updateNodeCount(p1p); updateNodeCount(p1n); updateNodeCount(p2p); updateNodeCount(p2n);
    }
    void addProbe(const std::string& name, int nPos, int nNeg) {
        if (isFrozen) return;
        probes.push_back({name, nPos, nNeg});
        // NOTE: Probes do NOT call updateNodeCount because they are passive observers
        // and should not inflate the matrix dimensions.
    }

    // Dependent Sources
    void addVCVS(int outPos, int outNeg, int ctrlPos, int ctrlNeg, double gain, std::string role = "") {
        if (isFrozen) return;
        voltageControlledVoltageSources.push_back({outPos, outNeg, ctrlPos, ctrlNeg, gain, role});
        updateNodeCount(outPos); updateNodeCount(outNeg); updateNodeCount(ctrlPos); updateNodeCount(ctrlNeg);
    }
    void addVCCS(int outPos, int outNeg, int ctrlPos, int ctrlNeg, double gm, std::string role = "") {
        if (isFrozen) return;
        voltageControlledCurrentSources.push_back({outPos, outNeg, ctrlPos, ctrlNeg, gm, role});
        updateNodeCount(outPos); updateNodeCount(outNeg); updateNodeCount(ctrlPos); updateNodeCount(ctrlNeg);
    }
    void addCCVS(int outPos, int outNeg, int ctrlPos, int ctrlNeg, double rm, std::string role = "") {
        if (isFrozen) return;
        // Sensing via ctrlPos-ctrlNeg (small sense resistor model in stamps.h)
        currentControlledVoltageSources.push_back({outPos, outNeg, ctrlPos, ctrlNeg, rm, role});
        updateNodeCount(outPos); updateNodeCount(outNeg); updateNodeCount(ctrlPos); updateNodeCount(ctrlNeg);
    }
    void addCCCS(int outPos, int outNeg, int ctrlPos, int ctrlNeg, double alpha, std::string role = "") {
        if (isFrozen) return;
        currentControlledCurrentSources.push_back({outPos, outNeg, ctrlPos, ctrlNeg, alpha, role});
        updateNodeCount(outPos); updateNodeCount(outNeg); updateNodeCount(ctrlPos); updateNodeCount(ctrlNeg);
    }
};

// ============================================================================
// SUBCIRCUIT DEFINITIONS (Hierarchical IC Packages)
// ============================================================================

/*
 * SubCircuitDefinition - Represents the internal circuitry of an IC package.
 * Used for hierarchical simulation where ICs like ULN2003A expand to their
 * internal Darlington transistors, resistors, and protection diodes.
 *
 * Example: ULN2003A has 16 pins, 7 Darlington channels internally.
 * Pin mapping connects external package pins to internal node indices.
 */
struct SubCircuitDefinition {
    std::string name;                         // e.g., "ULN2003A"
    std::string description;                  // Human-readable description
    
    // Package Information
    int pinCount = 0;                         // Total pins (e.g., 16 for DIP-16)
    std::vector<std::string> pinNames;        // e.g., {"1B", "2B", ..., "GND", "VCC"}
    
    // Internal Circuitry (uses local node indices starting from 0)
    TensorBlock internalBlock;
    
    // Pin-to-Internal-Node Mapping
    // Key: pin index (0-based), Value: internal node index
    std::map<int, int> pinToInternalNode;

    // Pin-to-NetUUID Mapping (Persistence)
    // Stores the original NetUUID string of the net connected to this pin at creation time.
    std::vector<std::string> pinNetUUIDs;
    
    // Helper to get internal node for a given pin
    int getInternalNode(int pinIndex) const {
        auto it = pinToInternalNode.find(pinIndex);
        return (it != pinToInternalNode.end()) ? it->second : -1;
    }
};

/*
 * ICPackageInstance - An instance of a SubCircuitDefinition placed on the schematic.
 * Stores the mapping from package pins to global netlist nodes.
 */
struct ICPackageInstance {
    std::string instanceName;                 // e.g., "U1"
    std::string subcircuitName;               // Reference to SubCircuitDefinition
    
    // Pin-to-Global-Node Mapping (set during netlist generation)
    std::map<int, int> pinToGlobalNode;       // Key: pin index, Value: global node
    
    // Position (for GUI)
    double x = 0.0, y = 0.0;
    double rotation = 0.0;
};

// BlockInstance - A placed instance of a TensorBlock in the global netlist.

struct BlockInstance {
    std::string blockName;                // Name of the TensorBlock template
    std::vector<int> nodeMapping;         // Maps local node index -> global node index
    std::map<std::string, double> parameters; // Instance-specific parameters (e.g. "W", "L")
    std::string name;                     // hierarchical instance path (X1.X2), for branch-current names
};

// BlockState - Dynamic state for a block instance (Solving the Shared History Bug)
struct BlockState {
    struct History {
        std::vector<double> v; // [t-1, t-2, ...]
        std::vector<double> i; // [t-1, t-2, ...]
        void resize(size_t n) { v.resize(n, 0.0); i.resize(n, 0.0); }
    };

    // Charge history of a nonlinear device after the last accepted step, per
    // charge branch (type domain): MOSFET gs, gd, gb (Meyer + overlap), bd, bs;
    // diode: the junction; BJT: be, bc, bx (external base), substrate.
    struct DeviceCharge {
        double v[3][5] = {};  // branch voltages, last three accepted steps (v[0] latest; LTE)
        double q[5] = {}, i[5] = {};
        double half[3] = {};  // MOSFET Meyer half-capacitances gs, gd, gb (ngspice averages two)
        // Diode / BJT junction limiting (ngspice state0): the limited junction
        // voltages of the last Newton iteration (diode vd; BJT vbe, vbc).
        // jvState 0: none yet -> start at vcrit (ngspice MODEINITJCT).
        double jv[2] = {};
        int jvState = 0;
    };

    std::vector<History> capacitorState;
    std::vector<History> inductorState;
    std::vector<DeviceCharge> mosfetState, diodeState, bjtState, jfetState;
    std::vector<char> swOn, swPrev;  // per Switch: the state the last stamp used / the last accepted one
    // Lossless line: the outgoing waves w = v + Z0*i at both ports after every accepted step (t[0] = 0),
    // read back TD later; breaks: where a kink in w reaches the far port (the transient lands on them)
    struct LineHistory { std::vector<double> t, w1, w2; std::set<double> breaks; };
    std::vector<LineHistory> lineState;
    std::vector<std::shared_ptr<osdi::Instance>> osdiState;  // per OsdiDevice (TensorNetlist::instantiateOsdi)
    std::vector<std::vector<BusProbe::DigitalSample>> busProbeState; 
    // Future: BJT OP, etc.

    // Initialize state based on block definition
    void initialize(const TensorBlock& block) {
        capacitorState.resize(block.capacitors.size());
        for(auto& h : capacitorState) h.resize(3); // default 3 slots

        inductorState.resize(block.inductors.size());
        for(auto& h : inductorState) h.resize(3);

        swOn.resize(block.switches.size());
        swPrev.resize(block.switches.size());
        for (size_t i = 0; i < block.switches.size(); ++i) swOn[i] = swPrev[i] = block.switches[i].init;
        mosfetState.resize(block.mosfets.size());
        diodeState.resize(block.diodes.size());
        bjtState.resize(block.bjts.size());
        jfetState.resize(block.jfets.size());
        osdiState.resize(block.osdiDevices.size());
        busProbeState.resize(block.busProbes.size());
        lineState.resize(block.transmissionLines.size());
    }
};

// TensorNetlist - The top-level hierarchical circuit container.

class TensorNetlist {
public:
    std::map<std::string, TensorBlock> blockDefinitions; // Library of blocks
    TensorBlock globalBlock;                             // Top-level components
    
    // State Storage (Per-instance)
    BlockState globalState;
    std::vector<BlockState> instanceStates;              // Index matches instances vector

    std::vector<BlockInstance> instances;                // Placed block instances
    int numGlobalNodes = 0;

    // Create the OSDI runtime instance of every OSDI device (global block and
    // every placed instance) whose state slot is empty; tempK is the device
    // temperature handed to setup_instance. Throws on model errors.
    void instantiateOsdi(double tempK);
    
    // Hierarchical Support
    std::vector<std::pair<std::string, std::string>> probes; // Name, GlobalNodeString
    std::map<std::string, std::shared_ptr<TensorBlock>> subBlocks; // Resolved sub-circuits

    // Global Simulation Context (Phase 2.9)
    SimulationEnvironment environment;

    // SPICE .MODEL card library
    std::map<std::string, ModelCard> modelCards;

    // .ic V(node)=value: forced during the transient's DC bias, or the initial
    // node voltages with UIC
    std::vector<std::pair<int, double>> nodeICs;

    // .nodeset V(node)=value: held at value through the first DC solve, then released
    std::vector<std::pair<int, double>> nodeSets;

    // Node names for messages: [i] = name of global node i, hierarchical for subcircuit internals ("x1.n"); "" if none
    std::vector<std::string> nodeNames;

    void addModelCard(const ModelCard& card) {
        modelCards[card.name] = card;
    }

    const ModelCard* findModelCard(const std::string& name) const {
        auto it = modelCards.find(name);
        return (it != modelCards.end()) ? &it->second : nullptr;
    }

    TensorNetlist() { 
        globalBlock.name = "GLOBAL"; 
        globalState.initialize(globalBlock); // Init empty global state
    }

    void defineBlock(TensorBlock block) {
        block.freeze();
        blockDefinitions[block.name] = block;
    }
    
    void addInstance(const std::string& blockName, std::vector<int> nodeMap, const std::string& name = "") {
        // Sync numGlobalNodes with user-supplied mapping to avoid collisions
        for (int n : nodeMap) if (n > numGlobalNodes) numGlobalNodes = n;

        // Validation and Auto-Expansion of Node Map
        auto it = blockDefinitions.find(blockName);
        if (it != blockDefinitions.end()) {
             const TensorBlock& block = it->second;
             // Ensure nodeMap defines at least the pins? 
             // Ideally we check against pinCount.
             
             // Expand nodeMap for internal nodes
             // Typically pin nodes are 1..pinCount.
             // Internal nodes are pinCount+1..numInternalNodes.
             // Map them to new global nodes.
             
             // Local nodes are 1..N and every consumer reads nodeMapping[local-1]
             // (tensorizeNetlist, stampAllElements), so N entries. The old N+1
             // allocated one phantom floating node per instance.
             int requiredSize = block.numInternalNodes;
             
             if ((int)nodeMap.size() < requiredSize) {
                 int currentSize = nodeMap.size();
                 for (int i = currentSize; i < requiredSize; ++i) {
                     // Allocate new global node
                     numGlobalNodes++;
                     nodeMap.push_back(numGlobalNodes);
                 }
             }
        }
        
        instances.push_back({blockName, nodeMap, {}, name});
        
        // Initialize state for new instance
        BlockState newState;
        if (it != blockDefinitions.end()) {
             newState.initialize(it->second);
        }
        instanceStates.push_back(newState);


        for (int n : nodeMap) if (n > numGlobalNodes) numGlobalNodes = n;
    }
    
    // --- Convenience Methods for Global Block ---
    void addResistor(int n1, int n2, double R) {
        globalBlock.addResistor(n1, n2, R);
        if (n1 > numGlobalNodes) numGlobalNodes = n1;
        if (n2 > numGlobalNodes) numGlobalNodes = n2;
    }
    void addResistor(const std::string& name, int n1, int n2, double R) {
        globalBlock.addResistor(name, n1, n2, R);
        if (n1 > numGlobalNodes) numGlobalNodes = n1;
        if (n2 > numGlobalNodes) numGlobalNodes = n2;
    }
    void addCapacitor(int n1, int n2, double C) {
        addCapacitor("", n1, n2, C);
    }
    void addCapacitor(const std::string& name, int n1, int n2, double C) {
        globalBlock.addCapacitor(name, n1, n2, C);
        // Sync global state
        globalState.capacitorState.emplace_back();
        globalState.capacitorState.back().resize(3);

        if (n1 > numGlobalNodes) numGlobalNodes = n1;
        if (n2 > numGlobalNodes) numGlobalNodes = n2;
    }
    void addInductor(int n1, int n2, double L) {
        addInductor("", n1, n2, L);
    }
    void addInductor(const std::string& name, int n1, int n2, double L) {
        globalBlock.addInductor(name, n1, n2, L);
        globalState.inductorState.emplace_back();
        globalState.inductorState.back().resize(3);

        if (n1 > numGlobalNodes) numGlobalNodes = n1;
        if (n2 > numGlobalNodes) numGlobalNodes = n2;
    }
    void addVoltageSource(int nPos, int nNeg, double V, std::string type = "") {
        globalBlock.addVoltageSource(nPos, nNeg, V, type);
        if (nPos > numGlobalNodes) numGlobalNodes = nPos;
        if (nNeg > numGlobalNodes) numGlobalNodes = nNeg;
    }
    void addCurrentSource(int nPos, int nNeg, double I) {
        globalBlock.addCurrentSource(nPos, nNeg, I);
        if (nPos > numGlobalNodes) numGlobalNodes = nPos;
        if (nNeg > numGlobalNodes) numGlobalNodes = nNeg;
    }
    void addSignalSource(const std::string& name, int nPos, int nNeg, 
                         SignalSource::WaveformType type, double amp, double freq) {
        globalBlock.addSignalSource(name, nPos, nNeg, type, amp, freq);
        if (nPos > numGlobalNodes) numGlobalNodes = nPos;
        if (nNeg > numGlobalNodes) numGlobalNodes = nNeg;
    }
    void addDiode(const std::string& name, int anode, int cathode, double Is) {
        globalBlock.addDiode(name, anode, cathode, Is);
        globalState.diodeState.emplace_back();
        if (anode > numGlobalNodes) numGlobalNodes = anode;
        if (cathode > numGlobalNodes) numGlobalNodes = cathode;
    }
    void addDiode(int anode, int cathode, double Is) {
        globalBlock.addDiode(anode, cathode, Is);
        globalState.diodeState.emplace_back();
        if (anode > numGlobalNodes) numGlobalNodes = anode;
        if (cathode > numGlobalNodes) numGlobalNodes = cathode;
    }
    void addMosfet(const std::string& name, int D, int G, int S, int B,
                   const std::string& model, double W, double L) {
        globalBlock.addMosfet(name, D, G, S, B, model, W, L);
        globalState.mosfetState.emplace_back();
        numGlobalNodes = std::max({numGlobalNodes, D, G, S, B});
    }
    void addBJT(const std::string& name, int C, int B, int E, 
                const std::string& model, bool isNPN = true,
                double betaF = 100.0, double betaR = 1.0, 
                double Is = 1e-14, double Vt = PhysicsConstants::V_THERMAL_NOM) {
        globalBlock.addBJT(name, C, B, E, model, isNPN, betaF, betaR, Is, Vt);
        globalState.bjtState.emplace_back();
        numGlobalNodes = std::max({numGlobalNodes, C, B, E});
    }
    void addJFET(const std::string& name, int D, int G, int S, 
                 const std::string& model, bool isNChannel = true,
                 double beta = 1e-4, double Vto = -2.0, double lambda = 0.01) {
        globalBlock.addJFET(name, D, G, S, model, isNChannel, beta, Vto, lambda);
        numGlobalNodes = std::max({numGlobalNodes, D, G, S});
    }
    void addTransmissionLine(int p1p, int p1n, int p2p, int p2n, double Z0, double Td, const std::string& name = "") {
        globalBlock.addTransmissionLine(p1p, p1n, p2p, p2n, Z0, Td, name);
        numGlobalNodes = std::max({numGlobalNodes, p1p, p1n, p2p, p2n});
    }
    void addProbe(const std::string& name, int nPos, int nNeg) {
        globalBlock.addProbe(name, nPos, nNeg);
        // NOTE: Probes do NOT inflate numGlobalNodes because they are passive observers.
    }
    void addVCVS(int outPos, int outNeg, int ctrlPos, int ctrlNeg, double gain, std::string role = "") {
        globalBlock.addVCVS(outPos, outNeg, ctrlPos, ctrlNeg, gain, role);
        numGlobalNodes = std::max({numGlobalNodes, outPos, outNeg, ctrlPos, ctrlNeg});
    }
    void addVCCS(int outPos, int outNeg, int ctrlPos, int ctrlNeg, double gm, std::string role = "") {
        globalBlock.addVCCS(outPos, outNeg, ctrlPos, ctrlNeg, gm, role);
        numGlobalNodes = std::max({numGlobalNodes, outPos, outNeg, ctrlPos, ctrlNeg});
    }
    void addCCVS(int outPos, int outNeg, int ctrlPos, int ctrlNeg, double rm, std::string role = "") {
        globalBlock.addCCVS(outPos, outNeg, ctrlPos, ctrlNeg, rm, role);
        numGlobalNodes = std::max({numGlobalNodes, outPos, outNeg, ctrlPos, ctrlNeg});
    }
    void addCCCS(int outPos, int outNeg, int ctrlPos, int ctrlNeg, double alpha, std::string role = "") {
        globalBlock.addCCCS(outPos, outNeg, ctrlPos, ctrlNeg, alpha, role);
        numGlobalNodes = std::max({numGlobalNodes, outPos, outNeg, ctrlPos, ctrlNeg});
    }
    void addBusProbe(const std::string& name, const std::vector<int>& nodes) {
        BusProbe bp;
        bp.name = name;
        bp.bit_width = nodes.size();
        bp.nodeIndices = nodes;
        globalBlock.busProbes.push_back(bp);
    }
};

// ============================================================================
// SUBCIRCUIT EXTRACTION (Phase 1.7 - Component Abstraction)
// ============================================================================

/**
 * PortInfo - Describes an external port for subcircuit extraction.
 * Used to specify which nodes in the source netlist become pins in the subcircuit.
 */
struct PortInfo {
    std::string pinName;      // Name for the pin (e.g., "VIN", "VOUT", "GND")
    int globalNodeId;         // Node ID in the source TensorNetlist
    bool isPowerPin = false;  // True for VDD/VSS pins (affects symbol rendering)
    std::string netUUIDStr;   // Serialized NetUUID for persistence
};

/**
 * extractSubCircuit
 * 
 * Converts a flat TensorNetlist into a reusable SubCircuitDefinition.
 * 
 * @param source      The TensorNetlist to extract from (typically a Bench/Block design)
 * @param name        Name for the new subcircuit (e.g., "MyInverter")
 * @param description Human-readable description
 * @param ports       List of PortInfo specifying which nodes become external pins
 * @return            A SubCircuitDefinition ready for library storage
 * 
 * The algorithm:
 * 1. Create a new TensorBlock from the source's globalBlock.
 * 2. Build a node remapping table (global node → local node).
 * 3. Port nodes are mapped to sequential local indices (1, 2, 3...).
 * 4. Internal-only nodes are mapped after port nodes.
 * 5. All component node references are rewritten using the remap table.
 * 6. The resulting SubCircuitDefinition has pinToInternalNode set correctly.
 */
inline SubCircuitDefinition extractSubCircuit(
    const TensorNetlist& source,
    const std::string& name,
    const std::string& description,
    const std::vector<PortInfo>& ports)
{
    SubCircuitDefinition result;
    result.name = name;
    result.description = description;
    result.pinCount = static_cast<int>(ports.size());
    
    // Build the pin names and port-to-node mapping
    // Local node 0 = internal ground (if present)
    // Local nodes 1..N = external pins
    // Local nodes N+1.. = internal nodes
    
    std::map<int, int> globalToLocal;  // Global node → Local node
    int nextLocalNode = 1;  // Start at 1 (0 is reserved for ground)
    
    // Step 1: Map port nodes first (in order specified)
    for (size_t i = 0; i < ports.size(); ++i) {
        const auto& port = ports[i];
        result.pinNames.push_back(port.pinName);
        result.pinNetUUIDs.push_back(port.netUUIDStr);
        
        if (globalToLocal.find(port.globalNodeId) == globalToLocal.end()) {
            globalToLocal[port.globalNodeId] = nextLocalNode;
            result.pinToInternalNode[static_cast<int>(i)] = nextLocalNode;
            nextLocalNode++;
        } else {
            // Port references an already-mapped node (shared pin)
            result.pinToInternalNode[static_cast<int>(i)] = globalToLocal[port.globalNodeId];
        }
    }
    
    // Step 2: Collect all nodes used in the source netlist
    std::set<int> allNodes;
    const auto& block = source.globalBlock;
    
    for (const auto& r : block.resistors) {
        allNodes.insert(r.nodeTerminal1);
        allNodes.insert(r.nodeTerminal2);
    }
    for (const auto& c : block.capacitors) {
        allNodes.insert(c.nodePlate1);
        allNodes.insert(c.nodePlate2);
    }
    for (const auto& l : block.inductors) {
        allNodes.insert(l.nodeCoil1);
        allNodes.insert(l.nodeCoil2);
    }
    for (const auto& vs : block.voltageSources) {
        allNodes.insert(vs.nodePositive);
        allNodes.insert(vs.nodeNegative);
    }
    for (const auto& cs : block.currentSources) {
        allNodes.insert(cs.nodePositive);
        allNodes.insert(cs.nodeNegative);
    }
    for (const auto& d : block.diodes) {
        allNodes.insert(d.anode);
        allNodes.insert(d.cathode);
    }
    for (const auto& m : block.mosfets) {
        allNodes.insert(m.drain);
        allNodes.insert(m.gate);
        allNodes.insert(m.source);
        allNodes.insert(m.body);
    }
    for (const auto& b : block.bjts) {
        allNodes.insert(b.nodeCollector);
        allNodes.insert(b.base);
        allNodes.insert(b.emitter);
        allNodes.insert(b.substrate);
    }
    for (const auto& o : block.osdiDevices)
        for (int n : o.nodes) allNodes.insert(n);
    
    // Step 3: Map internal-only nodes
    for (int node : allNodes) {
        if (node == 0) continue;  // Ground stays as 0
        if (globalToLocal.find(node) == globalToLocal.end()) {
            globalToLocal[node] = nextLocalNode++;
        }
    }
    globalToLocal[0] = 0;  // Ground is always 0
    
    // Helper lambda to remap a node
    auto remap = [&](int globalNode) -> int {
        auto it = globalToLocal.find(globalNode);
        return (it != globalToLocal.end()) ? it->second : 0;
    };
    
    // Step 4: Copy and remap all components
    TensorBlock& out = result.internalBlock;
    out.name = name + "_internal";
    out.numInternalNodes = nextLocalNode - 1;
    out.pinNames = block.pinNames; // Copy pin metadata
    
    // Resistors
    for (const auto& r : block.resistors) {
        Resistor newR = r;
        newR.nodeTerminal1 = remap(r.nodeTerminal1);
        newR.nodeTerminal2 = remap(r.nodeTerminal2);
        out.resistors.push_back(newR);
    }
    
    // Capacitors
    for (const auto& c : block.capacitors) {
        Capacitor newC = c;
        newC.nodePlate1 = remap(c.nodePlate1);
        newC.nodePlate2 = remap(c.nodePlate2);
        out.capacitors.push_back(newC);
    }
    
    // Inductors
    for (const auto& l : block.inductors) {
        Inductor newL = l;
        newL.nodeCoil1 = remap(l.nodeCoil1);
        newL.nodeCoil2 = remap(l.nodeCoil2);
        out.inductors.push_back(newL);
    }
    
    // Voltage Sources (Note: These become internal sources in the subcircuit)
    for (const auto& vs : block.voltageSources) {
        VoltageSource newVS = vs;
        newVS.nodePositive = remap(vs.nodePositive);
        newVS.nodeNegative = remap(vs.nodeNegative);
        out.voltageSources.push_back(newVS);
    }
    
    // Current Sources
    for (const auto& cs : block.currentSources) {
        CurrentSource newCS = cs;
        newCS.nodePositive = remap(cs.nodePositive);
        newCS.nodeNegative = remap(cs.nodeNegative);
        out.currentSources.push_back(newCS);
    }
    
    // Diodes
    for (const auto& d : block.diodes) {
        Diode newD = d;
        newD.anode = remap(d.anode);
        newD.cathode = remap(d.cathode);
        out.diodes.push_back(newD);
    }
    
    // MOSFETs
    for (const auto& m : block.mosfets) {
        Mosfet newM = m;
        newM.drain = remap(m.drain);
        newM.gate = remap(m.gate);
        newM.source = remap(m.source);
        newM.body = remap(m.body);
        out.mosfets.push_back(newM);
    }
    
    // BJTs
    for (const auto& b : block.bjts) {
        BJT newB = b;
        newB.nodeCollector = remap(b.nodeCollector);
        newB.base = remap(b.base);
        newB.emitter = remap(b.emitter);
        newB.substrate = remap(b.substrate);
        out.bjts.push_back(newB);
    }
    for (const auto& o : block.osdiDevices) {
        OsdiDevice newO = o;
        for (auto& n : newO.nodes) n = remap(n);
        out.osdiDevices.push_back(newO);
    }
    
    // Note: Probes, BusProbes, and other runtime-only structures are NOT copied
    // to the subcircuit definition (they are measurement tools, not circuit elements).
    
    return result;
}

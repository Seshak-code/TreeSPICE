#pragma once
/**
 * physics_constants.h
 * 
 * Centralized Physical Constants for SPICE-Class Simulation.
 * Eliminates magic numbers scattered throughout the solver.
 */

namespace PhysicsConstants {

// =============================================================================
// THERMAL & FUNDAMENTAL
// =============================================================================


/** Boltzmann constant */
constexpr double K_BOLTZMANN = 1.380649e-23; // J/K

/** Electron charge */
constexpr double Q_ELECTRON = 1.602176634e-19; // Coulombs
/** Thermal voltage kT/q at the nominal 27 C (300.15 K), as ngspice: 25.865 mV.
 *  Built-in devices scale it by T / 300.15; OSDI devices get the temperature itself. */
constexpr double V_THERMAL_NOM = K_BOLTZMANN * 300.15 / Q_ELECTRON;

// =============================================================================
// DEVICE DEFAULTS (Silicon)
// =============================================================================

/** Default diode saturation current (Silicon PN Junction) */
constexpr double IS_DEFAULT_DIODE = 1e-14; // Amperes

/** Default emission coefficient (Ideal diode) */
constexpr double N_DEFAULT_DIODE = 1.0;

/** Default BJT saturation current */
constexpr double IS_DEFAULT_BJT = 1e-14; // Amperes

/** Default BJT forward beta */
constexpr double BETA_F_DEFAULT = 100.0;

/** Default BJT reverse beta */
constexpr double BETA_R_DEFAULT = 1.0;


// =============================================================================
// SOLVER HEURISTICS
// =============================================================================

/** Voltage source internal resistance (Norton equivalent) */
constexpr double R_VSOURCE_INTERNAL = 1e-3; // Ohms

// =============================================================================
// TOLERANCES
// =============================================================================

/** Relative tolerance for Newton-Raphson convergence */
constexpr double RELTOL = 1e-3;

/** Absolute voltage tolerance */
constexpr double VNTOL = 1e-6; // Volts

/** Absolute current tolerance (pico-Ampere resolution) */
constexpr double ABSTOL = 1e-12; // Amperes

/** KCL residual tolerance */
constexpr double KCL_TOL = 1e-12; // Amperes

/** Relative KCL tolerance (componentwise backward error): node i's residual
 *  only counts beyond KCL_RELTOL * (|A||v|)_i. Stiff stamps (a 999k-gain E
 *  source is 1e10 S; a 1 kV Norton source is 1e6 A of cancelling terms) leave
 *  floating-point residuals far above KCL_TOL that no iteration can remove.
 *  At ordinary nodes the allowance is ~1e-13 A, so nothing loosens there. */
constexpr double KCL_RELTOL = 1e-10;
// ngspice PIVTOL: a DC solution factored through a pivot at or below it is singular
constexpr double PIVTOL = 1e-13;

/** Conductance used to short an inductor at DC. Same 1 mOhm as the Norton
 *  voltage-source stamp, so stamps and KCL audits stay consistent.
 *  ponytail: not an ideal short (I*1mOhm error); exact fix is an MNA branch
 *  current row, which the voltage-domain-only matrix doesn't have yet. */
constexpr double DC_SHORT_G = 1e3; // Siemens

/** Junction exponent cap: above Vj = JUNCTION_EXP_ARG_MAX * N*Vt every diode/BJT
 *  kernel continues linearly instead of exponentially (overflow guard only).
 *  Was 30: with SPICE's BJT IS=1e-16 the junction went linear at 1.07 mA,
 *  giving Vbe = 1.18 V at 18 mA (true exponential: 0.83 V). At 50 the knee is
 *  >= 5 kA even for IS = 1e-18, exp() is 13 orders below double overflow, and
 *  pnjlim (which bypasses limiting above this cap) bounds NR overshoot. */
constexpr double JUNCTION_EXP_ARG_MAX = 50.0;

/** PCG solver tolerance */
constexpr double PCG_TOL = 1e-10;

/** Minimum time step to prevent underflow */
constexpr double DT_MIN = 1e-18; // Seconds

/** Maximum NR iterations */
constexpr int MAX_NR_ITER = 50;

/** Maximum PCG iterations */
constexpr int MAX_PCG_ITER = 1000;

/** ngspice GMIN: 1e-12 S in parallel with every PN junction (diode, BJT),
 *  in the AoS models, every SoA kernel and both KCL audits. Never node-to-ground. */
constexpr double JUNCTION_GMIN = 1e-12;  // the default; a run's .options gmin travels in each device's parameters (.gmin)

/** GMIN stepping stages */
constexpr int GMIN_STEPS = 10;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/** Compute thermal voltage at a given temperature */
inline double thermalVoltage(double tempK) {
    return (K_BOLTZMANN * tempK) / Q_ELECTRON;
}

} // namespace PhysicsConstants

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

/** Junction exponent cap: above Vj = JUNCTION_EXP_ARG_MAX * N*Vt every diode/BJT
 *  kernel continues linearly instead of exponentially (overflow guard only).
 *  Was 30: with SPICE's BJT IS=1e-16 the junction went linear at 1.07 mA,
 *  giving Vbe = 1.18 V at 18 mA (true exponential: 0.83 V). At 50 the knee is
 *  >= 5 kA even for IS = 1e-18, exp() is 13 orders below double overflow, and
 *  pnjlim (which bypasses limiting above this cap) bounds NR overshoot. */
constexpr double JUNCTION_EXP_ARG_MAX = 50.0;

/** Maximum NR iterations */
constexpr int MAX_NR_ITER = 50;

/** ngspice GMIN: 1e-12 S in parallel with every PN junction (diode, BJT),
 *  in every junction model and the KCL audits. Never node-to-ground. */
constexpr double JUNCTION_GMIN = 1e-12;  // the default; a run's .options gmin travels in each device's parameters (.gmin)

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/** Compute thermal voltage at a given temperature */
inline double thermalVoltage(double tempK) {
    return (K_BOLTZMANN * tempK) / Q_ELECTRON;
}

} // namespace PhysicsConstants

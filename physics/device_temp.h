#pragma once
#include <cmath>
#include "physics_constants.h"

// Temperature laws shared by the diode, BJT and MOS level 1 (ngspice diotemp / bjttemp / mos1temp).
// The parser applies them once per circuit temperature, so the device models run at their final values.
namespace devtemp {

constexpr double REFTEMP = 300.15;  // ngspice: 27 C

inline double vt(double T) { return PhysicsConstants::K_BOLTZMANN * T / PhysicsConstants::Q_ELECTRON; }

// Silicon band gap [eV]
inline double egfet(double T) { return 1.16 - 7.02e-4 * T * T / (T + 1108.0); }

// Junction potential shift (ngspice pbfact)
inline double pbfact(double T) {
    const double k = PhysicsConstants::K_BOLTZMANN;
    const double arg = -egfet(T) / (2 * k * T) + 1.1150877 / (k * (REFTEMP + REFTEMP));
    return -2 * vt(T) * (1.5 * std::log(T / REFTEMP) + PhysicsConstants::Q_ELECTRON * arg);
}

// 1 + c1 dt + c2 dt^2: the TC1/TC2 style coefficient law
inline double poly(double c1, double c2, double dt) { return 1.0 + c1 * dt + c2 * dt * dt; }

// Saturation current factor exp((T/Tn - 1) EG/(N Vt) + XTI/N ln(T/Tn)) (diode; the BJT passes N = 1)
inline double isFactor(double T, double Tn, double eg, double xti, double n) {
    return std::exp((T / Tn - 1) * eg / (n * vt(T)) + xti / n * std::log(T / Tn));
}

// A junction's potential pot (at Tn) moved to T, and the factor its zero-bias capacitance gets, for grading
// mj (ngspice's pbo / gma law, the same in diotemp, bjttemp and mos1temp).
inline double junctionAt(double &pot, double mj, double Tn, double T) {
    const double fact1 = Tn / REFTEMP, fact2 = T / REFTEMP;
    const double pbo = (pot - pbfact(Tn)) / fact1, gmaold = (pot - pbo) / pbo;
    pot = pbfact(T) + fact2 * pbo;
    const double gmanew = (pot - pbo) / pbo;
    return (1 + mj * (4e-4 * (T - REFTEMP) - gmanew)) / (1 + mj * (4e-4 * (Tn - REFTEMP) - gmaold));
}

} // namespace devtemp

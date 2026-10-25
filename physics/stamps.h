#pragma once
#include "../math/linalg.h"
#include <vector>
#include <cmath>
#include <algorithm>

/*
 * stamps.h - the G source's transconductance stamp
 */


/**
 * VCCS - Voltage-Controlled Current Source
 * Iout = gm * Vcontrol
 *
 * ctrlPos/ctrlNeg: Control voltage terminals (sense)
 * outPos/outNeg: Output current terminals (current flows from outPos to outNeg)
 * gm: Transconductance (Siemens)
 */
inline void stampVCCS(int ctrlPos, int ctrlNeg, int outPos, int outNeg,
                      double gm, MatrixConstructor& mat, std::vector<double>& /*rhs*/)
{
    // VCCS stamps transconductance directly into the matrix
    // I_out = gm * (V_ctrl+ - V_ctrl-)
    // Current enters outPos, leaves outNeg

    if (outPos != 0 && ctrlPos != 0) mat.add(outPos - 1, ctrlPos - 1, gm);
    if (outPos != 0 && ctrlNeg != 0) mat.add(outPos - 1, ctrlNeg - 1, -gm);
    if (outNeg != 0 && ctrlPos != 0) mat.add(outNeg - 1, ctrlPos - 1, -gm);
    if (outNeg != 0 && ctrlNeg != 0) mat.add(outNeg - 1, ctrlNeg - 1, gm);
}

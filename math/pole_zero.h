#pragma once
#include <complex>
#include <string>
#include <vector>

namespace acutesim {
// Finite roots s of det(A + sB) = 0 for dense row-major n x n A, B (B may be singular: the infinite
// roots are dropped). Appends them to `roots`; returns an error text ("" on success).
// ponytail: dense O(n^3) (balancing, Hessenberg reduction, complex QR on B^-1 A after eliminating the
// algebraic rows); refuses n above kMaxPoleZeroN.
constexpr int kMaxPoleZeroN = 500;
std::string finiteRoots(std::vector<double> A, std::vector<double> B, int n, std::vector<std::complex<double>>& roots);
}  // namespace acutesim

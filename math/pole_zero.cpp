#include "math/pole_zero.h"
#include <algorithm>
#include <cmath>

namespace acutesim {
namespace {
using C = std::complex<double>;

// Eigenvalues of a real n x n row-major matrix: power-of-two balancing, Householder reduction to upper
// Hessenberg form, then complex QR steps (Wilkinson shift, Givens rotations) with deflation. False if
// an eigenvalue takes more than 60 steps.
bool eigenvalues(std::vector<double> M, int n, std::vector<C>& w) {
  for (bool changed = true; changed;) {  // D^-1 M D with D = 2^k: same eigenvalues, smaller norm
    changed = false;
    for (int i = 0; i < n; ++i) {
      double c = 0, r = 0;
      for (int j = 0; j < n; ++j) if (j != i) { c += std::abs(M[j * n + i]); r += std::abs(M[i * n + j]); }
      if (c == 0 || r == 0) continue;
      const double f = std::exp2(std::round(0.5 * std::log2(r / c)));
      if (f == 1 || c * f + r / f >= 0.95 * (c + r)) continue;
      for (int j = 0; j < n; ++j) { M[i * n + j] /= f; M[j * n + i] *= f; }
      changed = true;
    }
  }
  std::vector<C> H(M.begin(), M.end());
  auto h = [&](int i, int j) -> C& { return H[i * n + j]; };
  for (int k = 0; k + 2 < n; ++k) {  // Householder: zero column k below the subdiagonal
    double a = 0;
    for (int i = k + 1; i < n; ++i) a += std::norm(h(i, k));
    if (a == 0) continue;
    a = std::sqrt(a);
    std::vector<C> v(n, 0.0);
    const C x0 = h(k + 1, k);
    v[k + 1] = x0 + (std::abs(x0) > 0 ? x0 / std::abs(x0) : C(1)) * a;
    for (int i = k + 2; i < n; ++i) v[i] = h(i, k);
    double vv = 0;
    for (int i = k + 1; i < n; ++i) vv += std::norm(v[i]);
    for (int j = 0; j < n; ++j) {  // H = (I - 2 v v^H / v^H v) H (I - 2 v v^H / v^H v)
      C t = 0;
      for (int i = k + 1; i < n; ++i) t += std::conj(v[i]) * h(i, j);
      t *= 2.0 / vv;
      for (int i = k + 1; i < n; ++i) h(i, j) -= t * v[i];
    }
    for (int i = 0; i < n; ++i) {
      C t = 0;
      for (int j = k + 1; j < n; ++j) t += h(i, j) * v[j];
      t *= 2.0 / vv;
      for (int j = k + 1; j < n; ++j) h(i, j) -= t * std::conj(v[j]);
    }
    for (int i = k + 2; i < n; ++i) h(i, k) = 0;
  }
  double norm = 0;
  for (const C& x : H) norm = std::max(norm, std::abs(x));
  const double eps = 1e-15;
  std::vector<C> cs(n), sn(n);
  for (int hi = n - 1, its = 0; hi >= 0;) {
    int lo = hi;  // the active window [lo, hi]: no negligible subdiagonal inside
    for (; lo > 0; --lo) {
      const double s = std::abs(h(lo - 1, lo - 1)) + std::abs(h(lo, lo));
      if (std::abs(h(lo, lo - 1)) <= eps * (s > 0 ? s : norm)) { h(lo, lo - 1) = 0; break; }
    }
    if (lo == hi) { w.push_back(h(hi, hi)); --hi; its = 0; continue; }
    if (++its > 60) return false;
    // Shift: the eigenvalue of the trailing 2x2 nearer h(hi, hi); every 10th step an exceptional one
    const C a = h(hi - 1, hi - 1), b = h(hi - 1, hi), c = h(hi, hi - 1), d = h(hi, hi);
    const C half = 0.5 * (a + d), disc = std::sqrt(half * half - (a * d - b * c));
    C mu = std::abs(half + disc - d) < std::abs(half - disc - d) ? half + disc : half - disc;
    if (its % 10 == 0) mu = d + 0.75 * std::abs(c);
    for (int i = lo; i <= hi; ++i) h(i, i) -= mu;
    for (int k = lo; k < hi; ++k) {  // H - mu I = Q R: Givens rotation G_k zeroes h(k+1, k)
      const C x = h(k, k), y = h(k + 1, k);
      const double r = std::hypot(std::abs(x), std::abs(y));
      cs[k] = r > 0 ? x / r : C(1);
      sn[k] = r > 0 ? y / r : C(0);
      for (int j = k; j <= hi; ++j) {
        const C t1 = h(k, j), t2 = h(k + 1, j);
        h(k, j) = std::conj(cs[k]) * t1 + std::conj(sn[k]) * t2;
        h(k + 1, j) = cs[k] * t2 - sn[k] * t1;
      }
    }
    for (int k = lo; k < hi; ++k)  // R Q: the rotations from the right
      for (int i = lo; i <= std::min(k + 1, hi); ++i) {
        const C t1 = h(i, k), t2 = h(i, k + 1);
        h(i, k) = t1 * cs[k] + t2 * sn[k];
        h(i, k + 1) = t2 * std::conj(cs[k]) - t1 * std::conj(sn[k]);
      }
    for (int i = lo; i <= hi; ++i) h(i, i) += mu;
  }
  for (C& x : w)  // a real matrix: a real eigenvalue carries rounding-level imaginary parts in complex QR
    if (std::abs(x.imag()) <= 1e-10 * std::abs(x)) x = x.real();
  return true;
}

// X = B^-1 X for row-major n x n B (copied) and X, by LU with partial pivoting; false if B is numerically singular.
bool solveMulti(std::vector<double> B, std::vector<double>& X, int n) {
  double bmax = 0;
  for (double v : B) bmax = std::max(bmax, std::abs(v));
  for (int c = 0; c < n; ++c) {
    int pr = c;
    for (int r = c + 1; r < n; ++r) if (std::abs(B[r * n + c]) > std::abs(B[pr * n + c])) pr = r;
    if (std::abs(B[pr * n + c]) <= 1e-13 * bmax) return false;
    if (pr != c) for (int k = 0; k < n; ++k) { std::swap(B[pr * n + k], B[c * n + k]); std::swap(X[pr * n + k], X[c * n + k]); }
    for (int r = c + 1; r < n; ++r) {
      const double f = B[r * n + c] / B[c * n + c];
      if (f == 0) continue;
      for (int k = c; k < n; ++k) B[r * n + k] -= f * B[c * n + k];
      for (int k = 0; k < n; ++k) X[r * n + k] -= f * X[c * n + k];
    }
  }
  for (int c = n - 1; c >= 0; --c)
    for (int k = 0; k < n; ++k) {
      double v = X[c * n + k];
      for (int j = c + 1; j < n; ++j) v -= B[c * n + j] * X[j * n + k];
      X[c * n + k] = v / B[c * n + c];
    }
  return true;
}
}  // namespace

std::string finiteRoots(std::vector<double> A, std::vector<double> B, int n, std::vector<std::complex<double>>& roots) {
  if (n > kMaxPoleZeroN) return "pole-zero: " + std::to_string(n) + " unknowns exceed the dense limit of " + std::to_string(kMaxPoleZeroN);
  auto maxabs = [](const std::vector<double>& M) { double m = 0; for (double v : M) m = std::max(m, std::abs(v)); return m; };
  const double nb = maxabs(B);
  if (nb == 0) return "";  // no reactive element: no finite root
  const double rho = maxabs(A) / nb;  // roots are found in units of rho, so A and rho*B are comparable
  for (double& v : B) v *= rho;
  const double aTol = 1e-13 * maxabs(A), bTol = 1e-11 * maxabs(B);
  // Infinite roots come from B being singular. Row operations (determinant-preserving) bring B to echelon form;
  // each row left with B = 0 is an algebraic constraint, eliminated with a column of its A part (a Schur
  // complement, which keeps the pencil linear because that row has no s term). Repeat until B is nonsingular:
  // then every remaining root is finite and is an eigenvalue of -B^-1 A. Nilpotent blocks never reach the eigen solver.
  int m = n;
  for (;;) {
    int r = 0;
    for (int c = 0; c < m && r < m; ++c) {
      int p = r;
      for (int k = r + 1; k < m; ++k) if (std::abs(B[k * m + c]) > std::abs(B[p * m + c])) p = k;
      if (std::abs(B[p * m + c]) <= bTol) continue;
      if (p != r) for (int l = 0; l < m; ++l) { std::swap(A[p * m + l], A[r * m + l]); std::swap(B[p * m + l], B[r * m + l]); }
      for (int k = r + 1; k < m; ++k) {
        const double f = B[k * m + c] / B[r * m + c];
        if (f == 0) continue;
        for (int l = 0; l < m; ++l) { A[k * m + l] -= f * A[r * m + l]; B[k * m + l] -= f * B[r * m + l]; }
        B[k * m + c] = 0;
      }
      ++r;
    }
    if (r == m) break;
    std::vector<char> rowGone(m, 0), colGone(m, 0);
    for (int i = r; i < m; ++i) {
      int j = -1;
      for (int l = 0; l < m; ++l) if (!colGone[l] && (j < 0 || std::abs(A[i * m + l]) > std::abs(A[i * m + j]))) j = l;
      if (j < 0 || std::abs(A[i * m + j]) <= aTol) return "pole-zero: the system matrix is singular (det(G + sC) is identically zero)";
      for (int k = 0; k < m; ++k) {
        if (k == i) continue;
        const double fa = A[k * m + j] / A[i * m + j], fb = B[k * m + j] / A[i * m + j];
        if (fa == 0 && fb == 0) continue;
        for (int l = 0; l < m; ++l) { A[k * m + l] -= fa * A[i * m + l]; B[k * m + l] -= fb * A[i * m + l]; }
      }
      rowGone[i] = colGone[j] = 1;
    }
    std::vector<int> kr, kc;
    for (int l = 0; l < m; ++l) { if (!rowGone[l]) kr.push_back(l); if (!colGone[l]) kc.push_back(l); }
    const int q = (int)kr.size();
    std::vector<double> A2(q * q), B2(q * q);
    for (int a = 0; a < q; ++a) for (int b = 0; b < q; ++b) { A2[a * q + b] = A[kr[a] * m + kc[b]]; B2[a * q + b] = B[kr[a] * m + kc[b]]; }
    A = std::move(A2); B = std::move(B2); m = q;
    if (m == 0) return "";
  }
  std::vector<double>& X = A;  // X = B^-1 A
  if (!solveMulti(B, X, m)) return "pole-zero: the reactive part of the system is singular";
  std::vector<C> mu;
  if (!eigenvalues(X, m, mu)) return "pole-zero: QR iteration did not converge";
  for (auto e : mu) roots.push_back(-e * rho);
  return "";
}
}  // namespace acutesim

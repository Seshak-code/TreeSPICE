// test_contraction_solver.cpp — tensor-contraction solver: backward error, plans, reuse.
// Covers: grid (unsymmetric), circuit-like graph with supply-net hubs and
// VCVS rows (1e10 off-diagonals: pivoting stress), disconnected components,
// n=1, long chain; prints plan size and timings up to 1e6 unknowns. With a GPU,
// fronts >= TREESPICE_GPU_FRONT (default 512) take the FP32 GPU path.
#include "math/contraction_solver.h"
#include "math/gpu_front_lu.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>

static int g_fail = 0;
static void check(bool ok, const char* what, double val) {
    std::printf("  [%s] %-58s %.3e\n", ok ? "PASS" : "FAIL", what, val);
    if (!ok) ++g_fail;
}

// Build CSR from (row,col)->value map.
static Csr_matrix toCsr(int n, const std::map<std::pair<int, int>, double>& e) {
    Csr_matrix A;
    A.rows = A.cols = n;
    A.row_pointer.assign(n + 1, 0);
    for (auto& [rc, v] : e) A.row_pointer[rc.first + 1]++;
    for (int i = 0; i < n; ++i) A.row_pointer[i + 1] += A.row_pointer[i];
    for (auto& [rc, v] : e) { A.col_indices.push_back(rc.second); A.values.push_back(v); }
    A.nnz = (int)A.values.size();
    return A;
}
static void conductance(std::map<std::pair<int, int>, double>& e, int a, int b, double g) {
    if (a >= 0) e[{a, a}] += g;
    if (b >= 0) e[{b, b}] += g;
    if (a >= 0 && b >= 0) { e[{a, b}] -= g; e[{b, a}] -= g; }
}
static double relErr(const std::vector<double>& x, const std::vector<double>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < x.size(); ++i) { num = std::max(num, std::abs(x[i] - ref[i])); den = std::max(den, std::abs(ref[i])); }
    return num / (den > 0 ? den : 1);
}
static std::vector<double> rhsFor(int n, unsigned seed) {
    std::mt19937 rng(seed); std::uniform_real_distribution<double> u(-1, 1);
    std::vector<double> b(n); for (auto& v : b) v = u(rng); return b;
}
static double secondsSince(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// 2D grid of conductances, grounded at the corner, with an unsymmetric VCCS-like perturbation
static Csr_matrix grid(int w, bool unsym) {
    std::map<std::pair<int, int>, double> e;
    auto id = [w](int x, int y) { return y * w + x; };
    for (int y = 0; y < w; ++y)
        for (int x = 0; x < w; ++x) {
            if (x + 1 < w) conductance(e, id(x, y), id(x + 1, y), 1e-3);
            if (y + 1 < w) conductance(e, id(x, y), id(x, y + 1), 1e-3);
            e[{id(x, y), id(x, y)}] += 1e-12;  // GMIN
            if (unsym && x + 1 < w && (x + y) % 7 == 0) e[{id(x, y), id(x + 1, y)}] += 4e-4;
        }
    conductance(e, 0, -1, 1.0);
    return toCsr(w * w, e);
}

// Circuit-like graph: random resistor network + 3 supply hubs + VCVS rows
static Csr_matrix circuit(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, n - 1);
    std::map<std::pair<int, int>, double> e;
    for (int i = 1; i < n; ++i) conductance(e, i, std::max(0, i - 1 - (int)(rng() % 8)), 1e-3 * (1 + rng() % 10));
    for (int h = 0; h < 3; ++h) {  // supply nets touching ~n/3 nodes each
        int hub = pick(rng);
        for (int k = 0; k < n / 3; ++k) conductance(e, hub, pick(rng), 1e-4);
        conductance(e, hub, -1, 1e3);  // Norton source
    }
    for (int k = 0; k < n / 50; ++k) {  // VCVS: out row has g_int on diag, -gain*g_int on control
        int out = pick(rng), cp = pick(rng), cn = pick(rng);
        if (out == cp || out == cn || cp == cn) continue;
        double g = 1e4, gm = 1e3 * g;
        e[{out, out}] += g; e[{out, cp}] -= gm; e[{out, cn}] += gm;
        conductance(e, cn, out, 1e-4);  // feedback resistor keeps it nonsingular
    }
    for (int i = 0; i < n; ++i) e[{i, i}] += 1e-12;
    return toCsr(n, e);
}

// Componentwise backward error max_i |b - Ax|_i / (|A||x| + |b|)_i: the right
// accuracy metric (dense LU is not ground truth on ill-conditioned systems).
static double backwardError(const Csr_matrix& A, const std::vector<double>& x, const std::vector<double>& b) {
    std::vector<double> Ax(A.rows);
    matrixMultiplication(A, x.data(), Ax.data());
    double w = 0;
    for (int i = 0; i < A.rows; ++i) {
        double s = std::abs(b[i]);
        for (int k = A.row_pointer[i]; k < A.row_pointer[i + 1]; ++k) s += std::abs(A.values[k] * x[A.col_indices[k]]);
        w = std::max(w, std::abs(Ax[i] - b[i]) / s);
    }
    return w;
}

static void compareDense(const char* name, const Csr_matrix& A) {  // the componentwise backward error of a direct solve
    auto b = rhsFor(A.rows, 7);
    auto plan = planContraction(A);
    auto rc = solveContraction(A, b, plan);
    char label[160];
    std::snprintf(label, sizeof label, "%s n=%d fronts=%zu: backward error", name, A.rows, plan.fronts.size());
    check(rc.converged && backwardError(A, rc.solution, b) <= 1e-13, label, rc.converged ? backwardError(A, rc.solution, b) : 1.0);
}

int main() {
    std::printf("== correctness (backward error) ==\n");
    if (gpuFrontLuAvailable()) {  // grid 300: peak front ~570, past the default GPU cutoff (512)
        auto A = grid(300, true); auto b = rhsFor(A.rows, 3);
        auto r = solveContraction(A, b, planContraction(A));
        check(r.gpuFronts > 0 && !r.fp64Refactor && backwardError(A, r.solution, b) <= 1e-13,
              "GPU FP32 fronts + FP64 refinement: backward error", backwardError(A, r.solution, b));
    } else std::printf("  (no GPU: FP32 front path not exercised)\n");
    compareDense("grid 30x30 unsymmetric", grid(30, true));
    {   // a singular system has no solution: an empty one, never zeros a caller could take for an answer
        MatrixConstructor M;
        M.setDimensions(2, 2);
        M.add(0, 0, 1.0); M.add(0, 1, 1.0); M.add(1, 0, 1.0); M.add(1, 1, 1.0);
        const auto S = M.createCsr();
        const auto r = solveContraction(S, {1.0, 2.0}, planContraction(S));
        check(!r.converged && r.solution.empty(), "singular: no solution (converged=0, size)", (double)r.solution.size());
    }
    compareDense("circuit+hubs+VCVS", circuit(1500, 1));
    compareDense("circuit+hubs+VCVS", circuit(1500, 2));
    {   // 200 random small E-source networks: every one needs off-diagonal pivots
        double worst = 0; int bad = 0;
        for (unsigned seed = 1; seed <= 200; ++seed) {
            auto A = circuit(40 + seed % 160, seed); auto b = rhsFor(A.rows, seed);
            auto plan = planContraction(A, 8 + seed % 32);
            auto r = solveContraction(A, b, plan);
            double w = r.converged ? backwardError(A, r.solution, b) : 1.0;
            worst = std::max(worst, w); bad += (w > 1e-13);
        }
        check(bad == 0, "200 random E-source networks: worst backward error", worst);
    }
    {   // two disconnected grids
        auto g1 = grid(12, false);
        std::map<std::pair<int, int>, double> e;
        for (int pass = 0; pass < 2; ++pass)
            for (int i = 0; i < g1.rows; ++i)
                for (int k = g1.row_pointer[i]; k < g1.row_pointer[i + 1]; ++k)
                    e[{i + pass * g1.rows, g1.col_indices[k] + pass * g1.rows}] = g1.values[k];
        compareDense("disconnected components", toCsr(2 * g1.rows, e));
    }
    {   // per-front dirty flags: after one changed entry, the cached solve reuses the
        // untouched subtrees and is bit-identical to a full refactorization
        auto A = circuit(1500, 3);
        auto plan = planContraction(A);
        ContractionCache cache;
        auto b = rhsFor(A.rows, 3);
        solveContraction(A, b, plan, 2, &cache);
        const size_t first = cache.refactored;
        A.values[A.row_pointer[7]] *= 1.5;  // one entry changes
        auto cached = solveContraction(A, b, plan, 2, &cache);
        auto fresh = solveContraction(A, b, plan, 2);
        const size_t reused = cache.reused, again = cache.refactored - first;
        check(cached.converged && cached.solution == fresh.solution && reused > 0 && again < first,
              "dirty flags: bit-identical to a refactorization; fronts refactored after one change", (double)again);
        auto same = solveContraction(A, b, plan, 2, &cache);  // nothing changed: every front reused
        check(same.solution == fresh.solution && cache.refactored == first + again,
              "dirty flags: an unchanged matrix refactors no front", (double)(cache.refactored - first - again));
    }
    {
        std::map<std::pair<int, int>, double> e{{{0, 0}, 2.0}};
        auto A = toCsr(1, e); auto plan = planContraction(A);
        auto r = solveContraction(A, {4.0}, plan);
        check(r.converged && std::abs(r.solution[0] - 2.0) < 1e-15, "n=1", std::abs(r.solution[0] - 2.0));
    }
    {   // singular matrix must be reported, not "solved"
        std::map<std::pair<int, int>, double> e{{{0, 0}, 1.0}, {{0, 1}, 1.0}, {{1, 0}, 1.0}, {{1, 1}, 1.0}};
        auto A = toCsr(2, e); auto plan = planContraction(A);
        auto r = solveContraction(A, {1.0, 2.0}, plan);
        check(!r.converged || !std::isfinite(r.finalResidual) || r.finalResidual > 1e-3, "singular matrix not reported as solved", r.finalResidual);
    }

    std::printf("== scale (plan / factor+solve, residual) ==\n");
    struct Case { const char* name; Csr_matrix A; };
    for (int w : {100, 316, 1000}) {
        auto A = grid(w, true);
        auto t0 = std::chrono::steady_clock::now();
        auto plan = planContraction(A);
        double tp = secondsSince(t0);
        auto b = rhsFor(A.rows, 3);
        t0 = std::chrono::steady_clock::now();
        auto r = solveContraction(A, b, plan);
        double ts = secondsSince(t0);
        std::vector<double> Ax(A.rows);
        matrixMultiplication(A, r.solution.data(), Ax.data());
        double rel = 0, bn = 0;
        for (int i = 0; i < A.rows; ++i) { rel = std::max(rel, std::abs(Ax[i] - b[i])); bn = std::max(bn, std::abs(b[i])); }
        std::printf("  grid %4dx%-4d n=%-8d fronts=%-7zu levels=%-3zu peak front=%.0f^2  plan %.3f s  solve %.3f s  |Ax-b|/|b|=%.1e\n",
                    w, w, A.rows, plan.fronts.size(), plan.levels.size(), std::sqrt(plan.peakEntries), tp, ts, rel / bn);
        if (w == 100) check(r.converged && tp + ts < 1.0 && rel / bn < 1e-10, "10^4-node DC solve under 1 s (Phase 1 gate)", tp + ts);
    }
    {
        int n = 200000;
        std::map<std::pair<int, int>, double> e;
        for (int i = 0; i + 1 < n; ++i) conductance(e, i, i + 1, 1e-3);
        conductance(e, 0, -1, 1.0);
        auto A = toCsr(n, e); auto b = rhsFor(n, 5);
        auto t0 = std::chrono::steady_clock::now();
        auto plan = planContraction(A); auto r = solveContraction(A, b, plan);
        std::printf("  chain n=%d: %.3f s, peak front %.0f^2\n", n, secondsSince(t0), std::sqrt(plan.peakEntries));
        check(r.converged, "200k-node chain solves", r.finalResidual);
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

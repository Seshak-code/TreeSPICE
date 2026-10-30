// solve_triplets <file> [refineSteps] — one contraction-solver solve, for tests/refcheck/refcheck.py, which checks the
// solution against mpmath's exact one. Input: "n nnz", then nnz lines "i j value" (0-based; duplicates summed), then n
// lines of b. Output: "ok <0|1> rank_deficient <0|1>", then the n entries of x at 17 significant digits. The solve is
// the engine's: planContraction + solveContraction in FP64 on the CPU (refineSteps defaults to the engine's 2).
#include "math/contraction_solver.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: solve_triplets <file> [refineSteps]\n"); return 2; }
    std::ifstream in(argv[1]);
    int n = 0; long nnz = 0;
    if (!(in >> n >> nnz) || n <= 0 || nnz < 0) { std::fprintf(stderr, "bad header\n"); return 2; }
    std::map<std::pair<int, int>, double> e;
    for (long k = 0; k < nnz; ++k) {
        int i, j; double v;
        if (!(in >> i >> j >> v) || i < 0 || j < 0 || i >= n || j >= n) { std::fprintf(stderr, "bad entry %ld\n", k); return 2; }
        e[{i, j}] += v;
    }
    std::vector<double> b(n);
    for (double& v : b) if (!(in >> v)) { std::fprintf(stderr, "short right-hand side\n"); return 2; }

    Csr_matrix A;
    A.rows = A.cols = n;
    A.row_pointer.assign(n + 1, 0);
    for (const auto& [rc, v] : e) ++A.row_pointer[rc.first + 1];
    for (int i = 0; i < n; ++i) A.row_pointer[i + 1] += A.row_pointer[i];
    for (const auto& [rc, v] : e) { A.col_indices.push_back(rc.second); A.values.push_back(v); }  // map order = CSR order
    A.nnz = (int)A.values.size();

    const int refine = argc > 2 ? std::atoi(argv[2]) : 2;
    const SolverResult r = solveContraction(A, b, planContraction(A), refine, nullptr, /*gpu=*/false);
    std::printf("ok %d rank_deficient %d\n", r.converged ? 1 : 0, r.rankDeficient ? 1 : 0);
    for (double x : r.solution) std::printf("%.17g\n", x);
    return r.converged ? 0 : 1;
}

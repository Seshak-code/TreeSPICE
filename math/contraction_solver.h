#pragma once

#include "linalg.h"
#include <cstdint>
#include <memory>
#include <vector>

/**
 * contraction_solver.h — Tensor-contraction direct solver (default linear solver)
 *
 * Solves the Newton-step system A x = b by contracting the circuit's tensor
 * network along a tree of "fronts":
 *
 *   plan (once per topology): nested dissection of the circuit graph gives a
 *     contraction tree. Each front owns a set of pivot variables (a separator
 *     or a small leaf subgraph) and a boundary index set (variables it shares
 *     with its ancestors). Fronts on the same tree level are independent.
 *
 *   factor (every NR iteration): bottom-up, each front assembles its entries of
 *     the stamped Jacobian, ADDS its children's boundary tensors (superposition
 *     of Schur complements), eliminates its pivots with partial pivoting, and
 *     hands its own boundary tensor to its parent. Levels run in parallel.
 *
 *   solve: forward pass (same tree) + top-down back-substitution, followed by
 *     iterative refinement against the original A.
 *
 *   GPU (Phase 4): fronts of dimension >= TREESPICE_GPU_FRONT (env, default 512,
 *     0 = never) are factored in FP32 on the GPU when one is available
 *     (gpu_front_lu.h); FP64 refinement then restores full accuracy. If it does
 *     not converge (kappa * 2^-24 too large), the solve refactors in FP64.
 *
 * The leaves are built from the assembled CSR Jacobian, not from a separate
 * device model, so the solver always solves exactly what the stamps produced.
 */

struct ContractionFront {
    std::vector<int> pivots;     // variables eliminated in this front
    std::vector<int> boundary;   // remaining variables (owned by ancestors), sorted
    int parent = -1;
    std::vector<int> children;

    // Precomputed numeric-phase maps (local index = position in pivots ++ boundary)
    std::vector<int> assembleK;       // CSR value index of each entry assembled here
    std::vector<int> assembleRow, assembleCol;  // its static local position (pivots ++ boundary)
    std::vector<std::vector<int>> childMap;  // per child: child boundary pos -> local pos
    int dim() const { return (int)(pivots.size() + boundary.size()); }
};

struct ContractionPlan {
    int n = 0;
    std::vector<int> rowPtr, colIdx;          // CSR pattern this plan was built for
    std::vector<int> rowOfVar;                // equation (A row) paired with each variable
    std::vector<ContractionFront> fronts;     // postorder: children before parents
    std::vector<std::vector<int>> levels;     // fronts grouped by height; independent within a level
    double flopEstimate = 0.0;                // factorization flops
    double peakEntries = 0.0;                 // largest front, dim^2
    bool matches(const Csr_matrix& A) const {
        return A.rows == n && A.row_pointer == rowPtr && A.col_indices == colIdx;
    }
};

/** Build the contraction tree for a CSR sparsity pattern (pattern is symmetrised). */
ContractionPlan planContraction(const Csr_matrix& A, int leafSize = 64);

/** Factors kept between solves of one plan (per-front dirty flags): a front whose
 *  assembled entries are bitwise equal to the previous solve's, and whose children
 *  were all reused, keeps its factor and update tensor. Exact reuse: the same
 *  inputs through the same arithmetic, so results are bit-identical to refactoring.
 *  A linear subtree (a power grid, constant-step capacitor companions) factors once. */
struct ContractionCache {
    struct Impl;
    std::shared_ptr<Impl> impl;
    size_t reused = 0, refactored = 0;  // fronts, cumulative
};

/** Factor + solve A x = b along `plan` (A must match plan's pattern), with
 *  up to `refineSteps` iterative-refinement passes against A. `cache`: reuse
 *  unchanged subtrees from the previous solve of the same plan. */
SolverResult solveContraction(const Csr_matrix& A, const std::vector<double>& b,
                              const ContractionPlan& plan, int refineSteps = 2,
                              ContractionCache* cache = nullptr, bool gpu = true);

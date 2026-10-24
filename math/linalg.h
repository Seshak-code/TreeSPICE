#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <numeric> // For std::iota

/*
 * linalg.h
 * Sparse linear algebra library optimized for MNA-based circuit simulation
 * workloads.
 *
 * This library provides a complete toolchain for solving sparse linear systems
 * (Ax = b):
 * 1. CSR (Compressed Sparse Row) matrix storage format.
 * 2. Triplet-based matrix assembly with duplicate handling.
 * 3. OpenMP-parallelized Sparse Matrix-Vector Multiplication (SpMV).
 * 4. Preconditioned Conjugate Gradient (PCG) iterative solver.
 */

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/*
 * Compressed Sparse Row (CSR) Matrix
 *
 * The CSR format is designed for memory efficiency by storing only non-zero
 * elements. It consists of three primary arrays:
 * - values: Stores the numerical values of non-zero entries in row-major order.
 * - col_indices: Stores the column index corresponding to each stored value.
 * - row_pointer: Stores the index in 'values' where each row begins.
 *
 * This structure enables fast matrix-vector operations while significantly
 * reducing memory overhead compared to a dense (N x N) grid.
 */

struct Csr_matrix {
  std::vector<double> values;   // Non-zero numerical data
  std::vector<int> col_indices; // Column indices for each value
  std::vector<int> row_pointer; // Pointers to the start of each row
  int rows;                     // Dimension of the matrix (rows)
  int cols;                     // Dimension of the matrix (columns)
  int nnz;                      // Total count of non-zero entries
};

/*
 * Triplet - A single matrix entry in Coordinate (COO) format.
 *
 * A Triplet represents a raw, uncompressed entry defined by its
 * (row, column, value) coordinates. Triplets are used as an intermediate
 * stage during matrix assembly before compression into CSR format.
 */

struct Triplet {
  int row;
  int col;
  double value;
};

// ============================================================================
// CSR PATTERN CACHE (Sparse Merge — Phase D5)
// ============================================================================

/**
 * CachedCsrPattern
 *
 * The frozen sparsity pattern (row_pointer + col_indices, columns sorted within a row) of a one-time
 * symbolic assembly pass, and the CSR value index of any (row, col): later Newton iterations scatter
 * their stamps through assembleFast() instead of createCsr()'s sort + dedup.
 */
struct CachedCsrPattern {
    std::vector<int> row_pointer;
    std::vector<int> col_indices;
    int rows = 0, cols = 0, nnz = 0;

    // The CSR value index for (row, col), or -1 if not in the pattern: a binary search of the row (a hash map
    // here was 60 bytes per entry, a quarter of a BSIM4 SRAM array's solver memory)
    int indexOf(int row, int col) const {
        if (row < 0 || row >= rows) return -1;
        const auto b = col_indices.begin() + row_pointer[row], e = col_indices.begin() + row_pointer[row + 1];
        const auto it = std::lower_bound(b, e, col);
        return it != e && *it == col ? (int)(it - col_indices.begin()) : -1;
    }
};

// ============================================================================
// MATRIX ASSEMBLY
// ============================================================================

/*
 * MatrixConstructor - Accumulator for building sparse matrices from individual
 * components.
 *
 * In circuit simulation, multiple components (resistors, sources) may connect
 * to the same nodes, contributing to the same matrix cell. This class collects
 * these individual "stamps" as Triplets and merges them during final
 * compression.
 */


class MatrixConstructor {
  std::vector<int> rows_, cols_;
  std::vector<double> vals_;
  int numRows = 0;
  int numCols = 0;
  uint64_t checksum_ = 0;
  // createCsr's pattern cache: the (row, col) sequence it last sorted, each triplet's
  // CSR value slot (-1: outside the matrix, dropped) and that CSR pattern
  std::vector<int> lastRows_, lastCols_, slot_;
  Csr_matrix pattern_{{}, {}, {}, -1, -1, 0};

public:
  // Resets the constructor for a new matrix assembly
  void clear() {
    rows_.clear();
    cols_.clear();
    vals_.clear();
    numRows = 0;
    numCols = 0;
    checksum_ = 0;
  }

  // Explicitly set matrix dimensions
  void setDimensions(int rows, int cols) {
    numRows = rows;
    numCols = cols;
  }

  // XOR of hash(row, col, value) over the triplets of the last createCsr()
  uint64_t getChecksum() const { return checksum_; }

  // Stamps a numerical value at the specified (row, col) coordinate.
  void add(int row, int col, double value) {
    assert(row >= 0 && "Row index can't be negative");
    assert(col >= 0 && "Column index can't be negative");
    rows_.push_back(row);
    cols_.push_back(col);
    vals_.push_back(value);
  }

  // Multiply every stamp in row r < s.size() by s[r]
  void scaleRows(const std::vector<double>& s) {
    for (size_t k = 0; k < rows_.size(); ++k)
      if (rows_[k] >= 0 && rows_[k] < (int)s.size()) vals_[k] *= s[rows_[k]];
  }

  void addBatch(const std::vector<int>& rows, const std::vector<int>& cols, const std::vector<double>& values) {
    rows_.insert(rows_.end(), rows.begin(), rows.end());
    cols_.insert(cols_.end(), cols.begin(), cols.end());
    vals_.insert(vals_.end(), values.begin(), values.end());
  }

  /*
   * Converts the triplets into CSR: columns sorted within a row, duplicates summed
   * (in stamping order), entries outside the matrix dropped. Devices stamp in the
   * same order every Newton iteration, so the sort runs only when the (row, col)
   * sequence differs from the last call's; otherwise values are scattered through
   * the cached slots (the sort was ~11 % of a BSIM4 inverter-chain transient).
   */
  Csr_matrix createCsr() {
    const size_t T = rows_.size();
    checksum_ = 0;
    for (size_t k = 0; k < T; ++k) {
      uint64_t v_bits;
      std::memcpy(&v_bits, &vals_[k], sizeof(double));
      checksum_ ^= ((static_cast<uint64_t>(rows_[k]) * 104729) ^ (static_cast<uint64_t>(cols_[k]) * 700067)) ^ v_bits;
    }
    if (pattern_.rows != numRows || pattern_.cols != numCols || rows_ != lastRows_ || cols_ != lastCols_) {
      std::vector<int> p(T);
      std::iota(p.begin(), p.end(), 0);
      std::stable_sort(p.begin(), p.end(), [&](int a, int b) {
        return rows_[a] < rows_[b] || (rows_[a] == rows_[b] && cols_[a] < cols_[b]);
      });
      pattern_ = Csr_matrix{{}, {}, std::vector<int>(numRows + 1, 0), numRows, numCols, 0};
      slot_.assign(T, -1);
      int last_r = -1, last_c = -1;
      for (int idx : p) {
        const int r = rows_[idx], c = cols_[idx];
        if (r >= numRows || c >= numCols) continue;
        if (r != last_r || c != last_c) {
          pattern_.col_indices.push_back(c);
          pattern_.row_pointer[r + 1]++;
          last_r = r;
          last_c = c;
        }
        slot_[idx] = (int)pattern_.col_indices.size() - 1;
      }
      for (int i = 0; i < numRows; ++i) pattern_.row_pointer[i + 1] += pattern_.row_pointer[i];
      pattern_.nnz = (int)pattern_.col_indices.size();
      lastRows_ = rows_;
      lastCols_ = cols_;
    }
    Csr_matrix matrix = pattern_;
    matrix.values.assign(pattern_.nnz, 0.0);
    for (size_t k = 0; k < T; ++k)
      if (slot_[k] >= 0) matrix.values[slot_[k]] += vals_[k];
    return matrix;
  }

  /**
   * buildPattern()
   *
   * Performs a full createCsr() pass and extracts the frozen sparsity pattern
   * (row_pointer + col_indices) for assembleFast's lookups.
   * Call once per topology change.  The returned pattern can then be handed to
   * assembleFast() on every subsequent NR iteration.
   */
  CachedCsrPattern buildPattern() {
      Csr_matrix csr = createCsr();
      CachedCsrPattern pat;
      pat.rows        = csr.rows;
      pat.cols        = csr.cols;
      pat.nnz         = csr.nnz;
      pat.row_pointer = csr.row_pointer;
      pat.col_indices = csr.col_indices;
      return pat;
  }

  /**
   * assembleFast()
   *
   * O(stamps) scatter into a pre-allocated values array using a frozen pattern.
   * Avoids sort+dedup; produces a CSR matrix with the same sparsity as 'pattern'
   * but with freshly accumulated values from the current stamp buffers.
   * A stamp outside the pattern is dropped; `complete` (if given) reports it.
   */
  Csr_matrix assembleFast(const CachedCsrPattern& pattern, bool* complete = nullptr) {
      if (complete) *complete = true;
      Csr_matrix matrix;
      matrix.rows        = pattern.rows;
      matrix.cols        = pattern.cols;
      matrix.nnz         = pattern.nnz;
      matrix.row_pointer = pattern.row_pointer;
      matrix.col_indices = pattern.col_indices;
      matrix.values.assign(static_cast<size_t>(pattern.nnz), 0.0);
      for (size_t k = 0; k < rows_.size(); ++k) {
          int idx = pattern.indexOf(rows_[k], cols_[k]);
          if (idx >= 0) {
              matrix.values[static_cast<size_t>(idx)] += vals_[k];
          } else if (complete) {
              *complete = false;
          }
      }
      return matrix;
  }
};

// ============================================================================
// SPARSE MATRIX-VECTOR MULTIPLICATION (SpMV)
// ============================================================================

/*
 * Performs Sparse Matrix-Vector Multiplication (SpMV): y = A * x.
 *
 * For each row in the sparse matrix, we compute the dot product of that row
 * with the dense input vector. Only non-zero entries of 'A' are processed.
 * Parallelizes independent row dot-products using OpenMP.
 */

inline void matrixMultiplication(const Csr_matrix &matrix,
                                 const double *inputVector,
                                 double *outputVector) {
  // NASA Rule #7: Validate pointers and dimensions
  assert(inputVector != nullptr && "inputVector cannot be null");
  assert(outputVector != nullptr && "outputVector cannot be null");
  assert(matrix.rows >= 0 && matrix.cols >= 0 &&
         "Matrix dimensions must be not negative");

  // Runtime safety checks
  assert(matrix.row_pointer.size() >= (size_t)matrix.rows + 1 &&
         "Row pointer size mismatch");

#pragma omp parallel for
  for (int rowIndex = 0; rowIndex < matrix.rows; rowIndex++) {
    double rowSum = 0.0;

    // Define the traversal range for the current row
    int rowStart = matrix.row_pointer[rowIndex];
    int rowEnd = matrix.row_pointer[rowIndex + 1];

    assert(rowStart >= 0 && rowEnd <= (int)matrix.values.size() &&
           "Row bounds out of range");

    // Accumulate dot product: rowSum = sum(value[j] * inputVector[column[j]])
    for (int elementIndex = rowStart; elementIndex < rowEnd; ++elementIndex) {
      int col = matrix.col_indices[elementIndex];
      assert(col >= 0 && col < matrix.cols && "Column index out of range");
      rowSum += matrix.values[elementIndex] * inputVector[col];
    }

    // Store the final accumulation for this dimension
    outputVector[rowIndex] = rowSum;
  }
}

// ============================================================================
// VECTOR OPERATIONS (OpenMP Optimized)
// ============================================================================

// Computes the dot product of two vectors: sum = sum(A[i] * B[i])

inline double dotProduct(const std::vector<double> &vectorA,
                         const std::vector<double> &vectorB) {

  assert(vectorA.size() == vectorB.size() &&
         "Vectors must be of the same size");
  assert(!vectorA.empty() && "Vectors should not be empty");

  double sum = 0.0;
  const int n = (int)vectorA.size();
  // Sequential accumulation required for bit-identical determinism.
  // Parallel reduction order is non-deterministic.
  for (int index = 0; index < n; index++)
    sum += vectorA[index] * vectorB[index];
  return sum;
}

// Computes result = vectorA + scalar * vectorB (AXPY operation)

inline void vectorAdd(std::vector<double> &result,
                      const std::vector<double> &vectorA,
                      const std::vector<double> &vectorB, double scalar) {
  assert(vectorA.size() == vectorB.size() &&
         "Input vectors must be of same size");
  assert(result.size() == vectorA.size() &&
         "Result vector must match input size");

  const int n = (int)vectorA.size();
#pragma omp parallel for
  for (int index = 0; index < n; index++)
    result[index] = vectorA[index] + scalar * vectorB[index];

  assert(n >= 0 && "Vector size logic error");
}

/*
 * Extracts the inverse of the diagonal elements of a CSR matrix.
 *
 * This is used for the Jacobi preconditioner. If a diagonal element is zero or
 * near-zero, the inverse is set to 1.0, effectively disabling preconditioning
 * for that row to preserve numerical stability.
 *
 * Returns: A vector containing 1/A[i][i] for each row i.
 */

inline std::vector<double> getDiagonalInverse(const Csr_matrix &matrix) {
  assert(matrix.rows >= 0 && "Matrix rows cannot be negative");

  const double zeroThreshold = 1e-18;
  std::vector<double> diagonalInverse(matrix.rows, 1.0);

  // Pre-allocation check
  assert(diagonalInverse.size() == (size_t)matrix.rows &&
         "Vector pre-allocation error");

#pragma omp parallel for
  for (int rowIndex = 0; rowIndex < matrix.rows; rowIndex++) {
    const int rowStart = matrix.row_pointer[rowIndex];
    const int rowEnd = matrix.row_pointer[rowIndex + 1];

    for (int elementIndex = rowStart; elementIndex < rowEnd; elementIndex++) {
      if (matrix.col_indices[elementIndex] == rowIndex) {
        if (std::abs(matrix.values[elementIndex]) > zeroThreshold)
          diagonalInverse[rowIndex] = 1.0 / matrix.values[elementIndex];

        break; // If a diagonal is found, no need to check further in this row
      }
    }
  }
  return diagonalInverse;
}

// Contains the results of a linear solver execution.

struct SolverResult {
  std::vector<double> solution;
  double finalResidual;
  int iterations;
  bool converged;
  // Phase 2.9: Rank intelligence (populated by solveLU_Pivoted)
  bool rankDeficient = false;
  int rankDeficientRow = -1;    // 0-based row of smallest pivot found
  double smallestPivot = 1e300; // Min |pivot| across LU factorization
  int gpuFronts = 0;            // contraction solver: fronts factored in FP32 on the GPU
  bool fp64Refactor = false;    // ... whose refinement failed, so the solve was redone in FP64
};

/*
 * This solves Ax = b using the Preconditioned Conjugate Gradient (PCG) method.
 *
 * Uses a Jacobi (diagonal) preconditioner for faster convergence on diagonal
 * dominant matrices.
 *
 * matrix: The sparse matrix A (must be symmetric positive definite).
 * rhsVector: The right-hand side vector b.
 * tolerance: Convergence tolerance (stops when ||r|| < tolerance).
 * maxIterations: Maximum number of solver iterations.
 *
 * Returns: The solution vector, final residual, and iteration count.
 */

inline SolverResult solvePCG(const Csr_matrix &matrix,
                             const std::vector<double> &rhsVector,
                             double tolerance = 1e-10,
                             int maxIterations = 1000) {

  assert(matrix.rows > 0 && "Matrix must have at least one row");
  assert(matrix.rows == (int)rhsVector.size() &&
         "Matrix rows must match RHS vector size");
  assert(maxIterations > 0 && maxIterations < 100000 &&
         "maxIterations must be in [1, 100000)");
  assert(tolerance > 0 && "Tolerance must be positive");
  assert(matrix.cols == matrix.rows && "PCG requires a square matrix");

  const double divisionByZeroThreshold = 1e-20;
  const int numUnknowns = matrix.rows;

  std::vector<double> solution(numUnknowns, 0.0);
  std::vector<double> residual = rhsVector; // residual = b - A*x, with x=0

  assert(residual.size() == (size_t)numUnknowns &&
         "Residual vector size error");

  // Getting Jacobi Preconditioner (M^-1 = 1/diag(A))
  std::vector<double> preconditionerInverse = getDiagonalInverse(matrix);

  // Apply preconditioner: preconditionedResidual = M^-1 * residual
  std::vector<double> preconditionedResidual(numUnknowns);
  for (int i = 0; i < numUnknowns; i++) {
    preconditionedResidual[i] = residual[i] * preconditionerInverse[i];
  }

  std::vector<double> searchDirection = preconditionedResidual;
  double residualDotPreconditioned =
      dotProduct(residual, preconditionedResidual);

  // Early exit if already converged
  double initialResidualNorm = std::sqrt(dotProduct(residual, residual));
  if (initialResidualNorm < tolerance)
    return {solution, initialResidualNorm, 0, true};

  int iteration = 0;
  double currentResidualNorm = initialResidualNorm;
  const int maxIters = maxIterations;
  for (iteration = 0; iteration < maxIters; iteration++) {
    // matrixTimesSearchDirection = A * searchDirection
    std::vector<double> matrixTimesSearchDirection(numUnknowns);
    matrixMultiplication(matrix, searchDirection.data(),
                         matrixTimesSearchDirection.data());

    double searchDirectionDotAp =
        dotProduct(searchDirection, matrixTimesSearchDirection);
    if (std::abs(searchDirectionDotAp) < divisionByZeroThreshold) {
      // Numerical instability detected. Instead of crashing, return partial
      // result as failed.
      return {solution, currentResidualNorm, iteration, false};
    }

    double stepSize = residualDotPreconditioned / searchDirectionDotAp;

    // Update solution: x = x + stepSize * p
    vectorAdd(solution, solution, searchDirection, stepSize);
    // Update residual: r = r - stepSize * Ap
    vectorAdd(residual, residual, matrixTimesSearchDirection, -stepSize);

    // Check for convergence
    currentResidualNorm = std::sqrt(dotProduct(residual, residual));

    // Sanity check on residual values
    assert(!std::isnan(currentResidualNorm) && "Residual norm is NaN");

    if (currentResidualNorm < tolerance)
      return {solution, currentResidualNorm, iteration + 1, true};

    // Apply Preconditioner: z = M^-1 * r
    for (int i = 0; i < numUnknowns; i++)
      preconditionedResidual[i] = residual[i] * preconditionerInverse[i];

    double newResidualDotPreconditioned =
        dotProduct(residual, preconditionedResidual);

    if (std::abs(residualDotPreconditioned) < divisionByZeroThreshold) {
      return {solution, currentResidualNorm, iteration, false};
    }

    double directionUpdateFactor =
        newResidualDotPreconditioned / residualDotPreconditioned;

    // Update search direction: p = z + beta * p
    vectorAdd(searchDirection, preconditionedResidual, searchDirection,
              directionUpdateFactor);

    residualDotPreconditioned = newResidualDotPreconditioned;
  }

  return {solution, currentResidualNorm, iteration, false};
}

// ============================================================================
// DIRECT DENSE SOLVER (LU DECOMPOSITION WITH PARTIAL PIVOTING)
// ============================================================================

/**
 * Converts a sparse CSR matrix to a dense row-major vector representation.
 * Optimized for small matrices where direct solvers are preferred.
 */
inline std::vector<double> convertCsrToDense(const Csr_matrix &matrix) {
    std::vector<double> dense(matrix.rows * matrix.cols, 0.0);
    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = matrix.row_pointer[i]; j < matrix.row_pointer[i+1]; ++j) {
            dense[i * matrix.cols + matrix.col_indices[j]] = matrix.values[j];
        }
    }
    return dense;
}

/**
 * solveLU_Pivoted - Robust Direct Solver for Non-Symmetric Matrices.
 * 
 * Performs Crout/Doolittle LU decomposition with Partial Pivoting (row swapping).
 * This is the fallback solver for circuits where iterative methods (PCG) fail
 * due to zero-diagonals or non-symmetry (active components like BJTs).
 */
inline SolverResult solveLU_Pivoted(const Csr_matrix &matrix, 
                                   const std::vector<double> &rhsVector) {
    const int n = matrix.rows;
    assert(matrix.rows == matrix.cols && "LU requires a square matrix");
    assert(matrix.rows == (int)rhsVector.size() && "Dimension mismatch");

    std::vector<double> A = convertCsrToDense(matrix);
    std::vector<double> b = rhsVector;
    std::vector<double> x(n, 0.0);
    std::vector<int> pivot(n);
    for (int i = 0; i < n; ++i) pivot[i] = i;

    // Phase 2.9: Rank-revealing pivot tracking
    constexpr double kPivotThreshold = 1e-18;
    double smallestPivot = 1e300;
    int smallestPivotRow = -1;

    // --- 1. LU Decomposition with Partial Pivoting ---
    for (int i = 0; i < n; ++i) {
        // Find pivot for column i
        double maxVal = 0.0;
        int maxRow = i;
        for (int k = i; k < n; ++k) {
            double val = std::abs(A[k * n + i]);
            // Deterministic tie-breaking: prefer larger value; if identical, prefer lower row index.
            if (val > maxVal) {
                maxVal = val;
                maxRow = k;
            } else if (val == maxVal && k < maxRow) {
                maxRow = k;
            }
        }

        // Track smallest pivot for rank-deficiency detection
        if (maxVal < smallestPivot) {
            smallestPivot = maxVal;
            smallestPivotRow = i;
        }

        // Check for singularity
        if (maxVal < 1e-25) {
            // Matrix is singular or near-singular. Return failure.
            std::cerr << "LU: Singular matrix at row " << i << " (maxVal=" << maxVal << ")\n";
            std::cerr << "Diagonals: ";
            for(int d=0; d<n; ++d) std::cerr << A[d*n+d] << " ";
            std::cerr << "\n";
            SolverResult failResult;
            failResult.solution = x;
            failResult.finalResidual = 1.0;
            failResult.iterations = 0;
            failResult.converged = false;
            failResult.rankDeficient = true;
            failResult.rankDeficientRow = smallestPivotRow;
            failResult.smallestPivot = smallestPivot;
            return failResult;
        }

        // Swap rows if necessary
        if (maxRow != i) {
            std::swap(pivot[i], pivot[maxRow]);
            for (int k = 0; k < n; ++k) {
                std::swap(A[i * n + k], A[maxRow * n + k]);
            }
            std::swap(b[i], b[maxRow]);
        }

        // Compute multipliers and eliminate
        for (int j = i + 1; j < n; ++j) {
            A[j * n + i] /= A[i * n + i];
            for (int k = i + 1; k < n; ++k) {
                A[j * n + k] -= A[j * n + i] * A[i * n + k];
            }
        }
    }

    // --- 2. Forward Substitution (Ly = b) ---
    std::vector<double> y(n, 0.0);
    for (int i = 0; i < n; ++i) {
        y[i] = b[i];
        for (int j = 0; j < i; ++j) {
            y[i] -= A[i * n + j] * y[j];
        }
    }

    // --- 3. Backward Substitution (Ux = y) ---
    for (int i = n - 1; i >= 0; --i) {
        x[i] = y[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= A[i * n + j] * x[j];
        }
        x[i] /= A[i * n + i];
    }

    // Verify residual (||Ax-b||)
    std::vector<double> check(n, 0.0);
    matrixMultiplication(matrix, x.data(), check.data());
    double residual = 0.0;
    for (int i = 0; i < n; ++i) {
        double err = check[i] - rhsVector[i];
        residual += err * err;
    }
    residual = std::sqrt(residual);

    SolverResult luResult;
    luResult.solution = x;
    luResult.finalResidual = residual;
    luResult.iterations = 1;
    luResult.converged = true;
    luResult.smallestPivot = smallestPivot;
    luResult.rankDeficientRow = smallestPivotRow;
    luResult.rankDeficient = (smallestPivot < kPivotThreshold);
    return luResult;
}

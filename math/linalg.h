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
 * CSR (Compressed Sparse Row) storage, triplet assembly with duplicate handling,
 * and sparse matrix-vector multiplication. The solver is math/contraction_solver.
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
  }

  // Explicitly set matrix dimensions
  void setDimensions(int rows, int cols) {
    numRows = rows;
    numCols = cols;
  }

  // XOR of hash(row, col, value) over the triplets of the last createCsr()

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

// Contains the results of a linear solver execution.

struct SolverResult {
  std::vector<double> solution;  // empty unless converged
  double finalResidual = 0.0;
  int iterations = 0;
  bool converged = false;
  // Rank intelligence (the contraction solver's pivots)
  bool rankDeficient = false;
  int rankDeficientRow = -1;    // 0-based row of smallest pivot found
  double smallestPivot = 1e300; // Min |pivot| across LU factorization
  int gpuFronts = 0;            // contraction solver: fronts factored in FP32 on the GPU
  bool fp64Refactor = false;    // ... whose refinement failed, so the solve was redone in FP64
};


/**
 * Converts a sparse CSR matrix to a dense row-major vector representation
 * (OpenVLSI's verification harness compares against it).
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


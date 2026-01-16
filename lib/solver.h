#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>

// ========================
// Configuration
// ========================
constexpr int B = 11;   // Number of variables per block

// ========================
// Lambda inversion utilities
// ========================
double L(double mu);
double dL(double mu);
double invert(double L_target);

// ========================
// Data structures
// ========================
struct SparseBlock {
    std::vector<int> row;
    std::vector<int> col;
    std::vector<double> val;
};

using DenseBlock = std::array<std::array<double, B>, B>;
using VecBlock = std::array<double, B>;

// ========================
// Dense utilities
// ========================
DenseBlock to_dense(const SparseBlock& S);

void matvec(const DenseBlock& A, const double x[B], double y[B]);
void matmul(const DenseBlock& A, const DenseBlock& Bm, DenseBlock& C);
void subtract_inplace(DenseBlock& A, const DenseBlock& Bm);

// ========================
// LU with partial pivoting
// ========================
void lu_factor(DenseBlock& A, std::array<int, B>& piv);

void lu_solve_vec(
    const DenseBlock& LU,
    const std::array<int, B>& piv,
    const double b_in[B],
    double x[B]
);

void lu_solve_mat(
    const DenseBlock& LU,
    const std::array<int, B>& piv,
    const DenseBlock& Bm,
    DenseBlock& X
);

// ========================
// Block-tridiagonal solver
// ========================
void solve_block_tridiag(
    const std::vector<SparseBlock>& L,
    const std::vector<SparseBlock>& D,
    const std::vector<SparseBlock>& R,
    const std::vector<VecBlock>& Q,
    std::vector<VecBlock>& X
);

// ========================
// Sparse helper
// ========================
void add(SparseBlock& B, int p, int q, double v);

// ========================
// Other utilities
// ========================
inline double H(double v) {
    const double delta = 1e-6;
    return 0.5 * (1.0 + std::tanh(v / delta));
}

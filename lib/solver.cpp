#include <stdexcept>
#include <sstream>
#include <cmath>
#include <array>
#include <algorithm>
#include <iomanip>
#include <string>
#include <vector>
#include <iostream>

#include "solver.h"

// ========================
// Lambda inversion
// ========================
double L(double mu) {
    return (2.0 - (mu * mu + 2.0) * std::sqrt(1.0 - mu * mu)) / (mu * mu * mu);
}

double dL(double mu) {
    double s = std::sqrt(1.0 - mu * mu);
    double ds = -mu / s;

    double A = 2.0 - (mu * mu + 2.0) * s;
    double dA = -(2.0 * mu * s + (mu * mu + 2.0) * ds);

    double Bm = mu * mu * mu;
    double dBm = 3.0 * mu * mu;

    return (dA * Bm - A * dBm) / (Bm * Bm);
}

double invert(double L_target) {
    double mu = 0.5;

    for (int i = 0; i < 30; ++i) {
        mu -= (L(mu) - L_target) / dL(mu);
        if (mu < 1e-6)   mu = 1e-6;
        if (mu > 0.9999) mu = 0.9999;
    }
    return mu;
}

// ========================
// Dense utilities
// ========================
DenseBlock to_dense(const SparseBlock& S) {
    DenseBlock M{};
    for (std::size_t k = 0; k < S.val.size(); ++k)
        M[S.row[k]][S.col[k]] = S.val[k];
    return M;
}

void matvec(const DenseBlock& A, const double x[B], double y[B]) {
    for (int i = 0; i < B; ++i) {
        double s = 0.0;
        for (int j = 0; j < B; ++j)
            s += A[i][j] * x[j];
        y[i] = s;
    }
}

void matmul(const DenseBlock& A, const DenseBlock& Bm, DenseBlock& C) {
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < B; ++j) {
            double s = 0.0;
            for (int k = 0; k < B; ++k)
                s += A[i][k] * Bm[k][j];
            C[i][j] = s;
        }
    }
}

void subtract_inplace(DenseBlock& A, const DenseBlock& Bm) {
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < B; ++j)
            A[i][j] -= Bm[i][j];
}

static std::string block_to_string(const DenseBlock& A) {
    std::ostringstream oss;
    oss << std::setprecision(16);
    for (int r = 0; r < B; ++r) {
        for (int c = 0; c < B; ++c)
            oss << std::setw(24) << A[r][c] << ' ';
        oss << '\n';
    }
    return oss.str();
}

static double block_max_abs(const DenseBlock& A) {
    double maxv = 0.0;
    for (int r = 0; r < B; ++r)
        for (int c = 0; c < B; ++c)
            maxv = std::max(maxv, std::fabs(A[r][c]));
    return maxv;
}

// ========================
// Column scaling
// ========================
static std::array<double, B> compute_column_scaling(
    const std::vector<DenseBlock>& Ld,
    const std::vector<DenseBlock>& Dd,
    const std::vector<DenseBlock>& Rd) {
    std::array<double, B> colMax{};
    for (int j = 0; j < B; ++j)
        colMax[j] = 0.0;

    const int Nx = static_cast<int>(Dd.size());

    for (int i = 0; i < Nx; ++i) {
        for (int r = 0; r < B; ++r) {
            for (int c = 0; c < B; ++c) {
                colMax[c] = std::max(colMax[c], std::fabs(Dd[i][r][c]));
                if (i > 0)
                    colMax[c] = std::max(colMax[c], std::fabs(Ld[i][r][c]));
                if (i < Nx - 1)
                    colMax[c] = std::max(colMax[c], std::fabs(Rd[i][r][c]));
            }
        }
    }

    std::array<double, B> scale{};
    for (int j = 0; j < B; ++j) {
        if (colMax[j] > 0.0)
            scale[j] = 1.0 / colMax[j];
        else
            scale[j] = 1.0;
    }

    return scale;
}

static void apply_column_scaling(DenseBlock& A, const std::array<double, B>& scale) {
    for (int r = 0; r < B; ++r)
        for (int c = 0; c < B; ++c)
            A[r][c] *= scale[c];
}

static void apply_column_scaling_to_system(
    std::vector<DenseBlock>& Ld,
    std::vector<DenseBlock>& Dd,
    std::vector<DenseBlock>& Rd,
    const std::array<double, B>& scale) {
    const int Nx = static_cast<int>(Dd.size());
    for (int i = 0; i < Nx; ++i) {
        apply_column_scaling(Dd[i], scale);
        if (i > 0)      apply_column_scaling(Ld[i], scale);
        if (i < Nx - 1) apply_column_scaling(Rd[i], scale);
    }
}

static void unscale_solution(
    std::vector<VecBlock>& X,
    const std::array<double, B>& scale) {
    const int Nx = static_cast<int>(X.size());
    for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < B; ++j)
            X[i][j] *= scale[j];
}

// ========================
// LU factorization
// ========================
void lu_factor(DenseBlock& A, std::array<int, B>& piv, int blockIndex, const char* blockName) {
    for (int i = 0; i < B; ++i)
        piv[i] = i;

    const double amax = block_max_abs(A);
    const double pivot_tol = 1e-14 * std::max(1.0, amax);

    for (int k = 0; k < B; ++k) {
        int p = k;
        double maxv = std::fabs(A[k][k]);

        for (int i = k + 1; i < B; ++i) {
            double v = std::fabs(A[i][k]);
            if (v > maxv) {
                maxv = v;
                p = i;
            }
        }

        if (maxv < pivot_tol) {
            std::ostringstream oss;
            oss << "LU factorization failed: singular or nearly singular block detected.\n"
                << "Block name      = " << blockName << '\n'
                << "Global index    = " << blockIndex << '\n'
                << "Pivot step k    = " << k << '\n'
                << "Pivot row p     = " << p << '\n'
                << "Pivot candidate = " << maxv << '\n'
                << "Tolerance       = " << pivot_tol << '\n'
                << "Block contents:\n"
                << block_to_string(A);

            std::cerr << oss.str() << std::endl;
            throw std::runtime_error("LU factorization failed");
        }

        if (p != k) {
            std::swap(piv[k], piv[p]);
            for (int j = 0; j < B; ++j)
                std::swap(A[k][j], A[p][j]);
        }

        for (int i = k + 1; i < B; ++i) {
            A[i][k] /= A[k][k];
            const double lik = A[i][k];
            for (int j = k + 1; j < B; ++j)
                A[i][j] -= lik * A[k][j];
        }
    }
}

void lu_solve_vec(
    const DenseBlock& LU,
    const std::array<int, B>& piv,
    const double b_in[B],
    double x[B]) {

    double y[B];
    for (int i = 0; i < B; ++i)
        y[i] = b_in[piv[i]];

    for (int i = 0; i < B; ++i)
        for (int j = 0; j < i; ++j)
            y[i] -= LU[i][j] * y[j];

    for (int i = 0; i < B; ++i)
        x[i] = 0.0;

    for (int i = B - 1; i >= 0; --i) {
        for (int j = i + 1; j < B; ++j)
            y[i] -= LU[i][j] * x[j];
        x[i] = y[i] / LU[i][i];
    }
}

void lu_solve_mat(
    const DenseBlock& LU,
    const std::array<int, B>& piv,
    const DenseBlock& Bm,
    DenseBlock& X) {

    for (int col = 0; col < B; ++col) {
        double b_col[B];
        double x_col[B];

        for (int i = 0; i < B; ++i)
            b_col[i] = Bm[i][col];

        lu_solve_vec(LU, piv, b_col, x_col);

        for (int i = 0; i < B; ++i)
            X[i][col] = x_col[i];
    }
}

// ========================
// Block-tridiagonal solver
// ========================
void solve_block_tridiag(
    const std::vector<SparseBlock>& L,
    const std::vector<SparseBlock>& D,
    const std::vector<SparseBlock>& R,
    const std::vector<VecBlock>& Q,
    std::vector<VecBlock>& X) {

    const int Nx = static_cast<int>(D.size());
    if (Nx == 0) return;

    std::vector<DenseBlock> Dd(Nx), Ld(Nx), Rd(Nx);
    for (int i = 0; i < Nx; ++i) {
        Dd[i] = to_dense(D[i]);
        if (i > 0)      Ld[i] = to_dense(L[i]);
        if (i < Nx - 1) Rd[i] = to_dense(R[i]);
    }

    // Global column scaling for all local variables
    const std::array<double, B> colScale = compute_column_scaling(Ld, Dd, Rd);
    apply_column_scaling_to_system(Ld, Dd, Rd, colScale);

    std::vector<VecBlock> Qm = Q;
    X.assign(Nx, VecBlock{});

    std::vector<std::array<int, B>> piv(Nx);
    std::vector<bool> factored(Nx, false);

    for (int i = 1; i < Nx; ++i) {
        const int im1 = i - 1;

        if (!factored[im1]) {
            lu_factor(Dd[im1], piv[im1], im1, "D");
            factored[im1] = true;
        }

        DenseBlock Xtemp{};
        lu_solve_mat(Dd[im1], piv[im1], Rd[im1], Xtemp);

        DenseBlock L_X{};
        matmul(Ld[i], Xtemp, L_X);
        subtract_inplace(Dd[i], L_X);

        double y[B], q_prev[B];
        for (int k = 0; k < B; ++k)
            q_prev[k] = Qm[im1][k];

        lu_solve_vec(Dd[im1], piv[im1], q_prev, y);

        double Ly[B];
        matvec(Ld[i], y, Ly);
        for (int k = 0; k < B; ++k)
            Qm[i][k] -= Ly[k];
    }

    if (!factored[Nx - 1]) {
        lu_factor(Dd[Nx - 1], piv[Nx - 1], Nx - 1, "D");
        factored[Nx - 1] = true;
    }

    {
        double rhs[B], sol[B];
        for (int k = 0; k < B; ++k)
            rhs[k] = Qm[Nx - 1][k];

        lu_solve_vec(Dd[Nx - 1], piv[Nx - 1], rhs, sol);

        for (int k = 0; k < B; ++k)
            X[Nx - 1][k] = sol[k];
    }

    for (int i = Nx - 2; i >= 0; --i) {
        if (!factored[i]) {
            lu_factor(Dd[i], piv[i], i, "D");
            factored[i] = true;
        }

        double RX[B];
        matvec(Rd[i], X[i + 1].data(), RX);

        double rhs[B];
        for (int k = 0; k < B; ++k)
            rhs[k] = Qm[i][k] - RX[k];

        double sol[B];
        lu_solve_vec(Dd[i], piv[i], rhs, sol);

        for (int k = 0; k < B; ++k)
            X[i][k] = sol[k];
    }

    // Recover physical variables: x = S * x_hat
    unscale_solution(X, colScale);
}

// ========================
// Sparse helper
// ========================
void add(SparseBlock& Bm, int p, int q, double v) {
    Bm.row.push_back(p);
    Bm.col.push_back(q);
    Bm.val.push_back(v);
}
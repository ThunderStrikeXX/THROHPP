#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <array>
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <cassert>
#include <string>

bool warnings = false;

#include "steel.h"
#include "liquid_sodium.h"
#include "vapor_sodium.h"

// =======================================================================
//
//                        [VARIOUS ALGORITHMS]
//
// =======================================================================

#pragma region find_mu

constexpr int B = 11;  // dimensione blocco

double L(double mu) {
    return (2.0 - (mu * mu + 2.0) * std::sqrt(1.0 - mu * mu)) / (mu * mu * mu);
}

double dL(double mu) {
    double s = std::sqrt(1.0 - mu * mu);
    double ds = -mu / s;

    double A = 2.0 - (mu * mu + 2.0) * s;
    double dA = -(2.0 * mu * s + (mu * mu + 2.0) * ds);

    double B = mu * mu * mu;
    double dB = 3.0 * mu * mu;

    return (dA * B - A * dB) / (B * B);
}

double invert(double L_target) {
    double mu = 0.5;

    for (int i = 0; i < 30; i++) {
        mu -= (L(mu) - L_target) / dL(mu);
        if (mu < 1e-6)   mu = 1e-6;
        if (mu > 0.9999) mu = 0.9999;
    }
    return mu;
}

#pragma endregion

#pragma region solving_functions

// Definition of data structures 
struct SparseBlock {
    std::vector<int> row;
    std::vector<int> col;
    std::vector<double> val;
};

using DenseBlock = std::array<std::array<double, B>, B>;
using VecBlock = std::array<double, B>;

// ------------------------- Utility dense -------------------------

// Converts a sparse matrix S to a dense matrix M
DenseBlock to_dense(const SparseBlock& S) {
    DenseBlock M{};
    for (std::size_t k = 0; k < S.val.size(); ++k) {
        int i = S.row[k];
        int j = S.col[k];
        M[i][j] = S.val[k];
    }
    return M;
}

// Executes the application of a dense matrix A to a vector x to get a vector y
void matvec(const DenseBlock& A, const double x[B], double y[B]) {
    for (int i = 0; i < B; ++i) {
        double s = 0.0;
        for (int j = 0; j < B; ++j)
            s += A[i][j] * x[j];
        y[i] = s;
    }
}

// Executes the multiplication between a dense matrix A and a dense matrix B to get a dense matrix C
void matmul(const DenseBlock& A, const DenseBlock& Bm, DenseBlock& C) {
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < B; ++j) {
            double s = 0.0;
            for (int k = 0; k < B; ++k)
                s += A[i][k] * Bm[k][j];
            C[i][j] = s;
        }
}

// Executes the subtraction of a matrix Bm from a matrix A
void subtract_inplace(DenseBlock& A, const DenseBlock& Bm) {
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < B; ++j)
            A[i][j] -= Bm[i][j];
}

// ------------------------- LU with pivoting -------------------------

// In-place LU factorization with partial pivoting, storing L below and U on/above the diagonal.
void lu_factor(DenseBlock& A, std::array<int, B>& piv) {
    for (int i = 0; i < B; ++i)
        piv[i] = i;

    for (int k = 0; k < B; ++k) {

        // Pivot
        int p = k;
        double maxv = std::fabs(A[k][k]);
        for (int i = k + 1; i < B; ++i) {
            double v = std::fabs(A[i][k]);
            if (v > maxv) {
                maxv = v;
                p = i;
            }
        }

        if (maxv == 0.0)
            throw std::runtime_error("LU: singular matrix");

        // Rows swapping
        if (p != k) {
            std::swap(piv[k], piv[p]);
            for (int j = 0; j < B; ++j)
                std::swap(A[k][j], A[p][j]);
        }

        // Elimination
        for (int i = k + 1; i < B; ++i) {
            A[i][k] /= A[k][k];
            double lik = A[i][k];
            for (int j = k + 1; j < B; ++j)
                A[i][j] -= lik * A[k][j];
        }
    }
}

// Solves Ax = b using the in-place LU factorization (with pivoting) via forward and backward substitution.
void lu_solve_vec(const DenseBlock& LU, const std::array<int, B>& piv,
    const double b_in[B], double x[B]) {

    // Applies pivot to b
    double y[B];
    for (int i = 0; i < B; ++i)
        y[i] = b_in[piv[i]];

    // Ly = Pb (forward)
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < i; ++j)
            y[i] -= LU[i][j] * y[j];
    }

    // Ux = y (backward)
    for (int i = B - 1; i >= 0; --i) {
        for (int j = i + 1; j < B; ++j)
            y[i] -= LU[i][j] * x[j];
        x[i] = y[i] / LU[i][i];
    }
}

// Solves LU·X = P·B column-wise by applying the LU-based vector solver to each column of B.
void lu_solve_mat(const DenseBlock& LU, const std::array<int, B>& piv,
    const DenseBlock& Bm, DenseBlock& X) {

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

// ------------------------- Solve block-tridiagonal -------------------------


// Block Thomas solver: performs forward elimination and back substitution on a block-tridiagonal system using per-block LU factorizations.
void solve_block_tridiag(
    const std::vector<SparseBlock>& L,
    const std::vector<SparseBlock>& D,
    const std::vector<SparseBlock>& R,
    const std::vector<VecBlock>& Q,
    std::vector<VecBlock>& X) {
    const int Nx = static_cast<int>(D.size());
    if (Nx == 0)
        return;

    // Dense copy of the blocks
    std::vector<DenseBlock> Dd(Nx);
    std::vector<DenseBlock> Ld(Nx);
    std::vector<DenseBlock> Rd(Nx);

    for (int i = 0; i < Nx; ++i) {
        Dd[i] = to_dense(D[i]);
        if (i > 0)     Ld[i] = to_dense(L[i]);
        if (i < Nx - 1)  Rd[i] = to_dense(R[i]);
    }

    std::vector<VecBlock> Qm = Q;             // Q changed during forward
    X.assign(Nx, VecBlock{});                 // Solution

    std::vector<std::array<int, B>> piv(Nx);
    std::vector<bool> factored(Nx, false);

    // -------- Forward elimination --------
    for (int i = 1; i < Nx; ++i) {
        int im1 = i - 1;

        if (!factored[im1]) {
            lu_factor(Dd[im1], piv[im1]);
            factored[im1] = true;
        }

        // Solve D[im1] * Xtemp = R[im1]
        DenseBlock Xtemp;
        lu_solve_mat(Dd[im1], piv[im1], Rd[im1], Xtemp);

        // D[i] = D[i] - L[i] * Xtemp
        DenseBlock L_X;
        matmul(Ld[i], Xtemp, L_X);

        subtract_inplace(Dd[i], L_X);

        // Solve D[im1] * y = Qm[im1]
        double y[B], q_prev[B];
        for (int k = 0; k < B; ++k)
            q_prev[k] = Qm[im1][k];
        lu_solve_vec(Dd[im1], piv[im1], q_prev, y);

        // Qm[i] = Qm[i] - L[i] * y
        double Ly[B];
        matvec(Ld[i], y, Ly);
        for (int k = 0; k < B; ++k)
            Qm[i][k] -= Ly[k];
    }

    // -------- Backward substitution --------

    // Last block
    if (!factored[Nx - 1]) {
        lu_factor(Dd[Nx - 1], piv[Nx - 1]);
        factored[Nx - 1] = true;
    }
    {
        double rhs[B];
        double sol[B];
        for (int k = 0; k < B; ++k)
            rhs[k] = Qm[Nx - 1][k];
        lu_solve_vec(Dd[Nx - 1], piv[Nx - 1], rhs, sol);
        for (int k = 0; k < B; ++k)
            X[Nx - 1][k] = sol[k];
    }

    // Previous blocks
    for (int i = Nx - 2; i >= 0; --i) {
        if (!factored[i]) {
            lu_factor(Dd[i], piv[i]);
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
}

// Add triplet to sparse block
auto add = [&](SparseBlock& B, int p, int q, double v) {
    B.row.push_back(p);
    B.col.push_back(q);
    B.val.push_back(v);
};

#pragma endregion

#pragma region other_functions

inline int H(double x) {
    return x > 0.0 ? 1 : 0;
}

#pragma endregion

inline double surf_ten(double T) {
    constexpr double Tm = 371.0;
    double val = 0.196 - 2.48e-4 * (T - Tm);
    return val > 0.0 ? val : 0.0;
}

int main() {

    #pragma region constants_and_variables

    // Mathematical constants
    const double M_PI = 3.14159265358979323846;

    // Physical properties
    const double emissivity = 0.9;          /// Wall emissivity [-]
    const double sigma = 5.67e-8;           /// Stefan-Boltzmann constant [W/m^2/K^4]
    const double Rv = 361.8;                /// Gas constant for the sodium vapor [J/(kg K)]
    const double Tc = 2509.46;              /// Critical temperature of sodium [K]
    double const eps_v = 1.0;               /// Surface fraction of the wick available for liquid passage [-]

    // Wick permeability parameters
    const double K = 1e-8;                  /// Permeability [m^2]
    const double CF = 1e4;                  /// Forchheimer coefficient [1/m]
    
    // Environmental boundary conditions
    const double h_conv = 10;               /// Convective heat transfer coefficient for external heat removal [W/m^2/K]
    const double power = 119;               /// Power at the evaporator side [W]
    const double T_env = 280.0;             /// External environmental temperature [K]

    // Evaporation and condensation parameters
    const double eps_s = 1.0;               /// Surface fraction of the wick available for phasic interface [-]
    const double sigma_e = 0.05;            /// Evaporation accomodation coefficient [-]. 1 means optimal evaporation
    const double sigma_c = 0.05;            /// Condensation accomodation coefficient [-]. 1 means optimal condensation
    double Omega = 1.0;                     /// Omega factor initialization

    // Geometric parameters
    const int N = 20;                                                           /// Number of axial nodes [-]
    const double l = 0.982; 			                                        /// Length of the heat pipe [m]
    const double dz = l / N;                                                    /// Axial discretization step [m]
    const double evaporator_length = 0.502;                                     /// Evaporator length [m]
    const double adiabatic_length = 0.188;                                      /// Adiabatic length [m]
    const double condenser_length = 0.292;                                      /// Condenser length [m]
    const double evaporator_nodes = std::floor(evaporator_length / dz);         /// Number of evaporator nodes
    const double condenser_nodes = std::ceil(condenser_length / dz);            /// Number of condenser nodes
    const double adiabatic_nodes = N - (evaporator_nodes + condenser_nodes);    /// Number of adiabatic nodes
    const double r_o = 0.01335;                                                 /// Outer wall radius [m]
    const double r_i = 0.0112;                                                  /// Wall-wick interface radius [m]
    const double r_v = 0.01075;                                                 /// Vapor-wick interface radius [m]

    // Surfaces 
    const double A_w_outer = 2 * M_PI * r_o * dz;                               /// Wall radial area (at r_o) [m^2]
    const double A_w_cross = M_PI * (r_o * r_o - r_i * r_i);                    /// Wall cross-sectional area [m^2]
    const double A_x_interface = 2 * M_PI * r_i * dz;                           /// Wick radial area (at r_i) [m^2]
    const double A_x_cross = M_PI * (r_i * r_i - r_v * r_v);                    /// Wick cross-sectional area [m^2]
    const double A_v_inner = 2 * M_PI * r_v * dz;                               /// Vapor radial area (at r_v) [m^2]
    const double A_v_cross = M_PI * r_v * r_v;                                  /// Vapor cross-sectional area [m^2]

    const double Eio1 = 2.0 / 3.0 * (r_o + r_i - 1 / (1 / r_o + 1 / r_i));
    const double Eio2 = 0.5 * (r_o * r_o + r_i * r_i);
    const double Evi1 = 2.0 / 3.0 * (r_i + r_v - 1 / (1 / r_i + 1 / r_v));
    const double Evi2 = 0.5 * (r_i * r_i + r_v * r_v);

    // Time-stepping parameters
    double dt = 1e-6;                                   /// Initial time step [s] (then it is updated according to the limits)
    const int tot_iter = 100000;                        /// Number of timesteps
    const double time_total = tot_iter * dt;            /// Total simulation time [s]

    // Numerical parameters
    const double tolerance = 1e-4;			/// Tolerance for the convergence of Picard loop [-]
    const int Kmax = 20;                    /// Maximum number of Picard iterations per timestep

    // Mesh z positions
    std::vector<double> mesh(N, 0.0);
    for (int i = 0; i < N; ++i) mesh[i] = i * dz;       /// Mesh discretization

    // Node partition
    const int N_e = static_cast<int>(std::floor(evaporator_length / dz));   /// Number of nodes of the evaporator region [-]
    const int N_c = static_cast<int>(std::ceil(condenser_length / dz));     /// Number of nodes of the condenser region [-]
    const int N_a = N - (N_e + N_c);                                        /// Number of nodes of the adiabadic region [-]

    const double T_full = 800.0;                                            /// Uniform temperature initialization [K]

    const double q_pp_evaporator = power / (2 * M_PI * evaporator_length * r_o);        /// Heat flux at evaporator from given power [W/m^2]
	std::vector<double> q_pp(N, 0.0);                                                   /// Heat flux profile [W/m^2]

    std::vector<double> rho_m(N, 0.01);
    std::vector<double> rho_l(N, 1000);
    std::vector<double> alpha_m(N, 0.9);
    std::vector<double> alpha_l(N, 0.1);
    std::vector<double> p_m(N);
    std::vector<double> p_l(N);
    std::vector<double> v_m(N, 1.0);
    std::vector<double> v_l(N, -0.001);
    std::vector<double> T_m(N);
    std::vector<double> T_l(N);
    std::vector<double> T_w(N);

    std::vector<double> T_sur(N);

    // Temperature initialization
    for (int i = 0; i < N; ++i) {

        const double f = double(i) / double(N - 1);
        const double T = 800.0 + f * (600.0 - 800.0);
        T_m[i] = T;
        T_l[i] = T;
        T_w[i] = T;
        T_sur[i] = T;

        p_m[i] = vapor_sodium::P_sat(T);
        p_l[i] = p_m[i];
    }

    // Old variables
    std::vector<double> rho_m_old = rho_m;
    std::vector<double> rho_l_old = rho_l;
    std::vector<double> alpha_m_old = alpha_m;
    std::vector<double> alpha_l_old = alpha_l;
    std::vector<double> p_m_old = p_m;
    std::vector<double> p_l_old = p_l;
    std::vector<double> v_m_old = v_m;
    std::vector<double> v_l_old = v_l;
    std::vector<double> T_m_old = T_m;
    std::vector<double> T_l_old = T_l;
    std::vector<double> T_w_old = T_w;

    std::vector<double> rho_m_iter(N);
    std::vector<double> rho_l_iter(N);
    std::vector<double> alpha_m_iter(N);
    std::vector<double> alpha_l_iter(N);
    std::vector<double> p_m_iter(N);
    std::vector<double> p_l_iter(N);
    std::vector<double> v_m_iter(N);
    std::vector<double> v_l_iter(N);
    std::vector<double> T_m_iter(N);
    std::vector<double> T_l_iter(N);
    std::vector<double> T_w_iter(N);

    // Blocks definition
    std::vector<SparseBlock> L(N), D(N), R(N);
    std::vector<VecBlock> Q(N), X(N);

    // Secondary useful variables
    std::vector<double> Gamma_xv(N, 0.0);
    std::vector<double> phi_x_v(N, 0.0);
    std::vector<double> heat_source_wall_liquid_flux(N, 0.0);
    std::vector<double> heat_source_liquid_wall_flux(N, 0.0);
    std::vector<double> heat_source_vapor_liquid_phase(N, 0.0);
    std::vector<double> heat_source_liquid_vapor_phase(N, 0.0);
    std::vector<double> heat_source_vapor_liquid_flux(N, 0.0);
    std::vector<double> heat_source_liquid_vapor_flux(N, 0.0);
    std::vector<double> p_saturation(N);

    // Create result folder
    int new_case = 0;
    std::string name = "case_0";
    while (true) {
        name = "case_" + std::to_string(new_case);
        if (!std::filesystem::exists(name)) {
            std::filesystem::create_directory(name);
            break;
        }
        new_case++;
    }

    // Print results in file
    std::ofstream mesh_output(name + "/mesh.txt", std::ios::trunc);
    std::ofstream time_output(name + "/time.txt", std::ios::trunc);

    std::ofstream v_velocity_output(name + "/vapor_velocity.txt", std::ios::trunc);
    std::ofstream v_pressure_output(name + "/vapor_pressure.txt", std::ios::trunc);
    std::ofstream v_temperature_output(name + "/vapor_temperature.txt", std::ios::trunc);
    std::ofstream v_rho_output(name + "/rho_vapor.txt", std::ios::trunc);

    std::ofstream l_velocity_output(name + "/liquid_velocity.txt", std::ios::trunc);
    std::ofstream l_pressure_output(name + "/liquid_pressure.txt", std::ios::trunc);
    std::ofstream l_temperature_output(name + "/liquid_temperature.txt", std::ios::trunc);
    std::ofstream l_rho_output(name + "/liquid_rho.txt", std::ios::trunc);

    std::ofstream w_temperature_output(name + "/wall_temperature.txt", std::ios::trunc);

    std::ofstream v_alpha_output(name + "/vapor_alpha.txt", std::ios::trunc);
    std::ofstream l_alpha_output(name + "/liquid_alpha.txt", std::ios::trunc);

    std::ofstream gamma_output(name + "/gamma_xv.txt", std::ios::trunc);
	std::ofstream phi_output(name + "/phi_xv.txt", std::ios::trunc);
    std::ofstream hs_wl_flux_output(name + "/heat_source_wall_liquid_flux.txt", std::ios::trunc);
    std::ofstream hs_lw_flux_output(name + "/heat_source_liquid_wall_flux.txt", std::ios::trunc);
    std::ofstream hs_vl_phase_output(name + "/heat_source_vapor_liquid_phase.txt", std::ios::trunc);
    std::ofstream hs_lv_phase_output(name + "/heat_source_liquid_vapor_phase.txt", std::ios::trunc);
    std::ofstream hs_vl_flux_output(name + "/heat_source_vapor_liquid_flux.txt", std::ios::trunc);
    std::ofstream hs_lv_flux_output(name + "/heat_source_liquid_vapor_flux.txt", std::ios::trunc);
    std::ofstream psat_output(name + "/p_saturation.txt", std::ios::trunc);
    std::ofstream tsur_output(name + "/T_sur.txt", std::ios::trunc);

    const int global_precision = 8;

    mesh_output << std::setprecision(global_precision);
	time_output << std::setprecision(global_precision);

    v_velocity_output << std::setprecision(global_precision);
    v_pressure_output << std::setprecision(global_precision);
    v_temperature_output << std::setprecision(global_precision);
    v_rho_output << std::setprecision(global_precision);

    l_velocity_output << std::setprecision(global_precision);
    l_pressure_output << std::setprecision(global_precision);
    l_temperature_output << std::setprecision(global_precision);
    l_rho_output << std::setprecision(global_precision);

    w_temperature_output << std::setprecision(global_precision);

    v_alpha_output << std::setprecision(global_precision);
    l_alpha_output << std::setprecision(global_precision);

    gamma_output << std::setprecision(global_precision);
	phi_output << std::setprecision(global_precision);
    hs_wl_flux_output << std::setprecision(global_precision);
    hs_lw_flux_output << std::setprecision(global_precision);
    hs_vl_phase_output << std::setprecision(global_precision);
    hs_lv_phase_output << std::setprecision(global_precision);
    hs_vl_flux_output << std::setprecision(global_precision);
    hs_lv_flux_output << std::setprecision(global_precision);
    psat_output << std::setprecision(global_precision);
    tsur_output << std::setprecision(global_precision);
   
    for (int i = 0; i < N; ++i) mesh_output << i * dz << " ";

    mesh_output.flush();
    mesh_output.close();

    #pragma endregion

    /// Print number of working threads
    std::cout << "Threads: " << omp_get_max_threads() << "\n";

    double start = omp_get_wtime();
    
	// Time-stepping loop
    for(int n = 0; n < tot_iter; ++n) {

        rho_m_iter = rho_m_old;
        rho_l_iter = rho_l_old;
        alpha_m_iter = alpha_m_old;
        alpha_l_iter = alpha_l_old;
        p_m_iter = p_m_old;
        p_l_iter = p_l_old;
        v_m_iter = v_m_old;
        v_l_iter = v_l_old;
        T_m_iter = T_m_old;
        T_l_iter = T_l_old;
        T_w_iter = T_w_old;
       
		// Picard iteration loop
        for (int k = 0; k < Kmax; ++k) {

		    // Space discretization loop
            for(int i = 1; i < N - 1; ++i) {

                // Physical properties
                const double k_w = steel::k(T_w_iter[i]);                                                /// Wall thermal conductivity [W/(m K)]
                const double k_x = liquid_sodium::k(T_l_iter[i]);                                        /// Liquid thermal conductivity [W/(m K)]
                const double k_m = vapor_sodium::k(T_m_iter[i], p_m_iter[i]);                                 /// Vapor thermal conductivity [W/(m K)]
                const double cp_m = vapor_sodium::cp(T_m_iter[i]);                                       /// Vapor specific heat [J/(kg K)]
                const double mu_v = vapor_sodium::mu(T_m_iter[i]);                                       /// Vapor dynamic viscosity [Pa*s]
                const double mu_l = liquid_sodium::mu(T_l_iter[i]);                                      /// Liquid dynamic viscosity
                const double Dh_v = 2.0 * r_v;                                                      /// Hydraulic diameter of the vapor core [m]
                const double Re_v = rho_m_iter[i] * std::fabs(v_m_iter[i]) * Dh_v / mu_v;                     /// Reynolds number [-]
                const double Pr_v = cp_m * mu_v / k_m;                                              /// Prandtl number [-]
                const double H_xm = vapor_sodium::h_conv(Re_v, Pr_v, k_m, Dh_v);                    /// Convective heat transfer coefficient at the vapor-wick interface [W/m^2/K]
                const double Psat = vapor_sodium::P_sat(T_sur[i]);                                  /// Saturation pressure [Pa]         
                const double dPsat_dT = Psat * std::log(10.0) * (7740.0 / (T_sur[i] * T_sur[i]));   /// Derivative of the saturation pressure wrt T [Pa/K]   
        
                double h_xv_v;      /// Specific enthalpy [J/kg] of vapor upon phase change between wick and vapor
                double h_vx_x;      /// Specific enthalpy [J/kg] of wick upon phase change between vapor and wick

                if (Gamma_xv[i] >= 0.0) {

                    // Evaporation case
                    h_xv_v = vapor_sodium::h(T_sur[i]);
                    h_vx_x = liquid_sodium::h(T_sur[i]);

                }
                else {

                    // Condensation case
                    h_xv_v = vapor_sodium::h(T_m_iter[i]);
                    h_vx_x = liquid_sodium::h(T_sur[i])
                        + (vapor_sodium::h(T_m_iter[i]) - vapor_sodium::h(T_sur[i]));
                }

                // Update heat fluxes at the interfaces
                if (i <= evaporator_nodes) q_pp[i] = q_pp_evaporator;                                  /// Evaporator imposed heat flux
                else if (i >= (N - condenser_nodes)) {

                    double conv = h_conv * (T_w_iter[i] - T_env);                                          /// Condenser convective heat flux
                    double irr = emissivity * sigma * (std::pow(T_w_iter[i], 4) - std::pow(T_env, 4));     /// Condenser irradiation heat flux

                    q_pp[i] = -(conv + irr);                                                           /// Heat flux at the outer wall (positive if to the wall)
                }

                const double beta = 1.0 / std::sqrt(2 * M_PI * Rv * T_sur[i]);
                const double b = -phi_x_v[i] / (p_m_iter[i] * std::sqrt(2.0 / (Rv * T_m_iter[i])));

                if (b < 0.1192) Omega = 1.0 + b * std::sqrt(M_PI);
                else if (b <= 0.9962) Omega = 0.8959 + 2.6457 * b;
                else Omega = 2.0 * b * std::sqrt(M_PI);

                const double fac = (2.0 * r_v * eps_s * beta) / (r_i * r_i);        /// Useful factor in the coefficients calculation [s / m^2]

                const double bGamma = -(Gamma_xv[i] / (2.0 * T_sur[i])) + fac * sigma_e * dPsat_dT; /// b coefficient [kg/(m3 s K)] 
                const double aGamma = 0.5 * Gamma_xv[i] + fac * sigma_e * dPsat_dT * T_sur[i];      /// a coefficient [kg/(m3 s)]
                const double cGamma = -fac * sigma_c * Omega;                                       /// c coefficient [s/m2]

                const double Ex3 = H_xm + (h_vx_x * r_i * r_i) / (2.0 * r_v) * bGamma;
                const double Ex4 = -k_x + H_xm * r_v + h_vx_x * r_i * r_i / 2.0 * bGamma;
                const double Ex5 = -2.0 * r_v * k_x + H_xm * r_v * r_v + h_vx_x * r_i * r_i / 2.0 * bGamma * r_v;
                const double Ex6 = -H_xm;
                const double Ex7 = (h_vx_x * r_i * r_i) / (2.0 * r_v) * cGamma;
                const double Ex8 = (h_vx_x * r_i * r_i) / (2.0 * r_v) * aGamma;

                const double alpha = 1.0 / (2 * r_o * (Eio1 - r_i) + r_i * r_i - Eio2);
                const double gamma = r_i * r_i + ((Ex5 - Evi2 * Ex3) * (Evi1 - r_i)) / (Ex4 - Evi1 * Ex3) - Evi2;

                // Delta coefficients
                const double C1 = - (Evi1 - r_i) / (Ex4 - Evi1 * Ex3) * Ex7;
                const double C2 = - (Evi1 - r_i) / (Ex4 - Evi1 * Ex3) * Ex6;
			    const double C3 = + (Evi1 - r_i) / (Ex4 - Evi1 * Ex3) * Ex3 + 1;
			    const double C4 = - 1;
			    const double C5 = - (Evi1 - r_i) / (Ex4 - Evi1 * Ex3) * (Ex8 - Ex7 * p_m[i]) + q_pp[i] / k_w * (Eio1 - r_i);

			    // c_x coefficients
			    const double C6 = (2 * k_w * (r_o - r_i) * alpha * C1 + k_x * Ex7 / (Ex4 - Evi1 * Ex3)) / (2 * (r_i - r_o) * k_w * alpha * gamma + k_x * (Ex5 - Evi2 * Ex3) / (Ex4 - Evi1 * Ex3) - 2 * r_i * k_x);
			    const double C7 = (2 * k_w * (r_o - r_i) * alpha * C2 + k_x * Ex6 / (Ex4 - Evi1 * Ex3)) / (2 * (r_i - r_o) * k_w * alpha * gamma + k_x * (Ex5 - Evi2 * Ex3) / (Ex4 - Evi1 * Ex3) - 2 * r_i * k_x);
			    const double C8 = (2 * k_w * (r_o - r_i) * alpha * C3 - k_x * Ex3 / (Ex4 - Evi1 * Ex3)) / (2 * (r_i - r_o) * k_w * alpha * gamma + k_x * (Ex5 - Evi2 * Ex3) / (Ex4 - Evi1 * Ex3) - 2 * r_i * k_x);
			    const double C9 = (2 * k_w * (r_o - r_i) * alpha * C4) / (2 * (r_i - r_o) * k_w * alpha * gamma + k_x * (Ex5 - Evi2 * Ex3) / (Ex4 - Evi1 * Ex3) - 2 * r_i * k_x);
			    const double C10 = (- q_pp[i] + 2 * k_w * (r_o - r_i) * alpha * C5 + k_x * (Ex8 - p_m[i] * Ex7) / (Ex4 - Evi1 * Ex3)) / (2 * (r_i - r_o) * k_w * alpha * gamma + k_x * (Ex5 - Evi2 * Ex3) / (Ex4 - Evi1 * Ex3) - 2 * r_i * k_x);

                // c_w coefficients
			    const double C11 = alpha * (C1 + gamma * C6);
			    const double C12 = alpha * (C2 + gamma * C7);
			    const double C13 = alpha * (C3 + gamma * C8);
			    const double C14 = alpha * (C4 + gamma * C9);
			    const double C15 = alpha * (C5 + gamma * C10);

			    // b_w coefficients
			    const double C16 = - 2 * r_o * C11;
			    const double C17 = - 2 * r_o * C12;
			    const double C18 = - 2 * r_o * C13;
			    const double C19 = - 2 * r_o * C14;
			    const double C20 = - 2 * r_o * C15 + q_pp[i] / k_w;

			    // a_w coefficients
			    const double C21 = (2 * r_o * Eio1 - Eio2) * C11;
			    const double C22 = (2 * r_o * Eio1 - Eio2) * C12;
			    const double C23 = (2 * r_o * Eio1 - Eio2) * C13;
			    const double C24 = (2 * r_o * Eio1 - Eio2) * C14 + 1;
                const double C25 = (2 * r_o * Eio1 - Eio2) * C15 - q_pp[i] * Eio1 / k_w;

			    // b_x coefficients
			    const double C26 = (- (Ex5 - Evi2 * Ex3) * C6 + Ex7) / (Ex4 - Evi1 * Ex3);
			    const double C27 = (- (Ex5 - Evi2 * Ex3) * C7 + Ex6) / (Ex4 - Evi1 * Ex3);
			    const double C28 = (- (Ex5 - Evi2 * Ex3) * C8 - Ex3) / (Ex4 - Evi1 * Ex3);
			    const double C29 = (- (Ex5 - Evi2 * Ex3) * C9) / (Ex4 - Evi1 * Ex3);
			    const double C30 = (- (Ex5 - Evi2 * Ex3) * C10 + Ex8 - p_m[i] * Ex7) / (Ex4 - Evi1 * Ex3);

			    // a_x coefficients
			    const double C31 = - Evi1 * C26 - Evi2 * C6;
			    const double C32 = - Evi1 * C27 - Evi2 * C7;
			    const double C33 = - Evi1 * C28 - Evi2 * C8 + 1;
			    const double C34 = - Evi1 * C29 - Evi2 * C9;
			    const double C35 = - Evi1 * C30 - Evi2 * C10;

                // T_sur coefficients
			    const double C36 = C31 + r_v * C26 + r_v * r_v * C6;
			    const double C37 = C32 + r_v * C27 + r_v * r_v * C7;
			    const double C38 = C33 + r_v * C28 + r_v * r_v * C8;
			    const double C39 = C34 + r_v * C29 + r_v * r_v * C9;
			    const double C40 = C35 + r_v * C30 + r_v * r_v * C10;

                // Mass source coefficients
			    const double C41 = bGamma * C36 + cGamma;
			    const double C42 = bGamma * C37;
			    const double C43 = bGamma * C38;
			    const double C44 = bGamma * C39;
			    const double C45 = bGamma * C40 - cGamma * p_m[i] + aGamma;

                // Heat source from mixture to liquid due to heat flux coefficients
			    const double C46 = - 2 * k_x * r_v / (r_i * r_i) * (C26 + 2 * r_v * C6);
			    const double C47 = - 2 * k_x * r_v / (r_i * r_i) * (C27 + 2 * r_v * C7);
			    const double C48 = - 2 * k_x * r_v / (r_i * r_i) * (C28 + 2 * r_v * C8);
			    const double C49 = - 2 * k_x * r_v / (r_i * r_i) * (C29 + 2 * r_v * C9);
			    const double C50 = - 2 * k_x * r_v / (r_i * r_i) * (C30 + 2 * r_v * C10);

			    // Heat source from liquid to mixture due to heat flux coefficients
			    const double C51 = 2 * H_xm * r_v / (r_i * r_i) * (C31 + C26 * r_v + C6 * r_v * r_v);
			    const double C52 = 2 * H_xm * r_v / (r_i * r_i) * (C32 + C27 * r_v + C7 * r_v * r_v - 1);
			    const double C53 = 2 * H_xm * r_v / (r_i * r_i) * (C33 + C28 * r_v + C8 * r_v * r_v);
			    const double C54 = 2 * H_xm * r_v / (r_i * r_i) * (C34 + C29 * r_v + C9 * r_v * r_v);
			    const double C55 = 2 * H_xm * r_v / (r_i * r_i) * (C35 + C30 * r_v + C10 * r_v * r_v);

			    // Heat source from mixture to liquid due to phase change coefficients
			    const double C56 = - h_vx_x * C41;
			    const double C57 = - h_vx_x * C42;
			    const double C58 = - h_vx_x * C43;
			    const double C59 = - h_vx_x * C44;
			    const double C60 = - h_vx_x * C45;

			    // Heat source from liquid to mixture due to phase change coefficients
			    const double C61 = h_xv_v * C41;
			    const double C62 = h_xv_v * C42;
			    const double C63 = h_xv_v * C43;
			    const double C64 = h_xv_v * C44;
			    const double C65 = h_xv_v * C45;

			    // Heat source from wall to liquid due to heat flux coefficients
			    const double C66 = 2 * k_w / r_i * (C16 + 2 * r_i * C11);
			    const double C67 = 2 * k_w / r_i * (C17 + 2 * r_i * C12);
			    const double C68 = 2 * k_w / r_i * (C18 + 2 * r_i * C13);
			    const double C69 = 2 * k_w / r_i * (C19 + 2 * r_i * C14);
			    const double C70 = 2 * k_w / r_i * (C20 + 2 * r_i * C15);

			    // Heat source from liquid to wall due to heat flux coefficients
			    const double C71 = - 2 * k_w * r_i / (r_o * r_o - r_i * r_i) * (C16 + 2 * r_i * C11);
			    const double C72 = - 2 * k_w * r_i / (r_o * r_o - r_i * r_i) * (C17 + 2 * r_i * C12);
			    const double C73 = - 2 * k_w * r_i / (r_o * r_o - r_i * r_i) * (C18 + 2 * r_i * C13);
			    const double C74 = - 2 * k_w * r_i / (r_o * r_o - r_i * r_i) * (C19 + 2 * r_i * C14);
			    const double C75 = - 2 * k_w * r_i / (r_o * r_o - r_i * r_i) * (C20 + 2 * r_i * C15);

			    T_sur[i] = C36 * p_m_iter[i] + C37 * T_m_iter[i] + C38 * T_l_iter[i] + C39 * T_w_iter[i] + C40;

                phi_x_v[i] = beta * (sigma_e * Psat - sigma_c * Omega * p_m_iter[i]);
                Gamma_xv[i] = 2 * r_v * eps_s / (r_i * r_i) * phi_x_v[i];

                // DPcap evaluation

                const double alpha_m0 = r_v * r_v / (r_i * r_i);
                const double r_p = 1e-5;
                const double surf_ten_value = surf_ten(T_l_iter[i]);

                const double Lambda = 3 * r_v / (eps_s * r_p) * (alpha_m_iter[i] - alpha_m0);
                double DPcap = 0.0;

                if (Lambda <= 0.0) DPcap = 0;
                else if (Lambda >= 2.0) DPcap = 2 * surf_ten_value / r_p;
                else {

                    double mu = invert(Lambda);
                    if (mu < 1e-3) {

                        DPcap = 2 * surf_ten_value / r_p * (mu + (3 * r_v) / (2 * eps_s * alpha_m0 * r_p) * 0.75);

                    }
                    else {

                        DPcap = 2 * surf_ten_value / r_p *
                            (mu + (9 * r_v) / (2 * eps_s * alpha_m0 * r_p) * (std::pow(1 - mu * mu, -0.5) - Lambda / mu));
                    }
                }

                // Mass mixture equation

                add(D[i], 0, 0,
                    alpha_m_old[i] / dt
                    + (alpha_m_iter[i] * v_m_iter[i] * H(v_m_iter[i])) / dz
                    - (alpha_m_iter[i] * v_m_iter[i - 1] * (1 - H(v_m_iter[i - 1]))) / dz
                );

                add(D[i], 0, 2,
                    + rho_m_old[i] / dt
                    + (rho_m_iter[i] * v_m_iter[i] * H(v_m_iter[i])) / dz
                    - (rho_m_iter[i] * v_m_iter[i - 1] * (1 - H(v_m_iter[i - 1]))) / dz
                );

                add(D[i], 0, 4,
                    - C41
			    );

                add(D[i], 0, 6,
                    + (alpha_m_iter[i] * rho_m_iter[i] * H(v_m_iter[i])) / dz
                    + (alpha_m_iter[i + 1] * rho_m_iter[i + 1] * (1 - H(v_m_iter[i]))) / dz
                );

                add(D[i], 0, 8, 
                    - C42
                );

                add(D[i], 0, 9, 
                    - C43
                );

                add(D[i], 0, 10, 
                    - C44
                );

                Q[i][0] = 
                    + C45
                    + 2 * ( 
                        + alpha_m_iter[i] * rho_m_iter[i] * v_m_iter[i] * H(v_m_iter[i])
                        + alpha_m_iter[i + 1] * rho_m_iter[i + 1] * v_m_iter[i] * (1 - H(v_m_iter[i]))
                        - alpha_m_iter[i - 1] * rho_m_iter[i - 1] * v_m_iter[i - 1] * H(v_m_iter[i - 1])
                        - alpha_m_iter[i] * rho_m_iter[i] * v_m_iter[i - 1] * (1 - H(v_m_iter[i - 1]))
                        ) / dz
                    + 2 * (rho_m_old[i] * alpha_m_old[i]) / dt
                    //+ mass_source[i]
                    ;

                add(L[i], 0, 0,
                    - (alpha_m_iter[i - 1] * v_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz
                );

                add(L[i], 0, 2,
                    - (rho_m_iter[i - 1] * v_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz
                );

                add(L[i], 0, 6,
                    - (alpha_m_iter[i - 1] * rho_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz
                    + (alpha_m_iter[i] * rho_m_iter[i] * (1 - H(v_m_iter[i - 1]))) / dz
                );

                add(R[i], 0, 0,
                    (alpha_m_iter[i + 1] * v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz
                );

                add(R[i], 0, 2,
                    (rho_m_iter[i + 1] * v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz
                );

                // Mass liquid equation

                add(D[i], 1, 1,
                    + eps_v * (alpha_l_old[i] / dt)
                    + eps_v * (alpha_l_iter[i] * v_l_iter[i] * H(v_l_iter[i])) / dz
                    - eps_v * (alpha_l_iter[i] * v_l_iter[i - 1] * (1 - H(v_l_iter[i - 1]))) / dz
                );

                add(D[i], 1, 3,
                    + eps_v * (rho_l_old[i] / dt)
                    + eps_v * (rho_l_iter[i] * v_l_iter[i] * H(v_l_iter[i])) / dz
                    - eps_v * (rho_l_iter[i] * v_l_iter[i - 1] * (1 - H(v_l_iter[i - 1]))) / dz
                );

                add(D[i], 1, 4,
                    + C41
			    );

                add(D[i], 1, 7,
                    + eps_v * (alpha_l_iter[i] * rho_l_iter[i] * H(v_l_iter[i])) / dz
                    + eps_v * (alpha_l_iter[i + 1] * rho_l_iter[i + 1] * (1 - H(v_l_iter[i]))) / dz
                );

                add(D[i], 1, 8, 
                    + C42
                );

                add(D[i], 1, 9, 
				    + C43
                );

                add(D[i], 1, 10, 
                    + C44
                );


                Q[i][1] = 
                    - C45
                    + 2 * (
                        + eps_v * (alpha_l_iter[i] * rho_l_iter[i] * v_l_iter[i] * H(v_l_iter[i]))
                        + eps_v * (alpha_l_iter[i + 1] * rho_l_iter[i + 1] * v_l_iter[i] * (1 - H(v_l_iter[i])))
                        - eps_v * (alpha_l_iter[i - 1] * rho_l_iter[i - 1] * v_l_iter[i - 1] * H(v_l_iter[i - 1]))
                        - eps_v * (alpha_l_iter[i] * rho_l_iter[i] * v_l_iter[i - 1] * (1 - H(v_l_iter[i - 1])))
                        ) / dz
                    + 2 * (
                        + eps_v * (rho_l_old[i] * alpha_l_old[i])
                        ) / dt
                    //- mass_source[i]
                    ;

                add(L[i], 1, 1,
                    - eps_v * (alpha_l_iter[i - 1] * v_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                );

                add(L[i], 1, 3,
                    - eps_v * (rho_l_iter[i - 1] * v_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                );

                add(L[i], 1, 7,
                    - eps_v * (alpha_l_iter[i - 1] * rho_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                    - eps_v * (alpha_l_iter[i] * rho_l_iter[i] * (1 - H(v_l_iter[i - 1]))) / dz
                );

                add(R[i], 1, 1,
                    + eps_v * (alpha_l_iter[i + 1] * v_l_iter[i] * (1 - H(v_l_iter[i]))) / dz
                );

                add(R[i], 1, 3,
                    + eps_v * (rho_l_iter[i + 1] * v_l_iter[i] * (1 - H(v_l_iter[i]))) / dz
                );

                // Mixture heat equation

                const double cp_m_p = vapor_sodium::cp(T_m_iter[i]);
                const double cp_m_l = vapor_sodium::cp(T_m_iter[i - 1]);
                const double cp_m_r = vapor_sodium::cp(T_m_iter[i + 1]);

                const double k_m_p = vapor_sodium::k(T_m_iter[i], p_m_iter[i]);
                const double k_m_l = vapor_sodium::k(T_m_iter[i - 1], p_m_iter[i - 1]);
                const double k_m_r = vapor_sodium::k(T_m_iter[i + 1], p_m_iter[i + 1]);

                add(D[i], 2, 0,
                    + (alpha_m_old[i] * cp_m_p * T_m_old[i]) / dt
                    + (alpha_m_iter[i] * cp_m_p * T_m_iter[i] * v_m_iter[i] * H(v_m_iter[i])) / dz
                    - (alpha_m_iter[i] * cp_m_p * T_m_iter[i] * v_m_iter[i - 1] * (1 - H(v_m_iter[i - 1]))) / dz
                );

                add(D[i], 2, 2,
                    + (T_m_old[i] * rho_m_old[i] * cp_m_p) / dt
                    + (rho_m_iter[i] * cp_m_p * T_m_iter[i] * v_m_iter[i] * H(v_m_iter[i])) / dz
                    - (rho_m_iter[i] * cp_m_p * T_m_iter[i] * v_m_iter[i - 1] * (1 - H(v_m_iter[i - 1]))) / dz
                    + p_m_iter[i] * (v_m_iter[i] * H(v_m_iter[i]) - v_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz
                    + p_m_old[i] / dt
                );

                add(D[i], 2, 4,
                    - C51 - C61
			    );

                add(D[i], 2, 6,
                    + (alpha_m_iter[i] * rho_m_iter[i] * cp_m_p * T_m_iter[i] * H(v_m_iter[i]) + alpha_m_iter[i + 1] * rho_m_iter[i + 1] * cp_m_r * T_m_iter[i + 1] * (1 - H(v_m_iter[i]))) / dz
                    + p_m_iter[i] * (alpha_m_iter[i] * H(v_m_iter[i]) + alpha_m_iter[i + 1] * (1 - H(v_m_iter[i]))) / dz
                );

                add(D[i], 2, 8,
                    + (alpha_m_old[i] * rho_m_old[i] * cp_m_p) / dt
                    + (alpha_m_iter[i] * rho_m_iter[i] * cp_m_p * v_m_iter[i] * H(v_m_iter[i])) / dz
                    - (alpha_m_iter[i] * rho_m_iter[i] * cp_m_p * v_m_iter[i - 1] * (1 - H(v_m_iter[i - 1]))) / dz
                    + (alpha_m_iter[i] * k_m_p * H(v_m_iter[i]) + alpha_m_iter[i + 1] * k_m_r * (1 - H(v_m_iter[i]))) / (dz * dz)
                    + (alpha_m_iter[i - 1] * k_m_l * H(v_m_iter[i - 1]) + alpha_m_iter[i] * k_m_p * (1 - H(v_m_iter[i - 1]))) / (dz * dz)
                    - C52 - C62
                );

                add(D[i], 2, 9, 
				    - C53 - C63
                );

                add(D[i], 2, 10, 
                    - C54 - C64
                );

                Q[i][2] = 
                    + 3 * (alpha_m_old[i] * cp_m_p * T_m_old[i] * rho_m_old[i]) / dt
                    + 3 * ( 
                        + alpha_m_iter[i] * rho_m_iter[i] * cp_m_p * T_m_iter[i] * v_m_iter[i] * H(v_m_iter[i])
                        + alpha_m_iter[i + 1] * rho_m_iter[i + 1] * cp_m_r * T_m_iter[i + 1] * v_m_iter[i ] * (1 - H(v_m_iter[i]))
                        - alpha_m_iter[i - 1] * rho_m_iter[i - 1] * cp_m_l * T_m_iter[i - 1] * v_m_iter[i - 1] * H(v_m_iter[i - 1])
                        - alpha_m_iter[i] * rho_m_iter[i] * cp_m_p * T_m_iter[i] * v_m_iter[i - 1] * (1 - H(v_m_iter[i - 1]))
                        ) / dz
                    + p_m[i] * (
                        + alpha_m_iter[i] * v_m_iter[i] * H(v_m_iter[i])
                        + alpha_m_iter[i + 1] * v_m_iter[i] * (1 - H(v_m_iter[i]))
                        - alpha_m_iter[i - 1] * v_m_iter[i - 1] * H(v_m_iter[i - 1])
                        - alpha_m_iter[i] * v_m_iter[i - 1] * (1 - H(v_m_iter[i - 1]))
                        ) / dz
                    + p_m_old[i] * alpha_m_old[i] / dt
                    + C55 + C65
                    // + heat_source_liquid_vapor_flux[i]
                    // + heat_source_liquid_vapor_phase[i]
                    ;

                add(L[i], 2, 0,
                    - (alpha_m_iter[i - 1] * cp_m_l * T_m_iter[i - 1] * v_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz
                );

                add(L[i], 2, 2,
                    - (rho_m_iter[i - 1] * cp_m_l * T_m_iter[i - 1] * v_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz
                    - p_m_iter[i] * (v_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz
                );

                add(L[i], 2, 6,
                    - (
                        + alpha_m_iter[i - 1] * rho_m_iter[i - 1] * cp_m_l * T_m_iter[i - 1] * H(v_m_iter[i - 1])
                        + alpha_m_iter[i] * rho_m_iter[i] * cp_m_l * T_m_iter[i] * (1 - H(v_m_iter[i - 1]))
                        ) / dz
                    + p_m[i] * (
                        + alpha_m_iter[i - 1] * H(v_m_iter[i - 1])
                        + alpha_m_iter[i] * (1 - H(v_m_iter[i - 1]))
                        ) / dz
                );

                add(L[i], 2, 8,
                    - (alpha_m_iter[i - 1] * rho_m_iter[i - 1] * cp_m_l * v_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz
                    - (alpha_m_iter[i - 1] * k_m_l * H(v_m_iter[i - 1]) + alpha_m_iter[i] * k_m_p * (1 - H(v_m_iter[i - 1]))) / (dz * dz)
                );

                add(R[i], 2, 0,
                    + (alpha_m_iter[i + 1] * cp_m_r * T_m_iter[i + 1] * v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz
                );

                add(R[i], 2, 2,
                    + (rho_m_iter[i + 1] * cp_m_r * T_m_iter[i + 1] * v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz
                    + p_m_iter[i] * (v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz
                );

                add(R[i], 2, 8,
                    + (alpha_m_iter[i + 1] * rho_m_iter[i + 1] * cp_m_r * v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz
                    - (alpha_m_iter[i] * k_m_p * H(v_m_iter[i]) + alpha_m_iter[i + 1] * k_m_r * (1 - H(v_m_iter[i]))) / (dz * dz)
                );

                // Heat liquid equation

                const double cp_l_p = liquid_sodium::cp(T_l_iter[i]);
                const double cp_l_l = liquid_sodium::cp(T_l_iter[i - 1]);
                const double cp_l_r = liquid_sodium::cp(T_l_iter[i + 1]);

                const double k_l_p = liquid_sodium::k(T_l_iter[i]);
                const double k_l_l = liquid_sodium::k(T_l_iter[i - 1]);
                const double k_l_r = liquid_sodium::k(T_l_iter[i + 1]);

                add(D[i], 3, 1,
                    + eps_v * (alpha_l_old[i] * T_l_old[i] * cp_l_p) / dt
                    + eps_v * (alpha_l_iter[i] * cp_l_p * T_l_iter[i] * v_l_iter[i] * H(v_l_iter[i])) / dz
                    - eps_v * (alpha_l_iter[i] * cp_l_p * T_l_iter[i] * v_l_iter[i - 1] * (1 - H(v_l_iter[i - 1]))) / dz
                );

                add(D[i], 3, 3,
                    + eps_v * (T_l_old[i] * rho_l_old[i] * cp_l_p) / dt
                    + eps_v * (rho_l_iter[i] * cp_l_p * T_l_iter[i] * v_l_iter[i] * H(v_l_iter[i])) / dz
                    - eps_v * (rho_l_iter[i] * cp_l_p * T_l_iter[i] * v_l_iter[i - 1] * (1 - H(v_l_iter[i - 1]))) / dz
                    + eps_v * p_l_iter[i] * (v_l_iter[i] * H(v_l_iter[i]) - v_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                    + eps_v * p_l_old[i] / dt
                );

                add(D[i], 3, 4,
                    - C46 - C56 - C66
                );

                add(D[i], 3, 7, 
                    + eps_v * (alpha_l_iter[i] * rho_l_iter[i] * cp_l_p * T_l_iter[i] * H(v_l_iter[i]) + alpha_l_iter[i + 1] * rho_l_iter[i + 1] * cp_l_r * T_l_iter[i + 1] * (1 - H(v_l_iter[i]))) / dz
                    + eps_v * (p_l_iter[i] * (alpha_l_iter[i] * H(v_l_iter[i]) + alpha_l_iter[i + 1] * (1 - H(v_l_iter[i]))) / dz)
                );

                add(D[i], 3, 8, 
                    - C47 - C57 - C67
                );

                add(D[i], 3, 9,
                    + eps_v * (alpha_l_old[i] * rho_l_old[i] * cp_l_p) / dt
                    + eps_v * (alpha_l_iter[i] * rho_l_iter[i] * cp_l_p * v_l_iter[i] * H(v_l_iter[i])) / dz
                    - eps_v * (alpha_l_iter[i] * rho_l_iter[i] * cp_l_p * v_l_iter[i - 1] * (1 - H(v_l_iter[i - 1]))) / dz
                    + eps_v * (alpha_l_iter[i] * k_l_p * H(v_l_iter[i]) + alpha_l_iter[i + 1] * k_l_r * (1 - H(v_l_iter[i]))) / (dz * dz)
                    + eps_v * (alpha_l_iter[i - 1] * k_l_l * H(v_l_iter[i - 1]) + alpha_l_iter[i] * k_l_p * (1 - H(v_l_iter[i - 1]))) / (dz * dz)
                    - C48 - C58 - C68

                );

                add(D[i], 3, 10, 
                    - C49 - C59 - C69
                );

                Q[i][3] = 
                    + eps_v * 3 * ( 
                        + alpha_l_old[i] * T_l_old[i] * cp_l_p * rho_l_old[i]) / dt
                    + eps_v * 3 * (
                        + alpha_l_iter[i] * rho_l_iter[i] * cp_l_p * T_l_iter[i] * v_l_iter[i] * H(v_l_iter[i])
                        + alpha_l_iter[i + 1] * rho_l_iter[i + 1] * cp_l_r * T_l_iter[i + 1] * v_l_iter[i] * (1 - H(v_l_iter[i]))
                        - alpha_l_iter[i - 1] * rho_l_iter[i - 1] * cp_l_l * T_l_iter[i - 1] * v_l_iter[i - 1] * H(v_l_iter[i - 1])
                        - alpha_l_iter[i] * rho_l_iter[i] * cp_l_p * T_l_iter[i] * v_l_iter[i] - 1 * (1 - H(v_l_iter[i - 1]))
                            ) / dz
                    + eps_v * p_l[i] * (
                            + alpha_l_iter[i] * v_l_iter[i] * H(v_l_iter[i])
                            + alpha_l_iter[i + 1] * v_l_iter[i] * (1 - H(v_l_iter[i]))
                            - alpha_l_iter[i - 1] * v_l_iter[i - 1] * H(v_l_iter[i - 1])
                            - alpha_l_iter[i] * v_l_iter[i - 1] * (1 - H(v_l_iter[i - 1]))
                            ) / dz
                    + eps_v * p_l_old[i] * alpha_l_old[i] / dt
                    + C50 + C60 + C70
                    // + heat_source_wall_liquid_flux[i]
                    // + heat_source_vapor_liquid_flux[i]
                    // + heat_source_vapor_liquid_phase[i]
                    ;

                add(L[i], 3, 1,
                    - eps_v * (alpha_l_iter[i - 1] * cp_l_l * T_l_iter[i - 1] * v_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                );

                add(L[i], 3, 3,
                    + eps_v * (rho_l_iter[i - 1] * cp_l_l * T_l_iter[i - 1] * v_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                    + eps_v * (p_l_iter[i] * v_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                );

                add(L[i], 3, 7,
                    - eps_v * (alpha_l_iter[i - 1] * rho_l_iter[i - 1] * cp_l_l * T_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                    - eps_v * (alpha_l_iter[i] * rho_l_iter[i] * cp_l_l * T_l_iter[i] * (1 - H(v_l_iter[i - 1]))) / dz
                    + eps_v * p_l_iter[i] * (alpha_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                    + eps_v * p_l_iter[i] * (alpha_l_iter[i] * (1 - H(v_l_iter[i - 1]))) / dz
                );

                add(L[i], 3, 9,
                    - eps_v * (alpha_l_iter[i - 1] * rho_l_iter[i - 1] * cp_l_l * v_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                    - eps_v * (alpha_l_iter[i - 1] * k_l_l * H(v_l_iter[i - 1]) + alpha_l_iter[i] * k_l_p * (1 - H(v_l_iter[i - 1]))) / (dz * dz)
                );

                add(R[i], 3, 1,
                    + eps_v * (alpha_l_iter[i + 1] * cp_l_r * T_l_iter[i + 1] * v_l_iter[i] * (1 - H(v_l_iter[i]))) / dz
                );

                add(R[i], 3, 3,
                    + eps_v * (rho_l_iter[i + 1] * cp_l_r * T_l_iter[i + 1] * v_l_iter[i] * (1 - H(v_l_iter[i]))) / dz
                    + eps_v * p_l_iter[i] * v_l_iter[i] * (1 - H(v_l_iter[i])) / dz
                );

                add(R[i], 3, 9,
                    + eps_v * (alpha_l_iter[i + 1] * rho_l_iter[i + 1] * cp_l_r * v_l_iter[i] * (1 - H(v_l_iter[i]))) / dz
                    - eps_v * (alpha_l_iter[i] * k_l_p * H(v_l_iter[i]) + alpha_l_iter[i + 1] * k_l_r * (1 - H(v_l_iter[i]))) / (dz * dz)
                );

                // Heat wall equation

                const double rho_w_p = steel::rho(T_w_iter[i]);
                const double rho_w_l = steel::rho(T_w_iter[i - 1]);
                const double rho_w_r = steel::rho(T_w_iter[i + 1]);

                const double cp_w_p = steel::cp(T_w_iter[i]);
                const double cp_w_l = steel::cp(T_w_iter[i - 1]);
                const double cp_w_r = steel::cp(T_w_iter[i + 1]);

                const double k_w_p = steel::k(T_w_iter[i]);
                const double k_w_l = steel::k(T_w_iter[i - 1]);
                const double k_w_r = steel::k(T_w_iter[i + 1]);

                const double k_w_lf = 0.5 * (k_w_l + k_w_p);
                const double k_w_rf = 0.5 * (k_w_r + k_w_p);

                add(D[i], 4, 4, 
                    - C71
                );

                add(D[i], 4, 8, 
                    - C72
                );

                add(D[i], 4, 9, 
                    - C73
                );

                add(D[i], 4, 10,
                    + (rho_w_p * cp_w_p) / dt
                    + (k_w_lf + k_w_rf) / (dz * dz)
                    - C74
                );

                Q[i][4] =
                    q_pp[i] * 2 * r_o / (r_o * r_o - r_i * r_i)
                    + (rho_w_p * cp_w_p * T_w_old[i]) / dt
                    + C75
                    // + heat_source_liquid_wall_flux[i]
                    ;

                add(L[i], 4, 10,
                    - k_w_lf / (dz * dz)
                );

                add(R[i], 4, 10,
                    - k_w_rf / (dz * dz)
                );

                // Momentum mixture equation

                const double Re = rho_m_iter[i] * v_m_iter[i] * Dh_v / mu_v;
                const double fm = Re > 1187.4 ? 0.3164 * std::pow(Re, -0.25) : 64 * std::pow(Re, -1);
                const double Fm = fm * std::abs(v_m_iter[i]) / (4 * r_v);

                add(D[i], 5, 0,
                    + (alpha_m_old[i] * v_m_old[i]) / dt
                    + (alpha_m_iter[i] * v_m_iter[i] * v_m_iter[i] * H(v_m_iter[i])) / dz
                    - (alpha_m_iter[i] * v_m_iter[i - 1] * v_m_iter[i - 1] * (1 - H(v_m_iter[i - 1]))) / dz
                );

                add(D[i], 5, 2,
                    + (v_m_old[i] * rho_m_old[i]) / dt
                    + (rho_m_iter[i] * v_m_iter[i] * v_m_iter[i] * H(v_m_iter[i])) / dz
                    - (rho_m_iter[i] * v_m_iter[i - 1] * v_m_iter[i - 1] * (1 - H(v_m_iter[i - 1]))) / dz
                );

                add(D[i], 5, 4,
                    - (alpha_m_iter[i] * H(v_m_iter[i])) / dz
                    - (alpha_m_iter[i + 1] * (1 - H(v_m_iter[i]))) / dz
                );

                add(D[i], 5, 6,
                    + (alpha_m_old[i] * rho_m_old[i]) / dt
                    + 2 * (rho_m_iter[i] * alpha_m_iter[i] * v_m_iter[i] * H(v_m_iter[i])) / dz
                    + 2 * (rho_m_iter[i + 1] * alpha_m_iter[i + 1] * v_m_iter[i + 1] * (1 - H(v_m_iter[i]))) / dz
                    + (Fm * rho_m_iter[i] * std::abs(v_m_iter[i]) * H(v_m_iter[i])) / (4 * r_v)
                    + (Fm * rho_m_iter[i + 1] * std::abs(v_m_iter[i]) * (1 - H(v_m_iter[i]))) / (4 * r_v)
                );

                Q[i][5] = 
                    + 3 * (alpha_m_old[i] * rho_m_old[i] * v_m_old[i]) / dt
                    - 3 * (rho_m_iter[i] * alpha_m_iter[i] * v_m[i] * v_m_iter[i] * H(v_m_iter[i])) / dz
                    - 3 * (rho_m_iter[i + 1] * alpha_m_iter[i + 1] * v_m_iter[i] * v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz
                    + 3 * (rho_m_iter[i - 1] * alpha_m_iter[i - 1] * v_m_iter[i - 1] * v_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz
                    + 3 * (rho_m_iter[i] * alpha_m_iter[i] * v_m[i - 1] * v_m_iter[i - 1] * (1 - H(v_m_iter[i - 1]))) / dz;

                add(L[i], 5, 0,
                    - (alpha_m_iter[i - 1] * v_m_iter[i - 1] * v_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz
                );

                add(L[i], 5, 2,
                    - (rho_m_iter[i - 1] * v_m_iter[i - 1] * v_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz
                );

                add(L[i], 5, 6,
                    - 2 * (rho_m_iter[i - 1] * alpha_m_iter[i - 1] * v_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz
                    - 2 * (rho_m_iter[i] * alpha_m_iter[i] * v_m[i] * (1 - H(v_m_iter[i - 1]))) / dz
                );

                add(R[i], 5, 0,
                    + (alpha_m_iter[i + 1] * v_m_iter[i] * v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz
                );

                add(R[i], 5, 2,
                    + (rho_m_iter[i + 1] * v_m_iter[i] * v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz
                );

                add(R[i], 5, 4,
                    + (alpha_m_iter[i] * H(v_m_iter[i])) / dz
                    + (alpha_m_iter[i + 1] * (1 - H(v_m_iter[i]))) / dz
                );

                // Momentum liquid equation

                const double Fl = 8 * mu_l / (eps_v * (r_i - r_v) * (r_i - r_v));

                add(D[i], 6, 1,
                    + eps_v * (alpha_l_old[i] * v_l_old[i] / dt)
                    + eps_v * (alpha_l_iter[i] * v_l_iter[i] * v_l_iter[i] * H(v_l_iter[i])) / dz
                    - eps_v * (alpha_l_iter[i] * v_l_iter[i - 1] * v_l_iter[i - 1] * (1 - H(v_l_iter[i - 1]))) / dz
                );

                add(D[i], 6, 3,
                    + eps_v * (v_l_old[i] * rho_l_old[i] / dt)
                    + eps_v * (rho_l_iter[i] * v_l_iter[i] * v_l_iter[i] * H(v_l_iter[i])) / dz
                    - eps_v * (rho_l_iter[i] * v_l_iter[i - 1] * v_l_iter[i - 1] * (1 - H(v_l_iter[i - 1]))) / dz
                    - DPcap / dz);

                add(D[i], 6, 5,
                    - eps_v * (alpha_l_iter[i] * H(v_l_iter[i])) / dz
                    - eps_v * (alpha_l_iter[i + 1] * (1 - H(v_l_iter[i]))) / dz
                );

                add(D[i], 6, 7,
                    + eps_v * (alpha_l_old[i] * rho_l_old[i]) / dt
                    + 2 * eps_v * (rho_l_iter[i] * alpha_l_iter[i] * v_l_iter[i] * H(v_l_iter[i])) / dz
                    + 2 * eps_v * (rho_l_iter[i + 1] * alpha_l_iter[i + 1] * v_l_iter[i + 1] * (1 - H(v_l_iter[i]))) / dz
                    + (Fl * std::abs(v_l_iter[i]) * H(v_l_iter[i]))
                    + (Fl * std::abs(v_l_iter[i]) * (1 - H(v_l_iter[i])))
                );

                Q[i][6] =
                    + 3 * eps_v * (alpha_l_old[i] * rho_l_old[i] * v_l_old[i]) / dt
                    - 3 * eps_v * (rho_l_iter[i] * alpha_l_iter[i] * v_l_iter[i] * v_l_iter[i] * H(v_l_iter[i])) / dz
                    - 3 * eps_v * (rho_l_iter[i + 1] * alpha_l_iter[i + 1] * v_l_iter[i] * v_l_iter[i] * (1 - H(v_l_iter[i]))) / dz
                    + 3 * eps_v * (rho_l_iter[i - 1] * alpha_l_iter[i - 1] * v_l_iter[i - 1] * v_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                    + 3 * eps_v * (rho_l_iter[i] * alpha_l_iter[i] * v_l_iter[i - 1] * v_l_iter[i - 1] * (1 - H(v_l_iter[i - 1]))) / dz;

                add(L[i], 6, 1,
                    - eps_v * (alpha_l_iter[i - 1] * v_l_iter[i - 1] * v_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                );

                add(L[i], 6, 3,
                    - eps_v * (rho_l_iter[i - 1] * v_l_iter[i - 1] * v_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                );

                add(L[i], 6, 7,
                    - 2 * eps_v * (rho_l_iter[i - 1] * alpha_l_iter[i - 1] * v_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                    - 2 * eps_v * (rho_l_iter[i] * alpha_l_iter[i] * v_l_iter[i] * (1 - H(v_l_iter[i - 1]))) / dz
                );

                add(R[i], 6, 1,
                    + eps_v * (alpha_l_iter[i + 1] * v_l_iter[i] * v_l_iter[i] * (1 - H(v_l_iter[i]))) / dz
                );

                add(R[i], 6, 3,
                    + eps_v * (rho_l_iter[i + 1] * v_l_iter[i] * v_l_iter[i] * (1 - H(v_l_iter[i]))) / dz
                    + DPcap / dz
                );

                add(R[i], 6, 5,
                    + eps_v * (alpha_l_iter[i] * H(v_l_iter[i])) / dz
                    + eps_v * (alpha_l_iter[i + 1] * (1 - H(v_l_iter[i]))) / dz
                );

                // State mixture equation

                add(D[i], 7, 0, 
                    - T_m_old[i] * Rv
                );

                add(D[i], 7, 4,
                    1.0
                );

                add(D[i], 7, 8, 
                    - rho_m_old[i] * Rv
                );

                Q[i][7] = - rho_m_old[i] * T_m_old[i] * Rv;

                // State liquid equation

                add(D[i], 8, 1, 
                    1.0
                );

                add(D[i], 8, 9, 
                    -1.0 / Tc * (275.32 + 511.58 / (2 * std::sqrt(1 - T_l_iter[i] / Tc)))
                );

                Q[i][8] = 219.0 + 275.32 * (1.0 - T_l_iter[i] / Tc) + 511.58 * std::sqrt(1.0 - T_l_iter[i] / Tc) + T_l_iter[i] / Tc * (275.32 + 511.58 / (2 * std::sqrt(1.0 - T_l_iter[i] / Tc)));

                // Volume fraction sum

                add(D[i], 9, 2, 
                    1.0
                );

                add(D[i], 9, 3, 
                    1.0
                );

                Q[i][9] = 1.0;

                // Capillary equation

                add(D[i], 10, 4, 1);
                add(D[i], 10, 5, -1);

                Q[i][10] = DPcap;

                DenseBlock D_dense = to_dense(D[i]);

                Gamma_xv[i] = C41 * p_m_iter[i] + C42 * T_m_iter[i] + C43 * T_l_iter[i] + C44 * T_w_iter[i] + C45;

                heat_source_wall_liquid_flux[i] = C66 * p_m_iter[i] + C67 * T_m_iter[i] + C68 * T_l_iter[i] + C69 * T_w_iter[i] + C70;
                heat_source_liquid_wall_flux[i] = C71 * p_m_iter[i] + C72 * T_m_iter[i] + C73 * T_l_iter[i] + C74 * T_w_iter[i] + C75;

                heat_source_vapor_liquid_phase[i] = C56 * p_m_iter[i] + C57 * T_m_iter[i] + C58 * T_l_iter[i] + C59 * T_w_iter[i] + C60;
                heat_source_liquid_vapor_phase[i] = C61 * p_m_iter[i] + C62 * T_m_iter[i] + C63 * T_l_iter[i] + C64 * T_w_iter[i] + C65;

                heat_source_vapor_liquid_flux[i] = C46 * p_m_iter[i] + C47 * T_m_iter[i] + C48 * T_l_iter[i] + C49 * T_w_iter[i] + C50;
                heat_source_liquid_vapor_flux[i] = C51 * p_m_iter[i] + C52 * T_m_iter[i] + C53 * T_l_iter[i] + C54 * T_w_iter[i] + C55;

                p_saturation[i] = Psat;
				T_sur[i] = C36 * p_m_iter[i] + C37 * T_m_iter[i] + C38 * T_l_iter[i] + C39 * T_w_iter[i] + C40;
            }

            // First node boundary conditions

            add(D[0], 0, 0, 1.0);
            add(D[0], 1, 1, 1.0);
            add(D[0], 2, 2, 1.0);
            add(D[0], 3, 3, 1.0);
            add(D[0], 4, 4, 1.0);
            add(D[0], 5, 5, 1.0);
            add(D[0], 6, 6, 1.0);
            add(D[0], 7, 7, 1.0);
            add(D[0], 8, 8, 1.0);
            add(D[0], 9, 9, 1.0);
            add(D[0], 10, 10, 1.0);

            add(R[0], 0, 0, -1.0);
            add(R[0], 1, 1, -1.0);
            add(R[0], 2, 2, -1.0);
            add(R[0], 3, 3, -1.0);
            add(R[0], 4, 4, -1.0);
            add(R[0], 5, 5, -1.0);
            add(R[0], 6, 6, 0.0);
            add(R[0], 7, 7, 0.0);
            add(R[0], 8, 8, -1.0);
            add(R[0], 9, 9, -1.0);
            add(R[0], 10, 10, -1.0);

            Q[0][0] = 0.0;
            Q[0][1] = 0.0;
            Q[0][2] = 0.0;
            Q[0][3] = 0.0;
            Q[0][4] = 0.0;
            Q[0][5] = 0.0;
            Q[0][6] = 0.0;
            Q[0][7] = 0.0;
            Q[0][8] = 0.0;
            Q[0][9] = 0.0;
            Q[0][10] = 0.0;

            // Last node boundary conditions

            add(D[N - 1], 0, 0, 1.0);
            add(D[N - 1], 1, 1, 1.0);
            add(D[N - 1], 2, 2, 1.0);
            add(D[N - 1], 3, 3, 1.0);
            add(D[N - 1], 4, 4, 1.0);
            add(D[N - 1], 5, 5, 1.0);
            add(D[N - 1], 6, 6, 1.0);
            add(D[N - 1], 7, 7, 1.0);
            add(D[N - 1], 8, 8, 1.0);
            add(D[N - 1], 9, 9, 1.0);
            add(D[N - 1], 10, 10, 1.0);

            add(L[N - 1], 0, 0, -1.0);
            add(L[N - 1], 1, 1, -1.0);
            add(L[N - 1], 2, 2, -1.0);
            add(L[N - 1], 3, 3, -1.0);
            add(L[N - 1], 4, 4, -1.0);
            add(L[N - 1], 5, 5, -1.0);
            add(L[N - 1], 6, 6, 0.0);
            add(L[N - 1], 7, 7, 0.0);
            add(L[N - 1], 8, 8, -1.0);
            add(L[N - 1], 9, 9, -1.0);
            add(L[N - 1], 10, 10, -1.0);

            Q[N - 1][0] = 0.0;
            Q[N - 1][1] = 0.0;
            Q[N - 1][2] = 0.0;
            Q[N - 1][3] = 0.0;
            Q[N - 1][4] = 0.0;
            Q[N - 1][5] = 0.0;
            Q[N - 1][6] = 0.0;
            Q[N - 1][7] = 0.0;
            Q[N - 1][8] = 0.0;
            Q[N - 1][9] = 0.0;
            Q[N - 1][10] = 0.0;

            solve_block_tridiag(L, D, R, Q, X);

            // Calculate Picard error
            double L1 = 0.0;

			double Aold, Anew, denom, eps;

            for (int i = 0; i < N; ++i) {

                Aold = rho_m_iter[i];
                Anew = X[i][0];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L1 += eps;

                Aold = rho_l_iter[i];
                Anew = X[i][1];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L1 += eps;

                Aold = alpha_m_iter[i];
                Anew = X[i][2];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L1 += eps;

                Aold = alpha_l_iter[i];
                Anew = X[i][3];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L1 += eps;

                Aold = p_m_iter[i];
                Anew = X[i][4];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L1 += eps;

                Aold = p_l_iter[i];
                Anew = X[i][5];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L1 += eps;

                Aold = v_m_iter[i];
                Anew = X[i][6];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L1 += eps;

                Aold = v_l_iter[i];
                Anew = X[i][7];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L1 += eps;

                Aold = T_m_iter[i];
                Anew = X[i][8];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L1 += eps;

                Aold = T_l_iter[i];
                Anew = X[i][9];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L1 += eps;

                Aold = T_w_iter[i];
                Anew = X[i][10];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L1 += eps;
            }

            if (L1 < tolerance)
                break;   // Picard converged

            // Update Picard values
            for (int i = 0; i < N; ++i) {
                rho_m_iter[i] = X[i][0];
                rho_l_iter[i] = X[i][1];
                alpha_m_iter[i] = X[i][2];
                alpha_l_iter[i] = X[i][3];
                p_m_iter[i] = X[i][4];
                p_l_iter[i] = X[i][5];
                v_m_iter[i] = X[i][6];
                v_l_iter[i] = X[i][7];
                T_m_iter[i] = X[i][8];
                T_l_iter[i] = X[i][9];
                T_w_iter[i] = X[i][10];
            }
        }

		// Update solution
        for (int i = 0; i < N; ++i) {
            rho_m_old[i] = X[i][0];
            rho_l_old[i] = X[i][1];
            alpha_m_old[i] = X[i][2];
            alpha_l_old[i] = X[i][3];
            p_m_old[i] = X[i][4];
            p_l_old[i] = X[i][5];
            v_m_old[i] = X[i][6];
            v_l_old[i] = X[i][7];
            T_m_old[i] = X[i][8];
            T_l_old[i] = X[i][9];
            T_w_old[i] = X[i][10];
        }

        const int output_every = 10;

        if (n % output_every == 0) {
            for (int i = 0; i < N; ++i) {

                v_velocity_output << X[i][6] << " ";
                v_pressure_output << X[i][4] << " ";
                v_temperature_output << X[i][8] << " ";
                v_rho_output << X[i][0] << " ";

                l_velocity_output << X[i][7] << " ";
                l_pressure_output << X[i][5] << " ";
                l_temperature_output << X[i][9] << " ";
                l_rho_output << X[i][1] << " ";

                w_temperature_output << X[i][10] << " ";

                v_alpha_output << X[i][2] << " ";
                l_alpha_output << X[i][3] << " ";

                gamma_output << Gamma_xv[i] << " ";
                phi_output << phi_x_v[i] << " ";

                hs_wl_flux_output << heat_source_wall_liquid_flux[i] << " ";
                hs_lw_flux_output << heat_source_liquid_wall_flux[i] << " ";

                hs_vl_phase_output << heat_source_vapor_liquid_phase[i] << " ";
                hs_lv_phase_output << heat_source_liquid_vapor_phase[i] << " ";

                hs_vl_flux_output << heat_source_vapor_liquid_flux[i] << " ";
                hs_lv_flux_output << heat_source_liquid_vapor_flux[i] << " ";

                psat_output << p_saturation[i] << " ";
                tsur_output << T_sur[i] << " ";
            }

            time_output << n * dt << " ";

            v_velocity_output << "\n";
            v_pressure_output << "\n";
            v_temperature_output << "\n";
            v_rho_output << "\n";

            l_velocity_output << "\n";
            l_pressure_output << "\n";
            l_temperature_output << "\n";
            l_rho_output << "\n";

            w_temperature_output << "\n";

            v_alpha_output << "\n";
            l_alpha_output << "\n";

            gamma_output << "\n";
            phi_output << "\n";

            hs_wl_flux_output << "\n";
            hs_lw_flux_output << "\n";

            hs_vl_phase_output << "\n";
            hs_lv_phase_output << "\n";

            hs_vl_flux_output << "\n";
            hs_lv_flux_output << "\n";

            psat_output << "\n";
            tsur_output << "\n";


            v_velocity_output.flush();
            v_pressure_output.flush();
            v_temperature_output.flush();
            v_rho_output.flush();

            l_velocity_output.flush();
            l_pressure_output.flush();
            l_temperature_output.flush();
            l_rho_output.flush();

            w_temperature_output.flush();

            v_alpha_output.flush();
            l_alpha_output.flush();

            gamma_output.flush();
            phi_output.flush();
            hs_wl_flux_output.flush();
            hs_lw_flux_output.flush();
            hs_vl_phase_output.flush();
            hs_lv_phase_output.flush();
            hs_vl_flux_output.flush();
            hs_lv_flux_output.flush();
            psat_output.flush();
            tsur_output.flush();

            time_output.flush();
        }
    }

    v_velocity_output.close();
    v_pressure_output.close();
    v_temperature_output.close();
    v_rho_output.close();

    l_velocity_output.close();
    l_pressure_output.close();
    l_temperature_output.close();
    l_rho_output.close();

    w_temperature_output.close();

    v_alpha_output.close();
    l_alpha_output.close();

    gamma_output.close();
    phi_output.close();
    hs_wl_flux_output.close();
    hs_lw_flux_output.close();
    hs_vl_phase_output.close();
    hs_lv_phase_output.close();
    hs_vl_flux_output.close();
    hs_lv_flux_output.close();
    psat_output.close();
    tsur_output.close();

    time_output.close();

    double end = omp_get_wtime();
    printf("Execution time: %.6f s\n", end - start);
}
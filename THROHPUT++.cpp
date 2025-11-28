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

// =======================================================================
//
//                       [MATERIAL PROPERTIES]
//
// =======================================================================

#pragma region steel_properties

/**
 * @brief Provides material properties for steel.
 *
 * This namespace contains lookup tables and helper functions to retrieve
 * temperature-dependent thermodynamic properties of steel, specifically:
 * - Specific Heat Capacity (Cp)
 * - Density (rho)
 * - Thermal Conductivity (k)
 *
 * All functions accept temperature in Kelvin [K] and return values in
 * standard SI units unless otherwise specified.
 */
namespace steel {

    /// Temperature values of the Cp table [K]
    constexpr std::array<double, 15> T = { 300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700 };

    /// Specific heat values of the Cp table [J kg^-1 K^-1]
    constexpr std::array<double, 15> Cp_J_kgK = { 510.0296,523.4184,536.8072,550.1960,564.0032,577.3920,590.7808,604.1696,617.5584,631.3656,644.7544,658.1432,671.5320,685.3392,698.7280 };

    /**
    * @brief Specific heat interpolation in temperature with complexity O(1)
    * @param Tquery Temperature for which the Cp is needed.
    */
    inline double cp(double Tquery) {

        if (Tquery <= T.front()) return Cp_J_kgK.front();
        if (Tquery >= T.back())  return Cp_J_kgK.back();

        int i = static_cast<int>((Tquery - 300.0) / 100.0);

        if (i < 0) i = 0;

        int iMax = static_cast<int>(T.size()) - 2;

        if (i > iMax) i = iMax;

        double x0 = 300.0 + 100.0 * i, x1 = x0 + 100.0;
        double y0 = Cp_J_kgK[static_cast<std::size_t>(i)];
        double y1 = Cp_J_kgK[static_cast<std::size_t>(i + 1)];
        double t = (Tquery - x0) / (x1 - x0);

        return y0 + t * (y1 - y0);
    }

    /**
    * @brief Density [kg/m3] as a function of temperature T
    */
    inline double rho(double T) { return (7.9841 - 2.6560e-4 * T - 1.158e-7 * T * T) * 1e3; }

    /**
    * @brief Thermal conductivity [W/(m*K)] as a function of temperature T
    */
    inline double k(double T) { return (3.116e-2 + 1.618e-4 * T) * 100.0; }
}

#pragma endregion

#pragma region liquid_sodium_properties

/**
 * @brief Provides thermophysical properties for Liquid Sodium (Na).
 *
 * This namespace contains constant data and functions to calculate key
 * temperature-dependent properties of liquid sodium.
 * All functions accept temperature T in Kelvin [K] and return values
 * in standard SI units unless otherwise specified.
 * The function give warnings if the input temperature is below the
 * (constant) solidification temperature.
 */
namespace liquid_sodium {

    /// Critical temperature [K]
    constexpr double Tcrit = 2509.46;

    /// Solidification temperature [K]
    constexpr double Tsolid = 370.87;

    /**
    * @brief Density [kg/m3] as a function of temperature T
    */
    inline double rho(double T) {

        if (T < Tsolid && warnings == true) std::cout << "Warning, temperature " << T << " is below solidification temperature (" << Tsolid << ")!";
        return 219.0 + 275.32 * (1.0 - T / Tcrit) + 511.58 * pow(1.0 - T / Tcrit, 0.5);
    }

    /**
    * @brief Thermal conductivity [W/(m*K)] as a function of temperature T
    */
    inline double k(double T) {

        if (T < Tsolid && warnings == true) std::cout << "Warning, temperature " << T << " is below solidification temperature!";
        return 124.67 - 0.11381 * T + 5.5226e-5 * T * T - 1.1842e-8 * T * T * T;
    }

    /**
    * @brief Specific heat at constant pressure [J/(kg·K)] as a function of temperature
    */
    inline double cp(double T) {

        if (T < Tsolid && warnings == true) std::cout << "Warning, temperature " << T << " is below solidification temperature!";
        double dXT = T - 273.15;
        return 1436.72 - 0.58 * dXT + 4.627e-4 * dXT * dXT;
    }

    /**
    * @brief Dynamic viscosity [Pa·s] using Shpilrain et al. correlation, valid for 371 K < T < 2500 K
    */
    inline double mu(double T) {

        if (T < Tsolid && warnings == true) std::cout << "Warning, temperature " << T << " is below solidification temperature!";
        return std::exp(-6.4406 - 0.3958 * std::log(T) + 556.835 / T);
    }

    /**
      * @brief Liquid sodium enthalpy [J/kg]NIST Shomate coefficients for Na(l), 370.98–1170.525 K
      * Cp° = A + B*t + C*t^2 + D*t^3 + E/t^2  [J/mol/K]
      * H° - H°298.15 = A*t + B*t^2/2 + C*t^3/3 + D*t^4/4 - E/t + F - H  [kJ/mol]
      * with t = T/1000
      */
    inline double h(double T) {

        const double T_min = 370.98, T_max = 1170.525;
        if (T < T_min) T = T_min;
        if (T > T_max) T = T_max;

        const double A = 40.25707;
        const double B = -28.23849;
        const double C = 20.69402;
        const double D = -3.641872;
        const double E = -0.079874;
        const double F = -8.782300;
        const double H = 2.406001; // NIST “H” coeff (not temperature)

        const double t = T / 1000.0;

        // kJ/mol relative to 298.15 K
        const double H_kJ_per_mol =
            A * t + B * t * t / 2.0 + C * t * t * t / 3.0 + D * t * t * t * t / 4.0 - E / t + F - H;

        // Convert to J/kg
        const double M_kg_per_mol = 22.98976928e-3; // Molar mass Na
        return (H_kJ_per_mol * 1000.0) / M_kg_per_mol;
    }

    double surf_ten(double T) {
        return (200.6 - 0.0986 * (T - 273.15)) * 1e-3;
    }
}

#pragma endregion

#pragma region vapor_sodium_properties

/**
 * @brief Provides thermophysical and transport properties for Sodium Vapor.
 *
 * This namespace contains constant data and functions to calculate key properties
 * of sodium vapor.
 * All functions primarily accept temperature T in Kelvin [K] and return values
 * in standard SI units unless otherwise noted.
 */
namespace vapor_sodium {

    constexpr double Tcrit_Na = 2509.46;           ///< Critical temperature [K]
    constexpr double Ad_Na = 3.46;                 ///< Adiabatic factor [-]
    constexpr double m_g_Na = 23e-3;               ///< Molar mass [kg/mol]

    /**
    * @brief Functions that clamps a value x to the range [a, b]
    */
    constexpr inline double clamp(double x, double a, double b) { return std::max(a, std::min(x, b)); }

    /**
    * @brief 1D table interpolation in T over monotone grid
    */
    template<size_t N>
    double interp_T(const std::array<double, N>& Tgrid, const std::array<double, N>& Ygrid, double T) {

        if (T <= Tgrid.front()) return Ygrid.front();
        if (T >= Tgrid.back())  return Ygrid.back();

        // locate interval
        size_t i = 0;
        while (i + 1 < N && !(Tgrid[i] <= T && T <= Tgrid[i + 1])) ++i;

        return Ygrid[i] + (T - Tgrid[i]) / (Tgrid[i + 1] - Tgrid[i]) * (Ygrid[i + 1] - Ygrid[i]);
    }

    /**
      * @brief Enthalpy of sodium vapor [J/kg] from NIST Shomate equation.
      * Valid for 1170.525 K ≤ T ≤ 6000 K.
      * Reference state: H(298.15 K) = 0 (per NIST convention).
      *
      * @param T Temperature [K]
      * @return Enthalpy of sodium vapor [J/kg]
      */
    inline double h(double T) {
        constexpr double T_min = 1170.525;
        constexpr double T_max = 6000.0;
        if (T < T_min) T = T_min;
        if (T > T_max) T = T_max;

        const double A = 20.80573;
        const double B = 0.277206;
        const double C = -0.392086;
        const double D = 0.119634;
        const double E = -0.008879;
        const double F = 101.0386;
        const double H = 107.2999;

        double t = T / 1000.0;

        double H_kJ_per_mol = A * t
            + B * t * t / 2.0
            + C * t * t * t / 3.0
            + D * t * t * t * t / 4.0
            - E / t
            + F
            - H;

        const double M_kg_per_mol = 22.98976928e-3;
        return (H_kJ_per_mol * 1000.0) / M_kg_per_mol; // J/kg
    }

    /**
    * @brief Saturation pressure [Pa] as a function of temperature T
    */
    inline double P_sat(double T) {

        const double val_MPa = std::exp(11.9463 - 12633.7 / T - 0.4672 * std::log(T));
        return val_MPa * 1e6;
    }

    /**
    * @brief Derivative of saturation pressure with respect to temperature [Pa/K] as a function of temperature T
    */
    inline double dP_sat_dVT(double T) {

        const double val_MPa_per_K =
            (12633.73 / (T * T) - 0.4672 / T) * std::exp(11.9463 - 12633.73 / T - 0.4672 * std::log(T));
        return val_MPa_per_K * 1e6;
    }

    /**
    * @brief Density of saturated vapor [kg/m^3] as a function of temperature T
    */
    inline double rho(double T) {

        const double hv = vapor_sodium::h(T) - liquid_sodium::h(T);     // [J/kg]
        const double dPdVT = dP_sat_dVT(T);                             // [Pa/K]
        const double rhol = liquid_sodium::rho(T);                      // [kg/m^3]
        const double denom = hv / (T * dPdVT) + 1.0 / rhol;
        return 1.0 / denom;                                             // [kg/m^3]
    }

    /**
    * @brief Specific heat at constant pressure from table interpolation [J/(kg*K)] as a function of temperature T
    */
    inline double cp(double T) {

        static const std::array<double, 21> Tgrid = { 400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400 };
        static const std::array<double, 21> Cpgrid = { 860,1250,1800,2280,2590,2720,2700,2620,2510,2430,2390,2360,2340,2410,2460,2530,2660,2910,3400,4470,8030 };

        // Table also lists 2500 K = 417030; extreme near critical. If needed, extend:
        if (T >= 2500.0) return 417030.0;

        return interp_T(Tgrid, Cpgrid, T);
    }

    /**
    * @brief Specific heat at constant volume from table interpolation [J/(kg*K)] as a function of temperature T
    */
    inline double cv(double T) {

        static const std::array<double, 21> Tgrid = { 400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400 };
        static const std::array<double, 21> Cvgrid = { 490, 840,1310,1710,1930,1980,1920,1810,1680,1580,1510,1440,1390,1380,1360,1330,1300,1300,1340,1440,1760 };

        // Table also lists 2500 K = 17030; extreme near critical. If needed, extend:
        if (T >= 2500.0) return 17030.0;

        return interp_T(Tgrid, Cvgrid, T);
    }

    /**
    * @brief Dynamic viscosity of sodium vapor [Pa·s] as a function of temperature T
    */
    inline double mu(double T) { return 6.083e-9 * T + 1.2606e-5; }

    /**
     * @brief Thermal conductivity [W/(m*K)] of sodium vapor over an extended range.
     *
     * Performs bilinear interpolation inside the experimental grid.
     * Outside 900–1500 K or 981–98066 Pa, it extrapolates using kinetic-gas scaling (k ~ sqrt(T))
     * referenced to the nearest boundary. Prints warnings when extrapolating outside of the boundaries.
     *
     * @param T Temperature [K]
     * @param P Pressure [Pa]
     */
    inline double k(double T, double P) {

        static const std::array<double, 7> Tgrid = { 900,1000,1100,1200,1300,1400,1500 };
        static const std::array<double, 5> Pgrid = { 981,4903,9807,49033,98066 };

        static const double Ktbl[7][5] = {
            // P = 981,   4903,    9807,    49033,   98066  [Pa]
            {0.035796, 0.0379,  0.0392,  0.0415,  0.0422},   // 900 K
            {0.034053, 0.043583,0.049627,0.0511,  0.0520},   // 1000 K
            {0.036029, 0.039399,0.043002,0.060900,0.0620},   // 1100 K
            {0.039051, 0.040445,0.042189,0.052881,0.061133}, // 1200 K
            {0.042189, 0.042886,0.043816,0.049859,0.055554}, // 1300 K
            {0.045443, 0.045908,0.046373,0.049859,0.054508}, // 1400 K
            {0.048930, 0.049162,0.049511,0.051603,0.054043}  // 1500 K
        };

        // Clamping function
        auto clamp_val = [](double x, double minv, double maxv) {
            return (x < minv) ? minv : ((x > maxv) ? maxv : x);
            };

        auto idz = [](double x, const auto& grid) {
            size_t i = 0;
            while (i + 1 < grid.size() && x > grid[i + 1]) ++i;
            return i;
            };

        const double Tmin = Tgrid.front(), Tmax = Tgrid.back();
        const double Pmin = Pgrid.front(), Pmax = Pgrid.back();

        bool Tlow = (T < Tmin);
        bool Thigh = (T > Tmax);
        bool Plow = (P < Pmin);
        bool Phigh = (P > Pmax);

        double Tc = clamp_val(T, Tmin, Tmax);
        double Pc = clamp_val(P, Pmin, Pmax);

        const size_t iT = idz(Tc, Tgrid);
        const size_t iP = idz(Pc, Pgrid);

        const double T0 = Tgrid[iT], T1 = Tgrid[std::min(iT + 1ul, Tgrid.size() - 1)];
        const double P0 = Pgrid[iP], P1 = Pgrid[std::min(iP + 1ul, Pgrid.size() - 1)];

        const double q11 = Ktbl[iT][iP];
        const double q21 = Ktbl[std::min(iT + 1ul, Tgrid.size() - 1)][iP];
        const double q12 = Ktbl[iT][std::min(iP + 1ul, Pgrid.size() - 1)];
        const double q22 = Ktbl[std::min(iT + 1ul, Tgrid.size() - 1)][std::min(iP + 1ul, Pgrid.size() - 1)];

        double k_interp = 0.0;

        // Bilinear interpolation
        if ((T1 != T0) && (P1 != P0)) {
            const double t = (Tc - T0) / (T1 - T0);
            const double u = (Pc - P0) / (P1 - P0);
            k_interp = (1 - t) * (1 - u) * q11 + t * (1 - u) * q21 + (1 - t) * u * q12 + t * u * q22;
        }
        else if (T1 != T0) {
            const double t = (Tc - T0) / (T1 - T0);
            k_interp = q11 + t * (q21 - q11);
        }
        else if (P1 != P0) {
            const double u = (Pc - P0) / (P1 - P0);
            k_interp = q11 + u * (q12 - q11);
        }
        else {
            k_interp = q11;
        }

        // Extrapolation handling
        if (Tlow || Thigh || Plow || Phigh) {
            if (Tlow && warnings == true)
                std::cerr << "[Warning] Sodium vapor k: T=" << T << " < " << Tmin << " K. Using sqrt(T) extrapolation.\n";
            if (Thigh && warnings == true)
                std::cerr << "[Warning] Sodium vapor k: T=" << T << " > " << Tmax << " K. Using sqrt(T) extrapolation.\n";
            if ((Plow || Phigh) && warnings == true)
                std::cerr << "[Warning] Sodium vapor k: P outside ["
                << Pmin << "," << Pmax << "] Pa. Using constant-P approximation.\n";

            double Tref = (Tlow ? Tmin : (Thigh ? Tmax : Tc));
            double k_ref = k_interp;
            double k_extrap = k_ref * std::sqrt(T / Tref);
            return k_extrap;
        }

        return k_interp;
    }

    /**
    * @brief Friction factor [-] (Gnielinski correlation) as a function of Reynolds number.
    * Retrieves an error if Re < 0.
    */
    inline double f(double Re) {

        if (Re <= 0.0) throw std::invalid_argument("Error: Re < 0");

        const double t = 0.79 * std::log(Re) - 1.64;
        return 1.0 / (t * t);
    }

    /**
    * @brief Nusselt number [-] (Gnielinski correlation) as a function of Reynolds number
    * Retrieves an error if Re < 0 or Nu < 0.
    */
    inline double Nu(double Re, double Pr) {

        // If laminar, Nu is constant
        if (Re < 1000) return 4.36;

        if (Re <= 0.0 || Pr <= 0.0) throw std::invalid_argument("Error: Re or Pr < 0");

        const double f = vapor_sodium::f(Re);
        const double fp8 = f / 8.0;
        const double num = fp8 * (Re - 1000.0) * Pr;
        const double den = 1.0 + 12.7 * std::sqrt(fp8) * (std::cbrt(Pr * Pr) - 1.0); // Pr^(2/3)
        return num / den;
    }

    /**
    * @brief Convective heat transfer coefficient [W/m^2/K] (Gnielinski correlation) as a function of Reynolds number
    * Retrieves an error if Re < 0 or Nu < 0.
    */
    inline double h_conv(double Re, double Pr, double k, double Dh) {

        const double Nu = vapor_sodium::Nu(Re, Pr);
        return Nu * k / Dh;
    }
}

#pragma endregion

// =======================================================================
//
//                        [SOLVING FUNCTIONS]
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

inline int H(double x) {
    return x > 0.0 ? 1 : 0;
}

int main() {

    #pragma region constants_and_variables

    // Mathematical constants
    const double M_PI = 3.14159265358979323846;

    // Physical properties
    const double emissivity = 0.9;          ///< Wall emissivity [-]
    const double sigma = 5.67e-8;           ///< Stefan-Boltzmann constant [W/m^2/K^4]
    const double Rv = 361.8;                ///< Gas constant for the sodium vapor [J/(kg K)]
    
    // Environmental boundary conditions
    const double h_conv = 10;               ///< Convective heat transfer coefficient for external heat removal [W/m^2/K]
    const double power = 1e3;               ///< Power at the evaporator side [W]
    const double T_env = 280.0;             ///< External environmental temperature [K]

    // Evaporation and condensation parameters
    const double eps_s = 1.0;               ///< Surface fraction of the wick available for phasic interface [-]
    const double sigma_e = 0.05;             ///< Evaporation accomodation coefficient [-]. 1 means optimal evaporation
    const double sigma_c = 0.05;             ///< Condensation accomodation coefficient [-]. 1 means optimal condensation
    double Omega = 1.0;                     ///< Omega factor initialization

    const double Tc = 2509.46;
    double const eps_v = 1.0;

	const double tolerance = 1e-4;			///< Tolerance for the convergence of Picard loop [-]

    // Geometric parameters
    const int N = 20;                                                           ///< Number of axial nodes [-]
    const double l = 0.982; 			                                        ///< Length of the heat pipe [m]
    const double dz = l / N;                                                    ///< Axial discretization step [m]
    const double evaporator_length = 0.502;                                     ///< Evaporator length [m]
    const double adiabatic_length = 0.188;                                      ///< Adiabatic length [m]
    const double condenser_length = 0.292;                                      ///< Condenser length [m]
    const double evaporator_nodes = std::floor(evaporator_length / dz);         ///< Number of evaporator nodes
    const double condenser_nodes = std::ceil(condenser_length / dz);            ///< Number of condenser nodes
    const double adiabatic_nodes = N - (evaporator_nodes + condenser_nodes);    ///< Number of adiabatic nodes
    const double r_o = 0.01335;                                                 ///< Outer wall radius [m]
    const double r_i = 0.0112;                                                  ///< Wall-wick interface radius [m]
    const double r_v = 0.01075;                                                 ///< Vapor-wick interface radius [m]

    // Surfaces 
    const double A_w_outer = 2 * M_PI * r_o * dz;                               ///< Wall radial area (at r_o) [m^2]
    const double A_w_cross = M_PI * (r_o * r_o - r_i * r_i);                    ///< Wall cross-sectional area [m^2]
    const double A_x_interface = 2 * M_PI * r_i * dz;                           ///< Wick radial area (at r_i) [m^2]
    const double A_x_cross = M_PI * (r_i * r_i - r_v * r_v);                    ///< Wick cross-sectional area [m^2]
    const double A_v_inner = 2 * M_PI * r_v * dz;                               ///< Vapor radial area (at r_v) [m^2]
    const double A_v_cross = M_PI * r_v * r_v;                                  ///< Vapor cross-sectional area [m^2]

    // Time-stepping parameters
    double dt = 1e-3;                                   ///< Initial time step [s] (then it is updated according to the limits)
    const int tot_iter = 100000;                        ///< Number of timesteps
    const double time_total = tot_iter * dt;            ///< Total simulation time [s]

    // Wick permeability parameters
    const double K = 1e-8;                              ///< Permeability [m^2]
    const double CF = 1e4;                              ///< Forchheimer coefficient [1/m]

    // Mesh z positions
    std::vector<double> mesh(N, 0.0);
    for (int i = 0; i < N; ++i) mesh[i] = i * dz;       ///< Mesh discretization

    // Node partition
    const int N_e = static_cast<int>(std::floor(evaporator_length / dz));   ///< Number of nodes of the evaporator region [-]
    const int N_c = static_cast<int>(std::ceil(condenser_length / dz));     ///< Number of nodes of the condenser region [-]
    const int N_a = N - (N_e + N_c);                                        ///< Number of nodes of the adiabadic region [-]

    const double T_full = 600.0;                                            ///< Uniform temperature initialization [K]

    const double q_pp_evaporator = power / (2 * M_PI * evaporator_length * r_o);        ///< Heat flux at evaporator from given power [W/m^2]
	std::vector<double> q_pp(N, 0.0);                                                   ///< Heat flux profile [W/m^2]

    // Mass sources/fluxes
    std::vector<double> phi_x_v(N, 0.0);                ///< Mass flux [kg/m2/s] at the wick-vapor interface (positive if evaporation)

    // Create result folder
    int new_case = 0;
    std::string name = "case_0";
    while (true) {
        std::string name = "case_" + std::to_string(new_case);
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

    const int global_precision = 4;

    mesh_output << std::setprecision(global_precision);

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
   
    for (int i = 0; i < N; ++i) mesh_output << i * dz << " ";

    mesh_output.flush();
    mesh_output.close();

    #pragma endregion

    std::vector<double> rho_m(N, 0.01);
    std::vector<double> rho_l(N, 700);
    std::vector<double> alpha_m(N, 0.9);
    std::vector<double> alpha_l(N, 0.1);
    std::vector<double> p_m(N, 2650);
    std::vector<double> p_l(N, 2650);
    std::vector<double> v_m(N, 1.0);
    std::vector<double> v_l(N, -0.001);
    std::vector<double> T_m(N);
    std::vector<double> T_l(N);
    std::vector<double> T_w(N);

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

    for (int i = 0; i < N; ++i) {
        const double f = double(i) / double(N - 1);   // da 0 a 1
        const double T = 800.0 + f * (600.0 - 800.0); // lineare 800 → 600
        T_m[i] = T;
        T_l[i] = T;
        T_w[i] = T;

        // Inizializza P vapore alla P di saturazione locale per evitare flash istantanei
        p_m[i] = vapor_sodium::P_sat(T);
        p_l[i] = p_m[i]; // Assumi equilibrio meccanico iniziale
    }

	std::vector<double> mass_source(N, 0.0);

	std::vector<double> heat_source_wall_liquid_flux(N, 0.0);
	std::vector<double> heat_source_liquid_wall_flux(N, 0.0);

	std::vector<double> heat_source_vapor_liquid_phase(N, 0.0);
	std::vector<double> heat_source_liquid_vapor_phase(N, 0.0);

	std::vector<double> heat_source_vapor_liquid_flux(N, 0.0);
	std::vector<double> heat_source_liquid_vapor_flux(N, 0.0);

    std::vector<double> p_saturation(N, 0.0);

    std::vector<double> Gamma_xv(N, 0.0);
    std::vector<double> T_sur(N, 800.0);

    std::vector<SparseBlock> L(N), D(N), R(N);
    std::vector<VecBlock> Q(N), X(N), X_old(N), X_iter(N), X_new(N);

    const double Eio1 = 2.0 / 3.0 * (r_o + r_i - 1 / (1 / r_o + 1 / r_i));
    const double Eio2 = 0.5 * (r_o * r_o + r_i * r_i);
    const double Evi1 = 2.0 / 3.0 * (r_i + r_v - 1 / (1 / r_i + 1 / r_v));
    const double Evi2 = 0.5 * (r_i * r_i + r_v * r_v);

	const int Kmax = 20;    ///< Maximum number of Picard iterations per timestep

    for (int i = 0; i < N; ++i) {
        X[i][0] = rho_m[i];
        X[i][1] = rho_l[i];
        X[i][2] = alpha_m[i];
        X[i][3] = alpha_l[i];
        X[i][4] = p_m[i];
        X[i][5] = p_l[i];
        X[i][6] = v_m[i];
        X[i][7] = v_l[i];
        X[i][8] = T_m[i];
        X[i][9] = T_l[i];
        X[i][10] = T_w[i];
    }

    
	// Time-stepping loop
    for(int n = 0; n < tot_iter; ++n) {

		X_old = X;        // X contains the solution at the previous timestep
        X_iter = X_old;

        for (int i = 0; i < N; ++i) {
            rho_m_old[i] = X_old[i][0];
            rho_l_old[i] = X_old[i][1];
            alpha_m_old[i] = X_old[i][2];
            alpha_l_old[i] = X_old[i][3];
            p_m_old[i] = X_old[i][4];
            p_l_old[i] = X_old[i][5];
            v_m_old[i] = X_old[i][6];
            v_l_old[i] = X_old[i][7];
            T_m_old[i] = X_old[i][8];
            T_l_old[i] = X_old[i][9];
            T_w_old[i] = X_old[i][10];
        }
       
		// Picard iteration loop
        for (int k = 0; k < Kmax; ++k) {

            for (int i = 0; i < N; ++i) {
                rho_m[i] = X_iter[i][0];
                rho_l[i] = X_iter[i][1];
                alpha_m[i] = X_iter[i][2];
                alpha_l[i] = X_iter[i][3];
                p_m[i] = X_iter[i][4];
                p_l[i] = (X_iter[i][5] > 0 ? X_iter[i][5] : 0.0);
                v_m[i] = X_iter[i][6];
                v_l[i] = X_iter[i][7];
                T_m[i] = X_iter[i][8];
                T_l[i] = X_iter[i][9];
                T_w[i] = X_iter[i][10];
            }

		    // Space discretization loop
            for(int i = 1; i < N - 1; ++i) {

                // Physical properties
                const double k_w = steel::k(T_w[i]);                                                ///< Wall thermal conductivity [W/(m K)]
                const double k_x = liquid_sodium::k(T_l[i]);                                        ///< Liquid thermal conductivity [W/(m K)]
                const double k_m = vapor_sodium::k(T_m[i], p_m[i]);                                 ///< Vapor thermal conductivity [W/(m K)]
                const double cp_m = vapor_sodium::cp(T_m[i]);                                       ///< Vapor specific heat [J/(kg K)]
                const double mu_v = vapor_sodium::mu(T_m[i]);                                       ///< Vapor dynamic viscosity [Pa*s]
                const double mu_l = liquid_sodium::mu(T_l[i]);                                      ///< Liquid dynamic viscosity
                const double Dh_v = 2.0 * r_v;                                                      ///< Hydraulic diameter of the vapor core [m]
                const double Re_v = rho_m[i] * std::fabs(v_m[i]) * Dh_v / mu_v;                     ///< Reynolds number [-]
                const double Pr_v = cp_m * mu_v / k_m;                                              ///< Prandtl number [-]
                const double H_xm = vapor_sodium::h_conv(Re_v, Pr_v, k_m, Dh_v);                    ///< Convective heat transfer coefficient at the vapor-wick interface [W/m^2/K]
                const double Psat = vapor_sodium::P_sat(T_sur[i]);                                  ///< Saturation pressure [Pa]         
                const double dPsat_dT = Psat * std::log(10.0) * (7740.0 / (T_sur[i] * T_sur[i]));   ///< Derivative of the saturation pressure wrt T [Pa/K]   
        
                double h_xv_v;      ///< Specific enthalpy [J/kg] of vapor upon phase change between wick and vapor
                double h_vx_x;      ///< Specific enthalpy [J/kg] of wick upon phase change between vapor and wick

                if (phi_x_v[i] >= 0.0) {

                    // Evaporation case
                    h_xv_v = vapor_sodium::h(T_sur[i]);
                    h_vx_x = liquid_sodium::h(T_sur[i]);

                }
                else {

                    // Condensation case
                    h_xv_v = vapor_sodium::h(T_m[i]);
                    h_vx_x = liquid_sodium::h(T_sur[i])
                        + (vapor_sodium::h(T_m[i]) - vapor_sodium::h(T_sur[i]));
                }

                // Update heat fluxes at the interfaces
                if (i <= evaporator_nodes) q_pp[i] = q_pp_evaporator;                                  ///< Evaporator imposed heat flux
                else if (i >= (N - condenser_nodes)) {

                    double conv = h_conv * (T_w[i] - T_env);                                          ///< Condenser convective heat flux
                    double irr = emissivity * sigma * (std::pow(T_w[i], 4) - std::pow(T_env, 4));     ///< Condenser irradiation heat flux

                    q_pp[i] = -(conv + irr);                                                           ///< Heat flux at the outer wall (positive if to the wall)
                }

                const double beta = 1.0 / std::sqrt(2 * M_PI * Rv * T_sur[i]);
                const double b = -phi_x_v[i] / (p_m[i] * std::sqrt(2.0 / (Rv * T_m[i])));

                if (b < 0.1192) Omega = 1.0 + b * std::sqrt(M_PI);
                else if (b <= 0.9962) Omega = 0.8959 + 2.6457 * b;
                else Omega = 2.0 * b * std::sqrt(M_PI);

                const double fac = (2.0 * r_v * eps_s * beta) / (r_i * r_i);        ///< Useful factor in the coefficients calculation [s / m^2]

                const double bGamma = -(Gamma_xv[i] / (2.0 * T_sur[i])) + fac * sigma_e * dPsat_dT; ///< b coefficient [kg/(m3 s K)] 
                const double aGamma = 0.5 * Gamma_xv[i] + fac * sigma_e * dPsat_dT * T_sur[i];      ///< a coefficient [kg/(m3 s)]
                const double cGamma = -fac * sigma_c * Omega;                                       ///< c coefficient [s/m2]

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

			    T_sur[i] = C36 * p_m[i] + C37 * T_m[i] + C38 * T_l[i] + C39 * T_w[i] + C40;

                phi_x_v[i] = beta * (sigma_e * Psat - sigma_c * Omega * p_m[i]);
                Gamma_xv[i] = 2 * r_v * eps_s / (r_i * r_i) * phi_x_v[i];

                // DPcap evaluation

                const double alpha_m0 = r_v * r_v / (r_i * r_i);
                const double r_p = 1e-5;
                const double surf_ten_value = liquid_sodium::surf_ten(T_l[i]); 

                const double Lambda = 3 * r_v / (eps_s * r_p) * (alpha_m[i] - alpha_m0);
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
                    + (alpha_m[i] * v_m[i] * H(v_m[i])) / dz
                    - (alpha_m[i] * v_m[i - 1] * (1 - H(v_m[i - 1]))) / dz
                );

                add(D[i], 0, 2,
                    + rho_m_old[i] / dt
                    + (rho_m[i] * v_m[i] * H(v_m[i])) / dz
                    - (rho_m[i] * v_m[i - 1] * (1 - H(v_m[i - 1]))) / dz
                );

                add(D[i], 0, 4,
                    - C41
			    );

                add(D[i], 0, 6,
                    + (alpha_m[i] * rho_m[i] * H(v_m[i])) / dz
                    + (alpha_m[i + 1] * rho_m[i + 1] * (1 - H(v_m[i]))) / dz
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
                        + alpha_m[i] * rho_m[i] * v_m[i] * H(v_m[i])
                        + alpha_m[i + 1] * rho_m[i + 1] * v_m[i] * (1 - H(v_m[i]))
                        - alpha_m[i - 1] * rho_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])
                        - alpha_m[i] * rho_m[i] * v_m[i - 1] * (1 - H(v_m[i - 1]))
                        ) / dz
                    + 2 * (rho_m_old[i] * alpha_m_old[i]) / dt
                    //+ mass_source[i]
                    ;

                add(L[i], 0, 0,
                    - (alpha_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])) / dz
                );

                add(L[i], 0, 2,
                    - (rho_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])) / dz
                );

                add(L[i], 0, 6,
                    - (alpha_m[i - 1] * rho_m[i - 1] * H(v_m[i - 1])) / dz
                    + (alpha_m[i] * rho_m[i] * (1 - H(v_m[i - 1]))) / dz
                );

                add(R[i], 0, 0,
                    (alpha_m[i + 1] * v_m[i] * (1 - H(v_m[i]))) / dz
                );

                add(R[i], 0, 2,
                    (rho_m[i + 1] * v_m[i] * (1 - H(v_m[i]))) / dz
                );

                // Mass liquid equation

                add(D[i], 1, 1,
                    + eps_v * (alpha_l_old[i] / dt)
                    + eps_v * (alpha_l[i] * v_l[i] * H(v_l[i])) / dz
                    - eps_v * (alpha_l[i] * v_l[i - 1] * (1 - H(v_l[i - 1]))) / dz
                );

                add(D[i], 1, 3,
                    + eps_v * (rho_l_old[i] / dt)
                    + eps_v * (rho_l[i] * v_l[i] * H(v_l[i])) / dz
                    - eps_v * (rho_l[i] * v_l[i - 1] * (1 - H(v_l[i - 1]))) / dz
                );

                add(D[i], 1, 4,
                    + C41
			    );

                add(D[i], 1, 7,
                    + eps_v * (alpha_l[i] * rho_l[i] * H(v_l[i])) / dz
                    + eps_v * (alpha_l[i + 1] * rho_l[i + 1] * (1 - H(v_l[i]))) / dz
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
                        + eps_v * (alpha_l[i] * rho_l[i] * v_l[i] * H(v_l[i]))
                        + eps_v * (alpha_l[i + 1] * rho_l[i + 1] * v_l[i] * (1 - H(v_l[i])))
                        - eps_v * (alpha_l[i - 1] * rho_l[i - 1] * v_l[i - 1] * H(v_l[i - 1]))
                        - eps_v * (alpha_l[i] * rho_l[i] * v_l[i - 1] * (1 - H(v_l[i - 1])))
                        ) / dz
                    + 2 * (
                        + eps_v * (rho_l_old[i] * alpha_l_old[i])
                        ) / dt
                    //- mass_source[i]
                    ;

                add(L[i], 1, 1,
                    - eps_v * (alpha_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])) / dz
                );

                add(L[i], 1, 3,
                    - eps_v * (rho_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])) / dz
                );

                add(L[i], 1, 7,
                    - eps_v * (alpha_l[i - 1] * rho_l[i - 1] * H(v_l[i - 1])) / dz
                    - eps_v * (alpha_l[i] * rho_l[i] * (1 - H(v_l[i - 1]))) / dz
                );

                add(R[i], 1, 1,
                    + eps_v * (alpha_l[i + 1] * v_l[i] * (1 - H(v_l[i]))) / dz
                );

                add(R[i], 1, 3,
                    + eps_v * (rho_l[i + 1] * v_l[i] * (1 - H(v_l[i]))) / dz
                );

                // Mixture heat equation

                const double cp_m_p = vapor_sodium::cp(T_m[i]);
                const double cp_m_l = vapor_sodium::cp(T_m[i - 1]);
                const double cp_m_r = vapor_sodium::cp(T_m[i + 1]);

                const double k_m_p = vapor_sodium::k(T_m[i], p_m[i]);
                const double k_m_l = vapor_sodium::k(T_m[i - 1], p_m[i - 1]);
                const double k_m_r = vapor_sodium::k(T_m[i + 1], p_m[i + 1]);

                add(D[i], 2, 0,
                    + (alpha_m_old[i] * cp_m_p * T_m_old[i]) / dt
                    + (alpha_m[i] * cp_m_p * T_m[i] * v_m[i] * H(v_m[i])) / dz
                    - (alpha_m[i] * cp_m_p * T_m[i] * v_m[i - 1] * (1 - H(v_m[i - 1]))) / dz

                );

                add(D[i], 2, 2,
                    + (T_m_old[i] * rho_m_old[i] * cp_m_p) / dt
                    + (rho_m[i] * cp_m_p * T_m[i] * v_m[i] * H(v_m[i])) / dz
                    - (rho_m[i] * cp_m_p * T_m[i] * v_m[i - 1] * (1 - H(v_m[i - 1]))) / dz
                    + p_m[i] * (v_m[i] * H(v_m[i]) + v_m[i - 1] * H(v_m[i - 1])) / dz
                    + p_m_old[i] / dt
                );

                add(D[i], 2, 4,
                    - C51 - C61
			    );

                add(D[i], 2, 6,
                    + (alpha_m[i] * rho_m[i] * cp_m_p * T_m[i] * H(v_m[i]) + alpha_m[i + 1] * rho_m[i + 1] * cp_m_r * T_m[i + 1] * (1 - H(v_m[i]))) / dz
                    + p_m[i] * (alpha_m[i] * H(v_m[i]) + alpha_m[i + 1] * (1 - H(v_m[i]))) / dz
                );

                add(D[i], 2, 8,
                    + (alpha_m_old[i] * rho_m_old[i] * cp_m_p) / dt
                    + (alpha_m[i] * rho_m[i] * cp_m_p * v_m[i] * H(v_m[i])) / dz
                    - (alpha_m[i] * rho_m[i] * cp_m_p * v_m[i - 1] * (1 - H(v_m[i - 1]))) / dz
                    + (alpha_m[i] * k_m_p * H(v_m[i]) + alpha_m[i + 1] * k_m_r * (1 - H(v_m[i]))) / (dz * dz)
                    + (alpha_m[i - 1] * k_m_l * H(v_m[i - 1]) + alpha_m[i] * k_m_p * (1 - H(v_m[i - 1]))) / (dz * dz)
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
                        + alpha_m[i] * rho_m[i] * cp_m_p * T_m[i] * v_m[i] * H(v_m[i]) 
                        + alpha_m[i + 1] * rho_m[i + 1] * cp_m_r * T_m[i + 1] * v_m[i + 1] * (1 - H(v_m[i]))
                        - alpha_m[i - 1] * rho_m[i - 1] * cp_m_l * T_m[i - 1] * v_m[i - 1] * H(v_m[i - 1]) 
                        - alpha_m[i] * rho_m[i] * cp_m_p * T_m[i] * v_m[i - 1] * (1 - H(v_m[i - 1]))
                        ) / dz
                    + p_m[i] * (
                        + alpha_m[i] * v_m[i] * H(v_m[i])
                        + alpha_m[i + 1] * v_m[i] * (1 - H(v_m[i]))
                        - alpha_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])
                        - alpha_m[i] * v_m[i - 1] * (1 - H(v_m[i - 1]))
                        ) / dz
                    + p_m_old[i] * alpha_m_old[i] / dt
                    + C55 + C65
                    // + heat_source_liquid_vapor_flux[i]
                    // + heat_source_liquid_vapor_phase[i]
                    ;

                add(L[i], 2, 0,
                    - (alpha_m[i - 1] * cp_m_l * T_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])) / dz
                );

                add(L[i], 2, 2,
                    - (rho_m[i - 1] * cp_m_l * T_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])) / dz
                    - p_m[i] * (v_m[i - 1] * H(v_m[i - 1])) / dz
                );

                add(L[i], 2, 6,
                    - (
                        + alpha_m[i - 1] * rho_m[i - 1] * cp_m_l * T_m[i - 1] * H(v_m[i - 1]) 
                        + alpha_m[i] * rho_m[i] * cp_m_l * T_m[i] * (1 - H(v_m[i - 1]))
                        ) / dz
                    + p_m[i] * (
                        + alpha_m[i - 1] * H(v_m[i - 1]) 
                        + alpha_m[i] * (1 - H(v_m[i - 1]))
                        ) / dz
                );

                add(L[i], 2, 8,
                    - (alpha_m[i - 1] * rho_m[i - 1] * cp_m_l * v_m[i - 1] * H(v_m[i - 1])) / dz
                    - (alpha_m[i - 1] * k_m_l * H(v_m[i - 1]) + alpha_m[i] * k_m_p * (1 - H(v_m[i - 1]))) / (dz * dz)
                );

                add(R[i], 2, 0,
                    + (alpha_m[i + 1] * cp_m_r * T_m[i + 1] * v_m[i] * (1 - H(v_m[i]))) / dz
                );

                add(R[i], 2, 2,
                    + (rho_m[i + 1] * cp_m_r * T_m[i + 1] * v_m[i] * (1 - H(v_m[i]))) / dz
                    + p_m[i] * (v_m[i] * (1 - H(v_m[i]))) / dz
                );

                add(R[i], 2, 8,
                    + (alpha_m[i + 1] * rho_m[i + 1] * cp_m_r * v_m[i] * (1 - H(v_m[i]))) / dz
                    - (alpha_m[i] * k_m_p * H(v_m[i]) + alpha_m[i + 1] * k_m_r * (1 - H(v_m[i]))) / (dz * dz)
                );

                // Heat liquid equation

                const double cp_l_p = liquid_sodium::cp(T_l[i]);
                const double cp_l_l = liquid_sodium::cp(T_l[i - 1]);
                const double cp_l_r = liquid_sodium::cp(T_l[i + 1]);

                const double k_l_p = liquid_sodium::k(T_l[i]);
                const double k_l_l = liquid_sodium::k(T_l[i - 1]);
                const double k_l_r = liquid_sodium::k(T_l[i + 1]);

                add(D[i], 3, 1,
                    + eps_v * (alpha_l_old[i] * T_l_old[i] * cp_l_p) / dt
                    + eps_v * (alpha_l[i] * cp_l_p * T_l[i] * v_l[i] * H(v_l[i])) / dz
                    - eps_v * (alpha_l[i] * cp_l_p * T_l[i] * v_l[i - 1] * (1 - H(v_l[i - 1]))) / dz
                );

                add(D[i], 3, 3,
                    + eps_v * (T_l_old[i] * rho_l_old[i] * cp_l_p) / dt
                    + eps_v * (rho_l[i] * cp_l_p * T_l[i] * v_l[i] * H(v_l[i])) / dz
                    - eps_v * (rho_l[i] * cp_l_p * T_l[i] * v_l[i - 1] * (1 - H(v_l[i - 1]))) / dz
                    + eps_v * p_l[i] * (v_l[i] * H(v_l[i]) + v_l[i - 1] * H(v_l[i - 1])) / dz
                    + eps_v * p_l_old[i] / dt
                );

                add(D[i], 3, 4,
                    - C46 - C56 - C66
                );

                add(D[i], 3, 7, 
                    + eps_v * (alpha_l[i] * rho_l[i] * cp_l_p * T_l[i] * H(v_l[i]) + alpha_l[i + 1] * rho_l[i + 1] * cp_l_r * T_l[i + 1] * (1 - H(v_l[i]))) / dz
                    + eps_v * (p_l[i] * (alpha_l[i] * H(v_l[i]) + alpha_l[i + 1] * (1 - H(v_l[i]))) / dz)
                );

                add(D[i], 3, 8, 
                    - C47 - C57 - C67
                );

                add(D[i], 3, 9,
                    + eps_v * (alpha_l_old[i] * rho_l_old[i] * cp_l_p) / dt
                    + eps_v * (alpha_l[i] * rho_l[i] * cp_l_p * v_l[i] * H(v_l[i])) / dz
                    - eps_v * (alpha_l[i] * rho_l[i] * cp_l_p * v_l[i - 1] * (1 - H(v_l[i - 1]))) / dz
                    + eps_v * (alpha_l[i] * k_l_p * H(v_l[i]) + alpha_l[i + 1] * k_l_r * (1 - H(v_l[i]))) / (dz * dz)
                    + eps_v * (alpha_l[i - 1] * k_l_l * H(v_l[i - 1]) + alpha_l[i] * k_l_p * (1 - H(v_l[i - 1]))) / (dz * dz)
                    - C48 - C58 - C68

                );

                add(D[i], 3, 10, 
                    - C49 - C59 - C69
                );

                Q[i][3] = 
                    + eps_v * 3 * ( 
                        + alpha_l_old[i] * T_l_old[i] * cp_l_p * rho_l_old[i]) / dt
                    + eps_v * 3 * (
                        + alpha_l[i] * rho_l[i] * cp_l_p * T_l[i] * v_l[i] * H(v_l[i])
                        + alpha_l[i + 1] * rho_l[i + 1] * cp_l_r * T_l[i + 1] * v_l[i + 1] * (1 - H(v_l[i]))
                        - alpha_l[i - 1] * rho_l[i - 1] * cp_l_l * T_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])
                        - alpha_l[i] * rho_l[i] * cp_l_p * T_l[i] * v_l[i] - 1 * (1 - H(v_l[i - 1]))
                            ) / dz
                    + eps_v * p_l[i] * (
                            + alpha_l[i] * v_l[i] * H(v_l[i])
                            + alpha_l[i + 1] * v_l[i] * (1 - H(v_l[i]))
                            - alpha_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])
                            - alpha_l[i] * v_l[i - 1] * (1 - H(v_l[i - 1]))
                            ) / dz
                    + eps_v * p_l_old[i] * alpha_l_old[i] / dt
                    + C50 + C60 + C70
                    // + heat_source_wall_liquid_flux[i]
                    // + heat_source_vapor_liquid_flux[i]
                    // + heat_source_vapor_liquid_phase[i]
                    ;

                add(L[i], 3, 1,
                    - eps_v * (alpha_l[i - 1] * cp_l_l * T_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])) / dz
                );

                add(L[i], 3, 3,
                    + eps_v * (rho_l[i - 1] * cp_l_l * T_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])) / dz
                    + eps_v * (p_l[i] * v_l[i - 1] * H(v_l[i - 1])) / dz
                );

                add(L[i], 3, 7,
                    - eps_v * (alpha_l[i - 1] * rho_l[i - 1] * cp_l_l * T_l[i - 1] * H(v_l[i - 1])) / dz
                    - eps_v * (alpha_l[i] * rho_l[i] * cp_l_l * T_l[i] * (1 - H(v_l[i - 1]))) / dz
                    + eps_v * p_l[i] * (alpha_l[i - 1] * H(v_l[i - 1])) / dz
                    + eps_v * p_l[i] * (alpha_l[i] * (1 - H(v_l[i - 1]))) / dz
                );

                add(L[i], 3, 9,
                    - eps_v * (alpha_l[i - 1] * rho_l[i - 1] * cp_l_l * v_l[i - 1] * H(v_l[i - 1])) / dz
                    - eps_v * (alpha_l[i - 1] * k_l_l * H(v_l[i - 1]) + alpha_l[i] * k_l_p * (1 - H(v_l[i - 1]))) / (dz * dz)
                );

                add(R[i], 3, 1,
                    + eps_v * (alpha_l[i + 1] * cp_l_r * T_l[i + 1] * v_l[i] * (1 - H(v_l[i]))) / dz
                );

                add(R[i], 3, 3,
                    + eps_v * (rho_l[i + 1] * cp_l_r * T_l[i + 1] * v_l[i] * (1 - H(v_l[i]))) / dz
                    + eps_v * p_l[i] * v_l[i] * (1 - H(v_l[i])) / dz
                );

                add(R[i], 3, 9,
                    + eps_v * (alpha_l[i + 1] * rho_l[i + 1] * cp_l_r * v_l[i] * (1 - H(v_l[i]))) / dz
                    - eps_v * (alpha_l[i] * k_l_p * H(v_l[i]) + alpha_l[i + 1] * k_l_r * (1 - H(v_l[i]))) / (dz * dz)
                );

                // Heat wall equation

                const double rho_w_p = steel::rho(T_w[i]);
                const double rho_w_l = steel::rho(T_w[i - 1]);
                const double rho_w_r = steel::rho(T_w[i + 1]);

                const double cp_w_p = steel::cp(T_w[i]);
                const double cp_w_l = steel::cp(T_w[i - 1]);
                const double cp_w_r = steel::cp(T_w[i + 1]);

                const double k_w_p = steel::k(T_w[i]);
                const double k_w_l = steel::k(T_w[i - 1]);
                const double k_w_r = steel::k(T_w[i + 1]);

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

                const double Re = rho_m[i] * v_m[i] * Dh_v / mu_v;
                const double fm = Re > 1187.4 ? 0.3164 * std::pow(Re, -0.25) : 64 * std::pow(Re, -1);
                const double Fm = fm * v_m[i] / (4 * r_v);

                add(D[i], 5, 0,
                    + (alpha_m_old[i] * v_m_old[i]) / dt
                    + (alpha_m[i] * v_m[i] * v_m[i] * H(v_m[i])) / dz
                    - (alpha_m[i] * v_m[i - 1] * v_m[i - 1] * (1 - H(v_m[i - 1]))) / dz
                );

                add(D[i], 5, 2,
                    + (v_m_old[i] * rho_m_old[i]) / dt
                    + (rho_m[i] * v_m[i] * v_m[i] * H(v_m[i])) / dz
                    - (rho_m[i] * v_m[i - 1] * v_m[i - 1] * (1 - H(v_m[i - 1]))) / dz
                );

                add(D[i], 5, 4,
                    - (alpha_m[i] * H(v_m[i])) / dz
                    - (alpha_m[i + 1] * (1 - H(v_m[i]))) / dz
                );

                add(D[i], 5, 6,
                    + (alpha_m_old[i] * rho_m_old[i]) / dt
                    + 2 * (rho_m[i] * alpha_m[i] * v_m[i] * H(v_m[i])) / dz
                    + 2 * (rho_m[i + 1] * alpha_m[i + 1] * v_m[i + 1] * (1 - H(v_m[i]))) / dz
                    + (Fm * rho_m[i] * std::abs(v_m[i]) * H(v_m[i])) / (4 * r_v)
                    + (Fm * rho_m[i + 1] * std::abs(v_m[i]) * (1 - H(v_m[i]))) / (4 * r_v)
                );

                Q[i][5] = 
                    + 3 * (alpha_m_old[i] * rho_m_old[i] * v_m_old[i]) / dt
                    - 3 * (rho_m[i] * alpha_m[i] * v_m[i] * v_m[i] * H(v_m[i])) / dz
                    - 3 * (rho_m[i + 1] * alpha_m[i + 1] * v_m[i] * v_m[i] * (1 - H(v_m[i]))) / dz
                    + 3 * (rho_m[i - 1] * alpha_m[i - 1] * v_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])) / dz
                    + 3 * (rho_m[i] * alpha_m[i] * v_m[i - 1] * v_m[i - 1] * (1 - H(v_m[i - 1]))) / dz;

                add(L[i], 5, 0,
                    - (alpha_m[i - 1] * v_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])) / dz
                );

                add(L[i], 5, 2,
                    - (rho_m[i - 1] * v_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])) / dz
                );

                add(L[i], 5, 6,
                    - 2 * (rho_m[i - 1] * alpha_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])) / dz
                    - 2 * (rho_m[i] * alpha_m[i] * v_m[i] * (1 - H(v_m[i - 1]))) / dz
                );

                add(R[i], 5, 0,
                    + (alpha_m[i + 1] * v_m[i] * v_m[i] * (1 - H(v_m[i]))) / dz
                );

                add(R[i], 5, 2,
                    + (rho_m[i + 1] * v_m[i] * v_m[i] * (1 - H(v_m[i]))) / dz
                );

                add(R[i], 5, 4,
                    + (alpha_m[i] * H(v_m[i])) / dz
                    + (alpha_m[i + 1] * (1 - H(v_m[i]))) / dz
                );

                // Momentum liquid equation

                const double Fl = 8 * mu_l / (eps_v * (r_i - r_v) * (r_i - r_v));

                add(D[i], 6, 1,
                    + eps_v * (alpha_l_old[i] * v_l_old[i] / dt)
                    + eps_v * (alpha_l[i] * v_l[i] * v_l[i] * H(v_l[i])) / dz
                    - eps_v * (alpha_l[i] * v_l[i - 1] * v_l[i - 1] * (1 - H(v_l[i - 1]))) / dz
                );

                add(D[i], 6, 3,
                    + eps_v * (v_l_old[i] * rho_l_old[i] / dt)
                    + eps_v * (rho_l[i] * v_l[i] * v_l[i] * H(v_l[i])) / dz
                    - eps_v * (rho_l[i] * v_l[i - 1] * v_l[i - 1] * (1 - H(v_l[i - 1]))) / dz
                    - DPcap / dz);

                add(D[i], 6, 5,
                    - eps_v * (alpha_l[i] * H(v_l[i])) / dz
                    - eps_v * (alpha_l[i + 1] * (1 - H(v_l[i]))) / dz
                );

                add(D[i], 6, 7,
                    + eps_v * (alpha_l_old[i] * rho_l_old[i]) / dt
                    + 2 * eps_v * (rho_l[i] * alpha_l[i] * v_l[i] * H(v_l[i])) / dz
                    + 2 * eps_v * (rho_l[i + 1] * alpha_l[i + 1] * v_l[i + 1] * (1 - H(v_l[i]))) / dz
                    + (Fl * std::abs(v_l[i]) * H(v_l[i]))
                    + (Fl * std::abs(v_l[i]) * (1 - H(v_l[i])))
                );

                Q[i][6] =
                    + 3 * eps_v * (alpha_l_old[i] * rho_l_old[i] * v_l_old[i]) / dt
                    - 3 * eps_v * (rho_l[i] * alpha_l[i] * v_l[i] * v_l[i] * H(v_l[i])) / dz
                    - 3 * eps_v * (rho_l[i + 1] * alpha_l[i + 1] * v_l[i] * v_l[i] * (1 - H(v_l[i]))) / dz
                    + 3 * eps_v * (rho_l[i - 1] * alpha_l[i - 1] * v_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])) / dz
                    + 3 * eps_v * (rho_l[i] * alpha_l[i] * v_l[i - 1] * v_l[i - 1] * (1 - H(v_l[i - 1]))) / dz;

                add(L[i], 6, 1,
                    - eps_v * (alpha_l[i - 1] * v_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])) / dz
                );

                add(L[i], 6, 3,
                    - eps_v * (rho_l[i - 1] * v_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])) / dz
                );

                add(L[i], 6, 7,
                    - 2 * eps_v * (rho_l[i - 1] * alpha_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])) / dz
                    - 2 * eps_v * (rho_l[i] * alpha_l[i] * v_l[i] * (1 - H(v_l[i - 1]))) / dz
                );

                add(R[i], 6, 1,
                    + eps_v * (alpha_l[i + 1] * v_l[i] * v_l[i] * (1 - H(v_l[i]))) / dz
                );

                add(R[i], 6, 3,
                    + eps_v * (rho_l[i + 1] * v_l[i] * v_l[i] * (1 - H(v_l[i]))) / dz
                    + DPcap / dz
                );

                add(R[i], 6, 5,
                    + eps_v * (alpha_l[i] * H(v_l[i])) / dz
                    + eps_v * (alpha_l[i + 1] * (1 - H(v_l[i]))) / dz
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
                    -1.0 / Tc * (275.32 + 511.58 / (2 * std::sqrt(1 - T_l[i] / Tc)))
                );

                Q[i][8] = 219.0 + 275.32 * (1.0 - T_l[i] / Tc) + 511.58 * std::sqrt(1.0 - T_l[i] / Tc) + T_l[i] / Tc * (275.32 + 511.58 / (2 * std::sqrt(1.0 - T_l[i] / Tc)));

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

                mass_source[i] = C41 * p_m[i] + C42 * T_m[i] + C43 * T_l[i] + C44 * T_w[i] + C45;

                heat_source_wall_liquid_flux[i] = C66 * p_m[i] + C67 * T_m[i] + C68 * T_l[i] + C69 * T_w[i] + C70;
                heat_source_liquid_wall_flux[i] = C71 * p_m[i] + C72 * T_m[i] + C73 * T_l[i] + C74 * T_w[i] + C75;

                heat_source_vapor_liquid_phase[i] = C56 * p_m[i] + C57 * T_m[i] + C58 * T_l[i] + C59 * T_w[i] + C60;
                heat_source_liquid_vapor_phase[i] = C61 * p_m[i] + C62 * T_m[i] + C63 * T_l[i] + C64 * T_w[i] + C65;

                heat_source_vapor_liquid_flux[i] = C46 * p_m[i] + C47 * T_m[i] + C48 * T_l[i] + C49 * T_w[i] + C50;
                heat_source_liquid_vapor_flux[i] = C51 * p_m[i] + C52 * T_m[i] + C53 * T_l[i] + C54 * T_w[i] + C55;

                p_saturation[i] = Psat;
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

            solve_block_tridiag(L, D, R, Q, X_new);

            // Calculate Picard error
            double L1 = 0.0;
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < 11; ++j) {
                    double Aold = X_iter[i][j];
                    double Anew = X_new[i][j];

                    double denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                    double eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);

                    L1 += eps;
                }

            if (L1 < tolerance)
				break;   // Picard converged

            // Update Picard values
            X_iter = X_new;

        }

        X = X_new;

        const int output_every = 1000;

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

    time_output.close();
}
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
        return 4.53e-1 - 1.48e-4 * T;
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

struct SparseBlock {
    std::vector<int> row;
    std::vector<int> col;
    std::vector<double> val;
};

using DenseBlock = std::array<std::array<double, B>, B>;
using VecBlock = std::array<double, B>;

// ------------------------- Utility dense -------------------------

DenseBlock to_dense(const SparseBlock& S) {
    DenseBlock M{};
    for (std::size_t k = 0; k < S.val.size(); ++k) {
        int i = S.row[k];
        int j = S.col[k];
        M[i][j] = S.val[k];
    }
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
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < B; ++j) {
            double s = 0.0;
            for (int k = 0; k < B; ++k)
                s += A[i][k] * Bm[k][j];
            C[i][j] = s;
        }
}

void subtract_inplace(DenseBlock& A, const DenseBlock& Bm) {
    for (int i = 0; i < B; ++i)
        for (int j = 0; j < B; ++j)
            A[i][j] -= Bm[i][j];
}

// ------------------------- LU con pivoting -------------------------

void lu_factor(DenseBlock& A, std::array<int, B>& piv) {
    for (int i = 0; i < B; ++i)
        piv[i] = i;

    for (int k = 0; k < B; ++k) {
        // pivot
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
            throw std::runtime_error("LU: matrice singolare");

        if (p != k) {
            std::swap(piv[k], piv[p]);
            for (int j = 0; j < B; ++j)
                std::swap(A[k][j], A[p][j]);
        }

        // eliminazione
        for (int i = k + 1; i < B; ++i) {
            A[i][k] /= A[k][k];
            double lik = A[i][k];
            for (int j = k + 1; j < B; ++j)
                A[i][j] -= lik * A[k][j];
        }
    }
}

void lu_solve_vec(const DenseBlock& LU, const std::array<int, B>& piv,
    const double b_in[B], double x[B]) {
    // applica pivot a b
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

void lu_solve_mat(const DenseBlock& LU, const std::array<int, B>& piv,
    const DenseBlock& Bm, DenseBlock& X) {
    // risolve LU X = P B (colonna per colonna)
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

// L[i]: blocco sinistro (i>=1)
// D[i]: blocco diagonale
// R[i]: blocco destro (i<=Nx-2)
// Q[i]: termini noti di blocco (dimensione B)
// X[i]: soluzione per blocco i
void solve_block_tridiag(
    const std::vector<SparseBlock>& L,
    const std::vector<SparseBlock>& D,
    const std::vector<SparseBlock>& R,
    const std::vector<VecBlock>& Q,
    std::vector<VecBlock>& X) {
    const int Nx = static_cast<int>(D.size());
    if (Nx == 0)
        return;

    // copia dense dei blocchi
    std::vector<DenseBlock> Dd(Nx);
    std::vector<DenseBlock> Ld(Nx);
    std::vector<DenseBlock> Rd(Nx);

    for (int i = 0; i < Nx; ++i) {
        Dd[i] = to_dense(D[i]);
        if (i > 0)     Ld[i] = to_dense(L[i]);
        if (i < Nx - 1)  Rd[i] = to_dense(R[i]);
    }

    std::vector<VecBlock> Qm = Q;             // Q modificato durante forward
    X.assign(Nx, VecBlock{});                 // soluzione

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

    // ultimo blocco
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

    // blocchi precedenti
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
    const double Pr_t = 0.01;               ///< Prandtl turbulent number for sodium vapor [-]
    const double gamma = 1.66;              ///< TODO: MAKE THIS PROPERTY TEMPERATURE DEPENDENT Ratio between constant pressure specific heat and constant volume specific heat [-] 

    // Environmental boundary conditions
    const double h_conv = 10;               ///< Convective heat transfer coefficient for external heat removal [W/m^2/K]
    const double power = 1e3;               ///< Power at the evaporator side [W]
    const double T_env = 280.0;             ///< External environmental temperature [K]

    // Evaporation and condensation parameters
    const double eps_s = 1.0;                                           ///< Surface fraction of the wick available for phasic interface [-]
    const double sigma_e = 1.0;                                         ///< Evaporation accomodation coefficient [-]. 1 means optimal evaporation
    const double sigma_c = 1.0;                                         ///< Condensation accomodation coefficient [-]. 1 means optimal condensation
    double Omega = 1.0;

    const double Tc = 2509.46;
    double const eps_v = 1.0;

    // Geometric parameters
    const int N = 200;                                                         ///< Number of axial nodes [-]
    const double L = 0.982; 			                                        ///< Length of the heat pipe [m]
    const double dz = L / N;                                                    ///< Axial discretization step [m]
    const double evaporator_length = 0.502;                                     ///< Evaporator length [m]
    const double adiabatic_length = 0.188;                                      ///< Adiabatic length [m]
    const double condenser_length = 0.292;                                      ///< Condenser length [m]
    const double evaporator_nodes = std::floor(evaporator_length / dz);         ///< Number of evaporator nodes
    const double condenser_nodes = std::ceil(condenser_length / dz);            ///< Number of condenser nodes
    const double adiabatic_nodes = N - (evaporator_nodes + condenser_nodes);    ///< Number of adiabatic nodes
    const double r_o = 0.01335;                                             ///< Outer wall radius [m]
    const double r_i = 0.0112;                                          ///< Wall-wick interface radius [m]
    const double r_v = 0.01075;                                             ///< Vapor-wick interface radius [m]

    // Surfaces 
    const double A_w_outer = 2 * M_PI * r_o * dz;                                       ///< Wall radial area (at r_o) [m^2]
    const double A_w_cross = M_PI * (r_o * r_o - r_i * r_i);        ///< Wall cross-sectional area [m^2]
    const double A_x_interface = 2 * M_PI * r_i * dz;                               ///< Wick radial area (at r_i) [m^2]
    const double A_x_cross = M_PI * (r_i * r_i - r_v * r_v);        ///< Wick cross-sectional area [m^2]
    const double A_v_inner = 2 * M_PI * r_v * dz;                                       ///< Vapor radial area (at r_v) [m^2]
    const double A_v_cross = M_PI * r_v * r_v;                                      ///< Vapor cross-sectional area [m^2]

    // Time-stepping parameters
    double dt = 1e-7;                               ///< Initial time step [s] (then it is updated according to the limits)
    const int nSteps = 10000000;                          ///< Number of timesteps
    const double time_total = nSteps * dt;          ///< Total simulation time [s]

    // Wick permeability parameters
    const double K = 1e-4;                          ///< Permeability [m^2]
    const double CF = 0.0;                          ///< Forchheimer coefficient [1/m]

    // Mesh z positions
    std::vector<double> mesh(N, 0.0);
    for (int i = 0; i < N; ++i) mesh[i] = i * dz;

    // Node partition
    const int N_e = static_cast<int>(std::floor(evaporator_length / dz));   ///< Number of nodes of the evaporator region [-]
    const int N_c = static_cast<int>(std::ceil(condenser_length / dz));     ///< Number of nodes of the condenser region [-]
    const int N_a = N - (N_e + N_c);                                        ///< Number of nodes of the adiabadic region [-]

    const double T_full = 800.0;

    // Initialization of the initial temperatures using the extremes in a linear distribution
    std::vector<double> T_o_w(N, T_full);
    std::vector<double> T_w_bulk(N, T_full);
    std::vector<double> T_w_x(N, T_full);
    std::vector<double> T_x_bulk(N, T_full);
    std::vector<double> T_x_v(N, T_full);
    std::vector<double> T_v_bulk(N, T_full);

    const double q_pp = power / (2 * M_PI * L * r_o);     ///< Heat flux at evaporator from given power [W/m^2]

    // Mass sources/fluxes
    std::vector<double> phi_x_v(N, 0.0);            ///< Mass flux [kg/m2/s] at the wick-vapor interface (positive if evaporation)

    // Create result folder
    std::filesystem::create_directories("results");

    // Print results in file
    std::ofstream mesh_io("mesh.txt", std::ios::trunc);

    std::ofstream v_velocity_output("results/vapor_velocity.txt", std::ios::trunc);
    std::ofstream v_pressure_output("results/vapor_pressure.txt", std::ios::trunc);
    std::ofstream v_bulk_temperature_output("results/vapor_bulk_temperature.txt", std::ios::trunc);

    std::ofstream x_velocity_output("results/wick_velocity.txt", std::ios::trunc);
    std::ofstream x_pressure_output("results/wick_pressure.txt", std::ios::trunc);
    std::ofstream x_bulk_temperature_output("results/wick_bulk_temperature.txt", std::ios::trunc);

    std::ofstream x_v_temperature_output("results/wick_vapor_interface_temperature.txt", std::ios::trunc);
    std::ofstream w_x_temperature_output("results/wall_wick_interface_temperature.txt", std::ios::trunc);
    std::ofstream o_w_temperature_output("results/outer_wall_temperature.txt", std::ios::trunc);
    std::ofstream w_bulk_temperature_output("results/wall_bulk_temperature.txt", std::ios::trunc);

    std::ofstream x_v_mass_flux_output("results/wick_vapor_mass_source.txt", std::ios::trunc);

    std::ofstream o_w_heat_flux_output("results/outer_wall_heat_flux.txt", std::ios::trunc);
    std::ofstream w_x_heat_flux_output("results/wall_wick_heat_flux.txt", std::ios::trunc);
    std::ofstream x_v_heat_flux_output("results/wick_vapor_heat_flux.txt", std::ios::trunc);

    std::ofstream rho_output("results/rho_vapor.txt", std::ios::trunc);

    mesh_io << std::setprecision(8);

    v_velocity_output << std::setprecision(8);
    v_pressure_output << std::setprecision(8);
    v_bulk_temperature_output << std::setprecision(8);

    x_velocity_output << std::setprecision(8);
    x_pressure_output << std::setprecision(8);
    x_bulk_temperature_output << std::setprecision(8);

    x_v_temperature_output << std::setprecision(8);
    w_x_temperature_output << std::setprecision(8);
    o_w_temperature_output << std::setprecision(8);
    w_bulk_temperature_output << std::setprecision(8);

    x_v_mass_flux_output << std::setprecision(8);

    o_w_heat_flux_output << std::setprecision(8);
    w_x_heat_flux_output << std::setprecision(8);
    x_v_heat_flux_output << std::setprecision(8);

    rho_output << std::setprecision(8);

    for (int i = 0; i < N; ++i) {
        mesh_io << i * dz << " ";
    }

    mesh_io.flush();

    #pragma endregion

    std::vector<double> rho_m(N, 0.0);
    std::vector<double> rho_l(N, 0.0);
    std::vector<double> alpha_m(N, 0.0);
    std::vector<double> alpha_l(N, 0.0);
    std::vector<double> p_m(N, 0.0);
    std::vector<double> p_l(N, 0.0);
    std::vector<double> v_m(N + 1, 0.0);
    std::vector<double> v_l(N + 1, 0.0);
    std::vector<double> T_m(N, 0.0);
    std::vector<double> T_l(N, 0.0);
    std::vector<double> T_w(N, 0.0);

    std::vector<double> Gamma_xv(N, 0.0);

    std::vector<SparseBlock> L(N), D(N), R(N);
    std::vector<VecBlock> Q(N), X;

    for(int i = 1; i <= N - 2; ++i){

        const double k_w = steel::k(T_w_bulk[i]);                           ///< Wall thermal conductivity [W/(m K)]
        const double k_x = liquid_sodium::k(T_x_bulk[i]);                   ///< Liquid thermal conductivity [W/(m K)]
        const double k_m = vapor_sodium::k(T_m[i], p_m[i]);                 ///< Vapor thermal conductivity [W/(m K)]
        const double cp_m = vapor_sodium::cp(T_m[i]);                       ///< Vapor specific heat [J/(kg K)]
        const double mu_v = vapor_sodium::mu(T_v_bulk[i]);                  ///< Vapor dynamic viscosity [Pa*s]
        const double mu_l = liquid_sodium::mu(T_x_bulk[i]);                 ///< Liquid dynamic viscosity
        const double Dh_v = 2.0 * r_v;                                      ///< Hydraulic diameter of the vapor core [m]
        const double Re_v = rho_m[i] * std::fabs(v_m[i]) * Dh_v / mu_v;     ///< Reynolds number [-]
        const double Pr_v = cp_m * mu_v / k_m;                              ///< Prandtl number [-]
        const double H_xm = vapor_sodium::h_conv(Re_v, Pr_v, k_m, Dh_v);    ///< Convective heat transfer coefficient at the vapor-wick interface [W/m^2/K]
        const double Psat = vapor_sodium::P_sat(T_x_v[i]);                  ///< Saturation pressure [Pa]         
        const double dPsat_dT = 
            Psat * std::log(10.0) * (7740.0 / (T_x_v[i] * T_x_v[i]));       ///< Derivative of the saturation pressure wrt T [Pa/K]   
        const double beta = 1.0 / std::sqrt();                  
        const double fac = (2.0 * r_v * eps_s * beta) / (r_i * r_i);        ///< Useful factor in the coefficients calculation [s / m^2]
        const double b = std::abs(-phi_x_v[i] / (p_m[i] * std::sqrt(2.0 / (Rv * T_v_bulk[i]))));

        if (b < 0.1192) Omega = 1.0 + b * std::sqrt(M_PI);
        else if (b <= 0.9962) Omega = 0.8959 + 2.6457 * b;
        else Omega = 2.0 * b * std::sqrt(M_PI);

        double h_xv_v;      ///< Specific enthalpy [J/kg] of vapor upon phase change between wick and vapor
        double h_vx_x;      ///< Specific enthalpy [J/kg] of wick upon phase change between vapor and wick

        if (phi_x_v[i] >= 0.0) {

            // Evaporation case
            h_xv_v = vapor_sodium::h(T_x_v[i]);
            h_vx_x = liquid_sodium::h(T_x_v[i]);

        }
        else {

            // Condensation case
            h_xv_v = vapor_sodium::h(T_v_bulk[i]);
            h_vx_x = liquid_sodium::h(T_x_v[i])
                + (vapor_sodium::h(T_v_bulk[i]) - vapor_sodium::h(T_x_v[i]));
        }

        const double bGamma = -(Gamma_xv[i] / (2.0 * T_x_v[i])) + fac * sigma_e * dPsat_dT;   ///< b coefficient [kg/(m3 s K)] 
        const double aGamma = 0.5 * Gamma_xv[i] + fac * sigma_e * dPsat_dT * T_x_v[i];        ///< a coefficient [kg/(m3 s)]
        const double cGamma = -fac * sigma_c * Omega;                                         ///< c coefficient [s/m2]

        const double Eio1 = 2.0 / 3.0 * (r_o + r_i - 1 / (1 / r_o + 1 / r_i));
        const double Eio2 = 0.5 * (r_o * r_o + r_i * r_i);
        const double Evi1 = 2.0 / 3.0 * (r_i + r_v - 1 / (1 / r_i + 1 / r_v));
        const double Evi2 = 0.5 * (r_i * r_i + r_v * r_v);

        const double Ex3 = H_xm + (h_vx_x * r_i * r_i) / (2.0 * r_v) * bGamma;
        const double Ex4 =
            -k_x +
            H_xm * r_v +
            h_vx_x * r_i * r_i / 2.0 * bGamma;
        const double Ex5 =
            -2.0 * r_v * k_x +
            H_xm * r_v * r_v +
            h_vx_x * r_i * r_i / 2.0 * bGamma * r_v;
        const double Ex6 = -H_xm;
        const double Ex7 = (h_vx_x * r_i * r_i) / (2.0 * r_v) * cGamma;
        const double Ex8 = (h_vx_x * r_i * r_i) / (2.0 * r_v) * aGamma;

        const double alpha = 1.0 / (2 * r_o * (Eio1 - r_i) + r_i * r_i - Eio2);
        const double gamma = r_i * r_i + ((Ex5 - Evi2 * Ex3) * (Evi1 - r_i)) / (Ex4 - Evi1 * Ex3) - Evi2;

        const double C1 = 2 * k_w * (r_o - r_i) * alpha;
        const double C2 = (Evi1 - r_i) / (Ex4 - Evi1 * Ex3);
        const double C3 = C1 + C2 * Ex3;
        const double C4 = -C1;
        const double C5 = -C2 * Ex6 - C2 * Ex7 * rho_m[i] * Rv;
        const double C6 = -C2 * Ex7 * Rv * T_m[i];
        const double C7 = -q_pp + C1 * q_pp * (Eio1 - r_i) - C2 * Ex8 + C2 * Ex7 * rho_m[i] * T_m[i] * Rv;
        const double C8 = alpha + C2 * Ex3 + alpha * gamma * C3;
        const double C9 = -alpha + alpha * gamma * C4;
        const double C10 = -C2 * Ex6 - C2 * Ex7 * Rv * rho_m[i] + alpha * gamma * C5;
        const double C11 = -C2 * Ex7 * T_m[i] * Rv + alpha * gamma * C6; 
        const double C12 = alpha * q_pp / k_w * (Eio1 - r_i) - C2 * Ex8 + 2 * C2 * Ex7 * T_m[i] * Rv * rho_m[i] + alpha * gamma * C7;
        const double C13 = -2 * r_o * C8;
        const double C14 = -2 * r_o * C9;
        const double C15 = -2 * r_o * C10;
        const double C16 = -2 * r_o * C11;
        const double C17 = -2 * r_o * C17 + q_pp / k_w;
        const double C18 = 2 * r_o * Eio1 - Eio2;
        const double C19 = C18 * C8;
        const double C20 = C18 * C9 + 1;
        const double C21 = C18 * C10;
        const double C22 = C18 * C11;
        const double C23 = C18 * C12 - Eio1 * q_pp / k_w;
        const double C24 = 1.0 / (Ex4 - Evi1 * Ex3);
        const double C25 = Ex5 - Evi2 * Ex3;
        const double C26 = C24 * Ex3 - C25 * C24 * C3;
        const double C27 = -C25 * C24 * C4;
        const double C28 = C24 * Ex6 + C24 * Ex7 * Rv * rho_m[i] - C25 * C24 * C5;
        const double C29 = C24 * Ex7 * Rv * T_m[i] - C25 * C24 * C6;
        const double C30 = C24 * Ex8 - 2 * C24 * Ex7 * Rv * T_m[i] * rho_m[i] - C25 * C24 * C7;
        const double C31 = 1 - Evi1 * C26 - Evi2 * C3;
        const double C32 = -Evi1 * C27 - Evi2 * C4;
        const double C33 = -Evi1 * C28 - Evi2 * C5;
        const double C34 = -Evi1 * C29 - Evi2 * C6;
        const double C35 = -Evi1 * C30 - Evi2 * C7;
        const double C36 = bGamma * C31 + bGamma * r_v * C26 + bGamma * r_v * r_v * C3;
        const double C37 = bGamma * C32 + bGamma * r_v * C27 + bGamma * r_v * r_v * C4;
        const double C38 = bGamma * C33 + bGamma * r_v * C28 + bGamma * r_v * r_v * C5 + cGamma * Rv * rho_m[i];
        const double C39 = bGamma * C34 + bGamma * r_v * C29 + bGamma * r_v * r_v * C6 + cGamma * Rv * T_m[i];
        const double C40 = aGamma + C35 + bGamma * r_v * C30 + bGamma * r_v * r_v * C7 - 2 * cGamma * Rv * T_m[i] * rho_m[i];
        const double C41 = 2 * H_xm * r_v / (r_i * r_i);
        const double C42 = C41 * C31 + C41 * C26 * r_v + C41 * C3 * r_v * r_v;
        const double C43 = C41 * C32 + C41 * C27 * r_v + C41 * C4 * r_v * r_v;
        const double C44 = C41 * C33 + C41 * C28 * r_v + C41 * C5 * r_v * r_v - C41;
        const double C45 = C41 * C34 + C41 * C29 * r_v + C41 * C6 * r_v * r_v;
        const double C46 = C41 * C35 + C41 * C30 * r_v + C41 * C7 * r_v * r_v;
        const double C47 = -k_x * 2 * r_v / (r_i * r_i);
        const double C48 = C47 * C26 + 2 * r_v * C47 * C3;
        const double C49 = C47 * C27 + 2 * r_v * C47 * C4;
        const double C50 = C47 * C28 + 2 * r_v * C47 * C5;
        const double C51 = C47 * C29 + 2 * r_v * C47 * C6;
        const double C52 = C47 * C30 + 2 * r_v * C47 * C7;
        const double C53 = k_w * 2 / r_i;
        const double C54 = C53 * C13 + 2 * r_i * C53 * C8;
        const double C55 = C53 * C14 + 2 * r_i * C53 * C9;
        const double C56 = C53 * C15 + 2 * r_i * C53 * C10;
        const double C57 = C53 * C16 + 2 * r_i * C53 * C11;
        const double C58 = C53 * C17 + 2 * r_i * C53 * C12;
        const double C59 = 219.0;
        const double C60 = 275.32;
        const double C61 = 511.58;
        const double C62 = -1.0 / Tc * (C60 + C61 / (2 * std::sqrt(1 - T_l[i] / Tc)));
        const double C63 = C59 + C60 * (1.0 - T_l[i] / Tc) + C61 * std::sqrt(1.0 - T_l[i] / Tc) + T_l[i] / Tc * (C60 + C61 / (2 * std::sqrt(1.0 - T_l[i] / Tc)));
        const double C64 = -k_w * 2 * r_i / (r_o * r_o - r_i * r_i);

        // DPcap evaluation

        const double alpha_m0 = r_v * r_v / (r_i * r_i);
        const double r_p = 1e-5;
        const double surf_ten_value = liquid_sodium::surf_ten(T_l[i]);                                /// TOCHANGEEEEE

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
            alpha_m[i] / dt
            + (alpha_m[i] * v_m[i] * H(v_m[i])
            - alpha_m[i - 1] * v_m[i - 1] * (1 - H(v_m[i - 1]))) / dz
            - C39);

        add(D[i], 0, 2,
            rho_m[i] / dt
            + (rho_m[i] * v_m[i] * H(v_m[i])
            - rho_m[i] * v_m[i - 1] * (1 - H(v_m[i - 1]))) / dz);

        add(D[i], 0, 6,
            (alpha_m[i] * rho_m[i] * H(v_m[i])
            + alpha_m[i + 1] * rho_m[i + 1] * (1 - H(v_m[i]))) / dz);

        add(D[i], 0, 8, -C38);
        add(D[i], 0, 9, -C36);
        add(D[i], 0, 10, -C37);

        Q[i][0] = C40  
                    + 2 * ( + alpha_m[i] * rho_m[i] * v_m[i] * H(v_m[i])
                            + alpha_m[i + 1] * rho_m[i + 1] * v_m[i] * (1 - H(v_m[i]))
                            - alpha_m[i - 1] * rho_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])
                            - alpha_m[i] * rho_m[i] * v_m[i - 1] * (1 - H(v_m[i - 1]))) / dz
                    + 2 * (rho_m[i] * alpha_m[i]) / dt;

        add(L[i], 0, 2,
            -(rho_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])) / dz);

        add(L[i], 0, 4,
            -(alpha_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])) / dz);

        add(L[i], 0, 6,
            -(alpha_m[i - 1] * rho_m[i - 1] * H(v_m[i - 1])
            + alpha_m[i] * rho_m[i] * (1 - H(v_m[i - 1]))) / dz);

        add(R[i], 0, 0,
            (alpha_m[i + 1] * v_m[i] * (1 - H(v_m[i]))) / dz);

        add(R[i], 0, 2,
            (rho_m[i + 1] * v_m[i] * (1 - H(v_m[i]))) / dz);

        // Mass liquid equation

        add(D[i], 1, 0, +C39);

        add(D[i], 1, 1,
            alpha_l[i] / dt
            + (alpha_l[i] * v_l[i] * H(v_l[i])
            - alpha_l[i - 1] * v_l[i - 1] * (1 - H(v_l[i - 1]))) / dz);

        add(D[i], 1, 3,
            rho_l[i] / dt
            + (rho_l[i] * v_l[i] * H(v_l[i])
            - rho_l[i] * v_l[i - 1] * (1 - H(v_l[i - 1]))) / dz);

        add(D[i], 1, 7,
            (alpha_l[i] * rho_l[i] * H(v_l[i])
            + alpha_l[i + 1] * rho_l[i + 1] * (1 - H(v_l[i]))) / dz);

        add(D[i], 1, 8, +C38);
        add(D[i], 1, 9, +C36);
        add(D[i], 1, 10, +C37);


        Q[i][1] = -C40
            + 2 * (+alpha_l[i] * rho_l[i] * v_l[i] * H(v_l[i])
                + alpha_l[i + 1] * rho_l[i + 1] * v_l[i] * (1 - H(v_l[i]))
                - alpha_l[i - 1] * rho_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])
                - alpha_l[i] * rho_l[i] * v_l[i - 1] * (1 - H(v_l[i - 1]))) / dz
            + 2 * (rho_l[i] * alpha_l[i]) / dt;

        add(L[i], 1, 3,
            -(rho_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])) / dz);

        add(L[i], 1, 5,
            -(alpha_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])) / dz);

        add(L[i], 1, 7,
            -(alpha_l[i - 1] * rho_l[i - 1] * H(v_l[i - 1])
            + alpha_l[i] * rho_l[i] * (1 - H(v_l[i - 1]))) / dz);

        add(R[i], 1, 1,
            (alpha_l[i + 1] * v_l[i] * (1 - H(v_l[i]))) / dz);

        add(R[i], 1, 3,
            (rho_l[i + 1] * v_l[i] * (1 - H(v_l[i]))) / dz);

        // Mixture heat equation

        const double cp_m_p = vapor_sodium::cp(T_m[i]);
        const double cp_m_l = vapor_sodium::cp(T_m[i - 1]);
        const double cp_m_r = vapor_sodium::cp(T_m[i + 1]);

        const double k_m_p = vapor_sodium::k(T_m[i], p_m[i]);
        const double k_m_l = vapor_sodium::k(T_m[i - 1], p_m[i - 1]);
        const double k_m_r = vapor_sodium::k(T_m[i + 1], p_m[i + 1]);

        add(D[i], 2, 0,
            (alpha_m[i] * T_m[i] * cp_m_p) / dt
            + (alpha_m[i] * cp_m_p * T_m[i] * v_m[i] * H(v_m[i])) / dz
            - (alpha_m[i] * cp_m_p * T_m[i] * v_m[i] * (1 - H(v_m[i - 1]))) / dz
            + p_m[i] / dt
            - C45 - h_xv_v * C39);

        add(D[i], 2, 2,
            (T_m[i] * rho_m[i] * cp_m_p) / dt
            + (rho_m[i] * cp_m_p * T_m[i] * v_m[i] * H(v_m[i])) / dz
            - (rho_m[i] * cp_m_p * T_m[i] * v_m[i] * (1 - H(v_m[i - 1]))) / dz
            + p_m[i] * (v_m[i] * H(v_m[i]) + v_m[i - 1] * H(v_m[i - 1])) / dz);

        add(D[i], 2, 6,
            (T_m[i] * cp_m_p* alpha_m[i] * rho_m[i]) / dt
            + (alpha_m[i] * rho_m[i] * cp_m_p * T_m[i] * H(v_m[i]) + alpha_m[i + 1] * rho_m[i + 1] * cp_m_r * T_m[i + 1] * (1 - H(v_m[i]))) / dz
            + p_m[i] * (alpha_m[i] * H(v_m[i]) + alpha_m[i + 1] * (1 - H(v_m[i]))) / dz);

        add(D[i], 2, 8,
            + (alpha_m[i] * rho_m[i] * cp_m_p * v_m[i] * H(v_m[i])) / dz
            - (alpha_m[i] * rho_m[i] * cp_m_p * v_m[i] * (1 - H(v_m[i - 1]))) / dz
            + (alpha_m[i] * k_m_p * H(v_m[i]) + alpha_m[i + 1] * k_m_r * (1 - H(v_m[i]))) / (dz * dz)
            + (alpha_m[i - 1] * k_m_l * H(v_m[i - 1]) + alpha_m[i] * k_m_p * (1 - H(v_m[i - 1]))) / (dz * dz)
            - C44 - h_xv_v * C38);

        add(D[i], 2, 9, - C42 - h_vx_x * C36);
        add(D[i], 2, 10, -C43 - h_xv_v * C37);

        Q_i[2] = 3 * (alpha_m[i] * T_m[i] * rho_m[i]) / dt 
            + 3 * ( 
                + alpha_m[i] * rho_m[i] * cp_m_p * T_m[i] * v_m[i] * H(v_m[i]) 
                + alpha_m[i + 1] * rho_m[i + 1] * cp_m_r * T_m[i + 1] * v_m[i + 1] * (1 - H(v_m[i]))
                - alpha_m[i - 1] * rho_m[i - 1] * cp_m_l * T_m[i - 1] * v_m[i - 1] * H(v_m[i - 1]) 
                - alpha_m[i] * rho_m[i] * cp_m_p * T_m[i] * v_m[i] * (1 - H(v_m[i - 1]))
                ) / dz
            + p_m[i] * (
                + alpha_m[i] * v_m[i] * H(v_m[i])
                + alpha_m[i + 1] * v_m[i] * (1 - H(v_m[i]))
                - alpha_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])
                - alpha_m[i] * v_m[i - 1] * (1 - H(v_m[i - 1]))
                    ) / dz
            + p_m[i] * alpha_m[i] / dt
            + C46 + h_xv_v * C40;

        add(L[i], 2, 0,
            -(alpha_m[i - 1] * cp_m_l * T_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])) / dz);

        add(L[i], 2, 2,
            (rho_m[i - 1] * cp_m_l * T_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])) / dz
            + p_m[i] * (v_m[i - 1] * H(v_m[i - 1])) / dz);

        add(L[i], 2, 6,
            -(
                + alpha_m[i - 1] * rho_m[i - 1] * cp_m_l * T_m[i - 1] * H(v_m[i - 1]) 
                + alpha_m[i] * rho_m[i] * cp_m_l * T_m[i] * (1 - H(v_m[i - 1]))
                ) / dz
            - p_m[i] * (
                + alpha_m[i - 1] * H(v_m[i - 1]) 
                + alpha_m[i] * (1 - H(v_m[i - 1]))
                ) / dz);

        add(L[i], 2, 8,
            -(alpha_m[i - 1] * rho_m[i - 1] * cp_m_l * v_m[i - 1] * H(v_m[i - 1])) / dz
            - (alpha_m[i - 1] * k_m_l * H(v_m[i - 1]) + alpha_m[i] * k_m_p * (1 - H(v_m[i - 1]))) / (dz * dz));


        add(R[i], 2, 0,
            (alpha_m[i + 1] * cp_m_r * T_m[i + 1] * v_m[i] * (1 - H(v_m[i]))) / dz);

        add(R[i], 2, 2,
            (rho_m[i + 1] * cp_m_r * T_m[i + 1] * v_m[i] * (1 - H(v_m[i]))) / dz
            + p_m[i] * (v_m[i] * (1 - H(v_m[i]))) / dz);

        add(R[i], 2, 8,
            (alpha_m[i + 1] * rho_m[i + 1] * cp_m_r * v_m[i] * (1 - H(v_m[i]))) / dz
            - (alpha_m[i] * k_m_p * H(v_m[i]) + alpha_m[i + 1] * k_m_r * (1 - H(v_m[i]))) / (dz * dz));

        // Heat liquid equation

        const double cp_l_p = liquid_sodium::cp(T_l[i]);
        const double cp_l_l = liquid_sodium::cp(T_l[i - 1]);
        const double cp_l_r = liquid_sodium::cp(T_l[i + 1]);

        const double k_l_p = liquid_sodium::k(T_l[i]);
        const double k_l_l = liquid_sodium::k(T_l[i - 1]);
        const double k_l_r = liquid_sodium::k(T_l[i + 1]);

        add(D[i], 3, 0, -C57 - h_xv_v * C39 - C51);

        add(D[i], 3, 1,
            eps_v * 
                (alpha_l[i] * T_l[i] * cp_l_p) / dt
                + (alpha_l[i] * cp_l_p * T_l[i] * v_l[i] * H(v_l[i])) / dz
                - (alpha_l[i] * cp_l_p * T_l[i] * v_l[i] * (1 - H(v_l[i - 1]))) / dz
                + p_l[i] / dt);

        add(D[i], 3, 3,
            eps_v * (
                T_l[i] * rho_l[i] * cp_l_p) / dt
                + (rho_l[i] * cp_l_p * T_l[i] * v_l[i] * H(v_l[i])) / dz
                - (rho_l[i] * cp_l_p * T_l[i] * v_l[i] * (1 - H(v_l[i - 1]))) / dz
                + p_l[i] * (v_l[i] * H(v_l[i]) + v_l[i - 1] * H(v_l[i - 1])) / dz);

        add(D[i], 3, 7, 
            eps_v* (
                T_l[i] * cp_l_p * alpha_l[i] * rho_l[i]) / dt
                + (alpha_l[i] * rho_l[i] * cp_l_p * T_l[i] * H(v_l[i]) + alpha_l[i + 1] * rho_l[i + 1] * cp_l_r * T_l[i + 1] * (1 - H(v_l[i]))) / dz
                + p_l[i] * (alpha_l[i] * H(v_l[i]) + alpha_l[i + 1] * (1 - H(v_l[i]))) / dz);

        add(D[i], 3, 8, 
            -C56 - h_vx_x * C38 - C50);

        add(D[i], 3, 9,
            eps_v * (
                + alpha_l[i] * rho_l[i] * cp_l_p * v_l[i] * H(v_l[i])) / dz
                - (alpha_l[i] * rho_l[i] * cp_l_p * v_l[i] * (1 - H(v_l[i - 1]))) / dz
                + (alpha_l[i] * k_l_p * H(v_l[i]) + alpha_l[i + 1] * k_l_r * (1 - H(v_l[i]))) / (dz * dz)
                + (alpha_l[i - 1] * k_l_l * H(v_l[i - 1]) + alpha_l[i] * k_l_p * (1 - H(v_l[i - 1]))) / (dz * dz)
                - C44 - h_xv_v * C38
            - C54 - C48 - h_vx_x * C36);

        add(D[i], 3, 10, 
            -C55 - h_vx_x * C37 - C49);

        Q[i][2] = 
            eps_v * (
                + 3 * (alpha_l[i] * T_l[i] * rho_l[i]) / dt
                + 3 * (
                    +alpha_l[i] * rho_l[i] * cp_l_p * T_l[i] * v_l[i] * H(v_l[i])
                    + alpha_l[i + 1] * rho_l[i + 1] * cp_l_r * T_l[i + 1] * v_l[i + 1] * (1 - H(v_l[i]))
                    - alpha_l[i - 1] * rho_l[i - 1] * cp_l_l * T_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])
                    - alpha_l[i] * rho_l[i] * cp_l_p * T_l[i] * v_l[i] * (1 - H(v_l[i - 1]))
                    ) / dz
                + p_l[i] * (
                    +alpha_l[i] * v_l[i] * H(v_l[i])
                    + alpha_l[i + 1] * v_l[i] * (1 - H(v_l[i]))
                    - alpha_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])
                    - alpha_l[i] * v_l[i - 1] * (1 - H(v_l[i - 1]))
                    ) / dz
                + p_l[i] * alpha_l[i] / dt)
            + C52 + C58 + h_vx_x * C40;

        add(L[i], 3, 1,
            eps_v * (
            -(alpha_l[i - 1] * cp_l_l * T_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])) / dz));

        add(L[i], 3, 3,
            eps_v * (
                (rho_l[i - 1] * cp_l_l* T_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])) / dz
                + p_l[i] * (v_l[i - 1] * H(v_l[i - 1])) / dz));

        add(L[i], 3, 7,
            eps_v * (
                -(
                    +alpha_l[i - 1] * rho_l[i - 1] * cp_l_l * T_l[i - 1] * H(v_l[i - 1])
                    + alpha_l[i] * rho_l[i] * cp_l_l * T_l[i] * (1 - H(v_l[i - 1]))
                ) / dz
            - p_l[i] * (
                +alpha_l[i - 1] * H(v_l[i - 1])
                + alpha_l[i] * (1 - H(v_l[i - 1]))
                ) / dz));

        add(L[i], 3, 9,
            eps_v * (
            -(alpha_l[i - 1] * rho_l[i - 1] * cp_l_l * v_l[i - 1] * H(v_l[i - 1])) / dz
            - (alpha_l[i - 1] * k_l_l * H(v_l[i - 1]) + alpha_l[i] * k_l_p * (1 - H(v_l[i - 1]))) / (dz * dz)));

        add(R[i], 3, 1,
            eps_v * (
            (alpha_l[i + 1] * cp_l_r* T_l[i + 1] * v_l[i] * (1 - H(v_l[i]))) / dz));

        add(R[i], 3, 3,
            eps_v * (
            (rho_l[i + 1] * cp_l_r* T_l[i + 1] * v_l[i] * (1 - H(v_l[i]))) / dz
            + p_l[i] * (v_l[i] * (1 - H(v_l[i]))) / dz));

        add(R[i], 3, 9,
            eps_v * (
                (alpha_l[i + 1] * rho_l[i + 1] * cp_l_r* v_l[i] * (1 - H(v_l[i]))) / dz
                - (alpha_l[i] * k_l_p * H(v_l[i]) + alpha_l[i + 1] * k_l_r * (1 - H(v_l[i]))) / (dz * dz)));

        // Heat wall equation

        const double rho_w_p = steel::rho(T_w[i]);
        const double rho_w_l = steel::rho(T_w[i - 1]);
        const double rho_w_r = steel::rho(T_w[i + 1]);

        const double cp_w_p = steel::rho(T_w[i]);
        const double cp_w_l = steel::rho(T_w[i - 1]);
        const double cp_w_r = steel::rho(T_w[i + 1]);

        const double k_w_p = steel::rho(T_w[i]);
        const double k_w_l = steel::rho(T_w[i - 1]);
        const double k_w_r = steel::rho(T_w[i + 1]);

        const double k_w_lf = 0.5 * (k_w_l + k_w_p);
        const double k_w_rf = 0.5 * (k_w_r + k_w_p);

        add(D[i], 4, 0, -C57 * C64 / C53);

        add(D[i], 4, 8, C56* C64 / C53);

        add(D[i], 4, 9, -C54 * C64 / C53);

        add(D[i], 4, 10,
            (rho_w_p* cp_w_p) / dt
            - (C55 / C53) * C64
            + (k_w_lf + k_w_rf) / (dz * dz));

        Q[i][4] =
            q_pp * 2 * r_o / (r_o * r_o - r_i * r_i)
            + (rho_w_p * cp_w_p * T_w[i]) / dt
            + C58 * C64 / C53;

        add(L[i], 4, 10,
            -k_w_lf / (dz * dz));

        add(R[i], 4, 10,
            -k_w_rf / (dz * dz));

        // Momentum mixture equation

        add(D[i], 5, 0,
            +(alpha_m[i] * v_m[i]) / dt
            + (alpha_m[i] * v_m[i] * v_m[i] * H(v_m[i])) / dz
            - (alpha_m[i] * v_m[i - 1] * v_m[i - 1] * (1 - H(v_m[i - 1]))) / dz);

        add(D[i], 5, 2,
            +(v_m[i] * rho_m[i]) / dt
            + (rho_m[i] * v_m[i] * v_m[i] * H(v_m[i])) / dz
            - (rho_m[i] * v_m[i - 1] * v_m[i - 1] * (1 - H(v_m[i - 1]))) / dz);

        add(D[i], 5, 4,
            -(alpha_m[i] * H(v_m[i])) / dz
            - (alpha_m[i] * (1 - H(v_m[i]))) / dz);

        add(D[i], 5, 6,
            +(alpha_m[i] * rho_m[i]) / dt
            + 2 * (rho_m[i] * alpha_m[i] * v_m[i] * H(v_m[i])) / dz
            + 2 * (rho_m[i + 1] * alpha_m[i + 1] * v_m[i + 1] * (1 - H(v_m[i]))) / dz
            + (fm * rho_m[i] * std::abs(v_m[i]) * H(v_m[i])) / (4 * r_v)
            + (fm * rho_m[i + 1] * std::abs(v_m[i]) * (1 - H(v_m[i]))) / (4 * r_v));

        Q[i][5] = 
            + 3 * (alpha_m[i] * rho_m[i] * v_m[i]) / dt
            - 3 * (rho_m[i] * alpha_m[i] * v_m[i] * v_m[i] * H(v_m[i])) / dz
            - 3 * (rho_m[i + 1] * alpha_m[i + 1] * v_m[i] * v_m[i] * (1 - H(v_m[i]))) / dz
            + 3 * (rho_m[i - 1] * alpha_m[i - 1] * v_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])) / dz
            + 3 * (rho_m[i] * alpha_m[i] * v_m[i - 1] * v_m[i - 1] * (1 - H(v_m[i - 1]))) / dz;

        add(L[i], 5, 0,
            -(alpha_m[i - 1] * v_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])) / dz);

        add(L[i], 5, 2,
            -(rho_m[i - 1] * v_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])) / dz);

        add(L[i], 5, 6,
            -2 * (rho_m[i - 1] * alpha_m[i - 1] * v_m[i - 1] * H(v_m[i - 1])) / dz
            - 2 * (rho_m[i] * alpha_m[i] * v_m[i] * (1 - H(v_m[i - 1]))) / dz);

        add(R[i], 5, 0,
            + (alpha_m[i + 1] * v_m[i] * v_m[i] * (1 - H(v_m[i]))) / dz);

        add(R[i], 5, 2,
            + (rho_m[i + 1] * v_m[i] * v_m[i] * (1 - H(v_m[i]))) / dz);

        add(R[i], 5, 4,
            + (alpha_m[i] * H(v_m[i])) / dz
            + (alpha_m[i] * (1 - H(v_m[i]))) / dz);

        // Momentum liquid equation

        add(D[i], 6, 1,
            + eps_v * (alpha_l[i] * v_l[i]) / dt
            + eps_v * (alpha_l[i] * v_l[i] * v_l[i] * H(v_l[i])) / dz
            - eps_v * (alpha_l[i] * v_l[i - 1] * v_l[i - 1] * (1 - H(v_l[i - 1]))) / dz);

        add(D[i], 6, 3,
            + eps_v * (v_l[i] * rho_l[i]) / dt
            + eps_v * (rho_l[i] * v_l[i] * v_l[i] * H(v_l[i])) / dz
            - eps_v * (rho_l[i] * v_l[i - 1] * v_l[i - 1] * (1 - H(v_l[i - 1]))) / dz)
            - DPcap / dz;

        add(D[i], 5, 5,
            - eps_v * (alpha_l[i] * H(v_l[i])) / dz
            - eps_v * (alpha_l[i] * (1 - H(v_l[i]))) / dz);

        add(D[i], 6, 7,
            + eps_v * (alpha_l[i] * rho_l[i]) / dt
            + 2 * eps_v * (rho_l[i] * alpha_l[i] * v_l[i] * H(v_l[i])) / dz
            + 2 * eps_v * (rho_l[i + 1] * alpha_l[i + 1] * v_l[i + 1] * (1 - H(v_l[i]))) / dz
            + (fl * rho_l[i] * std::abs(v_l[i]) * H(v_l[i])) / (4 * r_v)
            + (fl * rho_l[i + 1] * std::abs(v_l[i]) * (1 - H(v_l[i]))) / (4 * r_v));

        Q[i][6] =
            + 3 * eps_v * (alpha_l[i] * rho_l[i] * v_l[i]) / dt
            - 3 * eps_v * (rho_l[i] * alpha_l[i] * v_l[i] * v_l[i] * H(v_l[i])) / dz
            - 3 * eps_v * (rho_l[i + 1] * alpha_l[i + 1] * v_l[i] * v_l[i] * (1 - H(v_l[i]))) / dz
            + 3 * eps_v * (rho_l[i - 1] * alpha_l[i - 1] * v_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])) / dz
            + 3 * eps_v * (rho_l[i] * alpha_l[i] * v_l[i - 1] * v_l[i - 1] * (1 - H(v_l[i - 1]))) / dz;

        add(L[i], 6, 1,
            - eps_v * (alpha_l[i - 1] * v_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])) / dz);

        add(L[i], 6, 3,
            - eps_v * (rho_l[i - 1] * v_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])) / dz);

        add(L[i], 6, 7,
            - 2 * eps_v * (rho_l[i - 1] * alpha_l[i - 1] * v_l[i - 1] * H(v_l[i - 1])) / dz
            - 2 * eps_v * (rho_l[i] * alpha_l[i] * v_l[i] * (1 - H(v_l[i - 1]))) / dz);

        add(R[i], 6, 1,
            + eps_v * (alpha_l[i + 1] * v_l[i] * v_l[i] * (1 - H(v_l[i]))) / dz);

        add(R[i], 6, 3,
            + eps_v * (rho_l[i + 1] * v_l[i] * v_l[i] * (1 - H(v_l[i]))) / dz)
            + DPcap / dz;

        add(R[i], 6, 5,
            + eps_v * (alpha_l[i] * H(v_l[i])) / dz
            + eps_v * (alpha_l[i] * (1 - H(v_l[i]))) / dz);

        // State mixture equation

        add(D[i], 7, 0, 
            -T_m[i] * Rv);

        add(D[i], 7, 4
            1.0);

        add(D[i], 7, 8, 
            -rho_m[i] * Rv);

        Q[i][7] = -rho_m[i] * T_m[i] * Rv;

        // State liquid equation

        add(D[i], 8, 1, 
            1.0);

        add(D[i], 8, 9, 
            C62);

        Q[i][8] = C63;

        // Volume fraction sum

        add(D[i], 9, 2, 
            1.0);

        add(D[i], 9, 3, 
            1.0);

        Q[i][9] = 1.0;

        // Capillary equation

        add(D[i], 10, 4, 1);
        add(D[i], 10, 5, -1);

        Q[i][10] = DPcap;
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
    Q[0][0] = 0.0;
    Q[0][0] = 0.0;
    Q[0][0] = 0.0;
    Q[0][0] = 0.0;
    Q[0][0] = 0.0;
    Q[0][0] = 0.0;
    Q[0][0] = 0.0;
    Q[0][0] = 0.0;
    Q[0][0] = 0.0;
    Q[0][0] = 0.0;

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
}
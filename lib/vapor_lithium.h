#pragma once
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <array>

namespace vapor_lithium {

    constexpr double ag = 2.2730e7;    // J/kg
    constexpr double bg = 2.95e3;      // J/(kg K)

    // Gas constant for lithium vapor [J/(kg K)]
    constexpr double Rv = 8314.46 / 6.941;  // = 1197.88

    // From h_g(T) = ag + bg * T  (THROHPUT linearization)
    inline double cp_g_linear() {
        return 2.95e3;   // J/(kg K)
    }

    inline double h_g_linear(double T) {
        return ag + bg * T;                // J/kg
    }

    inline double T_from_h_g_linear(double h) {
        return (h - ag) / bg;              // K
    }

    /**
     * @brief Saturation pressure [Pa] - Nesmeyanov correlation
     */
    inline double P_sat(double T) {
        return 133.322 * std::pow(10.0, 5.667 - 8310.0 / T);
    }

    /**
     * @brief dP_sat/dT [Pa/K]
     */
    inline double dP_sat_dT(double T) {
        return P_sat(T) * 8310.0 * std::log(10.0) / (T * T);
    }

    /**
     * @brief Dynamic viscosity of lithium vapor [Pa s]
     *   Approximate kinetic theory fit, valid 1000-2000 K
     */
    inline double mu(double T) {
        return 3.64e-6 + 1.83e-8 * T;
    }

    /**
     * @brief Thermal conductivity [W/(m K)] of lithium vapor
     *   Approximate fit, valid 1000-2000 K
     */
    inline double k(double T, double /*P*/ = 0.0) {
        return 0.028 + 4.5e-5 * T;
    }

    /**
     * @brief Darcy friction factor (Petukhov correlation, smooth pipe)
     */
    inline double friction_factor(double Re) {
        if (Re <= 0.0)
            throw std::invalid_argument("Re <= 0 in friction_factor");
        return std::pow(0.79 * std::log(Re) - 1.64, -2.0);
    }

    /**
     * @brief Nusselt number for internal flow
     *        Laminar + Petukhov-Gnielinski turbulent
     */
    inline double Nusselt(double Re, double Pr) {
        if (Re < 0.0 || Pr < 0.0)
            throw std::invalid_argument("Re or Pr <= 0 in Nusselt");

        constexpr double Nu_lam = 4.36;

        auto Nu_turb = [&](double Re_loc) {
            const double f = friction_factor(Re_loc);
            const double num = (f / 8.0) * (Re_loc - 1000.0) * Pr;
            const double den = 1.0 + 12.7 * std::sqrt(f / 8.0)
                * (std::pow(Pr, 2.0 / 3.0) - 1.0);
            return num / den;
            };

        constexpr double Re_lam = 2300.0;
        constexpr double Re_turb = 4000.0;

        if (Re <= Re_lam) return Nu_lam;
        if (Re >= Re_turb) return Nu_turb(Re);

        const double chi =
            (std::log(Re) - std::log(Re_lam)) /
            (std::log(Re_turb) - std::log(Re_lam));
        return (1.0 - chi) * Nu_lam + chi * Nu_turb(Re);
    }

    /**
     * @brief Convective heat transfer coefficient [W/m2/K]
     */
    inline double h_conv(double Re, double Pr, double k_val, double Dh) {
        if (Dh <= 0.0 || k_val <= 0.0)
            throw std::invalid_argument("Dh or k <= 0 in h_conv");
        return Nusselt(Re, Pr) * k_val / Dh;
    }

    /**
     * @brief Surface tension [N/m] - Shpilrain
     */
    inline double surf_ten(double T) {
        double val = 0.396 - 1.05e-4 * T;
        return val > 0.0 ? val : 0.0;
    }

    /**
     * @brief Ratio of specific heats [-]
     */
    inline double gamma(double /*T*/) {
        double cp_val = cp_g_linear();
        double cv_val = cp_val - Rv;
        return cp_val / cv_val;  // ~1.68
    }
}
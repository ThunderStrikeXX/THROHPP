#pragma once
#include <cmath>

namespace liquid_lithium {

    constexpr double al = -7.015e3;    // J/kg
    constexpr double bl = 1.04e3;      // J/(kg·K)

    // Critical temperature [K]
    constexpr double Tcrit = 3223.0;

    // Solidification temperature [K]
    constexpr double Tsolid = 453.69;

    /**
     * @brief Density [kg/m3] — Shpilrain correlation, valid 454–1600 K
     */
    inline double rho(double T) {
        return 534.2 - 0.04462 * T - 1.756e-6 * T * T;
    }

    /**
     * @brief Thermal conductivity [W/(m·K)] — Shpilrain/Vargaftik
     */
    inline double k(double T) {
        return 22.28 + 0.05292 * T - 1.838e-5 * T * T;
    }

    /**
     * @brief Dynamic viscosity [Pa·s] — Shpilrain
     */
    inline double mu(double T) {
        return std::exp(-4.164 - 0.6374 * std::log(T) + 294.8 / T);
    }

    // From h_l(T) = al + bl * T  (THROHPUT linearization)
    inline double cp_l_linear() {
        return 1.04e3;   // J/(kg·K)
    }

    inline double h_l_linear(double T) {
        return al + bl * T;                // J/kg
    }

    inline double T_from_h_l_linear(double h) {
        return (h - al) / bl;              // K
    }
}
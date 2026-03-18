#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <array>
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <cassert>
#include <string>
#include <sstream>

#include "steel.h"
#include "liquid_sodium.h"
#include "vapor_sodium.h"
#include "solver.h"

// Mathematical constants
const double pi = 3.14159265358979323846;

inline double ramp(double t) {
    const double T = 100.0;

    if (t <= 0.0) return 0.0;
    if (t >= T)   return 1.0;

    return std::sin(0.5 * pi * t / T);
}

int main() {

    // =======================================================================
    //
    //                        [CONSTANTS AND VARIABLES]
    //
    // =======================================================================

    #pragma region constants_and_variables

    // Physical properties
    const double emissivity = 0.5;           // Wall emissivity [-]
    const double sigma = 5.67e-8;            // Stefan-Boltzmann constant [W/(m2K4)]
    const double Rv = 361.5;                 // Gas constant for the sodium vapor [J/(kgK)]

    // Evaporation and condensation parameters
    const double eps_s = 0.5;                // Surface fraction of the liquid available for phasic interface [-]
    const double sigma_e = 0.5;             // Evaporation accomodation coefficient [-]. 1 means optimal evaporation
    const double sigma_c = 0.5;             // Condensation accomodation coefficient [-]. 1 means optimal condensation
    double Omega = 1.0;                      // Initialization of Omega parameter for evaporation/condensation model [-]

    // Wick permeability parameters
    const double K = 1e-10;                 // Permeability [m^2]
    const double CF = 1e5;                  // Forchheimer coefficient [1/m]
    double const eps_v = 1.0;               // Surface fraction of the wick available for liquid passage [-]

    // Geometric parameters
    const int N = 22;                                                       // Number of axial nodes [-]
    const double length = 1.0; 			                                    // Length of the heat pipe [m]
    const double dz = length / (N - 2);                                     // Axial discretization step [m]
    const double r_o = 0.01335;                                             // Outer wall radius [m]
    const double r_i = 0.0112;                                              // Wall-wick interface radius [m]
    const double r_v = 0.01075;                                             // Vapor-wick interface radius [m]
    const double Dh_v = 2.0 * r_v;                                          // Hydraulic diameter of the vapor core [m]
    const double vol_wall_cell = (r_o * r_o - r_i * r_i) * pi * dz;         // Volume of the wall cell [m3]
    const double vol_liquid_cell = (r_i * r_i - r_v * r_v) * pi * dz;       // Volume of the liquid cell [m3]
    const double vol_vapor_cell = r_v * r_v * pi * dz;                      // Volume of the vapor cell [m3]
    const double A_interface_cell = 2 * pi * r_i * dz;                      // Interfacial area between vapor and liquid for a cell [m2] const double lateral_surface = dz * 2 * pi * r_o;                         // Lateral outer surface [m2]
    const double Kgeom = 2.0 * r_v * eps_s / (r_i * r_i);                   // Geometric parameter for the mass flux

    // Environmental boundary conditions
    const double h_conv = 1;                                                // Convective heat transfer coefficient for external heat removal [W/m^2/K]
    const double power = 1000;                                              // Power at the evaporator side [W]
    const double T_env = 280.0;                                             // External environmental temperature [K]
   
    // Evaporator region parameters
    const double evaporator_start = 0.020;                                  // Evaporator begin [m]
    const double evaporator_end = 0.073;                                    // Evaporator end [m]
    const double condenser_length = 0.292;                                  // Condenser length [m]
    const double evaporator_length = evaporator_end - evaporator_start;     // Evaporator length [m]
    const double delta_h = 0.01;                                            // Evaporator ramp [m]
    const double evaporator_length_eff = evaporator_length + delta_h;       // Evaporator effective length [m]
    const double q0 = power / (2 * pi * evaporator_length_eff * r_o);       // Heat flux at evaporator from given power [W/m^2]

    // Condenser region parameters
    const double delta_c = 0.05;                                            // Condenser ramp [m]
    const double condenser_start = length - condenser_length;               // Condenser begin [m]

    // Constant geometrical parameters for the radial model
    const double Eio1 = 2.0 / 3.0 * (r_o + r_i - 1 / (1 / r_o + 1 / r_i));
    const double Eio2 = 0.5 * (r_o * r_o + r_i * r_i);
    const double Evi1 = 2.0 / 3.0 * (r_i + r_v - 1 / (1 / r_i + 1 / r_v));
    const double Evi2 = 0.5 * (r_i * r_i + r_v * r_v);

    // Time-stepping parameters
    double dt_user = 1e-3;                              // Initial time step [s] (then it is updated according to the limits)
    double dt = dt_user;                                // Actual used time step [s]
    double time_simulation = 5000;                      // Simulation total number [s]
    double time_total = 0.0;                            // Total time elapsed [s]
    int halves = 0;                                     // Number of times the time step has been halved [-]
    int n = 0;                                          // Iterations counter [-]

    // Printing parameters
    double t_last_print = 0.0;                          // Time from last print [s]
    const double print_interval = 1e-1;                 // Time interval for printing [s]

    // Picard loops parameters	          
    int pic = 0;                                        // Number of Picard iterations [-]
    const int max_picard = 10;                         // Maximum number of Picard iterations per timestep [-]
    std::array<double, B> L_pic;                        // Picard residuals [-]
    std::array<bool, B> conv_var;                       // Bool array if parameter converged or not [-]
    std::array<double, B> pic_tol = {                   // Tolerance for the convergence of Picard loop [-]
        1e-6,  // rho_m
        1e-6,  // rho_l
        1e-6,  // alpha_m
        1e-6,  // alpha_l
        1e-6,  // p_m
        1e-6,  // p_l
        1,     // v_m
        1,     // v_l
        1e-6,  // T_m
        1e-6,  // T_l
        1e-6   // T_w
    };

    // Mesh z positions
    std::vector<double> mesh(N - 2, 0.0);
    for (int i = 0; i < N - 2; ++i) mesh[i] = i * dz;     // Mesh discretization

    // State variables definition and initialization
    std::vector<double> rho_m(N);                   // Mixture density [kg/m3]
    std::vector<double> rho_l(N);                   // Liquid density [kg/m3]
    std::vector<double> p_m(N);                     // Mixture pressure [Pa]
    std::vector<double> p_l(N);                     // Liquid pressure [Pa]
    std::vector<double> v_m(N + 1);                 // Mixture velocity [m/s]
    std::vector<double> v_l(N + 1);                 // Liquid velocity [m/s]
    std::vector<double> T_m(N);                     // Mixture bulk temperature [K]
    std::vector<double> T_l(N);                     // Liquid bulk temperature [K]
    std::vector<double> T_w(N);                     // Wall bulk temperature [K]
    std::vector<double> alpha_m(N);                 // Mixture volume fraction [-]
    std::vector<double> alpha_l(N);                 // Liquid volume fraction [-]

    // Secondary useful variables
    std::vector<double> Gamma_xv(N, 0.0);                   // Exact mass volumetric source [kg/m3s]
    std::vector<double> Gamma_xv_lin(N, 0.0);               // Linearized mass volumetric source (with C coefficients) [kg/m3s]
    std::vector<double> Gamma_xv_approx_error(N, 0.0);      // Residual between approximation and exact mass volumetric source 1 [kg/m3s]
    std::vector<double> Gamma_xv_lin_error(N, 0.0);         // Residual between approximation and exact mass volumetric source 2 [kg/m3s]
    std::vector<double> Gamma_xv_diff_error(N, 0.0);
    std::vector<double> Gamma_xv_approx(N, 0.0);            // Approximated mass volumetric source (with gamma coefficients) [kg/m3s]
    std::vector<double> T_sur(N);                           // Wick-vapor surface temperature [K]
    std::vector<double> q_ow(N, 0.0);                       // Heat flux profile [W/m^2]
    std::vector<double> power_flux_wx(N, 0.0);              // Heat volumetric source from wall to liquid due to difference in temperature [W/m3]
    std::vector<double> power_flux_xw(N, 0.0);              // Heat volumetric source from liquid to wall due to difference in temperature [W/m3]
    std::vector<double> power_mass_vx(N, 0.0);              // Heat volumetric source from vapor to liquid due to phase change [W/m3]
    std::vector<double> power_mass_xv(N, 0.0);              // Heat volumetric source from liquid to vapor due to phase change [W/m3]
    std::vector<double> power_flux_vx(N, 0.0);              // Heat volumetric source from vapor to liquid due to difference in temperature [W/m3]
    std::vector<double> power_flux_xv(N, 0.0);              // Heat volumetric source from liquid to vapor due to difference in temperature [W/m3]
    std::vector<double> p_saturation(N);                    // Saturation pressure at the temperature of the wick-vapor surface [Pa]
    std::vector<double> dPsat_dT(N);
    std::vector<double> DPcap(N, 0.0);                      // Capillary pressure difference between mixture and vapor [Pa]
    std::vector<double> energy_wall(N, 0.0);                // Wall internal energy [J]
    std::vector<double> energy_liquid(N, 0.0);              // Liquid internal energy [J]
    std::vector<double> energy_vapor(N, 0.0);               // Mixture internal energy [J]
    std::vector<double> aGamma(N);                          // a coefficient for the mass volumetric source approximation [?]
    std::vector<double> bGamma(N);                          // b coefficient for the mass volumetric source approximation [?]
    std::vector<double> cGamma(N);                          // c coefficient for the mass volumetric source approximation [?]

    double h_xv_v;                                          // Specific enthalpy [J/kg] of vapor upon phase change between wick and vapor
    double h_vx_x;                                          // Specific enthalpy [J/kg] of wick upon phase change between vapor and wick

    const double T_left = 1010.0;                        /// First node initialization temperature [K]
    const double T_right = 990.0;                       /// Last node initialization temperature [K]

    // Variables initialization
    for (int i = 0; i < N; ++i) {

        const double s = static_cast<double>(i) / (N - 1);

        T_m[i] = T_left + s * (T_right - T_left);
        T_l[i] = T_left + s * (T_right - T_left);
        T_w[i] = T_left + s * (T_right - T_left);

        T_sur[i] = T_left + s * (T_right - T_left);

        p_m[i] = vapor_sodium::P_sat(T_sur[i]);
        p_l[i] = p_m[i];

        rho_m[i] = p_m[i] / (T_m[i] * Rv);
        rho_l[i] = liquid_sodium::rho(T_l[i]);

        // alpha_m[i] = 0.95 - 0.35 * s;   // 0.95 → 0.60
        // alpha_l[i] = 1.0 - alpha_m[i];  // 0.05 → 0.40

        alpha_m[i] = 0.90;
        alpha_l[i] = 0.10;
    }

    for (int i = 2; i < N - 1; i++) {

        v_m[i] = 1.0;
        v_l[i] = -0.1;
    }


    int n_evap = 0;
    double sum_w = 0.0;
    std::vector<double> w(N, 0.0);

    // Evaporator cosine distribution
    for (int i = 1; i < N - 2; ++i) {

        if (mesh[i] >= evaporator_start - delta_h &&
            mesh[i] <= evaporator_end + delta_h) {

            ++n_evap;

            if (mesh[i] >= evaporator_start - delta_h &&
                mesh[i] < evaporator_start) {

                double x = (mesh[i] - (evaporator_start - delta_h)) / delta_h;
                w[i] = 0.5 * (1.0 - std::cos(pi * x));
            }
            else if (mesh[i] >= evaporator_start &&
                mesh[i] <= evaporator_end) {

                w[i] = 1.0;
            }
            else if (mesh[i] > evaporator_end &&
                mesh[i] <= evaporator_end + delta_h) {

                double x = (mesh[i] - evaporator_end) / delta_h;
                w[i] = 0.5 * (1.0 + std::cos(pi * x));
            }
        }

        sum_w += w[i];
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

    // Iter variables
    std::vector<double> rho_m_iter = rho_m;
    std::vector<double> rho_l_iter = rho_l;
    std::vector<double> alpha_m_iter = alpha_m;
    std::vector<double> alpha_l_iter = alpha_l;
    std::vector<double> p_m_iter = p_m;
    std::vector<double> p_l_iter = p_l;
    std::vector<double> v_m_iter = v_m;
    std::vector<double> v_l_iter = v_l;
    std::vector<double> T_m_iter = T_m;
    std::vector<double> T_l_iter = T_l;
    std::vector<double> T_w_iter = T_w;

    std::vector<double> T_sur_iter = T_sur;
    std::vector<double> Gamma_xv_iter = Gamma_xv;

    std::vector<double> T_sur_old = T_sur;
    std::vector<double> Gamma_xv_old = Gamma_xv;

    // Blocks definition
    std::vector<SparseBlock> L(N), D(N), R(N);
    std::vector<VecBlock> Q(N), X(N);

    // Differences between the heat exchanged at interfaces due to linearization (should be zero to machine precision)
    std::vector<double> power_residual_xw(N, 0);
    std::vector<double> heat_residual_xv(N, 0);

    // Heat exchange balance liquid vapor
    std::vector<double> heat_conduction_flux(N, 0.0);
    std::vector<double> heat_convection_flux(N, 0.0);
    std::vector<double> heat_phase_flux(N, 0.0);

    std::vector<double> C1(N, 0.0), C2(N, 0.0), C3(N, 0.0), C4(N, 0.0), C5(N, 0.0), C6(N, 0.0), C7(N, 0.0), C8(N, 0.0), C9(N, 0.0), C10(N, 0.0), C11(N, 0.0), C12(N, 0.0)
        , C13(N, 0.0), C14(N, 0.0), C15(N, 0.0), C16(N, 0.0), C17(N, 0.0), C18(N, 0.0), C19(N, 0.0), C20(N, 0.0), C21(N, 0.0), C22(N, 0.0), C23(N, 0.0), C24(N, 0.0), C25(N, 0.0)
        , C26(N, 0.0), C27(N, 0.0), C28(N, 0.0), C29(N, 0.0), C30(N, 0.0), C31(N, 0.0), C32(N, 0.0), C33(N, 0.0), C34(N, 0.0), C35(N, 0.0), C36(N, 0.0), C37(N, 0.0), C38(N, 0.0)
        , C39(N, 0.0), C40(N, 0.0), C41(N, 0.0), C42(N, 0.0), C43(N, 0.0), C44(N, 0.0), C45(N, 0.0), C46(N, 0.0), C47(N, 0.0), C48(N, 0.0), C49(N, 0.0), C50(N, 0.0), C51(N, 0.0)
        , C52(N, 0.0), C53(N, 0.0), C54(N, 0.0), C55(N, 0.0), C56(N, 0.0), C57(N, 0.0), C58(N, 0.0), C59(N, 0.0), C60(N, 0.0), C61(N, 0.0), C62(N, 0.0), C63(N, 0.0), C64(N, 0.0)
        , C65(N, 0.0), C66(N, 0.0), C67(N, 0.0), C68(N, 0.0), C69(N, 0.0), C70(N, 0.0), C71(N, 0.0), C72(N, 0.0), C73(N, 0.0), C74(N, 0.0), C75(N, 0.0);

    std::vector<double> H_xm(N, 0.0);
    std::vector<double> Dh(N, 0.0);

    std::vector<double> heat_balance_xv_1(N, 0.0);
    std::vector<double> heat_balance_xv_2(N, 0.0);
    std::vector<double> heat_balance_xv_3(N, 0.0);
    std::vector<double> heat_balance_xv_4(N, 0.0);
    std::vector<double> heat_balance_xv_5(N, 0.0);

    std::vector<double> a_x(N, 0.0);
    std::vector<double> b_x(N, 0.0);
    std::vector<double> c_x(N, 0.0);
    std::vector<double> a_w(N, 0.0);
    std::vector<double> b_w(N, 0.0);
    std::vector<double> c_w(N, 0.0);

    std::vector<double> T_sur_lin(N, 0.0);
    std::vector<double> T_sur_diff(N, 0.0);

    std::vector<double> balance_condition(N, 0.0);

    std::vector<double> alpha1(N, 0.0);
    std::vector<double> alpha2(N, 0.0);
    std::vector<double> alpha3(N, 0.0);
    std::vector<double> alpha4(N, 0.0);
    std::vector<double> alpha5(N, 0.0);
    std::vector<double> alpha6(N, 0.0);
    std::vector<double> alpha7(N, 0.0);
    std::vector<double> alpha8(N, 0.0);
    std::vector<double> alpha9(N, 0.0);
    std::vector<double> alpha10(N, 0.0);
    std::vector<double> alpha11(N, 0.0);
    std::vector<double> alpha12(N, 0.0);
    std::vector<double> delta(N, 0.0);

    std::vector<double> residual1(N, 0.0);
    std::vector<double> residual2(N, 0.0);
    std::vector<double> residual3(N, 0.0);
    std::vector<double> residual4(N, 0.0);
    std::vector<double> residual5(N, 0.0);
    std::vector<double> residual6(N, 0.0);

    std::vector<double> Ex3(N, 0.0);
    std::vector<double> Ex4(N, 0.0);
    std::vector<double> Ex5(N, 0.0);
    std::vector<double> Ex6(N, 0.0);
    std::vector<double> Ex7(N, 0.0);
    std::vector<double> Ex8(N, 0.0);

    std::vector<double> power_residual_xv(N, 0.0);

    // ------ Properties

    std::vector<double> cp_m(N, vapor_sodium::cp_g_linear());
    std::vector<double> cp_m_old = cp_m;

    std::vector<double> cp_l(N, liquid_sodium::cp_l_linear());
    std::vector<double> cp_l_old = cp_l;

    std::vector<double> cp_w(N);
    std::vector<double> cp_w_old = cp_w;

    std::vector<double> k_m(N);
    std::vector<double> k_l(N);

    std::vector<double> k_w(N);
    std::vector<double> rho_w(N);

    std::vector<double> mu_l(N);
    std::vector<double> mu_m(N);

    std::vector<double> Re_v(N);
    std::vector<double> Pr_v(N);

    std::vector<double> Re_l(N);
    
    std::vector<double> fm(N, 0.0);
    std::vector<double> fl(N, 0.0);

    bool heat_sources_xw = 1;
    bool heat_sources_xv_heat = 1;
    bool external_heat = 1;

    #pragma endregion

    #pragma region output

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

    // ---------- PRIMARY VARIABLES --------------

    std::ofstream mesh_output(name + "/mesh.txt", std::ios::trunc);

    std::ofstream time_output(name + "/time.txt", std::ios::app);
    std::ofstream dt_output(name + "/dt.txt", std::ios::app);
    std::ofstream simulation_time_output(name + "/simulation_time.txt", std::ios::app);
    std::ofstream clock_time_output(name + "/clock_time.txt", std::ios::app);

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

    std::ofstream power_flux_ow_output(name + "/power_flux_ow.txt", std::ios::trunc);
    std::ofstream power_flux_wx_output(name + "/power_flux_wx.txt", std::ios::trunc);
    std::ofstream power_flux_xw_output(name + "/power_flux_xw.txt", std::ios::trunc);
    std::ofstream power_mass_vx_output(name + "/power_mass_vx.txt", std::ios::trunc);
    std::ofstream power_mass_xv_output(name + "/power_mass_xv.txt", std::ios::trunc);
    std::ofstream power_flux_vx_output(name + "/power_flux_vx.txt", std::ios::trunc);
    std::ofstream power_flux_xv_output(name + "/power_flux_xv.txt", std::ios::trunc);

    std::ofstream dpcap_output(name + "/delta_p_capillary.txt", std::ios::trunc);
    std::ofstream psat_output(name + "/p_saturation.txt", std::ios::trunc);
    std::ofstream tsur_output(name + "/T_sur.txt", std::ios::trunc);

    // -------------- BALANCES ------------

    std::ofstream acc_mass_v_output(name + "/acc_mass_v.txt", std::ios::trunc);
    std::ofstream acc_mass_l_output(name + "/acc_mass_l.txt", std::ios::trunc);

    std::ofstream acc_energy_v_output(name + "/acc_energy_v.txt", std::ios::trunc);
    std::ofstream acc_energy_l_output(name + "/acc_energy_l.txt", std::ios::trunc);

    std::ofstream acc_mom_v_output(name + "/acc_mom_v.txt", std::ios::trunc);
    std::ofstream acc_mom_l_output(name + "/acc_mom_l.txt", std::ios::trunc);

    std::ofstream bal_mass_v_output(name + "/bal_mass_v.txt", std::ios::trunc);
    std::ofstream bal_mass_l_output(name + "/bal_mass_l.txt", std::ios::trunc);

    std::ofstream bal_energy_v_output(name + "/bal_energy_v.txt", std::ios::trunc);
    std::ofstream bal_energy_l_output(name + "/bal_energy_l.txt", std::ios::trunc);

    std::ofstream bal_mom_v_output(name + "/bal_mom_v.txt", std::ios::trunc);
    std::ofstream bal_mom_l_output(name + "/bal_mom_l.txt", std::ios::trunc);

    std::ofstream diff_mass_v_output(name + "/diff_mass_v.txt", std::ios::trunc);
    std::ofstream diff_mass_l_output(name + "/diff_mass_l.txt", std::ios::trunc);

    std::ofstream diff_energy_v_output(name + "/diff_energy_v.txt", std::ios::trunc);
    std::ofstream diff_energy_l_output(name + "/diff_energy_l.txt", std::ios::trunc);

    std::ofstream diff_mom_v_output(name + "/diff_mom_v.txt", std::ios::trunc);
    std::ofstream diff_mom_l_output(name + "/diff_mom_l.txt", std::ios::trunc);

    std::ofstream acc_energy_w_output(name + "/acc_energy_w.txt", std::ios::trunc);
    std::ofstream bal_energy_w_output(name + "/bal_energy_w.txt", std::ios::trunc);
    std::ofstream diff_energy_w_output(name + "/diff_energy_w.txt", std::ios::trunc);

    std::ofstream global_energy_balance_output(name + "/global_heat_balance.txt", std::ios::trunc);

    // ---------- RESIDUALS -------------

    std::ofstream power_residual_wx_output(name + "/power_residual_wx.txt", std::ios::trunc);
    std::ofstream power_residual_xv_output(name + "/power_residual_xv.txt", std::ios::trunc);

    std::ofstream residual_1_output(name + "/residual_1.txt", std::ios::trunc);
    std::ofstream residual_2_output(name + "/residual_2.txt", std::ios::trunc);
    std::ofstream residual_3_output(name + "/residual_3.txt", std::ios::trunc);
    std::ofstream residual_4_output(name + "/residual_4.txt", std::ios::trunc);
    std::ofstream residual_5_output(name + "/residual_5.txt", std::ios::trunc);
    std::ofstream residual_6_output(name + "/residual_6.txt", std::ios::trunc);

    std::ofstream T_sur_diff_output(name + "/T_sur_diff.txt", std::ios::trunc);

    std::ofstream Gamma_xv_approx_error_output(name + "/Gamma_xv_approx_error.txt", std::ios::trunc);
    std::ofstream Gamma_xv_lin_error_output(name + "/Gamma_xv_lin_error.txt", std::ios::trunc);
    std::ofstream Gamma_xv_diff_error_output(name + "/Gamma_xv_diff_error.txt", std::ios::trunc);

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
    power_flux_wx_output << std::setprecision(global_precision);
    power_flux_xw_output << std::setprecision(global_precision);
    power_mass_vx_output << std::setprecision(global_precision);
    power_mass_xv_output << std::setprecision(global_precision);
    power_flux_vx_output << std::setprecision(global_precision);
    power_flux_xv_output << std::setprecision(global_precision);
    psat_output << std::setprecision(global_precision);
    tsur_output << std::setprecision(global_precision);

	dpcap_output << std::setprecision(global_precision);
	power_flux_ow_output << std::setprecision(global_precision);

    acc_mass_v_output << std::setprecision(global_precision);
    acc_mass_l_output << std::setprecision(global_precision);

    acc_energy_v_output << std::setprecision(global_precision);
    acc_energy_l_output << std::setprecision(global_precision);

    acc_mom_v_output << std::setprecision(global_precision);
    acc_mom_l_output << std::setprecision(global_precision);

    bal_mass_v_output << std::setprecision(global_precision);
    bal_mass_l_output << std::setprecision(global_precision);

    bal_energy_v_output << std::setprecision(global_precision);
    bal_energy_l_output << std::setprecision(global_precision);

    bal_mom_v_output << std::setprecision(global_precision);
    bal_mom_l_output << std::setprecision(global_precision);

    diff_mass_v_output << std::setprecision(global_precision);
    diff_mass_l_output << std::setprecision(global_precision);

    diff_energy_v_output << std::setprecision(global_precision);
    diff_energy_l_output << std::setprecision(global_precision);

    diff_mom_v_output << std::setprecision(global_precision);
    diff_mom_l_output << std::setprecision(global_precision);

    acc_energy_w_output << std::setprecision(global_precision);
    bal_energy_w_output << std::setprecision(global_precision);
    diff_energy_w_output << std::setprecision(global_precision);

    global_energy_balance_output << std::setprecision(global_precision);

    power_residual_wx_output << std::setprecision(global_precision);
    power_residual_xv_output << std::setprecision(global_precision);

    residual_1_output << std::setprecision(global_precision);
    residual_2_output << std::setprecision(global_precision);
    residual_3_output << std::setprecision(global_precision);
    residual_4_output << std::setprecision(global_precision);
    residual_5_output << std::setprecision(global_precision);
    residual_6_output << std::setprecision(global_precision);

    T_sur_diff_output << std::setprecision(global_precision);

    Gamma_xv_approx_error_output << std::setprecision(global_precision);
    Gamma_xv_lin_error_output << std::setprecision(global_precision);
    Gamma_xv_diff_error_output << std::setprecision(global_precision);

    for (int i = 0; i < N - 2; ++i) mesh_output << mesh[i] << " ";

    mesh_output.flush();
    mesh_output.close();

    #pragma endregion

    double beta = 0.0;

    const double al = -2.359582e5;
    const double bl = 1.256230e3;

    const double ag = 4.683166e6;
    const double bg = 3.589755e2;

    std::cout << "Case: " << name << std::endl;

    // Start computational time measurement of whole simulation
    auto t_start_simulation = std::chrono::high_resolution_clock::now();

    // Time-stepping loop
    while (time_total < time_simulation) {

        double mass_sources = 0.0 /*ramp(time_total)*/;
        double heat_sources_xv_mass = 0.0 /*ramp(time_total)*/;

        n++;

        // Start computational time iteration
        auto t_start_timestep = std::chrono::high_resolution_clock::now();

        // Timestep selection
        dt = dt_user * pow(0.5, halves);    // Halfing of the timestep up to a lower bound     

        if (dt < 1e-12) {
            
            std::cout << "Timestep under 1e-12, convergence not achieved." << std::endl;
            std::cout << "Time: " << time_total << ", iteration: " << n << std::endl;
            system("pause");
            
            return 1;
        }

        // Updating all properties
        for (int i = 0; i < N; ++i) {

            cp_w[i] = steel::cp(T_w[i]);
            rho_w[i] = steel::rho(T_w[i]);
            k_w[i] = steel::k(T_w[i]);

            cp_l[i] = liquid_sodium::cp_l_linear();
            k_l[i] = liquid_sodium::k(T_l[i]);
            mu_l[i] = liquid_sodium::mu(T_l[i]);

            cp_m[i] = vapor_sodium::cp_g_linear();
            k_m[i] = vapor_sodium::k(T_m[i], p_m[i]);
            mu_m[i] = liquid_sodium::mu(T_m[i]);

        }

        const double s = (sum_w > 0.0) ? q0 / sum_w : 0.0;

        for (int i = 1; i < N - 2; ++i) {

            q_ow[i] = w[i] * s;

            double conv = h_conv * (T_w[i] - T_env);                                        // [W/m2]
            double irr = emissivity * sigma * (std::pow(T_w[i], 4) - std::pow(T_env, 4));   // [W/m2]

            if (mesh[i] >= condenser_start &&
                mesh[i] < condenser_start + delta_c) {

                double x = (mesh[i] - condenser_start) / delta_c;
                q_ow[i] = -(conv + irr) * 0.5 * (1.0 - std::cos(pi * x));
            }
            else if (mesh[i] >= condenser_start + delta_c) {
                q_ow[i] = -(conv + irr);
            }
        }

        // Picard iteration loop
        for (pic = 0; pic < max_picard; ++pic) {

            // Picard --> iter = new
            rho_m_iter = rho_m;
            rho_l_iter = rho_l;
            alpha_m_iter = alpha_m;
            alpha_l_iter = alpha_l;
            p_m_iter = p_m;
            p_l_iter = p_l;
            v_m_iter = v_m;
            v_l_iter = v_l;
            T_m_iter = T_m;
            T_l_iter = T_l;
            T_w_iter = T_w;

            T_sur_iter = T_sur;
            Gamma_xv_iter = Gamma_xv;

            // Cleaning all the blocks (the add function adds block and does not overwrite, so this is necessary)
            for (int i = 0; i < N; i++) {
                L[i].row.clear(); L[i].col.clear(); L[i].val.clear();
                D[i].row.clear(); D[i].col.clear(); D[i].val.clear();
                R[i].row.clear(); R[i].col.clear(); R[i].val.clear();
            }

            // Forcing boundary conditions

            v_m_iter[0] = 0.0;   // BC ingresso
            v_m_iter[N] = 0.0;   // BC uscita

            v_l_iter[0] = 0.0;
            v_l_iter[N] = 0.0;

            v_m[0] = 0.0;   // BC ingresso
            v_m[N] = 0.0;   // BC uscita

            v_l[0] = 0.0;
            v_l[N] = 0.0;

            T_m_iter[0] = T_m_iter[1];
            T_l_iter[0] = T_l_iter[1];

            T_m_iter[N - 1] = T_m_iter[N - 2];
            T_l_iter[N - 1] = T_l_iter[N - 2];

            alpha_m_iter[0] = alpha_m_iter[1];
            alpha_l_iter[0] = alpha_l_iter[1];

            alpha_m_iter[N - 1] = alpha_m_iter[N - 2];
            alpha_l_iter[N - 1] = alpha_l_iter[N - 2];

            rho_m_iter[0] = rho_m_iter[1];
            rho_l_iter[0] = rho_l_iter[1];

            rho_m_iter[N - 1] = rho_m_iter[N - 2];
            rho_l_iter[N - 1] = rho_l_iter[N - 2];

            p_m_iter[0] = p_m_iter[1];
            p_l_iter[0] = p_l_iter[1];

            p_m_iter[N - 1] = p_m_iter[N - 2];
            p_l_iter[N - 1] = p_l_iter[N - 2];

            // Space discretization loop
            for (int i = 1; i < N - 1; ++i) {

                // =======================================================================
                //
                //                              [COEFFICIENTS]
                //
                // =======================================================================

                #pragma region coefficients

                // Physical properties
                Re_v[i] = rho_m_iter[i] * std::fabs(v_m_iter[i]) * Dh_v / mu_m[i];              // Reynolds number [-]
                Pr_v[i] = cp_m[i] * mu_m[i] / k_m[i];                                                 // Prandtl number [-] 
                H_xm[i] = vapor_sodium::h_conv(Re_v[i], Pr_v[i], k_m[i], Dh_v);     // Convective heat transfer coefficient at the vapor-wick interface [W/m^2/K]
                p_saturation[i] = vapor_sodium::P_sat(T_sur_iter[i]);                           // Saturation pressure [Pa]         
                dPsat_dT[i] = vapor_sodium::dP_sat_dT(T_sur_iter[i]);                           // Derivative of the saturation pressure wrt T [Pa/K]   

                // Gamma coefficients definition (everything is calculated using iter (k-iteration) values)

                beta = 1.0 / std::sqrt(2.0 * pi * Rv * T_sur_iter[i]);

                cGamma[i] = - (Kgeom * beta * sigma_c);
                bGamma[i] = - (Gamma_xv_iter[i] / (2 * T_sur_iter[i])) + (Kgeom * beta * sigma_e * dPsat_dT[i]);
                aGamma[i] = Gamma_xv_iter[i] - bGamma[i] * T_sur_iter[i];
                
                // Definition of the enthalpies (as in THROHPUT)
                if (Gamma_xv_iter[i] >= 0.0) {

                    // Evaporation case
                    h_xv_v = vapor_sodium::h_g_linear(T_sur_iter[i]);
                    h_vx_x = liquid_sodium::h_l_linear(T_sur_iter[i]);

                }
                else {
                     
                    // Condensation case
                    h_xv_v = vapor_sodium::h_g_linear(T_m_iter[i]);
                    h_vx_x = liquid_sodium::h_l_linear(T_sur_iter[i])
                        + (vapor_sodium::h_g_linear(T_m_iter[i]) - vapor_sodium::h_g_linear(T_sur_iter[i]));
                }

                Dh[i] = h_xv_v - h_vx_x;

                // Radial model constants
                Ex3[i] = - H_xm[i] - (Dh[i] * r_i * r_i) / (2.0 * r_v) * bGamma[i];
                Ex4[i] = k_l[i] - H_xm[i] * r_v - (Dh[i] * r_i * r_i) / (2.0 * r_v) * bGamma[i] * r_v;
                Ex5[i] = 2.0 * r_v * k_l[i] - H_xm[i] * r_v * r_v - (Dh[i] * r_i * r_i) / (2.0 * r_v) * bGamma[i] * (r_v * r_v);
                Ex6[i] = -H_xm[i];
                Ex7[i] = (Dh[i] * r_i * r_i) / (2.0 * r_v) * cGamma[i];
                Ex8[i] = (Dh[i] * r_i * r_i) / (2.0 * r_v) * (aGamma[i] - cGamma[i] * p_m_iter[i]);

                alpha1[i] = 2 * r_o * Eio1 - Eio2;
                alpha2[i] = alpha1[i] - 2 * r_o * r_i + r_i * r_i;
                alpha3[i] = r_i - Eio1;
                alpha4[i] = 2 * k_w[i] * (r_i - r_o);
                alpha5[i] = 2 * r_i * Evi1 - Evi2;
                alpha6[i] = r_i - Evi1;
                alpha7[i] = alpha5[i] - r_i * r_i;
                alpha8[i] = alpha2[i] - alpha4[i] * alpha6[i] / k_l[i];
                alpha9[i] = alpha6[i] / k_l[i] - alpha3[i] / k_w[i];
                alpha10[i] = alpha4[i] * (Ex4[i] - Ex3[i] * Evi1) / k_l[i];
                alpha11[i] = Ex3[i] * alpha5[i] - 2 * r_i * Ex4[i] + Ex5[i];
                alpha12[i] = (Ex4[i] - Ex3[i] * Evi1) / k_l[i];

                delta[i] = alpha8[i] * alpha11[i] + alpha7[i] * alpha10[i];

                // c_w coefficients
                C1[i] = alpha7[i] * Ex7[i] / delta[i];
                C2[i] = alpha7[i] * Ex6[i] / delta[i];
                C3[i] = (alpha11[i] - alpha7[i] * Ex3[i]) / delta[i];
                C4[i] = -alpha11[i] / delta[i];
                C5[i] = (alpha9[i] * alpha11[i] * q_ow[i] + alpha7[i] * (Ex8[i] - alpha12[i] * q_ow[i])) / delta[i];

                // c_x coefficients
                C6[i] = alpha8[i] * Ex7[i] / delta[i];
                C7[i] = alpha8[i] * Ex6[i] / delta[i];
                C8[i] = -(alpha8[i] * Ex3[i] + alpha10[i]) / delta[i];
                C9[i] = alpha10[i] / delta[i];
                C10[i] = (alpha8[i] * (Ex8[i] - alpha12[i] * q_ow[i]) - alpha10[i] * alpha9[i] * q_ow[i]) / delta[i];

                // b_w coefficients
                C11[i] = -2.0 * r_o * C1[i];
                C12[i] = -2.0 * r_o * C2[i];
                C13[i] = -2.0 * r_o * C3[i];
                C14[i] = -2.0 * r_o * C4[i];
                C15[i] = q_ow[i] / k_w[i] - 2.0 * r_o * C5[i];

                // a_w coefficients
                C16[i] = alpha1[i] * C1[i];
                C17[i] = alpha1[i] * C2[i];
                C18[i] = alpha1[i] * C3[i];
                C19[i] = 1.0 + alpha1[i] * C4[i];
                C20[i] = -Eio1 * q_ow[i] / k_w[i] + alpha1[i] * C5[i];

                // b_x coefficients
                C21[i] = alpha4[i] * C1[i] / k_l[i] - 2.0 * r_i * C6[i];
                C22[i] = alpha4[i] * C2[i] / k_l[i] - 2.0 * r_i * C7[i];
                C23[i] = alpha4[i] * C3[i] / k_l[i] - 2.0 * r_i * C8[i];
                C24[i] = alpha4[i] * C4[i] / k_l[i] - 2.0 * r_i * C9[i];
                C25[i] = q_ow[i] / k_l[i] + alpha4[i] * C5[i] / k_l[i] - 2.0 * r_i * C10[i];

                // a_x coefficients
                C26[i] = -Evi1 * alpha4[i] / k_l[i] * C1[i] + alpha5[i] * C6[i];
                C27[i] = -Evi1 * alpha4[i] / k_l[i] * C2[i] + alpha5[i] * C7[i];
                C28[i] = 1.0 - Evi1 * alpha4[i] / k_l[i] * C3[i] + alpha5[i] * C8[i];
                C29[i] = -Evi1 * alpha4[i] / k_l[i] * C4[i] + alpha5[i] * C9[i];
                C30[i] = -Evi1 * q_ow[i] / k_l[i] - Evi1 * alpha4[i] / k_l[i] * C5[i] + alpha5[i] * C10[i];

                // T_sur coefficients [checked]
                C31[i] = C26[i] + r_v * C21[i] + r_v * r_v * C6[i];
                C32[i] = C27[i] + r_v * C22[i] + r_v * r_v * C7[i];
                C33[i] = C28[i] + r_v * C23[i] + r_v * r_v * C8[i];
                C34[i] = C29[i] + r_v * C24[i] + r_v * r_v * C9[i];
                C35[i] = C30[i] + r_v * C25[i] + r_v * r_v * C10[i];

                // Mass source coefficients [checked]
                C36[i] = bGamma[i] * C31[i] + cGamma[i];
                C37[i] = bGamma[i] * C32[i];
                C38[i] = bGamma[i] * C33[i];
                C39[i] = bGamma[i] * C34[i];
                C40[i] = bGamma[i] * C35[i] - cGamma[i] * p_m_iter[i] + aGamma[i];
                    
                // Heat source from mixture to liquid due to heat flux coefficients
                C41[i] = -2 * k_l[i] * r_v / (r_i * r_i) * (C21[i] + 2 * r_v * C6[i]);
                C42[i] = -2 * k_l[i] * r_v / (r_i * r_i) * (C22[i] + 2 * r_v * C7[i]);
                C43[i] = -2 * k_l[i] * r_v / (r_i * r_i) * (C23[i] + 2 * r_v * C8[i]);
                C44[i] = -2 * k_l[i] * r_v / (r_i * r_i) * (C24[i] + 2 * r_v * C9[i]);
                C45[i] = -2 * k_l[i] * r_v / (r_i * r_i) * (C25[i] + 2 * r_v * C10[i]);

                // Heat source from liquid to mixture due to heat flux coefficients
                C46[i] = 2 * H_xm[i] * r_v / (r_i * r_i) * (C26[i] + C21[i] * r_v + C6[i] * r_v * r_v);
                C47[i] = 2 * H_xm[i] * r_v / (r_i * r_i) * (C27[i] + C22[i] * r_v + C7[i] * r_v * r_v - 1);
                C48[i] = 2 * H_xm[i] * r_v / (r_i * r_i) * (C28[i] + C23[i] * r_v + C8[i] * r_v * r_v);
                C49[i] = 2 * H_xm[i] * r_v / (r_i * r_i) * (C29[i] + C24[i] * r_v + C9[i] * r_v * r_v);
                C50[i] = 2 * H_xm[i] * r_v / (r_i * r_i) * (C30[i] + C25[i] * r_v + C10[i] * r_v * r_v);

                // Heat source from mixture to liquid due to phase change coefficients [checked]
                C51[i] = -h_vx_x * C36[i];
                C52[i] = -h_vx_x * C37[i];
                C53[i] = -h_vx_x * C38[i];
                C54[i] = -h_vx_x * C39[i];
                C55[i] = -h_vx_x * C40[i];

                // Heat source from liquid to mixture due to phase change coefficients [checked]
                C56[i] = h_xv_v * C36[i];
                C57[i] = h_xv_v * C37[i];
                C58[i] = h_xv_v * C38[i];
                C59[i] = h_xv_v * C39[i];
                C60[i] = h_xv_v * C40[i];

                // Heat source from wall to liquid due to heat flux coefficients [checked]
                C61[i] = 2 * k_w[i] / r_i * (C11[i] + 2 * r_i * C1[i]);
                C62[i] = 2 * k_w[i] / r_i * (C12[i] + 2 * r_i * C2[i]);
                C63[i] = 2 * k_w[i] / r_i * (C13[i] + 2 * r_i * C3[i]);
                C64[i] = 2 * k_w[i] / r_i * (C14[i] + 2 * r_i * C4[i]);
                C65[i] = 2 * k_w[i] / r_i * (C15[i] + 2 * r_i * C5[i]);

                // Heat source from liquid to wall due to heat flux coefficients [checked]
                C66[i] = -2 * k_w[i] * r_i / (r_o * r_o - r_i * r_i) * (C11[i] + 2 * r_i * C1[i]);
                C67[i] = -2 * k_w[i] * r_i / (r_o * r_o - r_i * r_i) * (C12[i] + 2 * r_i * C2[i]);
                C68[i] = -2 * k_w[i] * r_i / (r_o * r_o - r_i * r_i) * (C13[i] + 2 * r_i * C3[i]);
                C69[i] = -2 * k_w[i] * r_i / (r_o * r_o - r_i * r_i) * (C14[i] + 2 * r_i * C4[i]);
                C70[i] = -2 * k_w[i] * r_i / (r_o * r_o - r_i * r_i) * (C15[i] + 2 * r_i * C5[i]);

                // DPcap evaluation
                const double alpha_m0 = r_v * r_v / (r_i * r_i);                        // Nominal vapor volume fraction (when r_c = infty) [-]
                const double r_p = 1e-5;                                                // Porosity radius [m]
                const double surf_ten_value = vapor_sodium::surf_ten(T_l_iter[i]);      // Surface tension [N/m]           

                const double Lambda = 3 * r_v * (alpha_m_iter[i] - alpha_m0) / (2 * alpha_m0 * eps_s * r_p);

                if (Lambda <= 0.0) DPcap[i] = 0;
                else if (Lambda >= 2.0) DPcap[i] = 2 * surf_ten_value / r_p;
                else {

                    double mu = invert(Lambda);
                    if (mu < 1e-3) {

                        DPcap[i] = 2 * surf_ten_value / r_p * (mu + (3 * r_v) / (2 * eps_s * alpha_m0 * r_p) * 0.75 * (alpha_m_iter[i] - alpha_m_old[i]));

                    }
                    else {

                        DPcap[i] = 2 * surf_ten_value / r_p *
                            (mu + (9 * r_v) / (2 * eps_s * alpha_m0 * r_p) * (std::pow(1 - mu * mu, -0.5) - Lambda / mu) * (alpha_m_iter[i] - alpha_m_old[i]));
                    }
                }

                #pragma endregion

                // =======================================================================
                //
                //                              [EQUATIONS]
                //
                // =======================================================================

                #pragma region equations

                // -------------- MASS MIXTURE EQUATION ------------------

                add(D[i], 0, 0, 0.0

                    // Temporal term
                    + alpha_m_iter[i] / dt

                    // Convective term
                    + (alpha_m_iter[i] * v_m_iter[i + 1] * H(v_m_iter[i + 1])) / dz
                    - (alpha_m_iter[i] * v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz
                );

                add(D[i], 0, 2, 0.0

                    // Temporal term
                    + rho_m_iter[i] / dt

                    // Convective term
                    + (rho_m_iter[i] * v_m_iter[i + 1] * H(v_m_iter[i + 1])) / dz
                    - (rho_m_iter[i] * v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz
                );

                add(D[i], 0, 4, 0.0

                    // Source term
                    - C36[i] * mass_sources                  // Mass source from wick
                );

                add(D[i], 0, 6, 0.0

                    // Convective term
                    - (alpha_m_iter[i - 1] * rho_m_iter[i - 1] * H(v_m_iter[i])) / dz
                    - (alpha_m_iter[i] * rho_m_iter[i] * (1 - H(v_m_iter[i]))) / dz
                );

                add(D[i], 0, 8, 0.0

                    // Source term
                    - C37[i] * mass_sources                      // Mass source from wick
                );

                add(D[i], 0, 9, 0.0

                    // Source term
                    - C38[i] * mass_sources                      // Mass source from wick
                );

                add(D[i], 0, 10, 0.0

                    // Source term
                    - C39[i] * mass_sources                     // Mass source from wick
                );

                Q[i][0] = 0.0

                    // Source term (implicit)
                    + C40[i] * mass_sources                 // Mass source from wick

                    // Temporal term
                    + (rho_m_old[i] * alpha_m_old[i]) / dt
                    + (rho_m_iter[i] * alpha_m_iter[i]) / dt

                    // Convective term
                    + 2 * (
                        + alpha_m_iter[i] * rho_m_iter[i] * v_m_iter[i + 1] * H(v_m_iter[i + 1])
                        + alpha_m_iter[i + 1] * rho_m_iter[i + 1] * v_m_iter[i + 1] * (1 - H(v_m_iter[i + 1]))
                        - alpha_m_iter[i - 1] * rho_m_iter[i - 1] * v_m_iter[i] * H(v_m_iter[i])
                        - alpha_m_iter[i] * rho_m_iter[i] * v_m_iter[i] * (1 - H(v_m_iter[i]))
                        ) / dz
                    ;

                add(L[i], 0, 0, 0.0

                    // Convective term
                    - (alpha_m_iter[i - 1] * v_m_iter[i] * H(v_m_iter[i])) / dz
                );

                add(L[i], 0, 2, 0.0

                    // Convective term
                    - (rho_m_iter[i - 1] * v_m_iter[i] * H(v_m_iter[i])) / dz
                );

                add(R[i], 0, 0, 0.0

                    // Convective term
                    + (alpha_m_iter[i + 1] * v_m_iter[i + 1] * (1 - H(v_m_iter[i + 1]))) / dz
                );

                add(R[i], 0, 2, 0.0

                    // Convective term
                    + (rho_m_iter[i + 1] * v_m_iter[i + 1] * (1 - H(v_m_iter[i + 1]))) / dz
                );

                add(R[i], 0, 6, 0.0

                    // Convective term
                    + (alpha_m_iter[i] * rho_m_iter[i] * H(v_m_iter[i + 1])) / dz
                    + (alpha_m_iter[i + 1] * rho_m_iter[i + 1] * (1 - H(v_m_iter[i + 1]))) / dz
                );

                // --------------- MASS LIQUID EQUATION -----------------

                add(D[i], 1, 1, 0.0

                    // Temporal term
                    + eps_v * (alpha_l_iter[i] / dt)

                    // Convective term
                    + eps_v * (alpha_l_iter[i] * v_l_iter[i + 1] * H(v_l_iter[i + 1])) / dz
                    - eps_v * (alpha_l_iter[i] * v_l_iter[i] * (1 - H(v_l_iter[i]))) / dz
                );

                add(D[i], 1, 3, 0.0

                    // Temporal term
                    + eps_v * (rho_l_iter[i] / dt)

                    // Convective term
                    + eps_v * (rho_l_iter[i] * v_l_iter[i + 1] * H(v_l_iter[i + 1])) / dz
                    - eps_v * (rho_l_iter[i] * v_l_iter[i] * (1 - H(v_l_iter[i]))) / dz
                );

                add(D[i], 1, 4, 0.0

                    // Source term
                    + C36[i] * mass_sources                    // Mass source from vapor
                );

                add(D[i], 1, 7, 0.0

                    // Convective term
                    - eps_v * (alpha_l_iter[i - 1] * rho_l_iter[i - 1] * H(v_l_iter[i])) / dz
                    - eps_v * (alpha_l_iter[i] * rho_l_iter[i] * (1 - H(v_l_iter[i]))) / dz
                );

                add(D[i], 1, 8, 0.0

                    // Source term
                    + C37[i] * mass_sources                      // Mass source from vapor
                );

                add(D[i], 1, 9, 0.0

                    // Source term
                    + C38[i] * mass_sources                     // Mass source from vapor
                );

                add(D[i], 1, 10, 0.0

                    // Source term
                    + C39[i] * mass_sources                      // Mass source from vapor
                );


                Q[i][1] =

                    // Temporal term
                    + eps_v * (rho_l_iter[i] * alpha_l_iter[i]) / dt
                    + eps_v * (rho_l_old[i] * alpha_l_old[i]) / dt

                    // Convective term
                    + 2 * (
                        + eps_v * (alpha_l_iter[i] * rho_l_iter[i] * v_l_iter[i + 1] * H(v_l_iter[i + 1]))
                        + eps_v * (alpha_l_iter[i + 1] * rho_l_iter[i + 1] * v_l_iter[i + 1] * (1 - H(v_l_iter[i + 1])))
                        - eps_v * (alpha_l_iter[i - 1] * rho_l_iter[i - 1] * v_l_iter[i] * H(v_l_iter[i]))
                        - eps_v * (alpha_l_iter[i] * rho_l_iter[i] * v_l_iter[i] * (1 - H(v_l_iter[i])))
                        ) / dz

                    // Source term (implicit)
                    - C40[i] * mass_sources                   // Mass source from vapor
                    ;

                add(L[i], 1, 1, 0.0

                    // Convective term
                    - eps_v * (alpha_l_iter[i - 1] * v_l_iter[i] * H(v_l_iter[i])) / dz
                );

                add(L[i], 1, 3, 0.0

                    // Convective term
                    - eps_v * (rho_l_iter[i - 1] * v_l_iter[i] * H(v_l_iter[i])) / dz
                );

                add(R[i], 1, 1, 0.0

                    // Convective term
                    + eps_v * (alpha_l_iter[i + 1] * v_l_iter[i + 1] * (1 - H(v_l_iter[i + 1]))) / dz
                );

                add(R[i], 1, 3, 0.0

                    // Convective term
                    + eps_v * (rho_l_iter[i + 1] * v_l_iter[i + 1] * (1 - H(v_l_iter[i + 1]))) / dz
                );

                add(R[i], 1, 7, 0.0

                    // Convective term
                    + eps_v * (alpha_l_iter[i] * rho_l_iter[i] * H(v_l_iter[i + 1])) / dz
                    + eps_v * (alpha_l_iter[i + 1] * rho_l_iter[i + 1] * (1 - H(v_l_iter[i + 1]))) / dz
                );

                // ---------- MIXTURE HEAT EQUATION ----------------

                add(D[i], 2, 0, 0.0

                    // Temporal term
                    + (alpha_m_iter[i] * cp_m[i] * T_m_iter[i]) / dt

                    // Convective term
                    + (alpha_m_iter[i] * cp_m[i] * T_m_iter[i] * v_m_iter[i + 1] * H(v_m_iter[i + 1])) / dz
                    - (alpha_m_iter[i] * cp_m[i] * T_m_iter[i] * v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz
                );

                add(D[i], 2, 2, 0.0

                    // Temporal term
                    + (T_m_iter[i] * rho_m_iter[i] * cp_m[i]) / dt

                    // Convective term
                    + (rho_m_iter[i] * cp_m[i] * T_m_iter[i] * v_m_iter[i + 1] * H(v_m_iter[i + 1])) / dz
                    - (rho_m_iter[i] * cp_m[i] * T_m_iter[i] * v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz

                    // Pressure I term
                    + p_m_iter[i] * (v_m_iter[i + 1] - v_m_iter[i]) / (2 * dz)

                    // Pressure II term
                    + p_m_iter[i] / dt
                );

                add(D[i], 2, 4, 0.0

                    // Source term
                    - C46[i] * heat_sources_xv_heat               // Heat source due to heat flux from wick
                    - C56[i] * heat_sources_xv_mass                 // Heat source due to mass flux from wick
                );

                add(D[i], 2, 6, 0.0

                    // Convective term
                    - (alpha_m_iter[i - 1] * rho_m_iter[i - 1] * cp_m[i - 1] * T_m_iter[i - 1] * H(v_m_iter[i])) / dz
                    - (alpha_m_iter[i] * rho_m_iter[i] * cp_m[i] * T_m_iter[i] * (1 - H(v_m_iter[i]))) / dz

                    // Pressure I term
                    - p_m_iter[i] * (alpha_m_iter[i] + alpha_m_iter[i + 1]) / (2 * dz)
                );

                add(D[i], 2, 8, 0.0

                    // Temporal term
                    + (alpha_m_iter[i] * rho_m_iter[i] * cp_m[i]) / dt

                    // Diffusion term
                    + (alpha_m_iter[i + 1] * k_m[i + 1] + 2 * alpha_m_iter[i] * k_m[i] + alpha_m_iter[i - 1] * k_m[i - 1]) / (2 * dz * dz)

                    // Convective term
                    + (alpha_m_iter[i] * rho_m_iter[i] * cp_m[i] * v_m_iter[i + 1] * H(v_m_iter[i + 1])) / dz
                    - (alpha_m_iter[i] * rho_m_iter[i] * cp_m[i] * v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz

                    // Source term
                    - C47[i] * heat_sources_xv_heat                    // Heat source due to heat flux from wick
                    - C57[i] * heat_sources_xv_mass                  // Heat source due to mass flux from wick
                );

                add(D[i], 2, 9, 0.0

                    // Source term
                    - C48[i] * heat_sources_xv_heat                     // Heat source due to heat flux from wick
                    - C58[i] * heat_sources_xv_mass                  // Heat source due to mass flux from wick
                );

                add(D[i], 2, 10, 0.0

                    // Source term
                    - C49[i] * heat_sources_xv_heat                   // Heat source due to heat flux from wick
                    - C59[i] * heat_sources_xv_mass                // Heat source due to mass flux from wick
                );  

                Q[i][2] = 0.0

                    // Temporal term (cross terms version)
                    + (alpha_m_iter[i] * cp_m_old[i] * T_m_iter[i] * rho_m_old[i]) / dt
                    + (alpha_m_iter[i] * cp_m_old[i] * T_m_old[i] * rho_m_iter[i]) / dt
                    + (alpha_m_old[i] * cp_m_old[i] * T_m_iter[i] * rho_m_iter[i]) / dt

                    // Convective term
                    + 3 * (
                        + alpha_m_iter[i] * rho_m_iter[i] * cp_m[i] * T_m_iter[i] * v_m_iter[i + 1] * H(v_m_iter[i + 1])
                        + alpha_m_iter[i + 1] * rho_m_iter[i + 1] * cp_m[i + 1] * T_m_iter[i + 1] * v_m_iter[i + 1] * (1 - H(v_m_iter[i + 1]))
                        - alpha_m_iter[i - 1] * rho_m_iter[i - 1] * cp_m[i - 1] * T_m_iter[i - 1] * v_m_iter[i] * H(v_m_iter[i])
                        - alpha_m_iter[i] * rho_m_iter[i] * cp_m[i] * T_m_iter[i] * v_m_iter[i] * (1 - H(v_m_iter[i]))
                        ) / dz

                    // Pressure I term
                    + p_m_iter[i] * (alpha_m_iter[i] + alpha_m_iter[i + 1]) * v_m_iter[i + 1] / (2 * dz)
                    - p_m_iter[i] * (alpha_m_iter[i] + alpha_m_iter[i - 1]) * v_m_iter[i] / (2 * dz)

                    // Pressure II term
                    + (p_m_iter[i] * alpha_m_old[i]) / dt

                    // Source term
                    + C50[i] * heat_sources_xv_heat                    // Heat source due to heat flux from wick
                    + C60[i] * heat_sources_xv_mass                 // Heat source due to mass flux from wick
                    ;

                add(L[i], 2, 0, 0.0

                    // Convective term
                    - (alpha_m_iter[i - 1] * cp_m[i - 1] * T_m_iter[i - 1] * v_m_iter[i] * H(v_m_iter[i])) / dz
                );

                add(L[i], 2, 2, 0.0

                    // Convective term
                    - (rho_m_iter[i - 1] * cp_m[i - 1] * T_m_iter[i - 1] * v_m_iter[i] * H(v_m_iter[i])) / dz

                    // Pressure I term
                    - p_m_iter[i] * (v_m_iter[i]) / (2 * dz)
                );

                add(L[i], 2, 8, 0.0

                    // Convective term
                    - (alpha_m_iter[i - 1] * rho_m_iter[i - 1] * cp_m[i - 1] * v_m_iter[i] * H(v_m_iter[i])) / dz

                    // Diffusion term
                    - (alpha_m_iter[i - 1] * k_m[i - 1] + alpha_m_iter[i] * k_m[i]) / (2 * dz * dz)
                );

                add(R[i], 2, 0, 0.0

                    // Convective term
                    + (alpha_m_iter[i + 1] * cp_m[i + 1] * T_m_iter[i + 1] * v_m_iter[i + 1] * (1 - H(v_m_iter[i + 1]))) / dz
                );

                add(R[i], 2, 2, 0.0

                    // Convective term
                    + (rho_m_iter[i + 1] * cp_m[i + 1] * T_m_iter[i + 1] * v_m_iter[i + 1] * (1 - H(v_m_iter[i + 1]))) / dz

                    // Pressure I term
                    + p_m_iter[i] * (v_m_iter[i + 1]) / (2 * dz)
                );

                add(R[i], 2, 6, 0.0

                    // Pressure I term
                    + p_m_iter[i] * (alpha_m_iter[i] + alpha_m_iter[i - 1]) / (2 * dz)

                    // Convective term
                    + (alpha_m_iter[i] * rho_m_iter[i] * cp_m[i] * T_m_iter[i] * H(v_m_iter[i + 1])) / dz
                    + (alpha_m_iter[i + 1] * rho_m_iter[i + 1] * cp_m[i + 1] * T_m_iter[i + 1] * (1 - H(v_m_iter[i + 1]))) / dz
                );

                add(R[i], 2, 8, 0.0

                    // Convective term
                    + (alpha_m_iter[i + 1] * rho_m_iter[i + 1] * cp_m[i + 1] * v_m_iter[i + 1] * (1 - H(v_m_iter[i + 1]))) / dz

                    // Diffusion term
                    - (alpha_m_iter[i + 1] * k_m[i + 1] + alpha_m_iter[i] * k_m[i]) / (2 * dz * dz)
                );

                // ---------- LIQUID HEAT EQUATION ----------------

                add(D[i], 3, 1, 0.0

                    // Temporal term
                    + eps_v * (alpha_l_iter[i] * cp_l[i] * T_l_iter[i]) / dt

                    // Convective term
                    + eps_v * (alpha_l_iter[i] * cp_l[i] * T_l_iter[i] * v_l_iter[i + 1] * H(v_l_iter[i + 1])) / dz
                    - eps_v * (alpha_l_iter[i] * cp_l[i] * T_l_iter[i] * v_l_iter[i] * (1 - H(v_l_iter[i]))) / dz
                );

                add(D[i], 3, 3, 0.0

                    // Temporal term
                    + eps_v * (T_l_iter[i] * rho_l_iter[i] * cp_l[i]) / dt

                    // Convective term
                    + eps_v * (rho_l_iter[i] * cp_l[i] * T_l_iter[i] * v_l_iter[i + 1] * H(v_l_iter[i + 1])) / dz
                    - eps_v * (rho_l_iter[i] * cp_l[i] * T_l_iter[i] * v_l_iter[i] * (1 - H(v_l_iter[i]))) / dz

                    // Pressure I term
                    + eps_v * p_l_iter[i] * (v_l_iter[i + 1] - v_l_iter[i]) / (2 * dz)

                    // Pressure II term
                    + eps_v * p_l_iter[i] / dt
                );

                add(D[i], 3, 4, 0.0

                    // Source term
                    - C41[i] * heat_sources_xw                     // Heat source due to heat flux from wall
                    - C51[i] * heat_sources_xv_mass                     // Heat source due to mass flux from vapor
                    - C61[i] * heat_sources_xv_heat                        // Heat source due to heat flux from vapor
                );

                add(D[i], 3, 7, 0.0

                    // Convective term
                    - eps_v * (alpha_l_iter[i - 1] * rho_l_iter[i - 1] * cp_l[i - 1] * T_l_iter[i - 1] * H(v_l_iter[i])) / dz
                    - eps_v * (alpha_l_iter[i] * rho_l_iter[i] * cp_l[i] * T_l_iter[i] * (1 - H(v_l_iter[i]))) / dz

                    // Pressure I term
                    - eps_v * p_l_iter[i] * (alpha_l_iter[i] + alpha_l_iter[i + 1]) / (2 * dz)
                );

                add(D[i], 3, 8, 0.0

                    // Source term
                    - C42[i] * heat_sources_xw                     // Heat source due to heat flux from wall
                    - C52[i] * heat_sources_xv_mass                     // Heat source due to mass flux from vapor 
                    - C62[i] * heat_sources_xv_heat                        // Heat source due to heat flux from vapor
                );

                add(D[i], 3, 9, 0.0

                    // Temporal term
                    + eps_v * (alpha_l_iter[i] * rho_l_iter[i] * cp_l[i]) / dt

                    // Diffusion term
                    + eps_v * (alpha_l_iter[i + 1] * k_l[i + 1] + 2 * alpha_l_iter[i] * k_l[i] + alpha_l_iter[i - 1] * k_l[i - 1]) / (2 * dz * dz)

                    // Convective term
                    + eps_v * (alpha_l_iter[i] * rho_l_iter[i] * cp_l[i] * v_l_iter[i + 1] * H(v_l_iter[i + 1])) / dz
                    - eps_v * (alpha_l_iter[i] * rho_l_iter[i] * cp_l[i] * v_l_iter[i] * (1 - H(v_l_iter[i]))) / dz

                    // Source term
                    - C43[i] * heat_sources_xw                     // Heat source due to heat flux from wall
                    - C53[i] * heat_sources_xv_mass                      // Heat source due to mass flux from vapor
                    - C63[i] * heat_sources_xv_heat                        // Heat source due to heat flux from vapor
                );

                add(D[i], 3, 10, 0.0

                    // Source term
                    - C44[i] * heat_sources_xw                     // Heat source due to heat flux from wall
                    - C54[i] * heat_sources_xv_mass                      // Heat source due to mass flux from vapor 
                    - C64[i] * heat_sources_xv_heat                       // Heat source due to heat flux from vapor
                );

                Q[i][3] = 0.0

                    // Temporal term (cross terms)
                    + eps_v * (alpha_l_iter[i] * cp_l[i] * T_l_iter[i] * rho_l_old[i]) / dt
                    + eps_v * (alpha_l_iter[i] * cp_l[i] * T_l_old[i] * rho_l_iter[i]) / dt
                    + eps_v * (alpha_l_old[i] * cp_l[i] * T_l_iter[i] * rho_l_iter[i]) / dt

                    // Convective term
                    + 3 * eps_v * (
                        + alpha_l_iter[i] * rho_l_iter[i] * cp_l[i] * T_l_iter[i] * v_l_iter[i + 1] * H(v_l_iter[i + 1])
                        + alpha_l_iter[i + 1] * rho_l_iter[i + 1] * cp_l[i +1] * T_l_iter[i + 1] * v_l_iter[i + 1] * (1 - H(v_l_iter[i + 1]))
                        - alpha_l_iter[i - 1] * rho_l_iter[i - 1] * cp_l[i - 1] * T_l_iter[i - 1] * v_l_iter[i] * H(v_l_iter[i])
                        - alpha_l_iter[i] * rho_l_iter[i] * cp_l[i] * T_l_iter[i] * v_l_iter[i] * (1 - H(v_l_iter[i]))
                        ) / dz

                    // Pressure I term
                    + eps_v * p_l_iter[i] * (alpha_l_iter[i] + alpha_l_iter[i + 1]) * v_l_iter[i + 1] / (2 * dz)
                    - eps_v * p_l_iter[i] * (alpha_l_iter[i] + alpha_l_iter[i - 1]) * v_l_iter[i] / (2 * dz)

                    // Pressure II term
                    + eps_v * (p_l_iter[i] * alpha_l_old[i]) / dt

                    // Source term
                    + C45[i] * heat_sources_xw                    // Heat source due to heat flux from wall
                    + C55[i] * heat_sources_xv_mass                      // Heat source due to mass flux from vapor
                    + C65[i] * heat_sources_xv_heat                        // Heat source due to heat flux from vapor
                    ;

                add(L[i], 3, 1, 0.0

                    // Convective term
                    - eps_v * (alpha_l_iter[i - 1] * cp_l[i - 1] * T_l_iter[i - 1] * v_l_iter[i] * H(v_l_iter[i])) / dz
                );

                add(L[i], 3, 3, 0.0

                    // Convective term
                    - eps_v * (rho_l_iter[i - 1] * cp_l[i - 1] * T_l_iter[i - 1] * v_l_iter[i] * H(v_l_iter[i])) / dz

                    // Pressure I term
                    - eps_v * p_l_iter[i] * (v_l_iter[i]) / (2 * dz)
                );

                add(L[i], 3, 9, 0.0

                    // Convective term
                    - eps_v * (alpha_l_iter[i - 1] * rho_l_iter[i - 1] * cp_l[i - 1] * v_l_iter[i] * H(v_l_iter[i])) / dz

                    // Diffusion term
                    - eps_v * (alpha_l_iter[i - 1] * k_l[i - 1] + alpha_l_iter[i] * k_l[i]) / (2 * dz * dz)
                );

                add(R[i], 3, 1, 0.0

                    // Convective term
                    + eps_v * (alpha_l_iter[i + 1] * cp_l[i +1] * T_l_iter[i + 1] * v_l_iter[i + 1] * (1 - H(v_l_iter[i + 1]))) / dz
                );

                add(R[i], 3, 3, 0.0

                    // Convective term
                    + eps_v * (rho_l_iter[i + 1] * cp_l[i +1] * T_l_iter[i + 1] * v_l_iter[i + 1] * (1 - H(v_l_iter[i + 1]))) / dz

                    // Pressure I term
                    + eps_v * p_l_iter[i] * (v_l_iter[i + 1]) / (2 * dz)
                );

                add(R[i], 3, 7, 0.0

                    // Pressure I term
                    + eps_v * p_l_iter[i] * (alpha_l_iter[i] + alpha_l_iter[i + 1]) / (2 * dz)

                    // Convective term
                    + eps_v * (alpha_l_iter[i] * rho_l_iter[i] * cp_l[i] * T_l_iter[i] * H(v_l_iter[i + 1])) / dz
                    + eps_v * (alpha_l_iter[i + 1] * rho_l_iter[i + 1] * cp_l[i +1] * T_l_iter[i + 1] * (1 - H(v_l_iter[i + 1]))) / dz
                );

                add(R[i], 3, 9, 0.0

                    // Convective term
                    + eps_v * (alpha_l_iter[i + 1] * rho_l_iter[i + 1] * cp_l[i +1] * v_l_iter[i + 1] * (1 - H(v_l_iter[i + 1]))) / dz

                    // Diffusion term
                    - eps_v * (alpha_l_iter[i + 1] * k_l[i + 1] + alpha_l_iter[i] * k_l[i]) / (2 * dz * dz)
                );

                // --------------- WALL HEAT EQUATION -------------------

                const double k_w_lf = 0.5 * (k_w[i - 1] + k_w[i]);
                const double k_w_rf = 0.5 * (k_w[i] + k_w[i + 1]);

                add(D[i], 4, 4, 0.0

                    // Source term
                    - C66[i] * heat_sources_xw                      // Heat source due to heat flux from wick
                );

                add(D[i], 4, 8, 0.0

                    // Source term
                    - C67[i] * heat_sources_xw                      // Heat source due to heat flux from wick
                );

                add(D[i], 4, 9, 0.0

                    // Source term
                    - C68[i] * heat_sources_xw                      // Heat source due to heat flux from wick
                );

                add(D[i], 4, 10, 0.0

                    // Temporal term
                    + (rho_w[i] * cp_w[i]) / dt

                    // Diffusion term
                    + (k_w_lf + k_w_rf) / (dz * dz)

                    // Source term
                    - C69[i] * heat_sources_xw                      // Heat source due to heat flux from wick
                );

                Q[i][4] = 0.0

                    // Source term 
                    + q_ow[i] * 2 * r_o / (r_o * r_o - r_i * r_i) * external_heat

                    // Temporal term
                    + (rho_w[i] * cp_w[i] * T_w_old[i]) / dt

                    // Source term
                    + C70[i] * heat_sources_xw                     // Heat source due to heat flux from wick
                    ;

                add(L[i], 4, 10, 0.0

                    // Diffusion term
                    - k_w_lf / (dz * dz)
                );

                add(R[i], 4, 10, 0.0

                    // Diffusion term
                    - k_w_rf / (dz * dz)
                );

                // -------- MOMENTUM MIXTURE EQUATION ---------

                fm[i] = Re_v[i] > 1187.4 ? 0.3164 * std::pow(Re_v[i], -0.25) : 64 * std::pow(Re_v[i], -1);
                fm[i] = Re_v[i] == 0 ? 0 : fm[i];

                add(D[i], 5, 0,

                    // Temporal term (central differences)
                    + (alpha_m_iter[i - 1] + alpha_m_iter[i]) * v_m_iter[i] / (4 * dt)

                    // Convective term
                    + (H(v_m_iter[i]) * alpha_m_iter[i] * v_m_iter[i] * v_m_iter[i]) / dz
                    + ((1 - H(v_m_iter[i])) * alpha_m_iter[i] * v_m_iter[i + 1] * v_m_iter[i + 1]) / dz
                );

                add(D[i], 5, 2,

                    // Temporal term (central differences)
                    + (rho_m_iter[i - 1] + rho_m_iter[i]) * v_m_iter[i] / (4 * dt)

                    // Convective term
                    + (H(v_m_iter[i]) * rho_m_iter[i] * v_m_iter[i] * v_m_iter[i]) / dz
                    + ((1 - H(v_m_iter[i])) * rho_m_iter[i] * v_m_iter[i + 1] * v_m_iter[i + 1]) / dz

                );

                add(D[i], 5, 4, 0.0

                    // Pressure term (central differences)
                    + (alpha_m_iter[i - 1] + alpha_m_iter[i]) / (2 * dz)
                );

                add(D[i], 5, 6,

                    // Temporal term (central differences)
                    + (alpha_m_iter[i - 1] * rho_m_iter[i - 1] + alpha_m_iter[i] * rho_m_iter[i]) / (2 * dt)

                    // Convective term
                    + 2 * (H(v_m_iter[i]) * alpha_m_iter[i] * rho_m_iter[i] * v_m_iter[i]) / dz
                    - 2 * ((1 - H(v_m_iter[i])) * alpha_m_iter[i - 1] * rho_m_iter[i - 1] * v_m_iter[i]) / dz

                    // Friction term (central differences)
                    + fm[i] * (rho_m_iter[i - 1] + rho_m_iter[i]) * std::abs(v_m_iter[i]) / (8 * r_v)
                );

                Q[i][5] =

                    // Temporal term (cross terms)
                    + (alpha_m_iter[i - 1] * rho_m_iter[i - 1] + alpha_m_iter[i] * rho_m_iter[i]) * v_m_old[i] / (2 * dt)
                    + ((alpha_m_iter[i - 1] + alpha_m_iter[i]) * v_m_iter[i] * (rho_m_old[i - 1] + rho_m_old[i])) / (4 * dt)
                    + ((rho_m_iter[i - 1] + rho_m_iter[i]) * v_m_iter[i] * (alpha_m_old[i - 1] + alpha_m_old[i])) / (4 * dt)

                    // Convective term
                    - 3 * H(v_m_iter[i]) * (
                        + alpha_m_iter[i - 1] * rho_m_iter[i - 1] * v_m_iter[i - 1] * v_m_iter[i - 1]
                        - alpha_m_iter[i] * rho_m_iter[i] * v_m_iter[i] * v_m_iter[i]
                        ) / dz
                    - 3 * (1 - H(v_m_iter[i])) * (
                        + alpha_m_iter[i - 1] * rho_m_iter[i - 1] * v_m_iter[i] * v_m_iter[i]
                        - alpha_m_iter[i] * rho_m_iter[i] * v_m_iter[i + 1] * v_m_iter[i + 1]
                        ) / dz
                    ;

                add(L[i], 5, 0, 0.0

                    // Temporal term (central differences)
                    + (alpha_m_iter[i - 1] + alpha_m_iter[i]) * v_m_iter[i] / (4 * dt)

                    // Convective term
                    - (H(v_m_iter[i]) * alpha_m_iter[i - 1] * v_m_iter[i - 1] * v_m_iter[i - 1]) / dz
                    - ((1 - H(v_m_iter[i])) * alpha_m_iter[i - 1] * v_m_iter[i] * v_m_iter[i]) / dz
                );

                add(L[i], 5, 2, 0.0

                    // Temporal term (central differences)
                    + (rho_m_iter[i - 1] + rho_m_iter[i]) * v_m_iter[i] / (4 * dt)

                    // Convective term
                    - (H(v_m_iter[i]) * rho_m_iter[i - 1] * v_m_iter[i - 1] * v_m_iter[i - 1]) / dz
                    - ((1 - H(v_m_iter[i])) * rho_m_iter[i - 1] * v_m_iter[i] * v_m_iter[i]) / dz
                );

                add(L[i], 5, 4, 0.0

                    // Pressure term (central differences)
                    - (alpha_m_iter[i - 1] + alpha_m_iter[i]) / (2 * dz)
                );

                add(L[i], 5, 6,

                    // Convective term
                    - 2 * (H(v_m_iter[i]) * alpha_m_iter[i - 1] * rho_m_iter[i - 1] * v_m_iter[i - 1]) / dz
                );

                add(R[i], 5, 6,

                    // Convective term
                    + 2 * ((1 - H(v_m_iter[i])) * alpha_m_iter[i] * rho_m_iter[i] * v_m_iter[i + 1]) / dz
                );

                // ------ MOMENTUM LIQUID EQUATION ------

                fl[i] = 8 * (mu_l[i - 1] + mu_l[i]) / (2 * eps_v * (r_i - r_v) * (r_i - r_v));

                add(D[i], 6, 1,

                    // Temporal term (central differences)
                    + eps_v * (alpha_l_iter[i - 1] + alpha_l_iter[i]) * v_l_iter[i] / (4 * dt)

                    // Convective term
                    + eps_v * (H(v_l_iter[i]) * alpha_l_iter[i] * v_l_iter[i] * v_l_iter[i]) / dz
                    + eps_v * ((1 - H(v_l_iter[i])) * alpha_l_iter[i] * v_l_iter[i + 1] * v_l_iter[i + 1]) / dz                    
                );

                add(D[i], 6, 3,

                    // Temporal term (central differences)
                    + eps_v * (rho_l_iter[i - 1] + rho_l_iter[i]) * v_l_iter[i] / (4 * dt)

                    // Convective term
                    + eps_v * (H(v_l_iter[i]) * rho_l_iter[i] * v_l_iter[i] * v_l_iter[i]) / dz
                    + eps_v * ((1 - H(v_l_iter[i])) * rho_l_iter[i] * v_l_iter[i + 1] * v_l_iter[i + 1]) / dz

                    // Capillary term (central differences)
                    + (DPcap[i - 1] + DPcap[i]) / (2 * dz)
                );

                add(D[i], 6, 5, 0.0

                    // Pressure term (central differences)
                    + eps_v * (alpha_l_iter[i - 1] + alpha_l_iter[i]) / (2 * dz)
                );

                add(D[i], 6, 7,

                    // Temporal term (central differences)
                    + eps_v * (alpha_l_iter[i - 1] * rho_l_iter[i - 1] + alpha_l_iter[i] * rho_l_iter[i]) / (2 * dt)

                    // Convective term
                    + 2 * eps_v * (H(v_l_iter[i]) * alpha_l_iter[i] * rho_l_iter[i] * v_l_iter[i]) / dz
                    - 2 * eps_v * ((1 - H(v_l_iter[i])) * alpha_l_iter[i - 1] * rho_l_iter[i - 1] * v_l_iter[i]) / dz

                    // Friction term
                    + fl[i] * std::abs(v_l_iter[i])
                );

                Q[i][6] =

                    // Temporal term (central differeces)
                    + eps_v * (alpha_l_iter[i - 1] * rho_l_iter[i - 1] + alpha_l_iter[i] * rho_l_iter[i]) * v_l_old[i] / (2 * dt)
                    + eps_v * ((alpha_l_iter[i - 1] + alpha_l_iter[i]) * v_l_iter[i] * (rho_l_old[i - 1] + rho_l_old[i])) / (4 * dt)
                    + eps_v * ((rho_l_iter[i - 1] + rho_l_iter[i]) * v_l_iter[i] * (alpha_l_old[i - 1] + alpha_l_old[i])) / (4 * dt)

                    // Convective term
                    - 3 * eps_v * H(v_l_iter[i]) * (
                        + alpha_l_iter[i - 1] * rho_l_iter[i - 1] * v_l_iter[i - 1] * v_l_iter[i - 1]
                        - alpha_l_iter[i] * rho_l_iter[i] * v_l_iter[i] * v_l_iter[i]
                        ) / dz
                    - 3 * eps_v * (1 - H(v_l_iter[i])) * (
                        + alpha_l_iter[i - 1] * rho_l_iter[i - 1] * v_l_iter[i] * v_l_iter[i]
                        - alpha_l_iter[i] * rho_l_iter[i] * v_l_iter[i + 1] * v_l_iter[i + 1]
                        ) / dz
                    ;

                add(L[i], 6, 1, 0.0

                    // Temporal term (central differences)
                    + eps_v * (alpha_l_iter[i - 1] + alpha_l_iter[i]) * v_l_iter[i] / (4 * dt)

                    // Convective term
                    - eps_v * (H(v_l_iter[i]) * alpha_l_iter[i - 1] * v_l_iter[i - 1] * v_l_iter[i - 1]) / dz
                    - eps_v * ((1 - H(v_l_iter[i])) * alpha_l_iter[i - 1] * v_l_iter[i] * v_l_iter[i]) / dz
                );

                add(L[i], 6, 3, 0.0

                    // Temporal term (central differences)
                    + eps_v * (rho_l_iter[i - 1] + rho_l_iter[i]) * v_l_iter[i] / (4 * dt)

                    // Convective term
                    - eps_v * (H(v_l_iter[i]) * rho_l_iter[i - 1] * v_l_iter[i - 1] * v_l_iter[i - 1]) / dz
                    - eps_v * ((1 - H(v_l_iter[i])) * rho_l_iter[i - 1] * v_l_iter[i] * v_l_iter[i]) / dz

                    // Capillary term (central differences)
                    - (DPcap[i - 1] + DPcap[i]) / (2 * dz)
                );

                add(L[i], 6, 5, 0.0

                    // Pressure term (central differences)
                    - eps_v * (alpha_l_iter[i - 1] + alpha_l_iter[i]) / (2 * dz)
                );

                add(L[i], 6, 7,

                    // Convective term
                    - 2 * eps_v * (H(v_l_iter[i]) * alpha_l_iter[i - 1] * rho_l_iter[i - 1] * v_l_iter[i - 1]) / dz
                );

                add(R[i], 6, 7,

                    // Convective term
                    + 2 * eps_v * ((1 - H(v_l_iter[i])) * alpha_l_iter[i] * rho_l_iter[i] * v_l_iter[i + 1]) / dz
                );

                // State mixture equation

                add(D[i], 7, 0, -T_m_iter[i] * Rv);
                add(D[i], 7, 4, 1.0);
                add(D[i], 7, 8, -rho_m_iter[i] * Rv);

                Q[i][7] = -rho_m_iter[i] * T_m_iter[i] * Rv;

                // State liquid equation

                add(D[i], 8, 1,
                    1.0
                );

                add(D[i], 8, 9,
                    +1.0 / liquid_sodium::Tcrit * (275.32 + 511.58 / (2 * std::sqrt(1 - T_l_iter[i] / liquid_sodium::Tcrit)))
                );

                Q[i][8] = 219.0 + 275.32 * (1.0 - T_l_iter[i] / liquid_sodium::Tcrit) + 511.58 * std::sqrt(1.0 - T_l_iter[i] / liquid_sodium::Tcrit) + T_l_iter[i] / liquid_sodium::Tcrit * (275.32 + 511.58 / (2 * std::sqrt(1.0 - T_l_iter[i] / liquid_sodium::Tcrit)));

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

                Q[i][10] = DPcap[i];

                #pragma endregion

                // Densification of the D block for debug purposes
                
                /*

                if(i == 1) {
                    DenseBlock D_dense = to_dense(D[i]);

                    std::cout << "Dense block D[" << i << "]\n";

                    for (int r = 0; r < B; ++r) {
                        for (int c = 0; c < B; ++c) {
                            std::cout << std::setw(16) << D_dense[r][c] << " ";
                        }
                        std::cout << "\n";
                    }
                }

                std::cout << std::endl;

                */
            }

            // After all equations are assembled, clear rows 6 and 7 for cell 1
            auto clearRow = [](auto& block, int row) {
                for (int k = block.row.size() - 1; k >= 0; --k) {
                    if (block.row[k] == row) {
                        block.row.erase(block.row.begin() + k);
                        block.col.erase(block.col.begin() + k);
                        block.val.erase(block.val.begin() + k);
                    }
                }
                };

            // Assemble ALL equations for cell 1 normally (including momentum)
            // Then AFTER assembly, override only row 6 (v_m) and row 7 (v_l):

            clearRow(L[1], 5); clearRow(D[1], 5); clearRow(R[1], 5);
            clearRow(L[1], 6); clearRow(D[1], 6); clearRow(R[1], 6);

            add(D[1], 5, 6, 1.0);
            add(D[1], 6, 7, 1.0);

            Q[1][5] = 0.0;
            Q[1][6] = 0.0;

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
            L_pic = {};

            double Aold, Anew, denom, eps;

            for (int i = 1; i < N - 1; ++i) {

                // rho_m
                Aold = rho_m_iter[i];
                Anew = X[i][0];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L_pic[0] += eps;

                // rho_l
                Aold = rho_l_iter[i];
                Anew = X[i][1];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L_pic[1] += eps;

                // alpha_m
                Aold = alpha_m_iter[i];
                Anew = X[i][2];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L_pic[2] += eps;

                // alpha_l
                Aold = alpha_l_iter[i];
                Anew = X[i][3];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L_pic[3] += eps;

                // p_m
                Aold = p_m_iter[i];
                Anew = X[i][4];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L_pic[4] += eps;

                // p_l
                Aold = p_l_iter[i];
                Anew = X[i][5];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L_pic[5] += eps;

                // v_m
                Aold = v_m_iter[i];
                Anew = X[i][6];
                denom = std::max({ std::abs(Aold), std::abs(Anew), 1e-3 });
                eps = std::abs(Anew - Aold) / denom;
                L_pic[6] = std::max(L_pic[6], eps);

                // v_l
                Aold = v_l_iter[i];
                Anew = X[i][7];
                denom = std::max({ std::abs(Aold), std::abs(Anew), 1e-5 });
                eps = std::abs(Anew - Aold) / denom;
                L_pic[7] = std::max(L_pic[7], eps);

                // T_m
                Aold = T_m_iter[i];
                Anew = X[i][8];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L_pic[8] += eps;

                // T_l
                Aold = T_l_iter[i];
                Anew = X[i][9];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L_pic[9] += eps;

                // T_w
                Aold = T_w_iter[i];
                Anew = X[i][10];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L_pic[10] += eps;
            }

            for (int k = 0; k < B; ++k)
                L_pic[k] /= N;

            double alpha = 1.0;

            // Update vectors from X
            for (int i = 0; i < N; ++i) {

                rho_m[i] = alpha * X[i][0] + (1 - alpha) * rho_m_iter[i];
                rho_l[i] = alpha * X[i][1] + (1 - alpha) * rho_l_iter[i];
                alpha_m[i] = alpha * X[i][2] + (1 - alpha) * alpha_m_iter[i];
                alpha_l[i] = alpha * X[i][3] + (1 - alpha) * alpha_l_iter[i];
                p_m[i] = alpha * X[i][4] + (1 - alpha) * p_m_iter[i];
                p_l[i] = alpha * X[i][5] + (1 - alpha) * p_l_iter[i];
                v_m[i] = alpha * X[i][6] + (1 - alpha) * v_m_iter[i];
                v_l[i] = alpha * X[i][7] + (1 - alpha) * v_l_iter[i];
                T_m[i] = alpha * X[i][8] + (1 - alpha) * T_m_iter[i];
                T_l[i] = alpha * X[i][9] + (1 - alpha) * T_l_iter[i];
                T_w[i] = alpha * X[i][10] + (1 - alpha) * T_w_iter[i];

                T_sur[i] = alpha * (C31[i] * p_m[i] + C32[i] * T_m[i] + C33[i] * T_l[i] + C34[i] * T_w[i] + C35[i]) + (1 - alpha) * T_sur_iter[i];
                Gamma_xv[i] = alpha * (Kgeom * beta * (sigma_e * vapor_sodium::P_sat(T_sur[i]) - sigma_c * p_m[i])) + (1 - alpha) * Gamma_xv_iter[i];
            }

            // Ghost cells
            Gamma_xv[0] = 0.0;
            Gamma_xv[N - 1] = 0.0;

            T_sur[0] = T_sur[1];
            T_sur[N - 1] = T_sur[N - 2];

            bool found_nan = false;

            // After solving the linear system, before updating variables
            for (int i = 0; i < N; ++i) {

                std::string var_name;
                double var_val;

                if (std::isnan(T_m[i])) { found_nan = true; var_name = "T_m";   var_val = T_m[i]; }
                if (std::isnan(p_m[i])) { found_nan = true; var_name = "p_m";   var_val = p_m[i]; }
                if (std::isnan(T_l[i])) { found_nan = true; var_name = "T_l";   var_val = T_l[i]; }
                if (std::isnan(T_w[i])) { found_nan = true; var_name = "T_w";   var_val = T_w[i]; }
                if (std::isnan(v_m[i])) { found_nan = true; var_name = "v_m";   var_val = v_m[i]; }
                if (std::isnan(alpha_m[i])) { found_nan = true; var_name = "alpha_m"; var_val = alpha_m[i]; }

                if (found_nan) {
                    std::cout << "=== NaN DETECTED ===" << std::endl;
                    std::cout << "Time: " << time_total << ", Picard iter: " << pic << ", Cell: " << i << std::endl;
                    std::cout << "First NaN variable: " << var_name << std::endl;
                    std::cout << "--- All variables at cell " << i << " ---" << std::endl;
                    std::cout << "T_m  = " << T_m[i] << std::endl;
                    std::cout << "p_m  = " << p_m[i] << std::endl;
                    std::cout << "T_l  = " << T_l[i] << std::endl;
                    std::cout << "T_w  = " << T_w[i] << std::endl;
                    std::cout << "u_m  = " << v_m[i] << std::endl;
                    std::cout << "alpha_m = " << alpha_m[i] << std::endl;
                    std::cout << "--- Neighbors ---" << std::endl;
                    if (i > 0) {
                        std::cout << "Cell " << i - 1 << ": T_m=" << T_m[i - 1]
                            << " p_m=" << p_m[i - 1] << " T_l=" << T_l[i - 1]
                            << " T_w=" << T_w[i - 1] << std::endl;
                    }
                    if (i < N - 1) {
                        std::cout << "Cell " << i + 1 << ": T_m=" << T_m[i + 1]
                            << " p_m=" << p_m[i + 1] << " T_l=" << T_l[i + 1]
                            << " T_w=" << T_w[i + 1] << std::endl;
                    }
                }
            }

            if (found_nan == true) {

                pic = max_picard;
                break;
            }

            /*
            if (time_total == 0) {
                std::cout << "=== Time " << time_total << " Picard iter " << pic << " == = " << std::endl;
                auto p = [](double v) {
                    std::ostringstream os;
                    os << std::setprecision(2) << std::scientific << v;
                    return os.str();
                    };
                std::cout << std::setw(3) << "i"
                    << std::setw(12) << "rho_m" << std::setw(12) << "rho_l"
                    << std::setw(12) << "alpha_m" << std::setw(12) << "alpha_l"
                    << std::setw(12) << "p_m" << std::setw(12) << "p_l"
                    << std::setw(12) << "v_m" << std::setw(12) << "v_l"
                    << std::setw(12) << "T_m" << std::setw(12) << "T_l"
                    << std::setw(12) << "T_w" << std::endl;
                for (int i = 0; i < N; ++i) {
                    std::cout << std::setw(3) << i
                        << std::setw(12) << p(rho_m[i]) << std::setw(12) << p(rho_l[i])
                        << std::setw(12) << p(alpha_m[i]) << std::setw(12) << p(alpha_l[i])
                        << std::setw(12) << p(p_m[i]) << std::setw(12) << p(p_l[i])
                        << std::setw(12) << p(v_m[i]) << std::setw(12) << p(v_l[i])
                        << std::setw(12) << p(T_m[i]) << std::setw(12) << p(T_l[i])
                        << std::setw(12) << p(T_w[i]) << std::endl;
                }
            }
            */

            bool valid_all = true;
            for (int i = 1; i < N - 1; ++i) {

                // Half timestep and redo iterations
                if (rho_m[i] < 1e-8 || rho_l[i] < 1e-8 || 
                    alpha_m[i] < -0.002 || alpha_l[i] < -0.002 || 
                    alpha_m[i] > 1.002 || alpha_l[i] > 1.002 || 
                    p_m[i] < 1e-4 ||
                    T_m[i] < 100 || T_l[i] < 100 || T_w[i] < 100) {

                    valid_all = false;
                }

                if ((alpha_m[i] < 0 && alpha_m[i] > -0.002)) {

                    alpha_m[i] = 0.0;
                    alpha_l[i] = 1.0;

                }

                if ((alpha_l[i] < 0 && alpha_l[i] > -0.002)) {

                    alpha_l[i] = 0.0;
                    alpha_m[i] = 1.0;

                }
            }

            if (valid_all == false) {

                pic = max_picard;
                break;
            }

            // Check if variable converged
            for (int k = 0; k < B; ++k)
                conv_var[k] = (L_pic[k] < pic_tol[k]);

            // Check if all variables converged
            bool conv_all = true;
            for (int k = 0; k < B; ++k)
                conv_all = conv_all && conv_var[k];

            for (int i = 1; i < N; i++) {

                power_flux_wx[i] = (C61[i] * p_m[i] + C62[i] * T_m[i] + C63[i] * T_l[i] + C64[i] * T_w[i] + C65[i]) * pi * (r_i * r_i) * dz;
                power_flux_xw[i] = (C66[i] * p_m[i] + C67[i] * T_m[i] + C68[i] * T_l[i] + C69[i] * T_w[i] + C70[i]) * pi * (r_o * r_o - r_i * r_i) * dz;

                power_flux_vx[i] = (C41[i] * p_m[i] + C42[i] * T_m[i] + C43[i] * T_l[i] + C44[i] * T_w[i] + C45[i]) * pi * (r_i * r_i) * dz;
                power_flux_xv[i] = (C46[i] * p_m[i] + C47[i] * T_m[i] + C48[i] * T_l[i] + C49[i] * T_w[i] + C50[i]) * pi * (r_i * r_i) * dz;

                power_mass_vx[i] = (C51[i] * p_m[i] + C52[i] * T_m[i] + C53[i] * T_l[i] + C54[i] * T_w[i] + C55[i]) * pi * (r_i * r_i) * dz;
                power_mass_xv[i] = (C56[i] * p_m[i] + C57[i] * T_m[i] + C58[i] * T_l[i] + C59[i] * T_w[i] + C60[i]) * pi * (r_i * r_i) * dz;
            }

            if (conv_all) {

                halves = std::max(0, --halves);
                break;                                      // Picard converged, so break the loops
            }
        }

        // Picard converged or max iterations reached
        if (pic != max_picard) {

            // Update total time elapsed
            time_total += dt;

            if (time_total >= t_last_print + print_interval) {

                // Time between timesteps [ms]
                auto t_now = std::chrono::high_resolution_clock::now();
                double simulation_time = std::chrono::duration<double, std::milli>(t_now - t_start_timestep).count();

                // Time from the start of the simulation
                double clock_time = std::chrono::duration<double>(t_now - t_start_simulation).count();

                // --------- PRIMARY VARIABLES -----------

                for (int i = 1; i < N - 1; ++i) {

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

                    gamma_output << Gamma_xv[i] * (pi * r_i * r_i * dz) << " ";

                    power_flux_ow_output << q_ow[i] * (2 * pi * r_o * dz) << " ";

                    power_flux_wx_output << power_flux_wx[i] << " ";
                    power_flux_xw_output << power_flux_xw[i] << " ";

                    power_mass_vx_output << power_mass_vx[i] << " ";
                    power_mass_xv_output << power_mass_xv[i] << " ";

                    power_flux_vx_output << power_flux_vx[i] << " ";
                    power_flux_xv_output << power_flux_xv[i] << " ";

                    psat_output << p_saturation[i] << " ";
                    tsur_output << T_sur[i] << " ";

					dpcap_output << DPcap[i] << " ";
                }

                // --------- RESIDUALS -----------

                for (int i = 1; i < N - 1; ++i) {

                    // Check parabolic profiles

                    c_w[i] = C1[i] * p_m[i] + C2[i] * T_m[i] + C3[i] * T_l[i] + C4[i] * T_w[i] + C5[i];
                    c_x[i] = C6[i] * p_m[i] + C7[i] * T_m[i] + C8[i] * T_l[i] + C9[i] * T_w[i] + C10[i];
                    b_w[i] = C11[i] * p_m[i] + C12[i] * T_m[i] + C13[i] * T_l[i] + C14[i] * T_w[i] + C15[i];
                    a_w[i] = C16[i] * p_m[i] + C17[i] * T_m[i] + C18[i] * T_l[i] + C19[i] * T_w[i] + C20[i];
                    b_x[i] = C21[i] * p_m[i] + C22[i] * T_m[i] + C23[i] * T_l[i] + C24[i] * T_w[i] + C25[i];
                    a_x[i] = C26[i] * p_m[i] + C27[i] * T_m[i] + C28[i] * T_l[i] + C29[i] * T_w[i] + C30[i];

                    residual1[i] = a_w[i] + Eio1 * b_w[i] + Eio2 * c_w[i] - T_w[i];
                    residual2[i] = a_x[i] + Evi1 * b_x[i] + Evi2 * c_x[i] - T_l[i];
                    residual3[i] = a_w[i] + r_i * b_w[i] + r_i * r_i * c_w[i] - a_x[i] - r_i * b_x[i] - r_i * r_i * c_x[i];
                    residual4[i] = k_w[i] * b_w[i] + 2 * r_i * k_w[i] * c_w[i] - k_l[i] * b_x[i] - 2 * r_i * k_l[i] * c_x[i];
                    residual5[i] = b_w[i] + 2 * r_o * c_w[i] - q_ow[i] / k_w[i];
                    residual6[i] = Ex3[i] * a_x[i] + Ex4[i] * b_x[i] + Ex5[i] * c_x[i] - Ex8[i] - Ex6[i] * T_m[i] - Ex7[i] * p_m[i];

                    residual_1_output << residual1[i] << " ";
                    residual_2_output << residual2[i] << " ";
                    residual_3_output << residual3[i] << " ";
                    residual_4_output << residual4[i] << " ";
                    residual_5_output << residual5[i] << " ";
                    residual_6_output << residual6[i] << " ";

                    // Check interface liquid vapor temperature

                    T_sur[i] = C31[i] * p_m[i] + C32[i] * T_m[i] + C33[i] * T_l[i] + C34[i] * T_w[i] + C35[i];
                    T_sur_lin[i] = a_x[i] + b_x[i] * r_v + c_x[i] * r_v * r_v;
                    T_sur_diff[i] = T_sur[i] - T_sur_lin[i];

                    T_sur_diff_output << T_sur_diff[i] << " ";

                    // Check mass exchange

                    Gamma_xv_approx[i] = aGamma[i] + bGamma[i] * T_sur[i] + cGamma[i] * (p_m[i] - p_m_iter[i]);
                    Gamma_xv_lin[i] = C36[i] * p_m[i] + C37[i] * T_m[i] + C38[i] * T_l[i] + C39[i] * T_w[i] + C40[i];

                    Gamma_xv_approx_error[i] = Gamma_xv[i] - Gamma_xv_approx[i];
                    Gamma_xv_lin_error[i] = Gamma_xv[i] - Gamma_xv_lin[i];
                    Gamma_xv_diff_error[i] = Gamma_xv_approx[i] - Gamma_xv_lin[i];

                    Gamma_xv_approx_error_output << Gamma_xv_approx_error[i] * dz * (pi * r_i * r_i) << " ";
                    Gamma_xv_lin_error_output << Gamma_xv_lin_error[i] * dz * (pi * r_i * r_i) << " ";
                    Gamma_xv_diff_error_output << Gamma_xv_diff_error[i] * dz * (pi * r_i * r_i) << " ";

                    // Check residual heat exchange liquid wall

                    power_residual_xw[i] = power_flux_wx[i] + power_flux_xw[i];

                    power_residual_wx_output << power_residual_xw[i] << " ";

                    // Check residual heat exchange liquid vapor

                    power_residual_xv[i] = +power_flux_vx[i] + power_flux_xv[i] + power_mass_xv[i] + power_mass_vx[i];

                    power_residual_xv_output << power_residual_xv[i] << " ";
                }

                // Check mass, energy and momentum accumulators, balances and differences for each of the phases

                double acc_mass_m = 0;
                double acc_mass_l = 0;

                double acc_energy_m = 0.0;
                double acc_energy_l = 0.0;

                double acc_mom_m = 0.0;
                double acc_mom_l = 0.0;

                double bal_mass_m = 0;
                double bal_mass_l = 0;

                double bal_energy_m = 0.0;
                double bal_energy_l = 0.0;

                double bal_mom_m = 0.0;
                double bal_mom_l = 0.0;

                double diff_mass_m = 0;
                double diff_mass_l = 0;

                double diff_energy_m = 0.0;
                double diff_energy_l = 0.0;

                double diff_mom_m = 0.0;
                double diff_mom_l = 0.0;

                double acc_energy_w = 0.0;
                double bal_energy_w = 0.0;
                double diff_energy_w = 0.0;

                double global_energy_balance = 0.0;

                for (int i = 1; i < N - 1; ++i) {

                    // ----- Accumulators

                    // acc_mass_m += (alpha_m[i] * rho_m[i] - alpha_m_old[i] * rho_m_old[i]) / dt * dz * (pi * r_i * r_i);
                    // acc_mass_l += eps_v * (alpha_l[i] * rho_l[i] - alpha_l_old[i] * rho_l_old[i]) / dt * dz * (pi * r_i * r_i);

                    acc_mass_m += + (((rho_m[i] * alpha_m_iter[i]) + (rho_m_iter[i] * alpha_m[i]))
                        - ((rho_m_old[i] * alpha_m_old[i]) + (rho_m_iter[i] * alpha_m_iter[i]))) * (pi * r_i * r_i * dz) / dt;
                    acc_mass_l += +eps_v * (((rho_l[i] * alpha_l_iter[i]) + (rho_l_iter[i] * alpha_l[i]))
                        - ((rho_l_old[i] * alpha_l_old[i]) + (rho_l_iter[i] * alpha_l_iter[i]))) * (pi * r_i * r_i * dz) / dt;

                    // double e_now_l = eps_v * alpha_l[i] * rho_l[i] * cp_l[i] * T_l[i];
                    // double e_old_l = eps_v * alpha_l_old[i] * rho_l_old[i] * cp_l_old[i] * T_l_old[i];

                    double e_now_l = eps_v * alpha_l[i] * rho_l_iter[i] * cp_l[i] * T_l_iter[i] + 
                                    eps_v * alpha_l_iter[i] * rho_l[i] * cp_l[i] * T_l_iter[i] + 
                                    eps_v * alpha_l_iter[i] * rho_l_iter[i] * cp_l[i] * T_l[i];

                    double e_old_l = eps_v * (rho_l_old[i] * alpha_l_iter[i] * cp_l_old[i] * T_l_iter[i]) +
                                    eps_v * (rho_l_iter[i] * alpha_l_iter[i] * cp_l_old[i] * T_l_old[i]) +
                                    eps_v * (rho_l_iter[i] * alpha_l_old[i] * cp_l_old[i] * T_l_iter[i]);

                    acc_energy_l += ((e_now_l - e_old_l) / dt * (pi * r_i * r_i * dz));

                    // double e_now_m = alpha_m[i] * rho_m[i] * cp_m[i] * T_m[i];
                    // double e_old_m = alpha_m_old[i] * rho_m_old[i] * cp_m_old[i] * T_m_old[i];

                    double e_now_m = alpha_m[i] * rho_m_iter[i] * cp_m[i] * T_m_iter[i] +
                                    alpha_m_iter[i] * rho_m[i] * cp_m[i] * T_m_iter[i] +
                                    alpha_m_iter[i] * rho_m_iter[i] * cp_m[i] * T_m[i];

                    double e_old_m = (rho_m_old[i] * alpha_m_iter[i] * cp_m[i] * T_m_iter[i]) +
                                    (rho_m_iter[i] * alpha_m_old[i] * cp_m[i] * T_m_iter[i]) +
                                    (rho_m_iter[i] * alpha_m_iter[i] * cp_m[i] * T_m_old[i]);

                    acc_energy_m += ((e_now_m - e_old_m) / dt * (pi * r_i * r_i * dz));

                    if (i != 1 && i != N - 2) {

                        // double arho_new_mm = (alpha_m[i - 1] * rho_m[i - 1] + alpha_m[i] * rho_m[i]) * v_m[i] / 2.0 / dt;
                        // double arho_old_mm = (alpha_m_old[i - 1] * rho_m_old[i - 1] + alpha_m_old[i] * rho_m_old[i]) *  v_m_old[i] / 2.0 / dt;

                        double arho_new_mm = (+ (alpha_m_iter[i - 1] + alpha_m_iter[i]) * (rho_m[i - 1] + rho_m[i]) * v_m_iter[i] / (4 * dt)
                                            + (rho_m_iter[i - 1] + rho_m_iter[i]) * (alpha_m[i - 1] + alpha_m[i]) * v_m_iter[i] / (4 * dt)
                                            + (alpha_m_iter[i - 1] * rho_m_iter[i - 1] + alpha_m_iter[i] * rho_m_iter[i]) * v_m[i] / (2 * dt));

                        double arho_old_mm = (+ (alpha_m_iter[i - 1] * rho_m_iter[i - 1] + alpha_m_iter[i] * rho_m_iter[i]) * v_m_old[i] / (2 * dt)
                                            + ((alpha_m_iter[i - 1] + alpha_m_iter[i]) * v_m_iter[i] * (rho_m_old[i - 1] + rho_m_old[i])) / (4 * dt)
                                            + ((rho_m_iter[i - 1] + rho_m_iter[i]) * v_m_iter[i] * (alpha_m_old[i - 1] + alpha_m_old[i]) / (4 * dt)));

                        acc_mom_m += (arho_new_mm - arho_old_mm) * (pi * r_i * r_i * dz);

                        // double arho_new_ml = eps_v * (alpha_l[i - 1] * rho_l[i - 1] + alpha_l[i] * rho_l[i]) / 2.0;
                        // double arho_old_ml = eps_v * (alpha_l_old[i - 1] * rho_l_old[i - 1] + alpha_l_old[i] * rho_l_old[i]) / 2.0;

                        double arho_new_ml = eps_v * (+ (alpha_l_iter[i - 1] + alpha_l_iter[i]) * (rho_l[i - 1] + rho_l[i]) * v_l_iter[i] / (4 * dt)
                                            + (rho_l_iter[i - 1] + rho_l_iter[i]) * (alpha_l[i - 1] + alpha_l[i]) * v_l_iter[i] / (4 * dt)
                                            + (alpha_l_iter[i - 1] * rho_l_iter[i - 1] + alpha_l_iter[i] * rho_l_iter[i]) * v_l[i] / (2 * dt));

                        double arho_old_ml = eps_v * (+ (alpha_l_iter[i - 1] * rho_l_iter[i - 1] + alpha_l_iter[i] * rho_l_iter[i]) * v_l_old[i] / (2 * dt)
                                            + (alpha_l_iter[i - 1] + alpha_l_iter[i]) * v_l_iter[i] * (rho_l_old[i - 1] + rho_l_old[i]) / (4 * dt)
                                            + (rho_l_iter[i - 1] + rho_l_iter[i]) * v_l_iter[i] * (alpha_l_old[i - 1] + alpha_l_old[i]) / (4 * dt));

                        acc_mom_l += (arho_new_ml - arho_old_ml) * (pi * r_i * r_i * dz);

                    }

                    acc_energy_w += rho_w[i] * cp_w[i] * (T_w[i] - T_w_old[i]) / dt * pi *  (r_o * r_o - r_i * r_i) * dz;

                    // ---------- Balances

                    // Check total mass vapor sources

                    bal_mass_m += Gamma_xv_lin[i] * (pi * r_i * r_i * dz);

                    // Check total mass liquid sources

                    bal_mass_l += -Gamma_xv_lin[i] * (pi * r_i * r_i * dz);

                    // Check total heat vapor sources

                    bal_energy_m += (power_flux_xv[i] + power_mass_xv[i])
                                - ((p_m_iter[i] * (v_m_iter[i + 1] - v_m_iter[i]) / (2 * dz)
                                    + p_m_iter[i] / dt) * alpha_m[i]
                                + (-p_m_iter[i] * (alpha_m_iter[i] + alpha_m_iter[i + 1]) / (2 * dz)) * v_m[i]
                                - (+p_m_iter[i] * (alpha_m_iter[i] + alpha_m_iter[i + 1]) * v_m_iter[i + 1] / (2 * dz)
                                    - p_m_iter[i] * (alpha_m_iter[i] + alpha_m_iter[i - 1]) * v_m_iter[i] / (2 * dz)
                                    + (p_m_iter[i] * alpha_m_old[i]) / dt)
                                - (p_m_iter[i] * (v_m_iter[i]) / (2 * dz)) * alpha_m[i - 1]
                                + (p_m_iter[i] * (v_m_iter[i + 1]) / (2 * dz)) * alpha_m[i + 1]
                                + (p_m_iter[i] * (alpha_m_iter[i] + alpha_m_iter[i + 1]) / (2 * dz)) * v_m[i + 1]) * (pi * r_i * r_i * dz);
                                

                    // Check total heat liquid sources

                    bal_energy_l += (power_flux_vx[i] + power_mass_vx[i] + power_flux_wx[i])
                                - ((eps_v * p_l_iter[i] * (v_l_iter[i + 1] - v_l_iter[i]) / (2 * dz)
                                    + eps_v * p_l_iter[i] / dt) * alpha_l[i]
                                + (-eps_v * p_l_iter[i] * (alpha_l_iter[i] + alpha_l_iter[i + 1]) / (2 * dz)) * v_l[i]
                                - (+eps_v * p_l_iter[i] * (alpha_l_iter[i] + alpha_l_iter[i + 1]) * v_l_iter[i + 1] / (2 * dz)
                                    - eps_v * p_l_iter[i] * (alpha_l_iter[i] + alpha_l_iter[i - 1]) * v_l_iter[i] / (2 * dz)
                                    + eps_v * (p_l_iter[i] * alpha_l_old[i]) / dt)
                                - (eps_v * p_l_iter[i] * (v_l_iter[i]) / (2 * dz)) * alpha_l[i - 1]
                                + (eps_v * p_l_iter[i] * (v_l_iter[i + 1]) / (2 * dz)) * alpha_l[i + 1]
                                + (eps_v * p_l_iter[i] * (alpha_l_iter[i] + alpha_l_iter[i + 1]) / (2 * dz)) * v_l[i + 1]) * (pi * r_i * r_i * dz);

                    if(i != 1 && i != N - 2) {

                        // Check total momentum sources vapor

                        bal_mom_m += -(p_m[i] - p_m[i - 1]) * (alpha_m_iter[i - 1] + alpha_m_iter[i]) / (2 * dz) * (pi * r_i * r_i * dz)
                                        - fm[i] * (rho_m_iter[i - 1] + rho_m_iter[i]) * std::abs(v_m_iter[i]) * v_m[i] / (8 * r_v) * (pi * r_i * r_i * dz);

                        // Check total momentum sources liquid

                        bal_mom_l += -eps_v * (p_l[i] - p_l[i - 1]) * (alpha_l_iter[i - 1] + alpha_l_iter[i]) / (2 * dz) * (pi * r_i * r_i * dz)
                                        - (alpha_l[i] - alpha_l[i - 1]) * (DPcap[i - 1] + DPcap[i]) / (2 * dz) * (pi * r_i * r_i * dz)
                                        - fl[i] * std::abs(v_l_iter[i]) * v_l[i] * (pi * r_i * r_i * dz);
                    }
                    // Check total heat wall sources

                    bal_energy_w += +q_ow[i] * 2 * pi * dz * r_o + power_flux_xw[i];
                
                    // Check total external heat sources

                    global_energy_balance += q_ow[i] * (2 * pi * r_o * dz);       
                }

                // --------- Differences

                diff_mass_m = (acc_mass_m - bal_mass_m);
                diff_mass_l = (acc_mass_l - bal_mass_l);

                diff_energy_m = (acc_energy_m - bal_energy_m);
                diff_energy_l = (acc_energy_l - bal_energy_l);

                diff_mom_m = (acc_mom_m - bal_mom_m);
                diff_mom_l = (acc_mom_l - bal_mom_l);

                diff_energy_w = (acc_energy_w - bal_energy_w);

                time_output << time_total << " ";
                dt_output << dt << " ";
                simulation_time_output << simulation_time << " ";
                clock_time_output << clock_time << " ";

                acc_mass_v_output << acc_mass_m << " ";
                acc_mass_l_output << acc_mass_l << " ";

                acc_energy_v_output << acc_energy_m << " ";
                acc_energy_l_output << acc_energy_l << " ";

                acc_mom_v_output << acc_mom_m << " ";
                acc_mom_l_output << acc_mom_l << " ";

                bal_mass_v_output << bal_mass_m << " ";
                bal_mass_l_output << bal_mass_l << " ";

                bal_energy_v_output << bal_energy_m << " ";
                bal_energy_l_output << bal_energy_l << " ";

                bal_mom_v_output << bal_mom_m << " ";
                bal_mom_l_output << bal_mom_l << " ";

                diff_mass_v_output << diff_mass_m << " ";
                diff_mass_l_output << diff_mass_l << " ";

                diff_energy_v_output << diff_energy_m << " ";
                diff_energy_l_output << diff_energy_l << " ";

                diff_mom_v_output << diff_mom_m << " ";
                diff_mom_l_output << diff_mom_l << " ";

                acc_energy_w_output << acc_energy_w << " ";
                bal_energy_w_output << bal_energy_w << " ";
                diff_energy_w_output << diff_energy_w << " ";

                global_energy_balance_output << global_energy_balance << " ";

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

                power_flux_wx_output << "\n";
                power_flux_xw_output << "\n";

                power_mass_vx_output << "\n";
                power_mass_xv_output << "\n";

                power_flux_vx_output << "\n";
                power_flux_xv_output << "\n";

                psat_output << "\n";
                tsur_output << "\n";

				dpcap_output << "\n";
				power_flux_ow_output << "\n";

                power_residual_wx_output << "\n";
                power_residual_xv_output << "\n";

                residual_1_output << "\n";
                residual_2_output << "\n";
                residual_3_output << "\n";
                residual_4_output << "\n";
                residual_5_output << "\n";
                residual_6_output << "\n";

                T_sur_diff_output << "\n";

                Gamma_xv_approx_error_output << "\n";
                Gamma_xv_lin_error_output << "\n";
                Gamma_xv_diff_error_output << "\n";

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
                power_flux_wx_output.flush();
                power_flux_xw_output.flush();
                power_mass_vx_output.flush();
                power_mass_xv_output.flush();
                power_flux_vx_output.flush();
                power_flux_xv_output.flush();
                psat_output.flush();
                tsur_output.flush();

                time_output.flush();
                dt_output.flush();
                simulation_time_output.flush();
                clock_time_output.flush();

				dpcap_output.flush();
				power_flux_ow_output.flush();

                acc_mass_v_output.flush();
                acc_mass_l_output.flush();

                acc_energy_v_output.flush();
                acc_energy_l_output.flush();

                acc_mom_v_output.flush();
                acc_mom_l_output.flush();

                bal_mass_v_output.flush();
                bal_mass_l_output.flush();

                bal_energy_v_output.flush();
                bal_energy_l_output.flush();

                bal_mom_v_output.flush();
                bal_mom_l_output.flush();

                diff_mass_v_output.flush();
                diff_mass_l_output.flush();

                diff_energy_v_output.flush();
                diff_energy_l_output.flush();

                diff_mom_v_output.flush();
                diff_mom_l_output.flush();

                acc_energy_w_output.flush();
                bal_energy_w_output.flush();
                diff_energy_w_output.flush();

                global_energy_balance_output.flush();

                power_residual_wx_output.flush();
                power_residual_xv_output.flush();

                residual_1_output.flush();
                residual_2_output.flush();
                residual_3_output.flush();
                residual_4_output.flush();
                residual_5_output.flush();
                residual_6_output.flush();

                T_sur_diff_output.flush();

                Gamma_xv_approx_error_output.flush();
                Gamma_xv_lin_error_output.flush();
                Gamma_xv_diff_error_output.flush();

                rho_m_old = rho_m;
                rho_l_old = rho_l;
                alpha_m_old = alpha_m;
                alpha_l_old = alpha_l;
                p_m_old = p_m;
                p_l_old = p_l;
                v_m_old = v_m;
                v_l_old = v_l;
                T_m_old = T_m;
                T_l_old = T_l;
                T_w_old = T_w;

                Gamma_xv_old = Gamma_xv;
                T_sur_old = T_sur;

                cp_l_old = cp_l;
                cp_m_old = cp_m;

                t_last_print += print_interval;
            }
        }
        else {

            // Rollback to previous time step (new = old) and halve dt

            rho_m = rho_m_old;
            rho_l = rho_l_old;
            alpha_m = alpha_m_old;
            alpha_l = alpha_l_old;
            p_m = p_m_old;
            p_l = p_l_old;
            v_m = v_m_old;
            v_l = v_l_old;
            T_m = T_m_old;
            T_l = T_l_old;
            T_w = T_w_old;

            cp_l = cp_l_old;
            cp_m = cp_m_old;

            T_sur = T_sur_old;
            Gamma_xv = Gamma_xv_old;

            halves += 1;        // Half again the timestep
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

    power_flux_wx_output.close();
    power_flux_xw_output.close();
    power_mass_vx_output.close();
    power_mass_xv_output.close();
    power_flux_vx_output.close();
    power_flux_xv_output.close();
    psat_output.close();
    tsur_output.close();

    time_output.close();
    dt_output.close();
    simulation_time_output.close();
    clock_time_output.close();

	dpcap_output.close();
	power_flux_ow_output.close();

    acc_mass_v_output.close();
    acc_mass_l_output.close();

    acc_energy_v_output.close();
    acc_energy_l_output.close();

    acc_mom_v_output.close();
    acc_mom_l_output.close();

    bal_mass_v_output.close();
    bal_mass_l_output.close();

    bal_energy_v_output.close();
    bal_energy_l_output.close();

    bal_mom_v_output.close();
    bal_mom_l_output.close();

    diff_mass_v_output.close();
    diff_mass_l_output.close();

    diff_energy_v_output.close();
    diff_energy_l_output.close();

    diff_mom_v_output.close();
    diff_mom_l_output.close();

    acc_energy_w_output.close();
    bal_energy_w_output.close();
    diff_energy_w_output.close();

    global_energy_balance_output.close();

    power_residual_wx_output.close();
    power_residual_xv_output.close();

    residual_1_output.close();
    residual_2_output.close();
    residual_3_output.close();
    residual_4_output.close();
    residual_5_output.close();
    residual_6_output.close();

    T_sur_diff_output.close();

    Gamma_xv_approx_error_output.close();
    Gamma_xv_lin_error_output.close();
    Gamma_xv_diff_error_output.close();

    return 0;
}
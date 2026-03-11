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

#include "steel.h"
#include "liquid_sodium.h"
#include "vapor_sodium.h"
#include "solver.h"

int main() {

    // =======================================================================
    //
    //                        [CONSTANTS AND VARIABLES]
    //
    // =======================================================================

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
    const double K = 1e-10;                 /// Permeability [m^2]
    const double CF = 1e5;                  /// Forchheimer coefficient [1/m]

    // Evaporation and condensation parameters
    const double eps_s = 0.01;               /// Surface fraction of the wick available for phasic interface [-]
    const double sigma_e = 0.05;            /// Evaporation accomodation coefficient [-]. 1 means optimal evaporation
    const double sigma_c = 0.05;            /// Condensation accomodation coefficient [-]. 1 means optimal condensation

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
    const double V_wall = dz * M_PI * (r_o * r_o - r_i * r_i);                  /// Volume of the wall cell [m3]
    const double V_liquid = dz * M_PI * (r_i * r_i - r_v * r_v);                /// Volume of the liquid cell [m3]
    const double V_vapor = dz * M_PI * r_v * r_v;                               /// Volume of the vapor cell [m3]
    const double lateral_surface = dz * 2 * M_PI * r_o;                         /// Lateral outer surface [m2]

    // Constant geometrical parameters for the radial model
    const double Eio1 = 2.0 / 3.0 * (r_o + r_i - 1 / (1 / r_o + 1 / r_i));
    const double Eio2 = 0.5 * (r_o * r_o + r_i * r_i);
    const double Evi1 = 2.0 / 3.0 * (r_i + r_v - 1 / (1 / r_i + 1 / r_v));
    const double Evi2 = 0.5 * (r_i * r_i + r_v * r_v);

    // Environmental boundary conditions
    const double h_conv = 1;                                                       /// Convective heat transfer coefficient for external heat removal [W/m^2/K]
    const double power = 1000;                                                      /// Power at the evaporator side [W]
    const double T_env = 280.0;                                                     /// External environmental temperature [K]
    const double q_pp_evaporator = power / (2 * M_PI * evaporator_length * r_o);    /// Heat flux at evaporator from given power [W/m^2]

    // Time-stepping parameters
    double dt_user = 1e-6;                              /// Initial time step [s] (then it is updated according to the limits)
    double dt = dt_user;                                /// Actual used time step [s]
    const int tot_iter = 1e8;                           /// Number of timesteps [-]
    double time_total = 0.0;                            /// Total time elapsed [s]
    int halves = 0;                                     /// Number of times the time step has been halved [-]

    // Picard loops parameters	          
    int pic = 0;                                        /// Number of Picard iterations [-]
    const int max_picard = 10;                         /// Maximum number of Picard iterations per timestep [-]
    std::array<double, B> L_pic;                        /// Picard residuals [-]
    std::array<bool, B> conv_var;                       /// Bool array if parameter converged or not [-]
    std::array<double, B> pic_tol = {                   /// Tolerance for the convergence of Picard loop [-]
        1e-4,  // rho_m
        1e-4,  // rho_l
        1e-4,  // alpha_m
        1e-4,  // alpha_l
        1e-2,  // p_m
        1e-2,  // p_l
        1e-2,  // v_m
        1e-2,  // v_l
        1e-4,  // T_m
        1e-4,  // T_l
        1e-4   // T_w
    };

    // Mesh z positions
    std::vector<double> mesh(N, 0.0);
    for (int i = 0; i < N; ++i) mesh[i] = i * dz;     /// Mesh discretization

    // State variables definition and initialization
    std::vector<double> rho_m(N, 0.01);                 /// Mixture density [kg/m3]
    std::vector<double> rho_l(N, 1000);                 /// Liquid density [kg/m3]
    std::vector<double> p_m(N);                         /// Mixture pressure [Pa]
    std::vector<double> p_l(N);                         /// Liquid pressure [Pa]
    std::vector<double> v_m(N + 1, 1.0);                  /// Mixture velocity [m/s]
    std::vector<double> v_l(N + 1, -0.1);                  /// Liquid velocity [m/s]
    std::vector<double> T_m(N);                         /// Mixture bulk temperature [K]
    std::vector<double> T_l(N);                         /// Liquid bulk temperature [K]
    std::vector<double> T_w(N);                         /// Wall bulk temperature [K]

    std::vector<double> alpha_m(N), alpha_l(N);

    for (int i = 0; i < N; ++i) {
        double s = static_cast<double>(i) / (N - 1);   // 0 → 1

        alpha_m[i] = 0.95 - 0.55 * s;   // 0.95 → 0.40
        alpha_l[i] = 0.05 + 0.55 * s;   // 0.05 → 0.60
    }

    // Secondary useful variables
    std::vector<double> Gamma_xv(N, 0.0);                           /// Exact mass volumetric source [kg/m3s]
    std::vector<double> Gamma_xv_lin(N, 0.0);                       /// Linearized mass volumetric source (with C coefficients) [kg/m3s]
    std::vector<double> Gamma_xv_approx_error(N, 0.0);                      /// Residual between approximation and exact mass volumetric source 1 [kg/m3s]
    std::vector<double> Gamma_xv_lin_error(N, 0.0);                     /// Residual between approximation and exact mass volumetric source 2 [kg/m3s]
    std::vector<double> Gamma_xv_diff_error(N, 0.0);
    std::vector<double> Gamma_xv_approx(N, 0.0);                    /// Approximated mass volumetric source (with gamma coefficients) [kg/m3s]
    std::vector<double> T_sur(N);                                   /// Wick-vapor surface temperature [K]
    std::vector<double> q_pp(N, 0.0);                               /// Heat flux profile [W/m^2]
    std::vector<double> phi_x_v(N, 0.0);                            /// Mass flux [kg/m2s]
    std::vector<double> heat_source_wall_liquid_flux(N, 0.0);       /// Heat volumetric source from wall to liquid due to difference in temperature [W/m3]
    std::vector<double> heat_source_liquid_wall_flux(N, 0.0);       /// Heat volumetric source from liquid to wall due to difference in temperature [W/m3]
    std::vector<double> heat_source_vapor_liquid_phase(N, 0.0);     /// Heat volumetric source from vapor to liquid due to phase change [W/m3]
    std::vector<double> heat_source_liquid_vapor_phase(N, 0.0);     /// Heat volumetric source from liquid to vapor due to phase change [W/m3]
    std::vector<double> heat_source_vapor_liquid_flux(N, 0.0);      /// Heat volumetric source from vapor to liquid due to difference in temperature [W/m3]
    std::vector<double> heat_source_liquid_vapor_flux(N, 0.0);      /// Heat volumetric source from liquid to vapor due to difference in temperature [W/m3]
    std::vector<double> p_saturation(N);                            /// Saturation pressure at the temperature of the wick-vapor surface [Pa]
    std::vector<double> DPcap(N, 0.0);                              /// Capillary pressure difference between mixture and vapor [Pa]
    std::vector<double> energy_wall(N, 0.0);                        /// Wall internal energy [J]
    std::vector<double> energy_liquid(N, 0.0);                      /// Liquid internal energy [J]
    std::vector<double> energy_vapor(N, 0.0);                       /// Mixture internal energy [J]
    std::vector<double> aGamma(N);                                  /// a coefficient for the mass volumetric source approximation [?]
    std::vector<double> bGamma(N);                                  /// b coefficient for the mass volumetric source approximation [?]
    std::vector<double> cGamma(N);                                  /// c coefficient for the mass volumetric source approximation [?]

    double h_xv_v;                                                  /// Specific enthalpy [J/kg] of vapor upon phase change between wick and vapor
    double h_vx_x;                                                  /// Specific enthalpy [J/kg] of wick upon phase change between vapor and wick

    const double T_left = 1000.0;                        /// First node initialization temperature [K]
    const double T_right = 1000.0;                       /// Last node initialization temperature [K]

    // Temperatures initialization
    for (int i = 0; i < N; ++i) {

        const double s = static_cast<double>(i) / (N - 1);

        T_m[i] = T_left + s * (T_right - T_left);
        T_l[i] = T_left + s * (T_right - T_left);
        T_w[i] = T_left + s * (T_right - T_left);

        T_sur[i] = T_left + s * (T_right - T_left);

        p_m[i] = vapor_sodium::P_sat(T_m[i]);
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
    std::vector<double> Gamma_xv_old = Gamma_xv;                           /// Exact mass volumetric source [kg/m3s]

    /// Iter variables
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
    std::vector<double> Gamma_xv_iter = Gamma_xv_old;                           /// Exact mass volumetric source [kg/m3s]

    std::vector<double> T_sur_iter = T_sur;

    std::vector<double> phi_x_v_iter(N, 0.0);

    // Blocks definition
    std::vector<SparseBlock> L(N), D(N), R(N);
    std::vector<VecBlock> Q(N), X(N);

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

	std::ofstream dpcap_output(name + "/delta_p_capillary.txt", std::ios::trunc);
	std::ofstream q_pp_output(name + "/q_pp.txt", std::ios::trunc);

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

	dpcap_output << std::setprecision(global_precision);

	q_pp_output << std::setprecision(global_precision);

    for (int i = 0; i < N; ++i) mesh_output << i * dz << " ";

    mesh_output.flush();
    mesh_output.close();

    // Vapor Equation of State update function. Updates density
    auto eos_update = [&](std::vector<double>& rho_, const std::vector<double>& p_, const std::vector<double>& T_) {

        for (int i = 0; i < N; i++) { rho_[i] = std::max(1e-6, p_[i] / (Rv * T_[i])); }

    }; eos_update(rho_m, p_m, T_m);

    std::vector<double> heat_balance_wx(N, 0);
    std::vector<double> heat_balance_xv(N, 0);

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

    std::vector<double> k_x(N, 0.0);
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

    std::vector<double> k_w(N, 0.0);

    std::vector<double> heat_balance_xv_full(N, 0.0);

    std::vector<double> conduction_err(N, 0.0);
    std::vector<double> convection_err(N, 0.0);
    std::vector<double> phase_change_err(N, 0.0);

    #pragma endregion

    /// Print number of working threads
    std::cout << "Threads: " << omp_get_max_threads() << "\n";

    double start = omp_get_wtime();

    // Time-stepping loop
    for (int n = 0; n < tot_iter; ++n) {

        // Timestep selection
        dt = std::max(dt_user * pow(0.5, halves), 1e-10);    // Halfing of the timestep up to a lower bound     

        // Picard iteration loop
        for (pic = 0; pic < max_picard; ++pic) {

            // Cleaning all the blocks (the add function adds block and does not overwrite, so this is necessary)
            for (int i = 0; i < N; i++) {
                L[i].row.clear(); L[i].col.clear(); L[i].val.clear();
                D[i].row.clear(); D[i].col.clear(); D[i].val.clear();
                R[i].row.clear(); R[i].col.clear(); R[i].val.clear();
            }

            /// Picard --> iter = new
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
            phi_x_v_iter = phi_x_v;
            // Gamma_xv_iter = Gamma_xv;

            v_m_iter[0] = 0.0;   // BC ingresso
            v_m_iter[N] = 0.0;   // BC uscita

            v_l_iter[0] = 0.0;
            v_l_iter[N] = 0.0;

            // Space discretization loop
            for (int i = 1; i < N - 1; ++i) {

                // =======================================================================
                //
                //                              [COEFFICIENTS]
                //
                // =======================================================================

                #pragma region coefficients

                // Physical properties
                k_w[i] = steel::k(T_w_iter[i]);                                   /// Wall thermal conductivity [W/(m K)]
                k_x[i] = liquid_sodium::k(T_l_iter[i]);                           /// Liquid thermal conductivity [W/(m K)]
                const double k_m = vapor_sodium::k(T_m_iter[i], p_m_iter[i]);               /// Vapor thermal conductivity [W/(m K)]
                const double cp_m = vapor_sodium::cp_g_linear();                            /// Vapor specific heat [J/(kg K)]
                const double mu_v = vapor_sodium::mu(T_m_iter[i]);                          /// Vapor dynamic viscosity [Pa*s]
                const double mu_l = liquid_sodium::mu(T_l_iter[i]);                         /// Liquid dynamic viscosity
                const double Dh_v = 2.0 * r_v;                                              /// Hydraulic diameter of the vapor core [m]
                const double Re_v = rho_m_iter[i] * std::fabs(v_m_iter[i]) * Dh_v / mu_v;   /// Reynolds number [-]
                const double Pr_v = cp_m * mu_v / k_m;                                      /// Prandtl number [-]
                // const double H_xm = vapor_sodium::h_conv(Re_v, Pr_v, k_m, Dh_v);            /// Convective heat transfer coefficient at the vapor-wick interface [W/m^2/K]
                H_xm[i] = (k_x[i] / Dh_v) * (5.0 + 0.66 * std::pow(std::abs(v_l[i]), 0.8));
                p_saturation[i] = vapor_sodium::P_sat(T_sur_iter[i]);                       /// Saturation pressure [Pa]         
                const double dPsat_dT = vapor_sodium::dP_sat_dT(T_sur_iter[i]);             /// Derivative of the saturation pressure wrt T [Pa/K]   

                double Omega = 1.0;
                const double Kgeom = 2.0 * r_v * eps_s / (r_i * r_i);

                // Gamma coefficients definition (everything is calculated using iter (k-iteration) values)

                const double beta = 1.0 / std::sqrt(2.0 * M_PI * Rv * T_sur_iter[i]);

                const double Psat = vapor_sodium::P_sat(T_sur_iter[i]);
                const double dPsat = vapor_sodium::dP_sat_dT(T_sur_iter[i]);

                cGamma[i] = - (Kgeom * beta * sigma_c);
                bGamma[i] = - (Gamma_xv[i] / (2 * T_sur_iter[i])) + (Kgeom * beta * sigma_e * dPsat);
                aGamma[i] = Gamma_xv[i] - bGamma[i] * T_sur_iter[i];

                Gamma_xv[i] = Kgeom * (sigma_e * p_saturation[i] - sigma_c * 1.0 * p_m_iter[i]);

                // Definition of the enthalpies (as in THROHPUT)
                if (Gamma_xv[i] >= 0.0) {

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
                Ex4[i] = k_x[i] - H_xm[i] * r_v - (Dh[i] * r_i * r_i) / (2.0 * r_v) * bGamma[i] * r_v;
                Ex5[i] = 2.0 * r_v * k_x[i] - H_xm[i] * r_v * r_v - (Dh[i] * r_i * r_i) / (2.0 * r_v) * bGamma[i] * (r_v * r_v);
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
                alpha8[i] = alpha2[i] - alpha4[i] * alpha6[i] / k_x[i];
                alpha9[i] = alpha6[i] / k_x[i] - alpha3[i] / k_w[i];
                alpha10[i] = alpha4[i] * (Ex4[i] - Ex3[i] * Evi1) / k_x[i];
                alpha11[i] = Ex3[i] * alpha5[i] - 2 * r_i * Ex4[i] + Ex5[i];
                alpha12[i] = (Ex4[i] - Ex3[i] * Evi1) / k_x[i];

                delta[i] = alpha8[i] * alpha11[i] + alpha7[i] * alpha10[i];

                // c_w coefficients
                C1[i] = alpha7[i] * Ex7[i] / delta[i];
                C2[i] = alpha7[i] * Ex6[i] / delta[i];
                C3[i] = (alpha11[i] - alpha7[i] * Ex3[i]) / delta[i];
                C4[i] = -alpha11[i] / delta[i];
                C5[i] = (alpha9[i] * alpha11[i] * q_pp[i] + alpha7[i] * (Ex8[i] - alpha12[i] * q_pp[i])) / delta[i];

                // c_x coefficients
                C6[i] = alpha8[i] * Ex7[i] / delta[i];
                C7[i] = alpha8[i] * Ex6[i] / delta[i];
                C8[i] = -(alpha8[i] * Ex3[i] + alpha10[i]) / delta[i];
                C9[i] = alpha10[i] / delta[i];
                C10[i] = (alpha8[i] * (Ex8[i] - alpha12[i] * q_pp[i]) - alpha10[i] * alpha9[i] * q_pp[i]) / delta[i];

                // b_w coefficients
                C11[i] = -2.0 * r_o * C1[i];
                C12[i] = -2.0 * r_o * C2[i];
                C13[i] = -2.0 * r_o * C3[i];
                C14[i] = -2.0 * r_o * C4[i];
                C15[i] = q_pp[i] / k_w[i] - 2.0 * r_o * C5[i];

                // a_w coefficients
                C16[i] = alpha1[i] * C1[i];
                C17[i] = alpha1[i] * C2[i];
                C18[i] = alpha1[i] * C3[i];
                C19[i] = 1.0 + alpha1[i] * C4[i];
                C20[i] = -Eio1 * q_pp[i] / k_w[i] + alpha1[i] * C5[i];

                // b_x coefficients
                C21[i] = alpha4[i] * C1[i] / k_x[i] - 2.0 * r_i * C6[i];
                C22[i] = alpha4[i] * C2[i] / k_x[i] - 2.0 * r_i * C7[i];
                C23[i] = alpha4[i] * C3[i] / k_x[i] - 2.0 * r_i * C8[i];
                C24[i] = alpha4[i] * C4[i] / k_x[i] - 2.0 * r_i * C9[i];
                C25[i] = q_pp[i] / k_x[i] + alpha4[i] * C5[i] / k_x[i] - 2.0 * r_i * C10[i];

                // a_x coefficients
                C26[i] = -Evi1 * alpha4[i] / k_x[i] * C1[i] + alpha5[i] * C6[i];
                C27[i] = -Evi1 * alpha4[i] / k_x[i] * C2[i] + alpha5[i] * C7[i];
                C28[i] = 1.0 - Evi1 * alpha4[i] / k_x[i] * C3[i] + alpha5[i] * C8[i];
                C29[i] = -Evi1 * alpha4[i] / k_x[i] * C4[i] + alpha5[i] * C9[i];
                C30[i] = -Evi1 * q_pp[i] / k_x[i] - Evi1 * alpha4[i] / k_x[i] * C5[i] + alpha5[i] * C10[i];

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
                C41[i] = -2 * k_x[i] * r_v / (r_i * r_i) * (C21[i] + 2 * r_v * C6[i]);
                C42[i] = -2 * k_x[i] * r_v / (r_i * r_i) * (C22[i] + 2 * r_v * C7[i]);
                C43[i] = -2 * k_x[i] * r_v / (r_i * r_i) * (C23[i] + 2 * r_v * C8[i]);
                C44[i] = -2 * k_x[i] * r_v / (r_i * r_i) * (C24[i] + 2 * r_v * C9[i]);
                C45[i] = -2 * k_x[i] * r_v / (r_i * r_i) * (C25[i] + 2 * r_v * C10[i]);

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
                    + (alpha_m_iter[i] * v_m_iter[i] * H(v_m_iter[i])) / dz
                    - (alpha_m_iter[i] * v_m_iter[i - 1] * (1 - H(v_m_iter[i - 1]))) / dz
                );

                add(D[i], 0, 2, 0.0

                    // Temporal term
                    + rho_m_iter[i] / dt

                    // Convective term
                    + (rho_m_iter[i] * v_m_iter[i] * H(v_m_iter[i])) / dz
                    - (rho_m_iter[i] * v_m_iter[i - 1] * (1 - H(v_m_iter[i - 1]))) / dz
                );

                add(D[i], 0, 4, 0.0

                    // Source term
                    - C36[i]                  // Mass source from wick
                );

                add(D[i], 0, 6, 0.0

                    // Convective term
                    + (alpha_m_iter[i] * rho_m_iter[i] * H(v_m_iter[i])) / dz
                    + (alpha_m_iter[i + 1] * rho_m_iter[i + 1] * (1 - H(v_m_iter[i]))) / dz
                );

                add(D[i], 0, 8, 0.0

                    // Source term
                    - C37[i]                   // Mass source from wick
                );

                add(D[i], 0, 9, 0.0

                    // Source term
                    - C38[i]                   // Mass source from wick
                );

                add(D[i], 0, 10, 0.0

                    // Source term
                    - C39[i]                  // Mass source from wick
                );

                Q[i][0] = 0.0

                    // Source term (implicit)
                    + C40[i]                  // Mass source from wick

                    // Temporal term
                    + (rho_m_iter[i] * alpha_m_old[i]) / dt
                    + (rho_m_old[i] * alpha_m_iter[i]) / dt

                    // Convective term
                    + 2 * (
                        + alpha_m_iter[i] * rho_m_iter[i] * v_m_iter[i] * H(v_m_iter[i])
                        + alpha_m_iter[i + 1] * rho_m_iter[i + 1] * v_m_iter[i] * (1 - H(v_m_iter[i]))
                        - alpha_m_iter[i - 1] * rho_m_iter[i - 1] * v_m_iter[i - 1] * H(v_m_iter[i - 1])
                        - alpha_m_iter[i] * rho_m_iter[i] * v_m_iter[i - 1] * (1 - H(v_m_iter[i - 1]))
                        ) / dz
                    ;

                add(L[i], 0, 0, 0.0

                    // Convective term
                    - (alpha_m_iter[i - 1] * v_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz
                );

                add(L[i], 0, 2, 0.0

                    // Convective term
                    - (rho_m_iter[i - 1] * v_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz
                );

                add(L[i], 0, 6, 0.0

                    // Convective term
                    - (alpha_m_iter[i - 1] * rho_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz
                    - (alpha_m_iter[i] * rho_m_iter[i] * (1 - H(v_m_iter[i - 1]))) / dz
                );

                add(R[i], 0, 0, 0.0

                    // Convective term
                    + (alpha_m_iter[i + 1] * v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz
                );

                add(R[i], 0, 2, 0.0

                    // Convective term
                    + (rho_m_iter[i + 1] * v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz
                );

                // --------------- MASS LIQUID EQUATION -----------------

                add(D[i], 1, 1, 0.0

                    // Temporal term
                    + eps_v * (alpha_l_iter[i] / dt)

                    // Convective term
                    + eps_v * (alpha_l_iter[i] * v_l_iter[i] * H(v_l_iter[i])) / dz
                    - eps_v * (alpha_l_iter[i] * v_l_iter[i - 1] * (1 - H(v_l_iter[i - 1]))) / dz
                );

                add(D[i], 1, 3, 0.0

                    // Temporal term
                    + eps_v * (rho_l_iter[i] / dt)

                    // Convective term
                    + eps_v * (rho_l_iter[i] * v_l_iter[i] * H(v_l_iter[i])) / dz
                    - eps_v * (rho_l_iter[i] * v_l_iter[i - 1] * (1 - H(v_l_iter[i - 1]))) / dz
                );

                add(D[i], 1, 4, 0.0

                    // Source term
                    + C36[i]                 // Mass source from vapor
                );

                add(D[i], 1, 7, 0.0

                    // Convective term
                    + eps_v * (alpha_l_iter[i] * rho_l_iter[i] * H(v_l_iter[i])) / dz
                    + eps_v * (alpha_l_iter[i + 1] * rho_l_iter[i + 1] * (1 - H(v_l_iter[i]))) / dz
                );

                add(D[i], 1, 8, 0.0

                    // Source term
                    + C37[i]                   // Mass source from vapor
                );

                add(D[i], 1, 9, 0.0

                    // Source term
                    + C38[i]                  // Mass source from vapor
                );

                add(D[i], 1, 10, 0.0

                    // Source term
                    + C39[i]                   // Mass source from vapor
                );


                Q[i][1] =

                    // Temporal term
                    + eps_v * (rho_l_old[i] * alpha_l_iter[i]) / dt
                    + eps_v * (rho_l_iter[i] * alpha_l_old[i]) / dt

                    // Convective term
                    + 2 * (
                        + eps_v * (alpha_l_iter[i] * rho_l_iter[i] * v_l_iter[i] * H(v_l_iter[i]))
                        + eps_v * (alpha_l_iter[i + 1] * rho_l_iter[i + 1] * v_l_iter[i] * (1 - H(v_l_iter[i])))
                        - eps_v * (alpha_l_iter[i - 1] * rho_l_iter[i - 1] * v_l_iter[i - 1] * H(v_l_iter[i - 1]))
                        - eps_v * (alpha_l_iter[i] * rho_l_iter[i] * v_l_iter[i - 1] * (1 - H(v_l_iter[i - 1])))
                        ) / dz

                    // Source term (implicit)
                    - C40[i]                // Mass source from vapor
                    ;

                add(L[i], 1, 1, 0.0

                    // Convective term
                    - eps_v * (alpha_l_iter[i - 1] * v_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                );

                add(L[i], 1, 3, 0.0

                    // Convective term
                    - eps_v * (rho_l_iter[i - 1] * v_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                );

                add(L[i], 1, 7, 0.0

                    // Convective term
                    - eps_v * (alpha_l_iter[i - 1] * rho_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                    - eps_v * (alpha_l_iter[i] * rho_l_iter[i] * (1 - H(v_l_iter[i - 1]))) / dz
                );

                add(R[i], 1, 1, 0.0

                    // Convective term
                    + eps_v * (alpha_l_iter[i + 1] * v_l_iter[i] * (1 - H(v_l_iter[i]))) / dz
                );

                add(R[i], 1, 3, 0.0

                    // Convective term
                    + eps_v * (rho_l_iter[i + 1] * v_l_iter[i] * (1 - H(v_l_iter[i]))) / dz
                );

                // ---------- MIXTURE HEAT EQUATION ----------------

                const double cv_m_p = vapor_sodium::cp_g_linear();
                const double cv_m_l = vapor_sodium::cp_g_linear();
                const double cv_m_r = vapor_sodium::cp_g_linear();

                const double k_m_p = vapor_sodium::k(T_m_iter[i], p_m_iter[i]);
                const double k_m_l = vapor_sodium::k(T_m_iter[i - 1], p_m_iter[i - 1]);
                const double k_m_r = vapor_sodium::k(T_m_iter[i + 1], p_m_iter[i + 1]);

                add(D[i], 2, 0, 0.0

                    // Temporal term
                    + (alpha_m_iter[i] * cv_m_p * T_m_iter[i]) / dt

                    // Convective term
                    + (alpha_m_iter[i] * cv_m_p * T_m_iter[i] * v_m_iter[i] * H(v_m_iter[i])) / dz
                    - (alpha_m_iter[i] * cv_m_p * T_m_iter[i] * v_m_iter[i - 1] * (1 - H(v_m_iter[i - 1]))) / dz
                );

                add(D[i], 2, 2, 0.0

                    // Temporal term
                    + (T_m_iter[i] * rho_m_iter[i] * cv_m_p) / dt

                    // Convective term
                    + (rho_m_iter[i] * cv_m_p * T_m_iter[i] * v_m_iter[i] * H(v_m_iter[i])) / dz
                    - (rho_m_iter[i] * cv_m_p * T_m_iter[i] * v_m_iter[i - 1] * (1 - H(v_m_iter[i - 1]))) / dz

                    // Pressure I term
                    + p_m_iter[i] * (v_m_iter[i] - v_m_iter[i - 1]) / (2 * dz)

                    // Pressure II term
                    + p_m_iter[i] / dt
                );

                add(D[i], 2, 4, 0.0

                    // Source term
                    - C46[i]                // Heat source due to heat flux from wick
                    - C56[i]                  // Heat source due to mass flux from wick
                );

                add(D[i], 2, 6, 0.0

                    // Convective term
                    + (alpha_m_iter[i] * rho_m_iter[i] * cv_m_p * T_m_iter[i] * H(v_m_iter[i])) / dz
                    + (alpha_m_iter[i + 1] * rho_m_iter[i + 1] * cv_m_r * T_m_iter[i + 1] * (1 - H(v_m_iter[i]))) / dz

                    // Pressure I term
                    + p_m_iter[i] * (alpha_m_iter[i] + alpha_m_iter[i + 1]) / (2 * dz)
                );

                add(D[i], 2, 8, 0.0

                    // Temporal term
                    + (alpha_m_iter[i] * rho_m_iter[i] * cv_m_p) / dt

                    // Diffusion term
                    + (alpha_m_iter[i + 1] * k_m_r + 2 * alpha_m_iter[i] * k_m_p + alpha_m_iter[i - 1] * k_m_l) / (2 * dz * dz)

                    // Convective term
                    + (alpha_m_iter[i] * rho_m_iter[i] * cv_m_p * v_m_iter[i] * H(v_m_iter[i])) / dz
                    - (alpha_m_iter[i] * rho_m_iter[i] * cv_m_p * v_m_iter[i - 1] * (1 - H(v_m_iter[i - 1]))) / dz

                    // Source term
                    - C47[i]                  // Heat source due to heat flux from wick
                    - C57[i]                  // Heat source due to mass flux from wick
                );

                add(D[i], 2, 9, 0.0

                    // Source term
                    - C48[i]                   // Heat source due to heat flux from wick
                    - C58[i]                  // Heat source due to mass flux from wick
                );

                add(D[i], 2, 10, 0.0

                    // Source term
                    - C49[i]                 // Heat source due to heat flux from wick
                    - C59[i]                 // Heat source due to mass flux from wick
                );  

                Q[i][2] = 0.0

                    // Temporal term (cross terms version)
                    + (alpha_m_iter[i] * cv_m_p * T_m_iter[i] * rho_m_old[i]) / dt
                    + (alpha_m_iter[i] * cv_m_p * T_m_old[i] * rho_m_iter[i]) / dt
                    + (alpha_m_old[i] * cv_m_p * T_m_iter[i] * rho_m_iter[i]) / dt

                    // Convective term
                    + 3 * (
                        + alpha_m_iter[i] * rho_m_iter[i] * cv_m_p * T_m_iter[i] * v_m_iter[i] * H(v_m_iter[i])
                        + alpha_m_iter[i + 1] * rho_m_iter[i + 1] * cv_m_r * T_m_iter[i + 1] * v_m_iter[i] * (1 - H(v_m_iter[i]))
                        - alpha_m_iter[i - 1] * rho_m_iter[i - 1] * cv_m_l * T_m_iter[i - 1] * v_m_iter[i - 1] * H(v_m_iter[i - 1])
                        - alpha_m_iter[i] * rho_m_iter[i] * cv_m_p * T_m_iter[i] * v_m_iter[i - 1] * (1 - H(v_m_iter[i - 1]))
                        ) / dz

                    // Pressure I term
                    + p_m_iter[i] * (alpha_m_iter[i] + alpha_m_iter[i + 1]) * v_m_iter[i] / (2 * dz)
                    - p_m_iter[i] * (alpha_m_iter[i] + alpha_m_iter[i - 1]) * v_m_iter[i - 1] / (2 * dz)

                    // Pressure II term
                    + (p_m_iter[i] * alpha_m_old[i]) / dt

                    // Source term
                    + C50[i]                  // Heat source due to heat flux from wick
                    + C60[i]                  // Heat source due to mass flux from wick
                    ;

                add(L[i], 2, 0, 0.0

                    // Convective term
                    - (alpha_m_iter[i - 1] * cv_m_l * T_m_iter[i - 1] * v_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz
                );

                add(L[i], 2, 2, 0.0

                    // Convective term
                    - (rho_m_iter[i - 1] * cv_m_l * T_m_iter[i - 1] * v_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz

                    // Pressure I term
                    - p_m_iter[i] * (v_m_iter[i - 1]) / (2 * dz)
                );

                add(L[i], 2, 6, 0.0

                    // Convective term
                    - (alpha_m_iter[i - 1] * rho_m_iter[i - 1] * cv_m_l * T_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz
                    - (alpha_m_iter[i] * rho_m_iter[i] * cv_m_p * T_m_iter[i] * (1 - H(v_m_iter[i - 1]))) / dz

                    // Pressure I term
                    - p_m_iter[i] * (alpha_m_iter[i] + alpha_m_iter[i - 1]) / (2 * dz)
                );

                add(L[i], 2, 8, 0.0

                    // Convective term
                    - (alpha_m_iter[i - 1] * rho_m_iter[i - 1] * cv_m_l * v_m_iter[i - 1] * H(v_m_iter[i - 1])) / dz

                    // Diffusion term
                    - (alpha_m_iter[i - 1] * k_m_l + alpha_m_iter[i] * k_m_p) / (2 * dz * dz)
                );

                add(R[i], 2, 0, 0.0

                    // Convective term
                    + (alpha_m_iter[i + 1] * cv_m_r * T_m_iter[i + 1] * v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz
                );

                add(R[i], 2, 2, 0.0

                    // Convective term
                    + (rho_m_iter[i + 1] * cv_m_r * T_m_iter[i + 1] * v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz

                    // Pressure I term
                    + p_m_iter[i] * (v_m_iter[i]) / (2 * dz)
                );

                add(R[i], 2, 8, 0.0

                    // Convective term
                    + (alpha_m_iter[i + 1] * rho_m_iter[i + 1] * cv_m_r * v_m_iter[i] * (1 - H(v_m_iter[i]))) / dz

                    // Diffusion term
                    - (alpha_m_iter[i + 1] * k_m_r + alpha_m_iter[i] * k_m_p) / (2 * dz * dz)
                );

                // ---------- LIQUID HEAT EQUATION ----------------

                const double cp_l_p = liquid_sodium::cp_l_linear();
                const double cp_l_l = liquid_sodium::cp_l_linear();
                const double cp_l_r = liquid_sodium::cp_l_linear();

                const double k_l_p = liquid_sodium::k(T_l_iter[i]);
                const double k_l_l = liquid_sodium::k(T_l_iter[i - 1]);
                const double k_l_r = liquid_sodium::k(T_l_iter[i + 1]);

                add(D[i], 3, 1, 0.0

                    // Temporal term
                    + eps_v * (alpha_l_iter[i] * cp_l_p * T_l_iter[i]) / dt

                    // Convective term
                    + eps_v * (alpha_l_iter[i] * cp_l_p * T_l_iter[i] * v_l_iter[i] * H(v_l_iter[i])) / dz
                    - eps_v * (alpha_l_iter[i] * cp_l_p * T_l_iter[i] * v_l_iter[i - 1] * (1 - H(v_l_iter[i - 1]))) / dz
                );

                add(D[i], 3, 3, 0.0

                    // Temporal term
                    + eps_v * (T_l_iter[i] * rho_l_iter[i] * cp_l_p) / dt

                    // Convective term
                    + eps_v * (rho_l_iter[i] * cp_l_p * T_l_iter[i] * v_l_iter[i] * H(v_l_iter[i])) / dz
                    - eps_v * (rho_l_iter[i] * cp_l_p * T_l_iter[i] * v_l_iter[i - 1] * (1 - H(v_l_iter[i - 1]))) / dz

                    // Pressure I term
                    + eps_v * p_l_iter[i] * (v_l_iter[i] - v_l_iter[i - 1]) / (2 * dz)

                    // Pressure II term
                    + eps_v * p_l_iter[i] / dt
                );

                add(D[i], 3, 4, 0.0

                    // Source term
                    - C41[i]                      // Heat source due to heat flux from wall
                    - C51[i]                      // Heat source due to mass flux from vapor
                    - C61[i]                      // Heat source due to heat flux from vapor
                );

                add(D[i], 3, 7, 0.0

                    // Convective term
                    + eps_v * (alpha_l_iter[i] * rho_l_iter[i] * cp_l_p * T_l_iter[i] * H(v_l_iter[i])) / dz
                    + eps_v * (alpha_l_iter[i + 1] * rho_l_iter[i + 1] * cp_l_r * T_l_iter[i + 1] * (1 - H(v_l_iter[i]))) / dz

                    // Pressure I term
                    + eps_v * p_l_iter[i] * (alpha_l_iter[i] + alpha_l_iter[i + 1]) / (2 * dz)
                );

                add(D[i], 3, 8, 0.0

                    // Source term
                    - C42[i]                      // Heat source due to heat flux from wall
                    - C52[i]                      // Heat source due to mass flux from vapor 
                    - C62[i]                      // Heat source due to heat flux from vapor
                );

                add(D[i], 3, 9, 0.0

                    // Temporal term
                    + eps_v * (alpha_l_iter[i] * rho_l_iter[i] * cp_l_p) / dt

                    // Diffusion term
                    + eps_v * (alpha_l_iter[i + 1] * k_l_r + 2 * alpha_l_iter[i] * k_l_p + alpha_l_iter[i - 1] * k_l_l) / (2 * dz * dz)

                    // Convective term
                    + eps_v * (alpha_l_iter[i] * rho_l_iter[i] * cp_l_p * v_l_iter[i] * H(v_l_iter[i])) / dz
                    - eps_v * (alpha_l_iter[i] * rho_l_iter[i] * cp_l_p * v_l_iter[i - 1] * (1 - H(v_l_iter[i - 1]))) / dz

                    // Source term
                    - C43[i]                      // Heat source due to heat flux from wall
                    - C53[i]                       // Heat source due to mass flux from vapor
                    - C63[i]                      // Heat source due to heat flux from vapor
                );

                add(D[i], 3, 10, 0.0

                    // Source term
                    - C44[i]                      // Heat source due to heat flux from wall
                    - C54[i]                      // Heat source due to mass flux from vapor 
                    - C64[i]                     // Heat source due to heat flux from vapor
                );

                Q[i][3] = 0.0

                    // Temporal term (cross terms)
                    + eps_v * (alpha_l_iter[i] * cp_l_p * T_l_iter[i] * rho_l_old[i]) / dt
                    + eps_v * (alpha_l_iter[i] * cp_l_p * T_l_old[i] * rho_l_iter[i]) / dt
                    + eps_v * (alpha_l_old[i] * cp_l_p * T_l_iter[i] * rho_l_iter[i]) / dt

                    // Convective term
                    + 3 * eps_v * (
                        +alpha_l_iter[i] * rho_l_iter[i] * cp_l_p * T_l_iter[i] * v_l_iter[i] * H(v_l_iter[i])
                        + alpha_l_iter[i + 1] * rho_l_iter[i + 1] * cp_l_r * T_l_iter[i + 1] * v_l_iter[i] * (1 - H(v_l_iter[i]))
                        - alpha_l_iter[i - 1] * rho_l_iter[i - 1] * cp_l_l * T_l_iter[i - 1] * v_l_iter[i - 1] * H(v_l_iter[i - 1])
                        - alpha_l_iter[i] * rho_l_iter[i] * cp_l_p * T_l_iter[i] * v_l_iter[i - 1] * (1 - H(v_l_iter[i - 1]))
                        ) / dz

                    // Pressure I term
                    + eps_v * p_l_iter[i] * (alpha_l_iter[i] + alpha_l_iter[i + 1]) * v_l_iter[i] / (2 * dz)
                    - eps_v * p_l_iter[i] * (alpha_l_iter[i] + alpha_l_iter[i - 1]) * v_l_iter[i - 1] / (2 * dz)

                    // Pressure II term
                    + eps_v * (p_l_iter[i] * alpha_l_old[i]) / dt

                    // Source term
                    + C45[i]                      // Heat source due to heat flux from wall
                    + C55[i]                      // Heat source due to mass flux from vapor
                    + C65[i]                      // Heat source due to heat flux from vapor
                    ;

                add(L[i], 3, 1, 0.0

                    // Convective term
                    - eps_v * (alpha_l_iter[i - 1] * cp_l_l * T_l_iter[i - 1] * v_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                );

                add(L[i], 3, 3, 0.0

                    // Convective term
                    - eps_v * (rho_l_iter[i - 1] * cp_l_l * T_l_iter[i - 1] * v_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz

                    // Pressure I term
                    - eps_v * p_l_iter[i] * (v_l_iter[i - 1]) / (2 * dz)
                );

                add(L[i], 3, 7, 0.0

                    // Convective term
                    - eps_v * (alpha_l_iter[i - 1] * rho_l_iter[i - 1] * cp_l_l * T_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz
                    - eps_v * (alpha_l_iter[i] * rho_l_iter[i] * cp_l_p * T_l_iter[i] * (1 - H(v_l_iter[i - 1]))) / dz

                    // Pressure I term
                    - eps_v * p_l_iter[i] * (alpha_l_iter[i] + alpha_l_iter[i - 1]) / (2 * dz)
                );

                add(L[i], 3, 9, 0.0

                    // Convective term
                    - eps_v * (alpha_l_iter[i - 1] * rho_l_iter[i - 1] * cp_l_l * v_l_iter[i - 1] * H(v_l_iter[i - 1])) / dz

                    // Diffusion term
                    - eps_v * (alpha_l_iter[i - 1] * k_l_l + alpha_l_iter[i] * k_l_p) / (2 * dz * dz)
                );

                add(R[i], 3, 1, 0.0

                    // Convective term
                    + eps_v * (alpha_l_iter[i + 1] * cp_l_r * T_l_iter[i + 1] * v_l_iter[i] * (1 - H(v_l_iter[i]))) / dz
                );

                add(R[i], 3, 3, 0.0

                    // Convective term
                    + eps_v * (rho_l_iter[i + 1] * cp_l_r * T_l_iter[i + 1] * v_l_iter[i] * (1 - H(v_l_iter[i]))) / dz

                    // Pressure I term
                    + eps_v * p_l_iter[i] * (v_l_iter[i]) / (2 * dz)
                );

                add(R[i], 3, 9, 0.0

                    // Convective term
                    + eps_v * (alpha_l_iter[i + 1] * rho_l_iter[i + 1] * cp_l_r * v_l_iter[i] * (1 - H(v_l_iter[i]))) / dz

                    // Diffusion term
                    - eps_v * (alpha_l_iter[i + 1] * k_l_r + alpha_l_iter[i] * k_l_p) / (2 * dz * dz)
                );

                // --------------- WALL HEAT EQUATION -------------------

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

                add(D[i], 4, 4, 0.0

                    // Source term
                    - C66[i]                      // Heat source due to heat flux from wick
                );

                add(D[i], 4, 8, 0.0

                    // Source term
                    - C67[i]                      // Heat source due to heat flux from wick
                );

                add(D[i], 4, 9, 0.0

                    // Source term
                    - C68[i]                      // Heat source due to heat flux from wick
                );

                add(D[i], 4, 10, 0.0

                    // Temporal term
                    + (rho_w_p * cp_w_p) / dt

                    // Diffusion term
                    + (k_w_lf + k_w_rf) / (dz * dz)

                    // Source term
                    - C69[i]                      // Heat source due to heat flux from wick
                );

                Q[i][4] = 0.0

                    // Source term 
                    + q_pp[i] * 2 * r_o / (r_o * r_o - r_i * r_i)

                    // Temporal term
                    + (rho_w_p * cp_w_p * T_w_old[i]) / dt

                    // Source term
                    + C70[i]                      // Heat source due to heat flux from wick
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

                const double Re = rho_m_iter[i] * std::abs(v_m_iter[i]) * Dh_v / mu_v;
                const double fm = Re > 1187.4 ? 0.3164 * std::pow(Re, -0.25) : 64 * std::pow(Re, -1);

                add(D[i], 5, 0,

                    // Temporal term (central differences)
                    + (alpha_m_iter[i] + alpha_m_iter[i + 1]) * v_m_iter[i] / (4 * dt)

                    // Convective term
                    - (H(v_m_iter[i]) * alpha_m_iter[i] * v_m_iter[i - 1] * v_m_iter[i - 1]) / dz
                    - ((1 - H(v_m_iter[i])) * alpha_m_iter[i] * v_m_iter[i] * v_m_iter[i]) / dz
                );

                add(D[i], 5, 2,

                    // Temporal term (central differences)
                    + (rho_m_iter[i] + rho_m_iter[i + 1]) * v_m_iter[i] / (4 * dt)

                    // Convective term
                    - (H(v_m_iter[i]) * rho_m_iter[i] * v_m_iter[i - 1] * v_m_iter[i - 1]) / dz
                    - ((1 - H(v_m_iter[i])) * rho_m_iter[i] * v_m_iter[i] * v_m_iter[i]) / dz

                );

                add(D[i], 5, 4, 0.0

                    // Pressure term (central differences)
                    - (alpha_m_iter[i] + alpha_m_iter[i + 1]) / (2 * dz)
                );

                add(D[i], 5, 6,

                    // Temporal term (central differences)
                    + (alpha_m_iter[i] * rho_m_iter[i] + alpha_m_iter[i + 1] * rho_m_iter[i + 1]) / (2 * dt)

                    // Convective term
                    + 2 * (H(v_m_iter[i]) * alpha_m_iter[i + 1] * rho_m_iter[i + 1] * v_m_iter[i]) / dz
                    - 2 * ((1 - H(v_m_iter[i])) * alpha_m_iter[i] * rho_m_iter[i] * v_m_iter[i]) / dz

                    // Friction term (central differences)
                    // WHaTCH ouT THiS TeRM GiVeS SoMe PRoBLeMS
                    // + fm * (rho_m_iter[i] + rho_m_iter[i + 1]) * std::abs(v_m_iter[i]) / (8 * r_v)
                );

                Q[i][5] =

                    // Temporal term (cross terms)
                    + (alpha_m_iter[i] * rho_m_iter[i] + alpha_m_iter[i + 1] * rho_m_iter[i + 1]) * v_m_old[i] / (2 * dt)
                    + ((alpha_m_iter[i] + alpha_m_iter[i + 1]) * v_m_iter[i] * (rho_m_old[i] + rho_m_old[i + 1])) / (4 * dt)
                    + ((rho_m_iter[i] + rho_m_iter[i + 1]) * v_m_iter[i] * (alpha_m_old[i] + alpha_m_old[i + 1])) / (4 * dt)

                    // Convective term
                    - 3 * H(v_m_iter[i]) * (
                        + alpha_m_iter[i] * rho_m_iter[i] * v_m_iter[i - 1] * v_m_iter[i - 1]
                        - alpha_m_iter[i + 1] * rho_m_iter[i + 1] * v_m_iter[i] * v_m_iter[i]
                        ) / dz
                    - 3 * (1 - H(v_m_iter[i])) * (
                        + alpha_m_iter[i] * rho_m_iter[i] * v_m_iter[i] * v_m_iter[i]
                        - alpha_m_iter[i + 1] * rho_m_iter[i + 1] * v_m_iter[i + 1] * v_m_iter[i + 1]
                        ) / dz
                    ;

                add(L[i], 5, 6,

                    // Convective term
                    - 2 * (H(v_m_iter[i]) * alpha_m_iter[i] * rho_m_iter[i] * v_m_iter[i - 1]) / dz
                );

                add(R[i], 5, 0,

                    // Temporal term (central differences)
                    + (alpha_m_iter[i] + alpha_m_iter[i + 1]) * v_m_iter[i] / (4 * dt)

                    // Convective term
                    + (H(v_m_iter[i]) * alpha_m_iter[i + 1] * v_m_iter[i] * v_m_iter[i]) / dz
                    + ((1 - H(v_m_iter[i])) * alpha_m_iter[i + 1] * v_m_iter[i + 1] * v_m_iter[i + 1]) / dz
                );

                add(R[i], 5, 2,

                    // Temporal term (central differences)
                    + (rho_m_iter[i] + rho_m_iter[i + 1]) * v_m_iter[i] / (4 * dt)

                    // Convective term
                    + (H(v_m_iter[i]) * rho_m_iter[i + 1] * v_m_iter[i] * v_m_iter[i]) / dz
                    + ((1 - H(v_m_iter[i])) * rho_m_iter[i + 1] * v_m_iter[i + 1] * v_m_iter[i + 1]) / dz
                );

                add(R[i], 5, 4, 0.0

                    // Pressure term (central differences)
                    + (alpha_m_iter[i] + alpha_m_iter[i + 1]) / (2 * dz)
                );

                add(R[i], 5, 6,

                    // Convective term
                    + 2 * ((1 - H(v_m_iter[i])) * alpha_m_iter[i + 1] * rho_m_iter[i + 1] * v_m_iter[i + 1]) / dz
                );

                // ------ MOMENTUM LIQUID EQUATION ------

                const double Fl = 8 * mu_l / (eps_v * (r_i - r_v) * (r_i - r_v));

                add(D[i], 6, 1,

                    // Temporal term (central differences)
                    + eps_v * (alpha_l_iter[i] + alpha_l_iter[i + 1]) * v_l_iter[i] / (4 * dt)

                    // Convective term
                    - eps_v * (H(v_l_iter[i]) * alpha_l_iter[i] * v_l_iter[i - 1] * v_l_iter[i - 1]) / dz
                    - eps_v * ((1 - H(v_l_iter[i])) * alpha_l_iter[i] * v_l_iter[i] * v_l_iter[i]) / dz                    
                );

                add(D[i], 6, 3,

                    // Temporal term (central differences)
                    + eps_v * (rho_l_iter[i] + rho_l_iter[i + 1]) * v_l_iter[i] / (4 * dt)

                    // Convective term
                    - eps_v * (H(v_l_iter[i]) * rho_l_iter[i] * v_l_iter[i - 1] * v_l_iter[i - 1]) / dz
                    - eps_v * ((1 - H(v_l_iter[i])) * rho_l_iter[i] * v_l_iter[i] * v_l_iter[i]) / dz

                    // Capillary term (central differences)
                    - (DPcap[i] + DPcap[i + 1]) / (2 * dz)
                );

                add(D[i], 6, 5, 0.0

                    // Pressure term (central differences)
                    - eps_v * (alpha_l_iter[i] + alpha_l_iter[i + 1]) / (2 * dz)
                );

                add(D[i], 6, 7,

                    // Temporal term (central differences)
                    + eps_v * (alpha_l_iter[i] * rho_l_iter[i] + alpha_l_iter[i + 1] * rho_l_iter[i + 1]) / (2 * dt)

                    // Convective term
                    + 2 * eps_v * (H(v_l_iter[i]) * alpha_l_iter[i + 1] * rho_l_iter[i + 1] * v_l_iter[i]) / dz
                    - 2 * eps_v * ((1 - H(v_l_iter[i])) * alpha_l_iter[i] * rho_l_iter[i] * v_l_iter[i]) / dz

                    // Friction term
                    // + Fl * std::abs(v_l_iter[i])
                );

                Q[i][6] =

                    // Temporal term (central differeces)
                    + eps_v * (alpha_l_iter[i] * rho_l_iter[i] + alpha_l_iter[i + 1] * rho_l_iter[i + 1]) * v_l_old[i] / (2 * dt)
                    + eps_v * ((alpha_l_iter[i] + alpha_l_iter[i + 1]) * v_l_iter[i] * (rho_l_old[i] + rho_l_old[i + 1])) / (4 * dt)
                    + eps_v * ((rho_l_iter[i] + rho_l_iter[i + 1]) * v_l_iter[i] * (alpha_l_old[i] + alpha_l_old[i + 1])) / (4 * dt)

                    // Convective term
                    - 3 * eps_v * H(v_l_iter[i]) * (
                        + alpha_l_iter[i] * rho_l_iter[i] * v_l_iter[i - 1] * v_l_iter[i - 1]
                        - alpha_l_iter[i + 1] * rho_l_iter[i + 1] * v_l_iter[i] * v_l_iter[i]
                        ) / dz
                    - 3 * eps_v * (1 - H(v_l_iter[i])) * (
                        + alpha_l_iter[i] * rho_l_iter[i] * v_l_iter[i] * v_l_iter[i]
                        - alpha_l_iter[i + 1] * rho_l_iter[i + 1] * v_l_iter[i + 1] * v_l_iter[i + 1]
                        ) / dz
                    ;

                add(L[i], 6, 6,

                    // Convective term
                    - 2 * eps_v * (H(v_l_iter[i]) * alpha_l_iter[i] * rho_l_iter[i] * v_l_iter[i - 1]) / dz
                );

                add(R[i], 6, 1,

                    // Temporal term (central differences)
                    + eps_v * (alpha_l_iter[i] + alpha_l_iter[i + 1]) * v_l_iter[i] / (4 * dt)

                    // Convective term
                    + eps_v * (H(v_l_iter[i]) * alpha_l_iter[i + 1] * v_l_iter[i] * v_l_iter[i]) / dz
                    + eps_v * ((1 - H(v_l_iter[i])) * alpha_l_iter[i + 1] * v_l_iter[i + 1] * v_l_iter[i + 1]) / dz
                );

                add(R[i], 6, 3,

                    // Temporal term (central differences)
                    + eps_v * (rho_l_iter[i] + rho_l_iter[i + 1]) * v_l_iter[i] / (4 * dt)

                    // Convective term
                    + eps_v * (H(v_l_iter[i]) * rho_l_iter[i + 1] * v_l_iter[i] * v_l_iter[i]) / dz
                    + eps_v * ((1 - H(v_l_iter[i])) * rho_l_iter[i + 1] * v_l_iter[i + 1] * v_l_iter[i + 1]) / dz

                    // Capillary term (central differences)
                    + (DPcap[i] + DPcap[i + 1]) / (2 * dz)
                );

                add(R[i], 6, 5, 0.0

                    // Pressure term (central differences)
                    + eps_v * (alpha_l_iter[i] + alpha_l_iter[i + 1]) / (2 * dz)
                );

                add(R[i], 6, 7,

                    // Convective term
                    + 2 * eps_v * ((1 - H(v_l_iter[i])) * alpha_l_iter[i + 1] * rho_l_iter[i + 1] * v_l_iter[i + 1]) / dz
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

                Q[i][10] = DPcap[i];

                #pragma endregion

                // Densification of the D block for debug purposes
                // DenseBlock D_dense = to_dense(D[i]);
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
            L_pic = {};

            double Aold, Anew, denom, eps;

            for (int i = 0; i < N; ++i) {

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
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L_pic[6] += eps;

                // v_l
                Aold = v_l_iter[i];
                Anew = X[i][7];
                denom = 0.5 * (std::abs(Aold) + std::abs(Anew));
                eps = denom > 1e-12 ? std::abs((Anew - Aold) / denom) : std::abs(Anew - Aold);
                L_pic[7] += eps;

                // v_l  — norma mista (robusta per v ~ 0)
                Aold = v_l_iter[i];
                Anew = X[i][7];

                // scala fisica per la velocità del liquido
                const double v_abs_tol = 1e-8;        // [m/s] rumore numerico accettabile
                const double v_scale = 1e-4;        // [m/s] scala fisica minima (regola pratica)

                denom = std::max({ std::abs(Aold), std::abs(Anew), v_scale, v_abs_tol });
                eps = std::abs(Anew - Aold) / denom;

                L_pic[7] += eps;

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

            // Update vectors from X
            for (int i = 0; i < N; ++i) {
                rho_m[i] = X[i][0];
                rho_l[i] = X[i][1];
                alpha_m[i] = X[i][2];
                alpha_l[i] = X[i][3];
                p_m[i] = X[i][4];
                p_l[i] = X[i][5];
                v_m[i] = X[i][6];
                v_l[i] = X[i][7];
                T_m[i] = X[i][8];
                T_l[i] = X[i][9];
                T_w[i] = X[i][10];

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
                residual4[i] = k_w[i] * b_w[i] + 2 * r_i * k_w[i] * c_w[i] - k_x[i] * b_x[i] - 2 * r_i * k_x[i] * c_x[i];
                residual5[i] = b_w[i] + 2 * r_o * c_w[i] - q_pp[i] / k_w[i];
                residual6[i] = Ex3[i] * a_x[i] + Ex4[i] * b_x[i] + Ex5[i] * c_x[i] - Ex8[i] - Ex6[i] * T_m[i] - Ex7[i] * p_m[i];

                // Check interface temperature
                T_sur[i] = C31[i] * p_m[i] + C32[i] * T_m[i] + C33[i] * T_l[i] + C34[i] * T_w[i] + C35[i];
                T_sur_lin[i] = a_x[i] + b_x[i] * r_v + c_x[i] * r_v * r_v;
                T_sur_diff[i] = T_sur[i] - T_sur_lin[i];

                // Check mass exchange
                Gamma_xv_approx[i] = aGamma[i] + bGamma[i] * T_sur[i] + cGamma[i] * (p_m[i] - p_m_iter[i]);
                Gamma_xv_lin[i] = C36[i] * p_m[i] + C37[i] * T_m[i] + C38[i] * T_l[i] + C39[i] * T_w[i] + C40[i];

                Gamma_xv_approx_error[i] = Gamma_xv[i] - Gamma_xv_approx[i];
                Gamma_xv_lin_error[i] = Gamma_xv[i] - Gamma_xv_lin[i];
                Gamma_xv_diff_error[i] = Gamma_xv_approx[i] - Gamma_xv_lin[i];

                // Check heat exchange balance liquid wall
                heat_source_wall_liquid_flux[i] = C61[i] * p_m[i] + C62[i] * T_m[i] + C63[i] * T_l[i] + C64[i] * T_w[i] + C65[i];
                heat_source_liquid_wall_flux[i] = C66[i] * p_m[i] + C67[i] * T_m[i] + C68[i] * T_l[i] + C69[i] * T_w[i] + C70[i];

                heat_source_wall_liquid_flux[i] *= r_i / 2;
                heat_source_liquid_wall_flux[i] *= ((r_o * r_o - r_i * r_i) / (2 * r_i));

                heat_balance_wx[i] = heat_source_wall_liquid_flux[i] + heat_source_liquid_wall_flux[i];

                // Check heat exchange balance liquid vapor
                heat_conduction_flux[i] = k_x[i] * (
                    (C21[i] + 2 * r_v * (C6[i])) * p_m[i] +
                    (C22[i] + 2 * r_v * (C7[i])) * T_m[i] +
                    (C23[i] + 2 * r_v * (C8[i])) * T_l[i] +
                    (C24[i] + 2 * r_v * (C9[i])) * T_w[i] +
                    (C25[i] + 2 * r_v * (C10[i])));

                heat_convection_flux[i] = H_xm[i] * (
                    (C26[i] + r_v * C21[i] + r_v * r_v * C6[i]) * p_m[i] +
                    (C27[i] + r_v * C22[i] + r_v * r_v * C7[i] - 1) * T_m[i] +
                    (C28[i] + r_v * C23[i] + r_v * r_v * C8[i]) * T_l[i] +
                    (C29[i] + r_v * C24[i] + r_v * r_v * C9[i]) * T_w[i] +
                    (C30[i] + r_v * C25[i] + r_v * r_v * C10[i]));

                heat_phase_flux[i] = (Dh[i] * r_i * r_i) / (2 * r_v) * (
                    C36[i] * p_m[i] +
                    C37[i] * T_m[i] +
                    C38[i] * T_l[i] +
                    C39[i] * T_w[i] +
                    C40[i]);

                balance_condition[i] = k_x[i] * (b_x[i] + 2 * c_x[i] * r_v)
                    - H_xm[i] * (a_x[i] + b_x[i] * r_v + c_x[i] * r_v * r_v - T_m[i])
                    - Dh[i] * r_i * r_i / (2 * r_v) * (aGamma[i] + bGamma[i] * (a_x[i] + b_x[i] * r_v + c_x[i] * r_v * r_v) + cGamma[i] * (p_m[i] - p_m_iter[i]));

                heat_balance_xv[i] = heat_conduction_flux[i] - heat_convection_flux[i] - heat_phase_flux[i];

                // Check heat sources definition

                heat_source_vapor_liquid_flux[i] = C41[i] * p_m[i] + C42[i] * T_m[i] + C43[i] * T_l[i] + C44[i] * T_w[i] + C45[i];
                heat_source_liquid_vapor_flux[i] = C46[i] * p_m[i] + C47[i] * T_m[i] + C48[i] * T_l[i] + C49[i] * T_w[i] + C50[i];

                heat_source_vapor_liquid_phase[i] = C51[i] * p_m[i] + C52[i] * T_m[i] + C53[i] * T_l[i] + C54[i] * T_w[i] + C55[i];
                heat_source_liquid_vapor_phase[i] = C56[i] * p_m[i] + C57[i] * T_m[i] + C58[i] * T_l[i] + C59[i] * T_w[i] + C60[i];

                conduction_err[i] = heat_source_vapor_liquid_flux[i] * (r_i * r_i) / (2 * r_v) + heat_conduction_flux[i];
                convection_err[i] = heat_source_liquid_vapor_flux[i] * (r_i * r_i) / (2 * r_v) - heat_convection_flux[i];
                phase_change_err[i] = (heat_source_liquid_vapor_phase[i] + heat_source_vapor_liquid_phase[i]) * (r_i * r_i) / (2 * r_v) - heat_phase_flux[i];

                heat_balance_xv_full[i] =
                    + heat_source_vapor_liquid_flux[i]
                    + heat_source_liquid_vapor_flux[i]
                    + heat_source_liquid_vapor_phase[i] 
                    + heat_source_vapor_liquid_phase[i];

                // Update heat fluxes at the interfaces
                if (i <= evaporator_nodes) q_pp[i] = q_pp_evaporator;       /// Evaporator imposed heat flux [W/m2]
                else if (i >= (N - condenser_nodes)) {

                    double conv = h_conv *
                        (T_w_iter[i] - T_env);                              /// Condenser convective heat flux [W/m2]
                    double irr = emissivity * sigma *
                        (std::pow(T_w_iter[i], 4) - std::pow(T_env, 4));    /// Condenser irradiation heat flux [W/m2]

                    q_pp[i] = -(conv + irr);                                /// Heat flux at the outer wall [W/m2] (positive if to the wall) 
                }
           
            }

            // Check if variable converged
            for (int k = 0; k < B; ++k)
                conv_var[k] = (L_pic[k] < pic_tol[k]);

            // Check if all variables converged
            bool conv_all = true;
            for (int k = 0; k < B; ++k)
                conv_all = conv_all && conv_var[k];

            if (conv_all) {

                halves = 0;             // Reset halves if Picard converged
                break;                  // Picard converged, so break the loops
            }
        }

        // Picard converged or max iterations reached
        if (pic != max_picard) {

            // Update n values (old = new) and update total time
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

            time_total += dt;       // Advance in time

            const int output_every = 1;

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

					dpcap_output << DPcap[i] << " ";

					q_pp_output << q_pp[i] << " ";
                }

                time_output << time_total << " ";

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

				dpcap_output << "\n";
				q_pp_output << "\n";

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

				dpcap_output.flush();
				q_pp_output.flush();
            }
        }
        else {

            // Rollback to previous time step (new = old) and halve dt
            for (int i = 0; i < N; ++i) {
                rho_m[i] = rho_m_old[i];
                rho_l[i] = rho_l_old[i];
                alpha_m[i] = alpha_m_old[i];
                alpha_l[i] = alpha_l_old[i];
                p_m[i] = p_m_old[i];
                p_l[i] = p_l_old[i];
                v_m[i] = v_m_old[i];
                v_l[i] = v_l_old[i];
                T_m[i] = T_m_old[i];
                T_l[i] = T_l_old[i];
                T_w[i] = T_w_old[i];

                Gamma_xv[i] = Gamma_xv_old[i];
            }

            halves += 1;        // Half again the timestep
            n -= 1;             // No time iteration considered
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

	dpcap_output.close();
	q_pp_output.close();

    double end = omp_get_wtime();
    printf("Execution time: %.6f s\n", end - start);
}
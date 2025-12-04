Below is a **clean, professional, publication-ready README** for your solver, assuming the final chosen name is **THRO-CXX**.
It is structured exactly as expected for a research-grade numerical code repository.

---

# THRO-CXX

**A Monolithic 1-D Sodium Heat-Pipe Solver in C++**

THRO-CXX is a fully coupled, monolithic, one-dimensional solver for transient sodium heat-pipe simulations.
It is inspired by the THROHPUT architecture but rewritten completely in modern C++.
The code resolves vapor, liquid wick, and solid wall regions simultaneously through a unified block-tridiagonal formulation and detailed phase-change closures.

This solver is intended for research on advanced heat-pipe modelling, thermofluid analysis, and reactor thermal-management systems.

---

## Main Features

### Multi-region physics

THRO-CXX solves three strongly coupled regions:

* **Vapor core** (compressible, convective, Navier–Stokes with friction correlations)
* **Porous wick (liquid)** (Darcy-Brinkman-Forchheimer, capillary pressure model)
* **Solid wall** (transient conduction)

### Monolithic solution algorithm

All equations for all variables are assembled into a **global system**
[
A , X = Q
]
with **11 variables per axial node**, solved simultaneously using a **block-tridiagonal Thomas algorithm with LU decomposition**.

### Phase-change model

* Kinetic interfacial mass flux with accommodation coefficients
* Ω–correction for rarefaction
* Temperature-dependent enthalpies (Shomate correlations)
* Interface temperatures reconstructed from full parabolic radial solutions

### Thermophysical properties

Accurate models over wide temperature ranges:

* Liquid sodium (ρ, cp, k, μ, surface tension, enthalpy)
* Vapor sodium (ρ, cp, k(T,P), μ, enthalpy, P_sat, dP/dT)
* Stainless steel 304L (ρ, cp, k via tabulated interpolation)

### Numerical scheme

* 1-D finite-volume discretization
* Upwind convection, central diffusion
* Explicit–implicit Picard iteration
* Adaptive interface closures with C-coefficients
* Boundary conditions for evaporator, adiabatic region, and condenser
* Fully transient time integration

### Output

At user-defined intervals, the solver writes:

* Vapor and liquid velocities
* Pressures
* Temperatures
* Densities
* Volume fractions
* Heat-flux terms
* Phase-change source terms
* Interface temperature (T_{\mathrm{sur}})
* Saturation pressure

All data are saved in a directory automatically created as:

```
case_0/, case_1/, case_2/, ...
```

---

## Code Structure

```
THRO-CXX/
│
├── main.cpp                # complete solver
├── material_models/        # steel, sodium liquid, sodium vapor property functions
├── block_solver/           # block-tridiagonal solver (LU factorization)
├── utils/                  # interpolation, clamp, helper arithmetic
└── README.md               # this file
```

---

## Dependencies

THRO-CXX uses only standard C++17 libraries:

* `<vector>`
* `<array>`
* `<cmath>`
* `<fstream>`
* `<iomanip>`
* `<algorithm>`
* `<stdexcept>`
* `<filesystem>`
* `<omp.h>` (optional, for timing and parallel sections)

No external libraries are required.

---

## Compilation

Example (Linux / macOS):

```bash
g++ -O3 -std=c++17 -fopenmp main.cpp -o throcxx
```

Windows (MSVC):

```powershell
cl /O2 /std:c++17 main.cpp /openmp
```

---

## Running

```bash
./throcxx
```

The program automatically creates the next available result directory:

```
case_0/
    mesh.txt
    time.txt
    vapor_velocity.txt
    vapor_pressure.txt
    vapor_temperature.txt
    vapor_alpha.txt
    ...
```

---

## Numerical Method Summary

### State vector per node (size 11)

1. ρ_v (vapor density)
2. ρ_l (liquid density)
3. α_v (vapor volume fraction)
4. α_l (liquid volume fraction)
5. p_v (vapor pressure)
6. p_l (liquid pressure)
7. u_v (vapor velocity)
8. u_l (liquid velocity)
9. T_v (vapor temperature)
10. T_l (liquid temperature)
11. T_w (wall temperature)

These form the block vector ( X_i \in \mathbb{R}^{11} ) for node ( i \in [0,N-1] ).

### System size

For N nodes:

[
X \in \mathbb{R}^{11N}, \quad
A \in \mathbb{R}^{11N \times 11N}
]

where A is block-tridiagonal with 11×11 blocks.

### Solver

A custom block-Thomas method performs:

* Forward block elimination
* In-place LU factorization with pivoting
* Backward substitution

Picard iterations handle nonlinearity.

---

## Progress and Validation

This solver reproduces the full physics structure of THROHPUT but in C++.
Validation steps currently include:

* Steady-state startup tests
* Response to imposed evaporator/condenser boundary conditions
* Comparison with literature correlations (e.g., saturation T, enthalpy curves)
* Numerical consistency (mass/energy residuals, Picard convergence)

---

## Planned Extensions

* Adaptive time stepping
* Parallel block solver
* Wick models with variable porosity
* Turbulence options for vapor region
* Python visualization scripts
* 2-D radial extensions

---

## License

To be defined (MIT recommended unless restrictions apply).

---

If you want, I can also generate:

• A shorter README
• A version formatted for GitHub with badges
• A version with equations rendered as LaTeX blocks
• A diagram of the solver architecture (ASCII or image)

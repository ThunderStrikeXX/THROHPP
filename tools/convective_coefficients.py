import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Parameters
# -------------------------------------------------
k  = 70.0       # thermal conductivity [W/(m K)]
Dh = 1e-2       # hydraulic diameter [m]
Pr = 0.01

Re1 = 2000.0
Re2 = 3000.0

# -------------------------------------------------
# Reynolds range
# -------------------------------------------------
Re = np.logspace(1, 6, 600)

# -------------------------------------------------
# Friction factor (Prandtl–von Kármán)
# -------------------------------------------------
def f(Re):
    t = 0.79 * np.log(Re) - 1.64
    return 1.0 / (t * t)

# -------------------------------------------------
# Gnielinski (standard)
# -------------------------------------------------
def Nu_gnielinski(Re, Pr):
    fp8 = f(Re) / 8.0
    num = fp8 * (Re - 1000.0) * Pr
    den = 1.0 + 12.7 * np.sqrt(fp8) * (Pr**(2.0/3.0) - 1.0)
    return num / den

# -------------------------------------------------
# Blended laminar + Gnielinski
# -------------------------------------------------
def h_blended_gnielinski(Re, Pr, k, Dh):
    Nu_lam = 4.36
    Nu = np.zeros_like(Re)

    for i, r in enumerate(Re):
        if r <= Re1:
            Nu[i] = Nu_lam
        elif r >= Re2:
            Nu[i] = Nu_gnielinski(r, Pr)
        else:
            chi = (r - Re1) / (Re2 - Re1)
            Nu[i] = (1.0 - chi) * Nu_lam + chi * Nu_gnielinski(r, Pr)

    return Nu * k / Dh

# -------------------------------------------------
# Dittus–Boelter
# -------------------------------------------------
def Nu_dittus_boelter(Re, Pr):
    return 0.023 * Re**0.8 * Pr**0.4

# -------------------------------------------------
# Blended laminar + Dittus–Boelter
# -------------------------------------------------
def h_blended_DB(Re, Pr, k, Dh):
    Nu_lam = 48.0 / 11.0
    Nu = np.zeros_like(Re)

    for i, r in enumerate(Re):
        if r <= Re1:
            Nu[i] = Nu_lam
        elif r >= Re2:
            Nu[i] = Nu_dittus_boelter(r, Pr)
        else:
            chi = (r - Re1) / (Re2 - Re1)
            Nu[i] = (1.0 - chi) * Nu_lam + chi * Nu_dittus_boelter(r, Pr)

    return Nu * k / Dh

# -------------------------------------------------
# Compute coefficients
# -------------------------------------------------
h_GN = h_blended_gnielinski(Re, Pr, k, Dh)
h_DB = h_blended_DB(Re, Pr, k, Dh)

# -------------------------------------------------
# Plot
# -------------------------------------------------
plt.figure()
plt.loglog(Re, h_GN, label="Laminar + Gnielinski (blended)")
plt.loglog(Re, h_DB, label="Laminar + Dittus–Boelter (blended)")
plt.xlabel("Reynolds number [-]")
plt.ylabel("Convective heat transfer coefficient h [W/(m² K)]")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()

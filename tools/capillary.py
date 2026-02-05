import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Physical / geometric params
# ----------------------------
r_i = 2.5e-3
r_v = np.sqrt(0.8) * r_i
r_p = 1.0e-5          # pore radius [m]
eps_s = 0.6           # wick porosity [-]
sigma = 0.16          # surface tension [N/m] (Na ~ 700–800 K)

# ----------------------------
# Derived quantities
# ----------------------------
alpha_m0 = (r_v / r_i)**2

# ----------------------------
# Lambda(mu) and inversion
# ----------------------------
def Lambda_of_mu(mu):
    # Eq. (5.6)
    return (1.0 / mu**3) * (2.0 - (mu**2 + 2.0) * np.sqrt(1.0 - mu**2))

def invert_Lambda(L):
    # Solve Lambda(mu) = L for mu in (0,1)
    if L <= 0.0:
        return 0.0
    if L >= 2.0:
        return 1.0

    mu = 0.5
    for _ in range(50):
        f = Lambda_of_mu(mu) - L
        dL_dmu = 3.0 * ((1.0 - mu**2)**(-0.5) - Lambda_of_mu(mu) / mu)
        mu -= f / dL_dmu
        mu = np.clip(mu, 1e-8, 1.0)
    return mu

# ----------------------------
# Capillary pressure model
# ----------------------------
def DPcap(alpha_m, alpha_m_old):
    Lambda = 3.0 * r_v * (alpha_m - alpha_m0) / (2.0 * alpha_m0 * eps_s * r_p)

    if Lambda <= 0.0:
        return 0.0
    if Lambda >= 2.0:
        return 2.0 * sigma / r_p

    mu = invert_Lambda(Lambda)
    delta_alpha = alpha_m - alpha_m_old

    if mu < 1e-3:
        dmu_dalpha = (3.0 * r_v) / (2.0 * eps_s * alpha_m0 * r_p) * 0.75
    else:
        dmu_dalpha = (9.0 * r_v) / (2.0 * eps_s * alpha_m0 * r_p) * (
            (1.0 - mu**2)**(-0.5) - Lambda / mu
        )

    return 2.0 * sigma / r_p * (mu + dmu_dalpha * delta_alpha)

# ----------------------------
# Sweep alpha_m
# ----------------------------
alpha_vals = np.linspace(0.8 * alpha_m0, 1.4 * alpha_m0, 400)
alpha_old = alpha_m0 * np.ones_like(alpha_vals)

DP_vals = np.array([DPcap(a, ao) for a, ao in zip(alpha_vals, alpha_old)])

# ----------------------------
# Plot
# ----------------------------
plt.figure()
plt.plot(alpha_vals, DP_vals)
plt.axvline(alpha_m0, linestyle="--")
plt.xlabel(r"$\alpha_m$")
plt.ylabel(r"$\Delta P_{\mathrm{cap}}\;[\mathrm{Pa}]$")
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters
k_l = 70.0     # W/m/K  (example: liquid sodium)
D_h = 1e-3     # m      (example hydraulic diameter)

# Velocity range
V = np.linspace(1e-4, 10.0, 500)   # m/s

# H(V)
H = (k_l / D_h) * (5.0 + 0.66 * V**0.8)

# Plot
plt.figure()
plt.plot(V, H)
plt.xlabel("V_l [m/s]")
plt.ylabel("H_xl [W/m²/K]")
plt.grid(True)
plt.show()

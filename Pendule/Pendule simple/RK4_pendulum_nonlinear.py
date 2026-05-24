import numpy as np
import matplotlib.pyplot as plt

# Pendule non-linéaire résolu avec RK4
g = 9.81
l = 0.25
m = 1.0

# Simulation
T = 10.0
dt = 0.001
n = int(T / dt) + 1

theta0 = 10.0
omega0 = 0.0

t = np.linspace(0, T, n)
theta = np.zeros(n)
omega = np.zeros(n)
theta[0] = theta0
omega[0] = omega0

def deriv(state):
    th, w = state
    return np.array([w, -(g / l) * np.sin(th)])

# RK4
for i in range(n - 1):
    y = np.array([theta[i], omega[i]])
    k1 = deriv(y)
    k2 = deriv(y + 0.5 * dt * k1)
    k3 = deriv(y + 0.5 * dt * k2)
    k4 = deriv(y + dt * k3)
    y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    theta[i + 1], omega[i + 1] = y_next

# Energies (exacte pour Ep)
Ec = 0.5 * m * (l * omega) ** 2
Ep = m * g * l * (1 - np.cos(theta))
E = Ec + Ep

# Tracé
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
ax1.plot(t, theta, label='θ (RK4 non-linéaire)')
ax1.set_ylabel('Angle (rad)')
ax1.legend()
ax1.grid(True)

ax2.plot(t, Ec, label='Ec')
ax2.plot(t, Ep, label='Ep')
ax2.plot(t, E, label='E totale', linewidth=1.2)
ax2.set_xlabel('Temps (s)')
ax2.set_ylabel('Energie (J)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
fig_path = 'RK4_pendulum_nonlinear.png'
plt.savefig(fig_path)
print('Figure saved to', fig_path)
plt.show()

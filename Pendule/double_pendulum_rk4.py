#!/usr/bin/env python3
'''
double_pendulum_rk4.py
Simulation d'un pendule double.
Options : intégration RK4 explicite ou solve_ivp de scipy (si installé).
Sauvegarde d'un CSV et animation Matplotlib.
Usage : python double_pendulum_rk4.py
'''
import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt

try:
    from scipy.integrate import solve_ivp
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# -------- Paramètres --------
g = 9.81
L1 = 1.0
L2 = 1.0
M1 = 1.0
M2 = 1.0

# Conditions initiales (degrés -> convertis en radians)
th1_deg = 120.0
w1_deg = 0.0
th2_deg = -10.0
w2_deg = 0.0
state0 = np.radians([th1_deg, w1_deg, th2_deg, w2_deg])

# Discrétisation
dt = 0.01
t_stop = 30.0
t = np.arange(0.0, t_stop + dt, dt)

# -------- Equations du mouvement (Lagrangien -> formes explicites) --------
def derivs(t_unused, state):
    th1, w1, th2, w2 = state
    delta = th2 - th1
    den1 = (M1 + M2) * L1 - M2 * L1 * cos(delta)**2
    den2 = (L2 / L1) * den1

    dw1 = (M2 * L1 * w1*w1 * sin(delta) * cos(delta)
           + M2 * g * sin(th2) * cos(delta)
           + M2 * L2 * w2*w2 * sin(delta)
           - (M1 + M2) * g * sin(th1)) / den1

    dw2 = (- M2 * L2 * w2*w2 * sin(delta) * cos(delta)
           + (M1 + M2) * g * sin(th1) * cos(delta)
           - (M1 + M2) * L1 * w1*w1 * sin(delta)
           - (M1 + M2) * g * sin(th2)) / den2

    return np.array([w1, dw1, w2, dw2])

# -------- Intégrateurs --------
def rk4_integrate(f, state0, t):
    y = np.empty((len(t), len(state0)))
    y[0] = state0.copy()
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        k1 = f(t[i-1], y[i-1])
        k2 = f(t[i-1] + dt/2, y[i-1] + dt*k1/2)
        k3 = f(t[i-1] + dt/2, y[i-1] + dt*k2/2)
        k4 = f(t[i-1] + dt, y[i-1] + dt*k3)
        y[i] = y[i-1] + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
    return y

def integrate(state0, t, method='rk4'):
    if method == 'solve_ivp' and SCIPY_AVAILABLE:
        sol = solve_ivp(derivs, (t[0], t[-1]), state0, t_eval=t, rtol=1e-9, atol=1e-9)
        print("scipy activé")
        return sol.y.T
        
    else:
        return rk4_integrate(derivs, state0, t)

# -------- Simulation --------
y = integrate(state0, t, method='rk4')  # ou method='solve_ivp' si scipy est disponible

# Positions cartésiennes
x1 = L1 * np.sin(y[:,0])
y1 = -L1 * np.cos(y[:,0])
x2 = x1 + L2 * np.sin(y[:,2])
y2 = y1 - L2 * np.cos(y[:,2])

# Sauvegarde CSV
import csv
with open('double_pendulum_simulation.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['t','th1','w1','th2','w2','x1','y1','x2','y2'])
    for i,ti in enumerate(t):
        writer.writerow([ti, y[i,0], y[i,1], y[i,2], y[i,3], x1[i], y1[i], x2[i], y2[i]])
print('CSV sauvegardé : double_pendulum_simulation.csv')

# Tracé simple (angles)
plt.figure(figsize=(8,4))
plt.plot(t, y[:,0], label='theta1')
plt.plot(t, y[:,2], label='theta2')
plt.legend()
plt.xlabel('t (s)')
plt.grid()
plt.show()

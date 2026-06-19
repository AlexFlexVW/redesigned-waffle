import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Simple comparison script: Euler semi-implicit vs RK4 for Benettin MLE

# Physical params
g = 9.81
l = 0.25
w0 = np.sqrt(g / l)

# Numerical params (keep small for quick test)
dt = 1e-4
n = 20000
epsilon = 1e-6
T_renorm = 500

output_dir = Path(r"C:\Users\vanwa\Documents\VSCode Local\resultat_code\Pendule_simple\compare")
output_dir.mkdir(parents=True, exist_ok=True)

def deriv(state):
    th, om = state
    return np.array([om, -w0**2 * np.sin(th)])

def step_euler_semi_implicit(Y, dt):
    th, om = Y
    om = om - w0**2 * np.sin(th) * dt
    th = th + om * dt
    return np.array([th, om])

def step_rk4(Y, dt):
    k1 = deriv(Y)
    k2 = deriv(Y + 0.5 * dt * k1)
    k3 = deriv(Y + 0.5 * dt * k2)
    k4 = deriv(Y + dt * k3)
    return Y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def benettin(step_func, theta0, omega0, n_iter, dt, epsilon, T_renorm):
    Y1 = np.array([theta0, omega0])
    Y2 = np.array([theta0 + epsilon, omega0])

    lambda_array = np.zeros(n_iter)
    sum_log = 0.0
    t = 0.0

    renorm_times = []
    cum_logs = []

    for i in range(n_iter - 1):
        Y1 = step_func(Y1, dt)
        Y2 = step_func(Y2, dt)
        t += dt

        if (i + 1) % T_renorm == 0:
            dth = (Y2[0] - Y1[0] + np.pi) % (2 * np.pi) - np.pi
            dom = Y2[1] - Y1[1]
            d = np.sqrt(dth**2 + dom**2)

            if d > 0:
                sum_log += np.log(d / epsilon)
                factor = epsilon / d
                # renormalize Y2 to distance epsilon from Y1
                Y2[0] = Y1[0] + dth * factor
                Y2[1] = Y1[1] + dom * factor

            lambda_array[i+1] = sum_log / t
            renorm_times.append(t)
            cum_logs.append(sum_log)
        else:
            lambda_array[i+1] = lambda_array[i]

    lyap_final = sum_log / (n_iter * dt)
    slope = None
    if len(renorm_times) >= 2:
        slope, _ = np.polyfit(renorm_times, cum_logs, 1)

    return lambda_array, lyap_final, slope, np.array(renorm_times), np.array(cum_logs)


def run_compare(theta0=1.0, omega0=0.0):
    print(f"Running comparison with dt={dt}, n={n}, epsilon={epsilon}, T_renorm={T_renorm}")

    lam_eu, lyap_eu, slope_eu, times_eu, logs_eu = benettin(step_euler_semi_implicit, theta0, omega0, n, dt, epsilon, T_renorm)
    lam_rk, lyap_rk, slope_rk, times_rk, logs_rk = benettin(step_rk4, theta0, omega0, n, dt, epsilon, T_renorm)

    print("Results:")
    print(f"  Euler semi-implicit -> MLE (sum_log/(n*dt)) = {lyap_eu:.6e}, slope fit = {slope_eu:.6e}")
    print(f"  RK4                -> MLE (sum_log/(n*dt)) = {lyap_rk:.6e}, slope fit = {slope_rk:.6e}")

    # Plot convergence (lambda_array vs iteration)
    it = np.arange(n)
    plt.figure(figsize=(10,5))
    plt.plot(it, lam_eu, label='Euler semi-implicit')
    plt.plot(it, lam_rk, label='RK4', alpha=0.8,linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Lyapunov estimate')
    plt.legend()
    plt.grid(alpha=0.3)
    fname = output_dir / f'compare_lambda_dt{dt}_n{n}.png'
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    print(f"Saved convergence plot: {fname.name}")

    # Plot cumulative log vs time and linear fits
    plt.figure(figsize=(10,5))
    if len(times_eu):
        plt.plot(times_eu, logs_eu, 'o-', label='Euler cum_log')
        if slope_eu is not None:
            plt.plot(times_eu, slope_eu * times_eu + (logs_eu[0] - slope_eu * times_eu[0]), '--', label=f'Euler fit slope={slope_eu:.4e}')
    if len(times_rk):
        plt.plot(times_rk, logs_rk, 'o-', label='RK4 cum_log')
        if slope_rk is not None:
            plt.plot(times_rk, slope_rk * times_rk + (logs_rk[0] - slope_rk * times_rk[0]), '--', label=f'RK4 fit slope={slope_rk:.4e}')

    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative log divergence (sum log(d/epsilon))')
    plt.legend()
    plt.grid(alpha=0.3)
    fname2 = output_dir / f'compare_cumlog_dt{dt}_n{n}.png'
    plt.tight_layout()
    plt.show()
    plt.savefig(fname2, dpi=200)
    print(f"Saved cumulative-log plot: {fname2.name}")


if __name__ == '__main__':
    run_compare()

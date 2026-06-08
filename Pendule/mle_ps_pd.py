import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from numba import njit

# Paramètres globaux d'affichage
plt.rcParams.update({'font.size': 15})

# ============================================================
# Paramètres physiques (communs)
# ============================================================
g  = 9.81
l  = 0.25
w0 = np.sqrt(g / l)
th0_1_pd=np.pi/2

# ============================================================
# Paramètres numériques
# ============================================================
dt       = 0.0001
n        = int(1e6)    # commun aux deux pendules
epsilon  = 1e-6
T_renorm = 500         # intervalle de renormalisation (pendule simple)

# ============================================================
# Pendule SIMPLE — Benettin / Euler semi-implicite
# ============================================================
@njit
def mle_benettin_Euler_simple(theta0, omega0, n_iter):
    th1 = theta0
    om1 = omega0
    th2 = theta0 + epsilon
    om2 = omega0

    sum_log = 0.0
    t       = 0.0

    for i in range(n_iter - 1):
        om1 = om1 - w0**2 * np.sin(th1) * dt
        th1 = th1 + om1 * dt

        om2 = om2 - w0**2 * np.sin(th2) * dt
        th2 = th2 + om2 * dt

        t += dt

        if (i + 1) % T_renorm == 0:
            dth = (th2 - th1 + np.pi) % (2 * np.pi) - np.pi
            dom = om2 - om1
            d   = np.sqrt(dth**2 + dom**2)
            if d > 0:
                sum_log += np.log(d / epsilon)
                factor = epsilon / d
                th2 = th1 + dth * factor
                om2 = om1 + dom * factor

    return sum_log / (n_iter * dt)


# ============================================================
# Pendule DOUBLE — Benettin / Euler semi-implicite
# ============================================================
@njit
def ddtheta(th1, th2, w1, w2):
    delta = th1 - th2
    sin_d = np.sin(delta)
    cos_d = np.cos(delta)
    det   = 2 - cos_d**2

    b1 = -(w2**2) * sin_d - 2 * (g / l) * np.sin(th1)
    b2 =  (w1**2) * sin_d -     (g / l) * np.sin(th2)

    d2th1 = (b1 - cos_d * b2) / det
    d2th2 = (2 * b2 - cos_d * b1) / det
    return d2th1, d2th2


@njit
def mle_benettin_Euler_double(theta_init, omega_init, n_iter):
    Y1 = np.array([theta_init[0], theta_init[1], omega_init[0], omega_init[1]])
    Y2 = np.array([theta_init[0] + epsilon, theta_init[1], omega_init[0], omega_init[1]])

    lyap_sum = 0.0

    for i in range(n_iter - 1):
        d2th1_1, d2th2_1 = ddtheta(Y1[0], Y1[1], Y1[2], Y1[3])
        w1_n1  = Y1[2] + d2th1_1 * dt
        w2_n1  = Y1[3] + d2th2_1 * dt
        th1_n1 = Y1[0] + w1_n1 * dt
        th2_n1 = Y1[1] + w2_n1 * dt

        d2th1_2, d2th2_2 = ddtheta(Y2[0], Y2[1], Y2[2], Y2[3])
        w1_n2  = Y2[2] + d2th1_2 * dt
        w2_n2  = Y2[3] + d2th2_2 * dt
        th1_n2 = Y2[0] + w1_n2 * dt
        th2_n2 = Y2[1] + w2_n2 * dt

        diff_th1 = (th1_n2 - th1_n1 + np.pi) % (2 * np.pi) - np.pi
        diff_th2 = (th2_n2 - th2_n1 + np.pi) % (2 * np.pi) - np.pi
        diff_w1  = w1_n2 - w1_n1
        diff_w2  = w2_n2 - w2_n1

        dist = np.sqrt(diff_th1**2 + diff_th2**2 + diff_w1**2 + diff_w2**2)

        if dist > 0:
            lyap_sum += np.log(dist / epsilon)

        Y1 = np.array([th1_n1, th2_n1, w1_n1, w2_n1])
        Y2 = np.array([
            th1_n1 + (epsilon / dist) * diff_th1,
            th2_n1 + (epsilon / dist) * diff_th2,
            w1_n1  + (epsilon / dist) * diff_w1,
            w2_n1  + (epsilon / dist) * diff_w2,
        ])

    return lyap_sum / (n_iter * dt)


# ============================================================
# Balayage en theta0
# ============================================================
N_POINTS      = 40
theta0_values = np.linspace(0.05, np.pi * 0.98, N_POINTS)
omega0        = 0.0

mle_simple = np.zeros(N_POINTS)
mle_double = np.zeros(N_POINTS)

print("Compilation JIT (chauffe)...")
_ = mle_benettin_Euler_simple(0.5, 0.0, 100)
_ = mle_benettin_Euler_double(np.array([0.5, 0.5]), np.array([0.0, 0.0]), 100)
print("Compilation terminée.\n")

print(f"Calcul MLE pour {N_POINTS} valeurs de θ₀  (n = {n:.0e}) ...")
for idx, th0 in enumerate(theta0_values):
    mle_simple[idx] = mle_benettin_Euler_simple(th0, omega0, n)
    mle_double[idx] = mle_benettin_Euler_double(
        np.array([th0, th0]),
        np.array([0.0, 0.0]),
        n
    )
    if (idx + 1) % 5 == 0:
        print(f"  {idx+1}/{N_POINTS}  |  θ₀ = {th0:.3f} rad  |  "
              f"λ_simple = {mle_simple[idx]:.4f}  |  λ_double = {mle_double[idx]:.4f}")

# ============================================================
# Vérification
# ============================================================
print("\n--- Vérification ---")
print(f"Pendule simple : MLE min = {mle_simple.min():.4f}, max = {mle_simple.max():.4f}")
print("  → Attendu : proche de 0 (système intégrable)")
print(f"Pendule double : MLE min = {mle_double.min():.4f}, max = {mle_double.max():.4f}")
print("  → Attendu : positif et croissant avec θ₀ (chaos)")

# ============================================================
# Graphique
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(theta0_values, mle_simple, 'o-',
        linewidth=2, markersize=5, color='steelblue',
        label='Pendule simple')
ax.plot(theta0_values, mle_double, 's-',
        linewidth=2, markersize=5, color='crimson',
        label='Pendule double ($\\theta_{1,0}=\\theta_{2,0}=\\theta_0$)')
ax.axhline(y=0, color='k', linestyle=':', alpha=0.5, linewidth=1.2, label='$\\lambda = 0$')

ax.set_xlabel("Angle initial $\\theta_0$ (rad)")
ax.set_ylabel("Exposant de Lyapunov maximal $\\lambda$")
ax.set_title(
    f"MLE en fonction de $\\theta_0$ — Méthode de Benettin / Euler semi-implicite\n"
    f"$\\varepsilon={epsilon}$, $n=10^6$"
)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
output_dir = Path(r"C:\Users\vanwa\Documents\VSCode Local\resultat_code\Pendule_simple_et_double\lyapunov_exposant")
output_dir.mkdir(parents=True, exist_ok=True)
out_path = output_dir / "mle_vs_theta0_pendules.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\nGraphique sauvegardé → {out_path}")
plt.close()
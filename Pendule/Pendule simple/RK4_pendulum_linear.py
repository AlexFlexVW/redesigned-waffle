import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paramètres globaux
plt.rcParams.update({'font.size': 15})

# Pendule linéarisé (petit angle) résolu avec RK4
g = 9.81
l = 0.25
m = 1.0

# Simulation
T = 10.0
dt = 0.001
n = int(T / dt) + 1

theta0 = 1.0
omega0 = 0.0

t = np.linspace(0, T, n)
theta = np.zeros(n)
omega = np.zeros(n)
theta[0] = theta0
omega[0] = omega0

def deriv_linear(state):
    th, w = state
    return np.array([w, -(g / l) * th])

# RK4
for i in range(n - 1):
    y = np.array([theta[i], omega[i]])
    k1 = deriv_linear(y)
    k2 = deriv_linear(y + 0.5 * dt * k1)
    k3 = deriv_linear(y + 0.5 * dt * k2)
    k4 = deriv_linear(y + dt * k3)
    y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    theta[i + 1], omega[i + 1] = y_next

# Energies (cohérentes avec l'approximation petit angle: Ep ≈ 1/2 m g l theta^2)
Ec = 0.5 * m * (l * omega) ** 2
Ep = 0.5 * m * g * l * theta ** 2
E = Ec + Ep
E0 = E[0]

# --- QUANTIFICATION DE LA CONSERVATION ---
sigma_rel  = np.std(E) / abs(E0) * 100               # en %
drift      = np.polyfit(np.arange(len(E)), E, 1)[0]  # J/itération
range_rel  = (E.max() - E.min()) / abs(E0) * 100     # en %

print(f"--- Conservation de l'énergie (RK4) ---")
print(f"σ(E)/E₀  = {sigma_rel:.4e} %")
print(f"Dérive   = {drift:.4e} J/itération")
print(f"Range/E₀ = {range_rel:.4e} %")
print("---------------------------------------")

# Solution analytique pour comparaison
w0 = np.sqrt(g / l)
theta_th = theta0 * np.cos(w0 * t)


# --- GESTION DES FICHIERS ---
# Définition du chemin absolu vers ton dossier cible
output_dir = Path(r"C:\Users\vanwa\Documents\VSCode Local\resultat_code\Pendule_simple\rk4")

# SÉCURITÉ : On vérifie si le dossier existe vraiment
if not output_dir.exists():
    # Si le dossier n'existe pas, on lève une erreur et on arrête le script
    raise FileNotFoundError(
        f"\n[ERREUR] Le dossier cible n'existe pas :\n'{output_dir}'\n"
        "Veuillez le créer manuellement avant de lancer le script."
    )

# 1. GRAPHIQUE DES ANGLES
plt.figure(figsize=(9, 5))
plt.plot(t, theta, label='θ (RK4 linéarisé)')
plt.plot(t, theta_th, label='Solution analytique (cos)', linestyle='--')
plt.xlabel('Temps (s)')
plt.ylabel('Angle (rad)')
plt.title(fr'Pendule simple linéarisé — RK4 ($\theta_0={theta0}$ rad)')
plt.legend()
plt.grid(True)
nom_fichier_angle = f"angle_plot_ps_RK4_approx_L_th0={theta0}.png"
angle_path = output_dir / nom_fichier_angle
plt.savefig(angle_path, bbox_inches='tight')


# 2. GRAPHIQUE DE L'ÉNERGIE MÉCANIQUE (ZOOM)
plt.figure(figsize=(9, 5))
plt.plot(t, E, label='E mécanique', linewidth=1.5, color='green')
plt.xlabel('Temps (s)')
plt.ylabel('Energie (J)')
plt.title(fr"Conservation de l'énergie mécanique — RK4 - Linear ($\theta_0={theta0}$ rad)")
plt.legend()
plt.grid(True)
nom_fichier_energie_mec = f"energy_mec_plot_ps_RK4_approx_L_th0={theta0}.png"
energy_tot_path = output_dir / nom_fichier_energie_mec
plt.savefig(energy_tot_path, bbox_inches='tight')

print(f"Graphiques sauvegardés avec succès dans :\n{output_dir}")

# Affichage final
plt.show()
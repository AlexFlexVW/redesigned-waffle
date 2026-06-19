import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paramètres globaux
plt.rcParams.update({'font.size': 15})

# Pendule non-linéaire résolu avec RK4
g = 9.81
l = 0.25
m = 1.0

# Simulation
T = 10.0
dt = 0.001
n = int(T / dt) + 1

theta0 = 1.6
omega0 = 0.0

t = np.linspace(0, T, n)
theta = np.zeros(n)
omega = np.zeros(n)
theta[0] = theta0
omega[0] = omega0

def deriv(state):
    th, w = state
    # Différence majeure ici : on garde le sin(th)
    return np.array([w, -(g / l) * np.sin(th)])

# RK4
for i in range(n - 1):
    y = np.array([theta[i], omega[i]])
    k1 = deriv(y)                   #pente au début du pas (= Euler classique)
    k2 = deriv(y + 0.5 * dt * k1)   #pente au milieu du pas, en utilisant k1 pour s'y projeter
    k3 = deriv(y + 0.5 * dt * k2)   #pente au milieu du pas, en utilisant k2 (estimation améliorée du milieu)
    k4 = deriv(y + dt * k3)         #pente à la fin du pas, en utilisant k3
    y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    #les pentes du milieu comptent double car elles représentent mieux le comportement sur tout l'intervalle (règle de Simpson)
    theta[i + 1], omega[i + 1] = y_next

# Energies (exactes pour Ep)
Ec = 0.5 * m * (l * omega) ** 2
Ep = m * g * l * (1 - np.cos(theta))
E = Ec + Ep
E0 = E[0]

# --- QUANTIFICATION DE LA CONSERVATION ---
sigma_rel  = np.std(E) / abs(E0) * 100               # en %
drift      = np.polyfit(np.arange(len(E)), E, 1)[0]  # J/itération
range_rel  = (E.max() - E.min()) / abs(E0) * 100     # en %

print(f"--- Conservation de l'énergie (RK4 Non-Linéaire) ---")
print(f"σ(E)/E₀  = {sigma_rel:.4e} %")
print(f"Dérive   = {drift:.4e} J/itération")
print(f"Range/E₀ = {range_rel:.4e} %")
print("----------------------------------------------------")


# --- GESTION DES FICHIERS ---
output_dir = Path(r"C:\Users\vanwa\Documents\VSCode Local\resultat_code\Pendule_simple\rk4")

# SÉCURITÉ : On vérifie si le dossier existe vraiment
if not output_dir.exists():
    raise FileNotFoundError(
        f"\n[ERREUR] Le dossier cible n'existe pas :\n'{output_dir}'\n"
        "Veuillez le créer manuellement avant de lancer le script."
    )

# 1. GRAPHIQUE DES ANGLES
plt.figure(figsize=(9, 5))
plt.plot(t, theta, label='θ (RK4 non-linéaire)')
# Note : Pas de solution analytique affichée ici car on est en régime non-linéaire
plt.xlabel('Temps (s)')
plt.ylabel('Angle (rad)')
plt.title(fr'Pendule simple non-linéaire — RK4 ($\theta_0={theta0}$ rad)')
plt.legend()
plt.grid(True)
nom_fichier_angle = f"angle_plot_ps_RK4_NL_th0={theta0}.png"
angle_path = output_dir / nom_fichier_angle
plt.savefig(angle_path, bbox_inches='tight')


# 2. GRAPHIQUE DE L'ÉNERGIE MÉCANIQUE (ZOOM)
plt.figure(figsize=(9, 5))
plt.plot(t, E, label='E totale', linewidth=1.5, color='green')
plt.xlabel('Temps (s)')
plt.ylabel('Energie mécanique(J)')
plt.title(fr"Conservation de l'énergie mécanique — RK4 - NL ($\theta_0={theta0}$ rad)")
plt.legend()
plt.grid(True)
nom_fichier_energie_mec = f"energy_mec_plot_ps_RK4_NL_th0={theta0}.png"
energy_tot_path = output_dir / nom_fichier_energie_mec
plt.savefig(energy_tot_path, bbox_inches='tight')

print(f"Graphiques sauvegardés avec succès dans :\n{output_dir}")

# Affichage final
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Paramètres physiques
g = 9.81          # gravité
l = 0.25          # longueur du pendule (m)
w0 = np.sqrt(g/l) # pulsation

# Paramètres numériques
dt = 0.0001       # très petit pas
n = 50000         # nombre d'itérations

# Conditions initiales
omega0 = 0.0

# Exposant de Lyapunov
epsilon = 0.001   # perturbation initiale très petite

def mle_single(theta0, omega0, n_iter):
    """
    Calcule l'exposant de Lyapunov pour une condition initiale donnée.
    Retourne:
    - lambda_array: tableau de lambda calculé à chaque itération i
    - lyapunov_exponent: valeur finale moyennée
    """
    # Stockage trajectoires
    theta1 = np.zeros(n_iter)
    omega1 = np.zeros(n_iter)
    theta2 = np.zeros(n_iter)
    omega2 = np.zeros(n_iter)
    
    # Conditions initiales
    theta1[0] = theta0
    omega1[0] = omega0
    theta2[0] = theta0 + epsilon
    omega2[0] = omega0
    
    lambda_array = np.zeros(n_iter)
    somme = 0.0
    
    # Méthode d'Euler pour deux trajectoires proches
    for i in range(n_iter - 1):
        # Trajectoire 1
        omega1[i+1] = omega1[i] - w0**2 * np.sin(theta1[i]) * dt
        theta1[i+1] = theta1[i] + omega1[i+1] * dt
        
        # Trajectoire 2 (perturbée)
        omega2[i+1] = omega2[i] - w0**2 * np.sin(theta2[i]) * dt
        theta2[i+1] = theta2[i] + omega2[i+1] * dt
        
        # Divergence
        delta_i = abs(theta2[i+1] - theta1[i+1])
        
        if delta_i > 0:
            somme += np.log(delta_i / epsilon)
            lambda_array[i+1] = somme / ((i + 1) * dt)
    
    lyapunov_exponent = somme / (n_iter * dt) 
    
    return lambda_array, lyapunov_exponent


# Graphique 1: Lambda en fonction de i pour theta0 initial
print("Calcul graphique 1: lambda(i) pour theta0 = 0.5 rad...")
lambda_vs_i, exp1 = mle_single(0.5, omega0, n)
iterations = np.arange(n)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(iterations, lambda_vs_i, linewidth=0.5)
ax.set_xlabel("Itération i")
ax.set_ylabel("Exposant de Lyapunov λ")
ax.set_title(f"Exposant de Lyapunov en fonction des itérations (θ₀ = 0.5 rad)")
ax.grid(True, alpha=0.3)
ax.axhline(y=exp1, color='r', linestyle='--', label=f'Valeur finale: {exp1:.6f}')
ax.legend()
plt.tight_layout()
plt.show()

# Graphique 2: Exposant de Lyapunov en fonction de theta0
print("Calcul graphique 2: Lyapunov exponent en fonction de θ₀...")
theta0_values = np.linspace(0.1, np.pi, 30)
lyapunov_exponents = np.zeros_like(theta0_values)

for idx, theta0 in enumerate(theta0_values):
    _, lyap = mle_single(theta0, omega0, n)
    lyapunov_exponents[idx] = lyap
    if (idx + 1) % 10 == 0:
        print(f"  Progression: {idx + 1}/{len(theta0_values)}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(theta0_values, lyapunov_exponents, 'o-', linewidth=2, markersize=6)
ax.set_xlabel("Angle initial θ₀ (rad)")
ax.set_ylabel("Exposant de Lyapunov λ")
ax.set_title("Exposant de Lyapunov en fonction de la condition initiale")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nCalculs terminés !")

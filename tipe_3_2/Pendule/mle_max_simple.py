"""
Calcul du MLE par méthode du maximum — Pendule double
======================================================

Définition utilisée :
    λ(t) = ln( δ(t) / δ(0) ) / t

où δ(t) est la distance entre deux trajectoires proches dans l'espace (θ₁, θ₂).
Le MLE est le MAXIMUM de λ(t) sur toute la durée d'observation.

Cette approche est adaptée aux pendules amortis car :
  - lim_{t→∞} λ(t) → -∞  (le frottement finit par dominer)
  - le max capture la phase chaotique transitoire avant stabilisation

Pré-requis :
    pip install numpy pandas matplotlib

Usage :
    python mle_max_simple.py --csv data.csv --col1 theta1 --col2 theta2 --dt 0.01
    python mle_max_simple.py --csv data.csv --col1 theta1 --col2 theta2 --dt 0.01 --plot
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_series(path, col1, col2, sep=None):
    """Charge deux colonnes angulaires depuis le CSV."""
    if sep is None:
        with open(path) as f:
            line = f.readline()
        sep = ';' if line.count(';') > line.count(',') else ','

    df = pd.read_csv(path, sep=sep)
    print(f"CSV chargé : {len(df)} lignes | colonnes : {list(df.columns)}")

    for col in [col1, col2]:
        if col not in df.columns:
            raise ValueError(f"Colonne '{col}' introuvable. "
                             f"Colonnes disponibles : {list(df.columns)}")

    theta1 = df[col1].dropna().values.astype(float)
    theta2 = df[col2].dropna().values.astype(float)
    n = min(len(theta1), len(theta2))
    return theta1[:n], theta2[:n]


def mle_par_max(theta1, theta2, dt):
    """
    Calcule λ(t) = ln( δ(t) / δ(0) ) / t  pour chaque t,
    puis retourne le maximum (= MLE estimé).

    δ(t) = distance euclidienne dans l'espace (θ₁, θ₂) entre
           la trajectoire enregistrée et une trajectoire "voisine"
           construite par petite perturbation initiale.

    ⚠️  Avec une seule trajectoire enregistrée, on ne peut pas
        mesurer δ(t) directement. On l'estime en comparant des
        segments temporellement proches (méthode des paires de voisins
        sur la trajectoire unique) :

        Pour chaque indice i, on trouve l'indice j le plus proche dans
        l'espace des phases (|i-j| > min_sep pour éviter la continuité
        triviale), et on suit δ_ij(t) = ‖(θ₁[i+t]-θ₁[j+t], θ₂[i+t]-θ₂[j+t])‖

        On moyenne ensuite ln(δ_ij(t)/δ_ij(0)) sur toutes les paires,
        ce qui donne une estimation robuste de λ(t).
    """
    N = len(theta1)
    traj = np.column_stack([theta1, theta2])   # shape (N, 2)

    # ── Paramètres ────────────────────────────────────────────────────────
    min_sep  = max(10, N // 20)   # séparation temporelle minimale entre voisins
    max_iter = N // 3             # durée de suivi des paires

    # ── Trouver les paires de voisins proches ─────────────────────────────
    from scipy.spatial import KDTree
    tree = KDTree(traj)
    # On demande assez de voisins pour pouvoir filtrer les proches temporels
    k = min(min_sep + 5, N - 1)
    distances, indices = tree.query(traj, k=k)

    pairs = []   # liste de (i, j, d0)
    for i in range(N - max_iter):
        for ki in range(1, k):
            j = indices[i, ki]
            if j >= N - max_iter:
                continue
            if abs(i - j) > min_sep:
                d0 = distances[i, ki]
                if d0 > 0:
                    pairs.append((i, j, d0))
                    break   # on garde seulement le meilleur voisin valide

    if len(pairs) == 0:
        raise ValueError("Aucune paire de voisins trouvée. "
                         "Vérifiez que le CSV contient suffisamment de points.")

    print(f"{len(pairs)} paires de voisins proches trouvées.")

    # ── Calcul de λ(t) = moyenne de ln(δ(t)/δ(0)) / t ────────────────────
    lambda_t = np.full(max_iter, np.nan)

    for step in range(1, max_iter):
        ln_ratios = []
        for (i, j, d0) in pairs:
            d_t = np.linalg.norm(traj[i + step] - traj[j + step])
            if d_t > 0:
                ln_ratios.append(np.log(d_t / d0))
        if ln_ratios:
            lambda_t[step] = np.mean(ln_ratios) / (step * dt)

    # ── MLE = maximum de λ(t) ─────────────────────────────────────────────
    valid_mask = ~np.isnan(lambda_t)
    lambda_valid = lambda_t[valid_mask]
    times = np.arange(max_iter)[valid_mask] * dt

    mle_idx  = np.argmax(lambda_valid)
    mle      = lambda_valid[mle_idx]
    mle_time = times[mle_idx]

    return times, lambda_valid, mle, mle_time


def main():
    parser = argparse.ArgumentParser(
        description="MLE par méthode du maximum — Pendule double"
    )
    parser.add_argument('--csv',  required=True,            help="Fichier CSV")
    parser.add_argument('--col1', required=True,            help="Colonne angle bras 1 (ex: theta1)")
    parser.add_argument('--col2', required=True,            help="Colonne angle bras 2 (ex: theta2)")
    parser.add_argument('--dt',   type=float, required=True, help="Pas de temps (s)")
    parser.add_argument('--sep',  default=None,             help="Séparateur CSV (auto si omis)")
    parser.add_argument('--plot', action='store_true',      help="Afficher le graphe")
    parser.add_argument('--save', default=None,             help="Sauvegarder le graphe (ex: mle.png)")
    args = parser.parse_args()

    # ── Chargement ────────────────────────────────────────────────────────
    theta1, theta2 = load_series(args.csv, args.col1, args.col2, sep=args.sep)
    N = len(theta1)
    print(f"Série : {N} points  |  durée = {N * args.dt:.2f} s")

    # ── Calcul MLE ────────────────────────────────────────────────────────
    print("\nCalcul de λ(t) en cours...")
    times, lambda_t, mle, mle_time = mle_par_max(theta1, theta2, args.dt)

    print(f"\n{'═'*50}")
    print(f"  MLE (méthode du max) = {mle:.5f}  s⁻¹")
    print(f"  Atteint à t = {mle_time:.4f} s")
    if mle > 0:
        print(f"  → Comportement CHAOTIQUE (MLE > 0)")
    else:
        print(f"  → Comportement STABLE (MLE ≤ 0)")
    print(f"{'═'*50}\n")

    # ── Graphe ────────────────────────────────────────────────────────────
    if args.plot or args.save:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("MLE — Pendule double (méthode du maximum)", fontsize=13)

        # Trajectoire θ₁, θ₂
        t_axis = np.arange(N) * args.dt
        axes[0].plot(t_axis, theta1, label=args.col1, lw=0.8)
        axes[0].plot(t_axis, theta2, label=args.col2, lw=0.8, alpha=0.7)
        axes[0].set_xlabel("Temps (s)")
        axes[0].set_ylabel("Angle (rad ou °)")
        axes[0].set_title("Trajectoires enregistrées")
        axes[0].legend()

        # λ(t) et son maximum
        axes[1].plot(times, lambda_t, color='steelblue', lw=1.5,
                     label='λ(t) = ⟨ln(δ(t)/δ₀)⟩ / t')
        axes[1].axhline(0, color='gray', lw=0.8, ls='--')
        axes[1].axvline(mle_time, color='red', lw=1.2, ls='--', alpha=0.6)
        axes[1].scatter([mle_time], [mle], color='red', zorder=5,
                        label=f'MLE = {mle:.4f} s⁻¹  (t={mle_time:.3f}s)')
        axes[1].set_xlabel("Temps (s)")
        axes[1].set_ylabel("λ(t)  [s⁻¹]")
        axes[1].set_title("Exposant de Lyapunov instantané")
        axes[1].legend(fontsize=9)

        plt.tight_layout()
        if args.save:
            plt.savefig(args.save, dpi=150, bbox_inches='tight')
            print(f"Graphe sauvegardé : {args.save}")
        if args.plot:
            plt.show()


if __name__ == "__main__":
    main()

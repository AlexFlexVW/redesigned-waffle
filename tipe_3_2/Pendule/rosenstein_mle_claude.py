"""
Calcul du Maximum Lyapunov Exponent (MLE) depuis une série temporelle CSV
Algorithme de Rosenstein, Collins & De Luca (1993)

Adapté aux pendules réels avec frottement :
  - La définition lim_{t→∞} ln(δ(t)/ε)/t → -∞ à cause de l'amortissement
  - On estime plutôt la PENTE MAXIMALE de la divergence moyenne des voisins
    proches, ce qui correspond au MLE "transitoire" avant stabilisation

Usage :
    python mle_pendule.py --csv data.csv --col angle --dt 0.01
    python mle_pendule.py --csv data.csv --col angle --dt 0.01 --embed 3 --tau 5
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import linregress


# ─────────────────────────────────────────────────────────────
# 1. RECONSTRUCTION DE L'ESPACE DES PHASES (Théorème de Takens)
# ─────────────────────────────────────────────────────────────

def mutual_information(x, max_tau=50, bins=64):
    """
    Estime le délai τ optimal via le premier minimum de l'Information Mutuelle.
    C'est la méthode recommandée (Fraser & Swinney 1986) plutôt que
    l'autocorrélation, car elle capture les dépendances non-linéaires.
    """
    tau_range = range(1, min(max_tau, len(x) // 4))
    mi_values = []

    x_norm = (x - x.min()) / (x.ptp() + 1e-12)

    for tau in tau_range:
        x1 = x_norm[:-tau]
        x2 = x_norm[tau:]
        hist2d, _, _ = np.histogram2d(x1, x2, bins=bins)
        hist2d = hist2d / hist2d.sum()
        hx  = -np.sum(p * np.log(p + 1e-12) for p in np.sum(hist2d, axis=1))
        hy  = -np.sum(p * np.log(p + 1e-12) for p in np.sum(hist2d, axis=0))
        hxy = -np.sum(hist2d * np.log(hist2d + 1e-12))
        mi_values.append(hx + hy - hxy)

    # Premier minimum local
    mi_values = np.array(mi_values)
    for i in range(1, len(mi_values) - 1):
        if mi_values[i] < mi_values[i-1] and mi_values[i] < mi_values[i+1]:
            return i + 1  # +1 car tau_range commence à 1
    return int(len(mi_values) * 0.1) + 1  # fallback : 10% de la plage


def false_nearest_neighbors(x, tau, max_dim=8, threshold=15.0):
    """
    Détermine la dimension d'embedding m optimale via les Faux Voisins Proches
    (Kennel, Brown & Abarbanel 1992).
    On cherche la dimension où le % de faux voisins devient négligeable.
    """
    fnn_fractions = []

    for m in range(1, max_dim + 1):
        # Construction de la matrice d'embedding
        N = len(x) - (m) * tau
        if N < 20:
            break
        embedded = np.column_stack([x[i*tau: i*tau + N] for i in range(m)])

        tree = KDTree(embedded)
        distances, indices = tree.query(embedded, k=2)

        fnn_count = 0
        valid = 0
        for j in range(N - tau):
            nn_idx = indices[j, 1]
            if nn_idx >= N - tau:
                continue
            d_m = distances[j, 1]
            if d_m < 1e-12:
                continue
            # Vérification dans m+1 dimensions
            d_m1 = abs(x[j + m*tau] - x[nn_idx + m*tau])
            if d_m1 / d_m > threshold:
                fnn_count += 1
            valid += 1

        fnn_fractions.append(fnn_count / valid if valid > 0 else 0)

    # Dimension où FNN tombe en dessous de 5%
    fnn_fractions = np.array(fnn_fractions)
    candidates = np.where(fnn_fractions < 0.05)[0]
    if len(candidates) > 0:
        return candidates[0] + 1
    return int(np.argmin(fnn_fractions)) + 1


def embed(x, m, tau):
    """Construit la matrice d'espace des phases retardé de taille (N', m)."""
    N = len(x) - (m - 1) * tau
    return np.column_stack([x[i*tau: i*tau + N] for i in range(m)])


# ─────────────────────────────────────────────────────────────
# 2. ALGORITHME DE ROSENSTEIN (MLE sur série temporelle)
# ─────────────────────────────────────────────────────────────

def rosenstein_mle(x, dt, m, tau, max_iter=None, min_tsep=None):
    """
    Calcule le MLE via l'algorithme de Rosenstein et al. (1993).

    Idée centrale :
        y(i) = (1/N) * Σ_j ln( d_j(i) )   pour i = 0, 1, ..., max_iter
    où d_j(i) est la distance entre la trajectoire j et son plus proche
    voisin au temps i.
    Le MLE est la pente de la partie LINÉAIRE de y(i) vs i*dt.

    Paramètres
    ----------
    x        : série temporelle 1D (numpy array)
    dt       : pas de temps (secondes)
    m        : dimension d'embedding
    tau      : délai (en indices)
    max_iter : nombre de pas de temps à suivre (défaut : len(x)//4)
    min_tsep : séparation temporelle minimale entre voisins (défaut : tau*m)
               évite les "voisins temporels" qui ne reflètent pas la géométrie

    Retourne
    --------
    times       : tableau des temps (len = max_iter)
    mean_ln_div : divergence logarithmique moyenne y(i)
    mle         : pente estimée = MLE (en 1/s)
    fit_start   : indice de début du fit linéaire
    fit_end     : indice de fin du fit linéaire
    """
    X = embed(x, m, tau)
    N = len(X)

    if max_iter is None:
        max_iter = N // 4
    if min_tsep is None:
        min_tsep = tau * m

    # Trouver le plus proche voisin (en excluant les voisins temporels proches)
    tree = KDTree(X)
    # On demande k=min_tsep+2 voisins pour pouvoir filtrer les proches temporels
    k_query = min(min_tsep + 10, N - 1)
    distances, indices = tree.query(X, k=k_query)

    nn_indices = np.full(N, -1, dtype=int)
    for j in range(N):
        for ki in range(1, k_query):
            idx = indices[j, ki]
            if abs(idx - j) > min_tsep:
                nn_indices[j] = idx
                break

    # Suivi de la divergence
    valid_pairs = [j for j in range(N) if nn_indices[j] != -1
                   and j + max_iter < N
                   and nn_indices[j] + max_iter < N]

    if len(valid_pairs) == 0:
        raise ValueError("Aucune paire de voisins valide trouvée. "
                         "Essayez de réduire min_tsep ou d'augmenter N.")

    divergence = np.zeros((len(valid_pairs), max_iter))
    for k, j in enumerate(valid_pairs):
        nj = nn_indices[j]
        for i in range(max_iter):
            d = np.linalg.norm(X[j + i] - X[nj + i])
            divergence[k, i] = np.log(d) if d > 0 else np.nan

    # Moyenne en ignorant les NaN
    mean_ln_div = np.nanmean(divergence, axis=0)
    times = np.arange(max_iter) * dt

    # ── Estimation de la pente (MLE) ──────────────────────────────────────
    # On cherche la région la plus linéaire de mean_ln_div.
    # Stratégie : fenêtre glissante de largeur ~20% de max_iter,
    # on garde celle avec le meilleur R².
    window = max(10, max_iter // 5)
    best_r2 = -np.inf
    fit_start, fit_end = 0, max_iter - 1

    for start in range(0, max_iter - window):
        end = start + window
        seg = mean_ln_div[start:end]
        if np.any(np.isnan(seg)):
            continue
        slope, intercept, r, *_ = linregress(times[start:end], seg)
        if r**2 > best_r2:
            best_r2 = r**2
            fit_start, fit_end = start, end

    # Fit final sur la région retenue
    seg_t = times[fit_start:fit_end]
    seg_y = mean_ln_div[fit_start:fit_end]
    mle, intercept, r_value, p_value, std_err = linregress(seg_t, seg_y)

    print(f"\n── Résultats du fit linéaire ──────────────────────────")
    print(f"   Fenêtre : t = [{seg_t[0]:.4f}s, {seg_t[-1]:.4f}s]")
    print(f"   R²      : {r_value**2:.4f}")
    print(f"   MLE     : {mle:.4f} ± {std_err:.4f}  [1/s]")
    if mle > 0:
        print(f"   → Comportement CHAOTIQUE détecté (MLE > 0)")
    else:
        print(f"   → Comportement STABLE (MLE ≤ 0, amortissement dominant)")
    print(f"──────────────────────────────────────────────────────")

    return times, mean_ln_div, mle, fit_start, fit_end, intercept


# ─────────────────────────────────────────────────────────────
# 3. LECTURE CSV + PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────

def load_csv(path, col, dt_col=None, sep=None):
    """
    Charge la série temporelle depuis le CSV.
    Détecte automatiquement le séparateur si non spécifié.
    """
    if sep is None:
        # Détection automatique : virgule ou point-virgule
        with open(path, 'r') as f:
            first_line = f.readline()
        sep = ';' if first_line.count(';') > first_line.count(',') else ','

    df = pd.read_csv(path, sep=sep)
    print(f"\nCSV chargé : {len(df)} lignes, colonnes : {list(df.columns)}")

    if col not in df.columns:
        raise ValueError(f"Colonne '{col}' introuvable. Colonnes disponibles : {list(df.columns)}")

    x = df[col].dropna().values.astype(float)
    print(f"Série '{col}' : {len(x)} points")
    return x


def main():
    parser = argparse.ArgumentParser(
        description="Calcul du MLE (Rosenstein 1993) depuis un CSV de pendule"
    )
    parser.add_argument('--csv',   required=True,          help="Chemin vers le fichier CSV")
    parser.add_argument('--col',   required=True,          help="Nom de la colonne de données (ex: 'angle')")
    parser.add_argument('--dt',    type=float, required=True, help="Pas de temps en secondes (ex: 0.01)")
    parser.add_argument('--embed', type=int,   default=None,  help="Dimension d'embedding m (auto si omis)")
    parser.add_argument('--tau',   type=int,   default=None,  help="Délai τ en indices (auto si omis)")
    parser.add_argument('--iter',  type=int,   default=None,  help="Nombre d'itérations de suivi (auto si omis)")
    parser.add_argument('--sep',   default=None,              help="Séparateur CSV (auto-détecté si omis)")
    parser.add_argument('--plot',  action='store_true',       help="Afficher les graphiques")
    parser.add_argument('--save',  default=None,              help="Sauvegarder le graphe (ex: mle.png)")
    args = parser.parse_args()

    # ── Chargement ────────────────────────────────────────────
    x = load_csv(args.csv, args.col, sep=args.sep)

    # Normalisation (stabilise le calcul)
    x = (x - np.mean(x)) / (np.std(x) + 1e-12)

    # ── Paramètres d'embedding automatiques ───────────────────
    if args.tau is None:
        print("\nEstimation du délai τ via Information Mutuelle...")
        tau = mutual_information(x)
        print(f"  → τ = {tau} indices ({tau * args.dt:.4f} s)")
    else:
        tau = args.tau
        print(f"  → τ fixé à {tau} indices")

    if args.embed is None:
        print("Estimation de la dimension d'embedding m via FNN...")
        m = false_nearest_neighbors(x, tau)
        print(f"  → m = {m}")
    else:
        m = args.embed
        print(f"  → m fixé à {m}")

    # ── Calcul MLE ─────────────────────────────────────────────
    print(f"\nCalcul MLE (Rosenstein) avec m={m}, τ={tau}...")
    times, mean_ln_div, mle, fs, fe, intercept = rosenstein_mle(
        x, args.dt, m, tau, max_iter=args.iter
    )

    # ── Visualisation ──────────────────────────────────────────
    if args.plot or args.save:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"MLE — {args.col} (m={m}, τ={tau})", fontsize=13)

        # Série temporelle
        axes[0].plot(np.arange(len(x)) * args.dt, x, lw=0.7, color='steelblue')
        axes[0].set_xlabel("Temps (s)")
        axes[0].set_ylabel("Amplitude normalisée")
        axes[0].set_title("Série temporelle")

        # Divergence + fit
        axes[1].plot(times, mean_ln_div, color='steelblue', lw=1.5,
                     label='Divergence moyenne ln⟨d(t)⟩')
        fit_line = intercept + mle * times[fs:fe]
        axes[1].plot(times[fs:fe], fit_line, 'r--', lw=2,
                     label=f'Régression linéaire\nMLE = {mle:.4f} s⁻¹')
        axes[1].axvspan(times[fs], times[fe-1], alpha=0.12, color='red',
                        label='Fenêtre de fit')
        axes[1].set_xlabel("Temps (s)")
        axes[1].set_ylabel("ln⟨d(t)⟩")
        axes[1].set_title("Divergence des trajectoires voisines")
        axes[1].legend(fontsize=9)

        plt.tight_layout()
        if args.save:
            plt.savefig(args.save, dpi=150, bbox_inches='tight')
            print(f"\nGraphe sauvegardé : {args.save}")
        if args.plot:
            plt.show()

    print(f"\n{'═'*54}")
    print(f"  MLE = {mle:.5f}  s⁻¹")
    print(f"{'═'*54}\n")
    return mle


if __name__ == "__main__":
    main()

# Analyse des fichiers Lyapunov

> Transcription de la conversation : revue des fichiers calculant l'exposant de Lyapunov pour le pendule simple et le pendule double.

## Question initiale

> Vérifie les fichiers `lyapunov_exposant` et dis-moi s'il y a des choses étonnantes ou qui pourraient être des erreurs.

## Fichiers analysés

- `Pendule/Pendule simple/Lyapunov_exposant(t) L1`
- `Pendule/Pendule simple/Lyapunov_exposant(t) L2`
- `Pendule/Pendule simple/Lyapunov_exposant(t)_avec_somme`
- `Pendule/Pendule simple/Lyapunov_exponant(theta0)`
- `Pendule/Double Pendulum/By me/Lyapunov Exposant`
- `Pendule/Double Pendulum/By me/Lyapunov Exposant (meth_Benettin)`

---

## Bugs réels

### `Lyapunov_exponant(theta0)` — le plus ancien

**1. `epsilon` négatif (ligne 23)**
```python
epsilon = -0.00001
```
Puis ligne 40 : `delta[0] = epsilon` → `delta[0] < 0`. Dans la boucle :
```python
somme += np.log(delta[i] / delta[0])
```
On calcule le log d'un nombre négatif (puisque `delta[i] > 0` toujours, car c'est une norme). Résultat : **NaN** partout.

**2. Variable `delta` masquée**
- Au niveau module : `delta = np.zeros(n)`
- Dans la fonction `mle` : `delta = np.zeros(n)` (variable locale qui masque)
- À la fin : `plt.plot(t, delta)` utilise la variable **module** (zéros), pas celle calculée dans `mle`.

Le graphe affiche donc des zéros.

**3. Décalage temporel dans `delta`**
```python
delta[i+1] = np.sqrt((theta2[i]-theta1[i])**2 + (omega2[i]-omega1[i])**2)
somme += np.log(delta[i] / delta[0])
```
L'écart à l'étape `i+1` est calculé avec les positions de l'étape `i`, puis on lit `delta[i]` (ancienne valeur) dans la même itération. Décalage d'un pas.

---

### `Lyapunov Exposant (meth_Benettin)` — ligne 111

```python
ax2.set_xlabel("Angle initial θ2_0 (rad) [θ1_0 fixé à ", th1_0, "]")
```
`set_xlabel` n'accepte pas plusieurs arguments comme `print`. Lève une `TypeError` à l'exécution.

**Correction :**
```python
ax2.set_xlabel(f"Angle initial θ2_0 (rad) [θ1_0 fixé à {th1_0}]")
```

---

### Tous les fichiers sauf Benettin — variable `limite` non initialisée

Dans L1, L2 et `avec_somme`, `limite` n'est définie qu'à l'intérieur du `if delta_i > 0:`. Si ce bloc n'est jamais exécuté, la ligne finale :
```python
lyapunov_exponent = limite / (n_iter * dt)
```
lève un **`UnboundLocalError`**. Risque faible mais réel.

---

## Choses étonnantes (pas des bugs mais à noter)

### L1 vs L2 — définition de δ différente

| Fichier | Mesure de δ |
|---|---|
| **L1** | `abs(θ₂ - θ₁)` — 1D dans l'espace des angles |
| **L2** | `sqrt(Δθ² + Δω²)` — norme dans l'espace de phase |

L2 est physiquement plus correct pour un exposant de Lyapunov : la divergence se mesure dans tout l'espace de phase, pas uniquement sur les angles.

### Absence de renormalisation

Sauf le fichier Benettin, aucun fichier ne renormalise la trajectoire perturbée. Sur 100 000 itérations avec un système chaotique :
- `delta` peut croître jusqu'à overflow,
- ou saturer (en pratique, `delta` est bornée par la taille de l'espace de phase),
- ou tomber à zéro.

La **méthode de Benettin** (renormalisation à chaque pas) est la seule méthode fiable pour les longues simulations. Seul le fichier `Lyapunov Exposant (meth_Benettin)` l'implémente correctement.

### Commentaire trompeur dans L1 et L2 (ligne 92)

```python
# CORRECTION ICI : La fonction mle_single retourne 4 valeurs, il faut bien les réceptionner
_, lyap_exp = mle_single(theta0, omega0, n)
```

La fonction retourne **2 valeurs**, pas 4. Reliquat d'une ancienne version copiée du fichier `avec_somme` (qui lui retourne effectivement 4 valeurs).

### `avec_somme` — pourquoi la "version somme" est incorrecte

La somme `∑ log(δᵢ/ε)` n'est pas la bonne formule. L'exposant de Lyapunov se calcule comme :
- `log(δₙ/ε) / T` (valeur finale, "version limite"), ou
- avec renormalisation à la Benettin.

La somme cumulative sans renormalisation donne un estimateur biaisé qui diverge linéairement avec le temps.

---

## Résumé par priorité

| Fichier | Problème | Sévérité |
|---|---|---|
| `Lyapunov_exponant(theta0)` | `epsilon < 0` → NaN | Critique |
| `Lyapunov_exponant(theta0)` | `delta` masquée → graphe vide | Critique |
| `Lyapunov Exposant (meth_Benettin)` | `set_xlabel` multi-args → crash | Critique |
| L1, L2, `avec_somme` | `limite` non initialisée | Potentiel |
| L1 | Mesure δ 1D seulement | Physique |
| Tous sauf Benettin | Pas de renormalisation | Méthode |
| L1, L2 | Commentaire "4 valeurs" faux | Mineur |

---

## Recommandations

1. **Supprimer ou réparer** `Lyapunov_exponant(theta0)` : trop de bugs, semble être une ancienne version.
2. **Corriger le `set_xlabel`** dans `Lyapunov Exposant (meth_Benettin)`.
3. **Initialiser `limite = 0`** avant la boucle dans L1, L2 et `avec_somme`.
4. **Unifier L1 et L2** : utiliser systématiquement la norme dans l'espace de phase (version L2).
5. **Privilégier la méthode de Benettin** (avec renormalisation) comme implémentation de référence, y compris pour le pendule simple.

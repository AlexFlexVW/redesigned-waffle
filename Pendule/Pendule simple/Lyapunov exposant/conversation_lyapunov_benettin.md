# Exposant de Lyapunov pour le pendule simple — méthode naïve vs Benettin

Transcription d'une conversation autour du calcul numérique de l'exposant de Lyapunov maximal (MLE) pour un pendule simple, et de la motivation de la méthode de Benettin avec renormalisation.

---

## 1. Interprétation d'un graphique λ(θ₀) obtenu par méthode naïve

**Question :** Le script `Lyapunov_exposant(t) L2` (deux trajectoires perturbées de ε, sans renormalisation, formule `λ = log(δ_final/ε)/T_total`) produit un graphe λ en fonction de θ₀ avec des valeurs entre 0.03 et 0.7, un creux vers θ₀ ≈ 0.9 et un pic vers θ₀ ≈ 1.4. Comment interpréter ?

**Réponse :**

**Point crucial : le pendule simple n'est PAS chaotique.** C'est un système intégrable (1 degré de liberté, conservatif). Le vrai exposant de Lyapunov devrait être **λ = 0** pour toutes les valeurs de θ₀.

### Ce qu'on observe réellement

Les valeurs non-nulles (0.03 à 0.7) ne sont **pas du chaos**, mais un **artefact numérique** dû à la méthode de mesure :

1. **Dérive de phase** : les deux trajectoires partent avec θ₀ et θ₀+ε. Comme la période d'un pendule non-linéaire **dépend de l'amplitude** (T augmente avec θ₀, surtout près de π), les deux trajectoires ont des périodes légèrement différentes → elles se déphasent.

2. **Croissance linéaire, pas exponentielle** : l'écart δ(t) entre les deux trajectoires croît **linéairement** dans le temps (avec des oscillations), pas exponentiellement comme dans le vrai chaos.

3. **La formule** `λ = log(δ_final/ε) / T_total` capte cette croissance linéaire. Si δ ~ A·t, alors λ_mesuré ~ log(A·t)/t → 0, mais **très lentement**. Pour n = 100 000 ce n'est pas encore convergé vers 0.

### Pourquoi cette forme de courbe ?

- **θ₀ → 0** : régime quasi-harmonique, T quasi-indépendant de l'amplitude (T ≈ T₀), donc très peu de déphasage → λ ≈ 0.
- **θ₀ croissant** : ∂T/∂θ₀ augmente, déphasage plus rapide → δ plus grand → λ apparent plus grand.
- **Le creux vers θ₀ ≈ 0.9** : effet d'interférence. À l'instant final t = n·dt, les deux trajectoires se retrouvent presque en phase (δ proche d'un minimum local de l'oscillation), donc log(δ/ε) est petit. Aucune signification physique — change `n` et le creux se déplacera.
- **Le pic vers θ₀ ≈ 1.4** : forte dépendance de T à l'amplitude proche de π/2, donc déphasage maximal.

### Conclusion

Le graphique **confirme** (involontairement) que le pendule simple n'est pas chaotique : avec n plus grand et une méthode propre (Benettin avec renormalisation), λ tend vers 0 partout. La méthode "deux trajectoires sans renormalisation" est trompeuse car elle confond **dérive de phase** (réversible, non chaotique) et **divergence exponentielle** (chaos).

---

## 2. Implémentation de la méthode de Benettin

Nouveau fichier `Lyapunov_exposant(t) L2 Benettin` :

```python
import numpy as np
import matplotlib.pyplot as plt

# Paramètres physiques
g = 9.81
l = 0.25
w0 = np.sqrt(g/l)

# Paramètres numériques
dt = 0.0001
n = 1000000

# Conditions initiales
omega0 = 0.0

# Perturbation initiale (norme conservée par renormalisation)
epsilon = 1e-6

# Intervalle de renormalisation (en nombre de pas)
T_renorm = 500


def mle_benettin(theta0, omega0, n_iter):
    """
    Méthode de Benettin : intégration de deux trajectoires
    + renormalisation périodique de l'écart à la norme epsilon,
    en sommant log(d/epsilon).
    """
    th1, om1 = theta0, omega0
    th2, om2 = theta0 + epsilon, omega0

    lambda_array = np.zeros(n_iter)
    sum_log = 0.0
    t = 0.0

    for i in range(n_iter - 1):
        # Euler pour les deux trajectoires
        om1 = om1 - w0**2 * np.sin(th1) * dt
        th1 = th1 + om1 * dt
        om2 = om2 - w0**2 * np.sin(th2) * dt
        th2 = th2 + om2 * dt
        t += dt

        # Renormalisation périodique
        if (i + 1) % T_renorm == 0:
            dth = th2 - th1
            dom = om2 - om1
            d = np.sqrt(dth**2 + dom**2)

            if d > 0:
                sum_log += np.log(d / epsilon)
                factor = epsilon / d
                th2 = th1 + dth * factor
                om2 = om1 + dom * factor

            lambda_array[i+1] = sum_log / t
        else:
            lambda_array[i+1] = lambda_array[i]

    lyapunov_exponent = sum_log / (n_iter * dt)
    return lambda_array, lyapunov_exponent
```

---

## 3. Différences entre les deux méthodes

### Méthode naïve

Deux trajectoires intégrées indépendamment du début à la fin :

```python
delta_i = sqrt((theta2 - theta1)**2 + (omega2 - omega1)**2)
lambda = log(delta_i / epsilon) / (i * dt)
```

Les deux trajectoires évoluent **librement** : δ part de ε, peut grossir, osciller, saturer. Aucune intervention.

### Méthode de Benettin

On **interrompt** régulièrement l'intégration (toutes les `T_renorm` itérations) pour :

1. Mesurer la divergence actuelle `d`.
2. Stocker `log(d/ε)` dans une somme cumulative.
3. **Rapprocher** la trajectoire perturbée de la référence, dans la même direction, à distance ε.

Le MLE final est alors :

$$\lambda = \frac{1}{T_{\text{total}}} \sum_k \log\!\left(\frac{d_k}{\varepsilon}\right)$$

---

## 4. Motivation détaillée de la renormalisation

### 4.1 La définition mathématique du MLE et pourquoi elle pose problème

$$\lambda = \lim_{t \to \infty} \lim_{\|\delta_0\| \to 0} \frac{1}{t} \log \frac{\|\delta(t)\|}{\|\delta_0\|}$$

**L'ordre des limites est crucial** : d'abord ε → 0 (perturbation infinitésimale), **puis** t → ∞. Cet ordre garantit qu'on mesure la dynamique de l'**équation linéarisée** autour de la trajectoire de référence, pas la dynamique non-linéaire complète.

Sur ordinateur on ne peut pas faire ε = 0 (limité par la précision machine), ni t = ∞. Le danger : si on prend t trop grand sans renormaliser, l'écart δ(t) sort du régime linéaire et la formule perd son sens.

### 4.2 Régime linéaire vs non-linéaire

Soit X(t) = (θ(t), ω(t)) la trajectoire de référence et X(t)+δ(t) la trajectoire perturbée. L'équation exacte pour δ :

$$\dot{\delta} = F(X + \delta) - F(X)$$

Tant que ‖δ‖ est petit :

$$\dot{\delta} \approx J(X(t)) \cdot \delta + O(\|\delta\|^2)$$

C'est l'**équation aux variations** : linéaire en δ, avec des coefficients qui dépendent du temps. Dans ce régime, ‖δ(t)‖ croît (ou décroît) exponentiellement en moyenne : c'est exactement ce que mesure λ.

Dès que ‖δ‖ devient comparable à l'échelle caractéristique du système (~1 rad pour le pendule), le terme O(‖δ‖²) n'est plus négligeable :

- Système chaotique : δ sature autour du diamètre de l'attracteur.
- Pendule simple : δ oscille de façon bornée car les deux trajectoires restent sur des orbites proches.

Dans les deux cas, mesurer `log(δ/ε)/t` après saturation ne donne plus le vrai λ.

### 4.3 Mécanisme de la renormalisation pas à pas

**Étape k (entre t_{k-1} et t_k = k·τ avec τ = T_renorm·dt) :**

1. À t_{k-1}, deux trajectoires séparées par δ_{k-1} avec ‖δ_{k-1}‖ = ε.
2. On intègre librement pendant τ. À t_k, l'écart est devenu δ_k avec ‖δ_k‖ = d_k.
3. Si τ est bien choisi, d_k ≪ taille du système → toujours dans le régime linéaire.
4. **Facteur d'expansion local** sur cet intervalle :

   $$\mu_k = \frac{d_k}{\varepsilon}$$

5. On enregistre `log μ_k`, puis on remet la trajectoire perturbée à distance ε **en conservant la direction** :

   $$\delta_k^{\text{nouveau}} = \varepsilon \cdot \frac{\delta_k}{d_k}$$

**Après N intervalles** :

$$\lambda \approx \frac{1}{N\tau} \sum_{k=1}^{N} \log \mu_k = \frac{1}{T_{\text{total}}} \log \prod_k \mu_k$$

Le **produit** des facteurs d'expansion locaux donne le facteur d'expansion total que δ aurait subi **si rien ne l'avait limité**.

### 4.4 Pourquoi conserver la direction lors de la remise à l'échelle

Le MLE est associé à un **vecteur propre dominant** (vecteur de Lyapunov principal) qui dépend du point X(t). Sous l'action de la dynamique linéarisée, **tout vecteur initial s'aligne exponentiellement vite avec ce vecteur dominant**.

Si à chaque renormalisation on repartait avec une direction **arbitraire** (par exemple toujours (ε, 0)), on détruirait cet alignement à chaque étape → λ sous-estimé.

En faisant `δ_k^{nouveau} = ε · δ_k / d_k`, on change **seulement la norme**, pas la direction.

### 4.5 Choix de T_renorm

Deux contraintes opposées :

**Borne supérieure** : τ doit être assez court pour que d_k = ε·exp(λ·τ) reste ≪ taille du système. Pour le pendule (échelle ~1 rad) avec λ ~ 1 s⁻¹ et ε = 1e-6 :

$$\tau \ll \frac{\log(1/\varepsilon)}{\lambda} \approx 14 \text{ s}$$

**Borne inférieure** : τ doit être assez long pour que δ ait le temps de s'aligner avec le vecteur de Lyapunov dominant après chaque renormalisation. Empiriquement τ doit être au moins de l'ordre du temps caractéristique du système (la période).

Avec `T_renorm = 500` et `dt = 1e-4` → τ = 0.05 s, plus court qu'une période du pendule (T₀ ≈ 1 s). Pour le pendule simple ce n'est pas grave (λ → 0 quand même) mais pour le double pendule, augmenter T_renorm à ~5000-10000.

### 4.6 Application au pendule simple

- Les deux trajectoires ont des **périodes légèrement différentes** (période dépend de l'amplitude).
- Sur un intervalle τ court, δ peut croître ou décroître selon la phase courante.
- `log μ_k` est tantôt positif, tantôt négatif, avec moyenne nulle.
- La somme `Σ log μ_k` reste bornée (marche aléatoire de moyenne nulle).
- Divisée par T_total qui croît linéairement, λ → 0. ✓

Sans renormalisation, δ saturerait et `log(δ_final/ε)/T` mesurerait essentiellement `log(δ_max/ε)/T` qui tend vers 0 **logarithmiquement** seulement → convergence très lente, valeurs apparentes ~0.3-0.7 sur des temps finis.

### 4.7 Universalité de la méthode

Pour un système chaotique (double pendule) :

- À chaque intervalle, δ est **systématiquement amplifié** par un facteur en moyenne > 1.
- `log μ_k` est en moyenne positif (= λ·τ).
- La somme croît linéairement avec N → λ tend vers une valeur finie positive.

Benettin donne le bon λ qu'il soit nul (intégrable), positif (chaotique), ou négatif (dissipatif convergent), parce qu'elle s'appuie toujours sur la dynamique linéaire **locale**.

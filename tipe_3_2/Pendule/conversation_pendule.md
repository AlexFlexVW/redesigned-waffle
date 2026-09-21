# Conversation — Pendule simple, pendule double et exposant de Lyapunov

Date : 2026-05-17

---

## 1. Explication des fichiers du dossier *Pendule simple*

Tous les fichiers utilisent la méthode d'Euler explicite pour résoudre
θ̈ = −ω₀²·sin(θ), avec g = 9.81 m/s² et l = 0.25 m (donc ω₀ ≈ 6.26 rad/s,
T₀ ≈ 1.0 s).

### Simulations de base

- **Simple Pendulum approx petit angle** — équation **linéarisée**
  θ̈ = −ω₀²·θ. Compare θ(t) numérique à θ₀·cos(ω₀t).
- **Simple pendulum** — version **non linéarisée** (avec sin(θ)). Détecte
  les maxima locaux pour mesurer la période, et la compare à T₀ et à la
  période théorique exacte (intégrale).

### Étude de la période T

- **Calcul de T** — T par maxima locaux, affiche T(occurrence) vs T₀.
- **T0 et T (theta0)** — T_num et T_théo (intégrale elliptique K(k))
  en fonction de θ₀. Montre l'anharmonicité.
- **T0-T (dt)** — erreur |T₀ − T_num| en fonction de dt (échelle log).
- **Différence de période théorique et analytique en fonction de w0** —
  même chose mais en faisant varier dt.
- **T en fonctin de w0(inutile actually)** — relation triviale T = 2π/ω₀.

### Exposant de Lyapunov (λ ≈ 0 attendu pour le pendule simple)

- **Lyapunov_exposant(t) L1** — δ = |Δθ| (norme L1).
- **Lyapunov_exposant(t) L2** — δ = √(Δθ² + Δω²) (norme L2 dans l'espace
  des phases).
- **Lyapunov_exposant(t)_avec_somme** — compare l'estimateur « limite »
  log(δ/ε)/t et l'estimateur « somme » Σlog(δ/ε)/t.
- **Lyapunov_exponant(theta0)** — variante avec L2 et somme.

---

## 2. Fichier *Trajectoires et Lyapunov L1 L2* (pendule simple)

Trois sous-graphes verticaux partageant l'axe du temps :

- a) θ₁(t) et θ₂(t) = θ₁(t) + ε
- b) ω₁(t) et ω₂(t)
- c) λ(t) avec norme L1 (|Δθ|) et norme L2 (√(Δθ² + Δω²))

Avec annotation de l'écart max en angle et en vitesse sur les graphes
correspondants.

---

## 3. Question : pourquoi le système semble peu chaotique ?

> Il semble que le système ne soit pas très chaotique car la distance se
> réduit et les exposants de Lyapunov aussi. Est-ce que ceci est attendu ?

**Oui, c'est attendu.** Le pendule simple non forcé, non amorti est un
système **intégrable**, **pas chaotique**. Son vrai exposant de Lyapunov
est λ = 0.

**Dans le code** :

1. `lambda = log(δ/ε) / t` : si δ reste **borné**, log(δ/ε) reste borné,
   donc en divisant par t → λ → 0.
2. La distance δ(t) **oscille** car θ₁ et θ₂ ont des périodes très
   légèrement différentes (anharmonicité : T dépend de θ₀). Elles se
   croisent et se recroisent → pics négatifs de λ.
3. La norme L1 (|Δθ|) s'annule à chaque croisement angulaire ; la norme
   L2 (qui inclut Δω) est plus régulière car Δθ et Δω ne s'annulent
   jamais simultanément.

Sur un pendule double avec une grande énergie, on observerait λ converger
vers une valeur **positive**.

---

## 4. Fichier équivalent pour le pendule double

Dans le dossier *Double Pendulum/By me/*, fichier **Trajectoires et
Lyapunov L1 L2** :

- Équations matricielles 2 − cos²δ (masses et longueurs égales).
- Intégration Euler semi-implicite.
- Perturbation ε = 10⁻³ sur θ₁(0).
- 3 sous-graphes :
  - a) θ₁ et θ₂ pour les deux trajectoires
  - b) ω₁ et ω₂ pour les deux trajectoires
  - c) λ(t) avec L1 = √(Δθ₁² + Δθ₂²) et L2 = √(Δθ₁² + Δθ₂² + Δω₁² + Δω₂²)

---

## 5. Pour ou contre l'ajout de la vitesse dans le calcul de λ ?

### Pour (norme dans tout l'espace des phases)

- **Définition mathématique correcte** (théorème d'Oseledec) : l'exposant
  est défini sur le flot dans R²ᴺ.
- **Évite les zéros accidentels** : Δθ peut s'annuler à un croisement
  même si les vitesses diffèrent. Δ phase complet ne s'annule presque
  jamais.
- **Plus régulier et converge plus vite** numériquement.
- **Cohérent avec l'algorithme de Benettin** qui renormalise le vecteur
  tangent complet.

### Contre (norme positionnelle seule)

- **Plus simple expérimentalement** : on filme l'angle, pas la vitesse.
- **Asymptotiquement équivalente** : pour t grand, λ_L1 et λ_L2
  convergent vers la même valeur (le MLE).
- **Argument dimensionnel** : mélanger rad et rad/s dans une même norme
  est dimensionnellement bancal sans pondération.

### Consensus dans la littérature

- **Théorie et calcul rigoureux** : norme dans l'espace des phases
  complet (Benettin et al. 1980, Wolf et al. 1985, Ott, Strogatz).
- **Séries temporelles expérimentales** : Rosenstein (1993), Kantz
  (1994) utilisent un espace des phases reconstruit par délais.
- **|Δθ| seul** : essentiellement dans des codes pédagogiques, présenté
  comme approximation valable asymptotiquement.

**TL;DR** : la norme complète (L2) est la bonne pour estimer le MLE
proprement.

---

## 6. Le calcul actuel est-il correct ? Faut-il Benettin ?

### Méthode actuelle (deux trajectoires séparées par ε)

- ✅ Correct sur le **régime linéaire** (t < quelques fois 1/λ).
- ❌ Biaisé à la baisse aux temps longs : quand δ atteint la taille de
  l'attracteur (~2π), il sature et log(δ/ε)/t décroît en 1/t.

### Méthode de Benettin–Galgani–Strelcyn (1980)

1. Deux trajectoires séparées par δ₀ = ε.
2. Tous les τ pas, mesurer δ_k = ‖x_B − x_A‖.
3. **Renormaliser** B à distance ε de A, dans la même direction.
4. Accumuler λ ≈ (1/(Nτ)) · Σ log(δ_k / ε).

**Avantages** : δ reste petit (régime linéaire toujours valide),
convergence vers le vrai MLE même à t → ∞. C'est l'algorithme standard.

Variante encore plus rigoureuse : **méthode tangente** — on intègre
δẋ = J(x)·δx en parallèle de la trajectoire de référence.

---

## 7. Perturber θ₁ plutôt que θ₂ : canonique ?

**Non, ce n'est pas canonique** — et ça n'a **aucune importance
asymptotique**.

Pourquoi : le MLE est associé à une direction propre (le vecteur de
Lyapunov dominant). Quelle que soit la perturbation initiale, tant
qu'elle a une composante non nulle sur cette direction (génériquement
vrai), la divergence s'aligne exponentiellement vite sur ce vecteur et
le taux mesuré converge vers le même MLE.

Conventions courantes :
- Vecteur unitaire arbitraire dans l'espace des phases, p.ex. δ₀ = ε·(1,0,0,0).
- Benettin renormalise dans la direction observée → corrige les
  mauvaises directions de départ automatiquement.
- Spectre complet : nécessite un ensemble orthonormal de perturbations
  (méthode QR).

---

## 8. Le MLE attendu est-il le même quelle que soit la quantité perturbée ?

**Oui, exactement**, à condition que :

1. La perturbation soit infinitésimale.
2. Elle ait une composante non nulle sur le vecteur de Lyapunov dominant
   (mesure pleine — le cas pathologique orthogonal est de mesure nulle).

**Démonstration sommaire** : la perturbation tangente vérifie
δx(t) = Σᵢ cᵢ e^{λᵢ t} vᵢ(t). Le terme dominant en e^{λ₁ t} écrase tous
les autres, donc ‖δx(t)‖ ∼ |c₁| e^{λ₁ t}. Le coefficient cᵢ initial
dépend de la direction perturbée mais entre dans un terme constant
log|c₁| qui devient négligeable une fois divisé par t.

**Conséquence pratique** :
- Perturber θ₁, θ₂, ω₁ ou ω₂ → même MLE.
- Seul le transitoire (durée avant stabilisation) peut différer.

**Nuance** : ceci vaut pour le MLE seul. Pour le spectre complet, il
faut suivre N perturbations orthogonales (Benettin–QR).

---

## 9. Faut-il renormaliser θ entre 0 et 2π ?

### Pour les calculs numériques : **non**

- L'équation utilise sin(θ) et cos(θ), 2π-périodiques → θ = 5π et θ = π
  donnent la même physique. Aucune nécessité de borner.
- Renormaliser introduit des **discontinuités** :
  - dérivée numérique : pic parasite.
  - différence entre trajectoires (Lyapunov) : δ artificiellement énorme
    quand une seule trajectoire saute de 2π → λ_L1 explose.
- ω n'est pas borné physiquement (rotation continue possible).
- Aucun problème de précision : double float garde ~15 chiffres
  significatifs même pour θ ~ 10⁴ rad.
- Bonne pratique : θ unwrapped pour tous les calculs.

### Pour la représentation graphique : **ça dépend**

| Usage | Renormaliser ? |
|-------|---|
| Intégration numérique | Non |
| Calcul d'exposant de Lyapunov | **Surtout pas** |
| Calcul d'énergie | Non (sin/cos s'en chargent) |
| Tracé θ(t) | Non (sauf besoin spécifique) |
| Section de Poincaré, portrait de phase | Oui, mod 2π |
| Animation visuelle | Sans objet |

Pour θ(t) on garde non renormalisé : voir un angle qui croît linéairement
révèle un régime rotatoire — information physique cruciale, surtout pour
le pendule double aux hautes énergies.

Pour les portraits de phase et sections de Poincaré on renormalise mod
2π : la dynamique vit sur un cylindre (θ ∈ S¹, ω ∈ R).

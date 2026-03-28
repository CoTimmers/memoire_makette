# Pivoting problem



## Push:

Hypothèses et principe de convergence vers les positions stables
L'analyse repose sur plusieurs hypothèses simplificatrices. Le bac est modélisé comme un solide rigide rectangulaire, et le contact avec les murs est supposé ponctuel lors de l'impact initial. La gravité constitue la seule force conservative du système, et les murs sont considérés comme rigides et fixes.

Lors du contact avec un mur, un choc réel engendre en pratique une série de phénomènes complexes : rebonds élastiques ou partiellement élastiques, impulsions normales et tangentielles au point de contact, vibrations structurelles, forces de frottement dynamique, et potentiellement plusieurs impacts successifs avant que le mouvement ne s'amortisse. De même, une vitesse angulaire initiale présente au moment du contact introduit une énergie cinétique de rotation supplémentaire, qui se manifeste par des oscillations ou rotations additionnelles autour du point de contact. Bien que ces effets soient présents dans la réalité, ils ne remettent pas en cause la conclusion finale sur la position d'équilibre atteinte.
En effet, par le principe de minimum d'énergie potentielle, le système converge nécessairement vers une configuration qui minimise la hauteur du centre de masse C. Quelle que soit la complexité du transitoire — rebonds, oscillations, impulsions, ou énergie cinétique de rotation initiale — toute cette énergie se dissipe progressivement par les pertes aux impacts et le frottement, et le bac se stabilise inévitablement dans l'un des minima de l'énergie potentielle. Comme l'illustre le graphe U(θ), ces minima correspondent exactement aux deux configurations où un côté entier du bac repose à plat contre un mur :

Cas 1 : le grand côté repose sur Wall 1 (minimum global, y_CoM = a/2)
Cas 2 : le petit côté repose sur Wall 2 (minimum local, y_CoM = b/2)

Le cas particulier (Case 3 — image 2), où le bac est coincé en appui simultané sur les deux murs sans qu'aucun côté ne soit aligné, correspond à une configuration intermédiaire qui n'est pas un minimum d'énergie et est donc intrinsèquement instable. Une injection d'énergie directionnelle suffit à en sortir, après quoi le système converge à nouveau vers l'un des deux états stables précédents.


* COINCEMENT : 

À l'issue de la phase précédente, le bac repose avec un côté entier sur mur 1. Son orientation θ est fixée. Le seul degré de liberté restant est la position `x` du coin A le long de mur 1 — le bac peut glisser horizontalement, mais ne peut plus tourner.

L'objectif de cette phase est d'amener le bac dans le coin, c'est-à-dire de faire glisser A vers x = 0 jusqu'à ce que le coin B vienne en contact avec mur 2.

### Principe énergétique — minimum dans l'espace de configuration (x, θ)

Le principe de minimum s'applique ici aussi, mais dans une direction différente de l'espace de configuration.

Durant la phase PUSH, le système cherchait un minimum en **θ** à position x quelconque — le bac cherchait l'orientation qui abaisse le plus le CoM. Ici, θ est fixé, et le système cherche un minimum en **x** — le bac cherche la position horizontale qui abaisse le plus le CoM.

Le câble de grue tire le CoM vers le point de suspension situé au-dessus du coin (x = 0). La composante horizontale de la tension est proportionnelle à x_CoM :

```
U_x = T · x_CoM
```

Cette énergie potentielle est minimale quand `x_CoM` est minimal, c'est-à-dire quand le bac est le plus proche du coin. La commande directionnelle vers le coin injecte de l'énergie dans ce sens, et le système converge naturellement vers x_CoM → 0.

Plus généralement, les deux phases ensemble minimisent la distance du CoM au point de suspension dans l'espace de configuration complet (x, θ) :

```
U(x, θ) ∝ distance(CoM, point de suspension)
```

Sous les contraintes imposées par les deux murs, le minimum global atteignable est précisément la configuration **coincée dans le coin** — un côté contre mur 1, le coin B contre mur 2. C'est l'unique point fixe stable du système contraint dans les deux directions simultanément.

### Convergence

La même logique dissipative que dans la phase PUSH s'applique : la commande injecte de l'énergie potentielle dans la direction −x, et la friction le long de mur 1 dissipe l'énergie cinétique. Le bac glisse vers le coin de façon monotone ou avec de petites oscillations amorties, jusqu'à ce que le moniteur détecte le contact de B avec mur 2 (`dist(B, mur 2) < tolérance`).

À ce stade, le bac est coincé dans le coin avec deux points de contact actifs — l'état initial requis pour la phase suivante (PIVOTEMENT).


## Pivotement

À l'issue de la phase PUSH, le bac repose avec deux points de contact actifs : le coin A sur Wall 1 et le coin B sur Wall 2. L'objectif est de faire pivoter le bac de 90° autour de A, en utilisant le mouvement de Wall 2, pour amener le grand côté à plat sur Wall 1.

### Problème de commande

Les forces de réaction aux contacts en A et B ne peuvent pas être calculées avec précision — elles dépendent de la micro-géométrie, de l'élasticité locale et de la dynamique d'impact. On adopte donc la même approche que pour les phases précédentes : raisonner en termes de **tendances énergétiques** et de **conditions géométriques suffisantes**.

### Principe du pendule

Le bac suspendu par le câble de la grue se comporte comme un **pendule** : il cherche à se positionner verticalement sous le point d'accroche. Lorsque la grue se déplace, le minimum d'énergie potentielle se déplace avec elle, et le bac suit naturellement. Ce principe est exploité pour piloter le pivotement sans modèle dynamique explicite.

### Secteur de force acceptable

À chaque instant, l'orientation θ du bac et l'angle ψ de Wall 2 définissent un **secteur angulaire acceptable** pour la direction de la force F, délimité par deux vecteurs :

- **−n_B** : normale sortante de Wall 2 (limite basse). En dessous, la force écarterait B du mur et briserait le contact.
- **CA** : vecteur du centre C vers le pivot A (limite haute). Au-delà, le couple s'inverse et le bac tournerait dans le mauvais sens.

Toute force F dans le secteur `[−nB, CA]` garantit un couple positif autour de A, avec A jouant le rôle de pivot naturel tant que la grue reste positionnée derrière Wall 1.

### Propriété remarquable : secteur constant

La largeur du secteur acceptable est **constante sur toute la course de pivotement** :

$$\psi = \arctan\left(\frac{b/2}{a/2}\right) = 53{,}1°$$

Ce qui implique que Wall 2 doit pivoter d'un angle **strictement supérieur à 36,9°** (= 90° − 53,1°) pour que la grue puisse se placer dans le secteur acceptable derrière Wall 2. On choisit donc **45°** comme angle de rotation de Wall 2 dans cette première phase, ce qui satisfait cette condition dans tous les cas.

### Stratégie de commande

La stratégie se déroule en deux temps :

1. **Phase de pivotement (0° à 45°)** : La grue est positionnée derrière Wall 2, dans le secteur acceptable. Wall 2 pivote lentement jusqu'à 45°. Le bac suit le mouvement par effet pendule — il cherche continuellement à se placer sous la grue, dont le minimum d'énergie potentielle se déplace avec le mur. Le mouvement quasi-statique de Wall 2 minimise les effets d'inertie et les risques de rebonds.

2. **Vérification au steady state** : À 45°, on attend que le bac atteigne son état stable, puis la caméra vérifie que le bac est dans la position attendue. L'action suivante n'est déterminée qu'à ce moment-là, sans réagir aux transitoires intermédiaires.

### Avantage de n'agir qu'au steady state

En n'agissant qu'une fois l'état stabilisé, on s'affranchit de la nécessité de modéliser les dynamiques transitoires (rebonds, oscillations, impulsions). Le contrôleur se contente de classifier un état statique observé par caméra et de décider la prochaine action. Cela réduit la complexité algorithmique, la bande passante requise, et la sensibilité aux erreurs de modèle. En contrepartie, le temps de cycle est plus long, ce qui est acceptable dès lors que la rapidité n'est pas la contrainte principale.

## GLISSEMENT :

À l'issue du pivotement, le bac a tourné de 90° et le mur 2 a également tourné de 90°. Les deux murs sont maintenant colinéaires — mur 1 et mur 2 forment une seule surface plane. Le bac repose avec son côté long à plat sur cette surface, avec le coin A en contact avec les deux murs.

L'objectif est de faire glisser le bac le long des murs pour l'éloigner du coin, jusqu'à ce qu'il ait atteint une position suffisamment dégagée.

## Pourquoi le principe de minimum ne s'applique plus

Dans les phases précédentes, le principe d'énergie potentielle guidait naturellement le bac vers un état d'équilibre géométrique défini. Ici la situation est différente : les deux murs étant colinéaires, il n'existe plus de configuration géométrique privilégiée le long de la surface. L'énergie potentielle est **neutre en translation** — le bac peut se trouver n'importe où le long des murs sans que l'énergie change. Il n'y a pas de puits de potentiel vers lequel converger.

On ne peut donc pas s'appuyer sur un argument énergétique pour garantir la convergence. On ne peut pas non plus calculer la force exacte nécessaire pour déplacer le bac d'une distance donnée, car la force de friction dépend de la réaction normale — elle-même impossible à calculer sans modèle de contact précis.

## Stratégie : feedback visuel itératif

On adopte une approche **itérative basée sur l'observation caméra** :

1. Appliquer une petite force de commande dans la direction souhaitée (le long des murs, en s'éloignant du coin)
2. Observer la nouvelle position du bac via la caméra
3. Vérifier si le déplacement obtenu est suffisant
4. Réitérer jusqu'à ce que le bac ait atteint la position cible

Cette approche est justifiée par le fait que la direction du mouvement souhaité est simple et connue — c'est la direction tangentielle le long des murs. On n'a pas besoin de connaître l'amplitude exacte du déplacement à chaque étape, seulement de détecter si le bac a bougé dans le bon sens et d'accumuler ces petits déplacements jusqu'à la position finale.

La caméra sert uniquement de **capteur de position** — elle mesure où se trouve le bac après chaque impulsion de force, sans chercher à estimer les forces internes ou les réactions de contact.

## Condition d'arrêt

La phase se termine lorsque la caméra détecte que le bac a suffisamment glissé, c'est-à-dire quand la distance entre le coin A et le coin du mur dépasse le seuil défini (`l ≥ l_slide_end`). À ce point, le bac est en position finale et la manipulation est terminée.


* FINAL      : 

La phase finale est symétrique à la phase de coincement (Phase 2), mais dans une configuration tournée de 90°. Après le glissement, le mur 2 se referme. Le bac — dont le côté long est aligné avec les murs — est poussé vers le coin par le mouvement de fermeture de mur 2.

## Principe énergétique — retour au minimum de potentiel

Contrairement à la phase de glissement, on peut ici invoquer à nouveau le principe de minimum d'énergie potentielle. La situation est la suivante :

- Le côté long du bac est posé sur mur 1 — θ est fixé, comme dans la phase COINCEMENT
- Le seul degré de liberté restant est la position x du bac le long de mur 1
- Le câble de grue tire le CoM vers le point de suspension situé au-dessus du coin (x = 0)

La composante horizontale de la tension est minimale quand x_CoM est minimal, c'est-à-dire quand le bac est le plus proche du coin. C'est exactement le même argument qu'en Phase 2 :

```
U_x = T · x_CoM   →   minimum quand x_CoM → 0
```

Le système converge donc naturellement vers le coin sous l'effet combiné de la commande et du mouvement de fermeture de mur 2. La friction le long de mur 1 dissipe l'énergie cinétique et amortit le mouvement.
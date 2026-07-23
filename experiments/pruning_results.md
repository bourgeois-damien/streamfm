# Élagage structuré en profondeur + fine-tuning de réparation (tâche derev)

Note de synthèse pour la rédaction du rapport. Tous les chiffres de ce document
sont mesurés ; les incertitudes et les formulations à éviter sont signalées en
fin de section.

---

## 1. Objectif et positionnement

Réduire le coût d'inférence du modèle StreamFM de déréverbération en streaming,
au-delà des optimisations d'exécution déjà en place (CUDA graph, `torch.compile`,
préallocation des tampons, `channels_last`), qui avaient déjà environ divisé par
deux le temps de calcul sans toucher au modèle.

Le levier retenu ici est **architectural** : supprimer des blocs résiduels
entiers, puis réparer la perte de qualité par un fine-tuning court. Ce choix a
été fait contre un réentraînement complet, pour des raisons de budget temps.

Cible de déploiement : GPU L4, pipeline streaming, **NFE=1**.

## 2. Modèle de départ

| | |
|---|---|
| Backbone | `CausalNCSNpp` (`sgmse/backbones/streaming_unet.py`) |
| `nf` | 128 |
| `ch_mult` | (1, 2, 2, 2) |
| `num_res_blocks` | 2 |
| `input_freqs` | 256 |
| Blocs résiduels | 25 |
| Paramètres du backbone | **27 895 944** |
| Checkpoint professeur | `checkpoints/streamfm_derev.ckpt` |
| Jeu de données | EARS-Reverb_v2_16k |

## 3. Stratégie d'élagage

### 3.1 Critère de sélection des blocs : Block Influence

Chaque bloc résiduel est noté par sa **Block Influence** (BI), définie comme

```
BI = 1 - cos(x_entrée, x_sortie)
```

c'est-à-dire l'angle entre l'entrée et la sortie du bloc. Un BI proche de zéro
signifie que le bloc transforme peu son entrée : il se comporte déjà presque
comme une identité, donc le remplacer par une identité coûte peu.

Mesure faite sur le jeu de test EARS-Reverb, code dans
`experiments/pruning/block_influence.py`, résultats complets dans
`results/pruning/block_influence_derev.json`.

### 3.2 Ce que le classement révèle

Les six blocs les moins influents sont **tous sur le chemin montant** (décodeur),
avec un écart net (BI < 0.20) avant le premier bloc du chemin descendant. C'est
un résultat intéressant en soi pour le rapport : dans ce U-Net causal, la
redondance est concentrée dans le décodeur.

Classement des blocs retenus pour la suppression :

| Rang | Bloc | BI | canaux | fréquences |
|---|---|---|---|---|
| 1 | `up_modules.lvl2_rnb1` | 0.137 | 256 | 64 |
| 2 | `up_modules.lvl1_rnb1` | 0.173 | 256 | 128 |
| 3 | `up_modules.lvl0_rnb1` | 0.187 | 128 | 256 |
| 4 | `up_modules.lvl1_rnb0` | 0.191 | 256 | 128 |
| 5 | `up_modules.lvl2_rnb0` | 0.196 | 256 | 64 |
| 6 | `up_modules.lvl3_rnb1` | 0.199 | 256 | 32 |

### 3.3 Choix de k = 3

On supprime les **trois** blocs de plus faible BI. Argument secondaire qui
conforte le choix : ces trois blocs tombent **un par niveau montant**
(lvl2, lvl1, lvl0), donc aucun niveau ne perd toute sa capacité résiduelle.

Mécanisme : substitution du bloc par un module `StreamingIdentity`
(`sgmse.backbones.streaming_unet.prune_resblocks_`). L'élagage est donc
*structuré en profondeur* — on retire des blocs entiers, pas des poids
individuels — ce qui garantit un gain de latence réel, sans dépendre d'un
support matériel de la parcimonie.

| | Paramètres backbone | Δ |
|---|---|---|
| k = 0 | 27 895 944 | — |
| k = 3 | 24 901 896 | **−10.73 %** |

### 3.4 Gain de latence

Campagne dédiée : pipeline audio streaming complet, CUDA graph total, fp16,
`channels_last`, 500 itérations après 100 de chauffe, GPU **L4**, NFE=1.
Résultats dans `results/pruning/prune_latency_derev.json`.

| k | latence moyenne / trame | p50 | p90 | fraction du budget 16 ms | Δ vs k=0 |
|---|---|---|---|---|---|
| 0 | 1.1384 ms | 1.1365 | 1.1430 | 7.12 % | — |
| 1 | 1.1028 ms | 1.0999 | 1.1080 | 6.89 % | −3.1 % |
| 2 | 1.0730 ms | 1.0686 | 1.0788 | 6.71 % | −5.7 % |
| **3** | **1.0378 ms** | 1.0358 | 1.0442 | 6.49 % | **−8.8 %** |

Le gain de latence (−8.8 %) est inférieur au gain de paramètres (−10.7 %), ce
qui est attendu : les blocs supprimés sont sur des cartes de grande résolution
fréquentielle mais le pipeline garde ses coûts fixes (STFT, iSTFT, solveur).

> **À ne pas faire :** ce couple 1.1384 → 1.0378 ms provient de sa propre
> campagne de mesure. Il ne doit **jamais** être mis dans le même tableau que le
> chiffre de 1.097 ms de la campagne STFTPR, qui a été mesuré séparément.

## 4. Pourquoi la réparation était indispensable

Ablation *zero-shot* : on charge le professeur complet, on remplace les blocs par
des identités, et on évalue **sans aucun réentraînement**. 50 fichiers de test,
NFE=1, pipeline offline, exécution eager. Résultats dans
`results/pruning/prune_ablation_derev.json`.

| k | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ | DistillMOS ↑ |
|---|---|---|---|---|
| 0 | 1.6082 | 0.7270 | −14.71 | 3.2939 |
| 1 | 1.4152 | 0.6626 | −16.99 | 2.9884 |
| 2 | 1.2241 | 0.5686 | −17.85 | 2.6894 |
| 3 | 1.0473 | 0.4339 | −21.44 | 1.7683 |

La dégradation est **catastrophique et monotone** : à k=3 le modèle est
essentiellement détruit (PESQ 1.05, DistillMOS 1.77). Un faible BI indique donc
qu'un bloc est *localement* proche de l'identité, mais **ne prédit pas** que le
réseau survivra à sa suppression — les erreurs se composent le long de la
profondeur. C'est le point méthodologique central de cette partie : le BI est un
bon critère de *classement*, pas un certificat d'innocuité.

## 5. Le fine-tuning de réparation

Configuration : `config/study_prune_streamfm_derev.yaml`.

Schéma : construire le modèle complet → charger le professeur en strict →
élaguer en place → fine-tuner. Les poids survivants partent donc exactement des
poids du professeur (démarrage à chaud), et non d'une initialisation aléatoire.

| Paramètre | Valeur | Justification |
|---|---|---|
| Initialisation | `checkpoints/streamfm_derev.ckpt` | démarrage à chaud depuis le professeur |
| Learning rate | 5e-5, constant | réparation courte, pas un réentraînement |
| Scheduler | aucun | idem |
| Gradient clipping | aucun | |
| `max_steps` | 25 000 | |
| Batch effectif | 12 | identique au protocole d'origine |
| Validation | toutes les 1 000 étapes, 50 batches | |
| Sauvegarde | toutes les 500 étapes | pour la sélection de checkpoint |

Exécution finale sur **1× L40S**, 12 workers de dataloader, environ 0.87 it/s
mesuré de bout en bout (le débit est limité par le dataloader, pas par le GPU :
`it/s ≈ num_workers / 15`).

## 6. Évolution de l'entraînement

### 6.1 Les métriques de qualité

Estimateur interne à la boucle d'entraînement : 20 fichiers de validation.

| Étape | PESQ | Lecture |
|---|---|---|
| 0 | 1.533 | modèle élagué, non réparé |
| 999 | 2.394 | **l'essentiel de la réparation est déjà faite** |
| 2 999 | 2.606 | |
| moyenne 2 999 – 7 999 | 2.546 | |
| moyenne 8 999 – 13 999 | 2.681 | dernier palier réel (≈ 2.5 σ) |
| moyenne 14 999 – 20 999 | 2.642 | plus aucun gain mesurable |

ESTOI et SI-SDR suivent la même forme. **La réparation converge très vite** :
~90 % du rattrapage avant l'étape 3 000, un dernier palier vers 9 000, puis
plateau. Les ~12 000 dernières étapes n'ont rien apporté de mesurable.

### 6.2 Les losses ne suivent pas les métriques

- `valid_loss` : reste entre 369 et 396 sur toute la course, avec une légère
  dérive **vers le haut** (début ≈ 374, fin ≈ 385).
- `train_loss` : descend de 338 à ≈ 290 jusqu'à l'étape 8 000, puis s'aplatit.

Ce n'est pas une anomalie. La loss de flow matching est **aveugle aux métriques
perceptives** : `sgmse/model.py:673` ne calcule que `0.5·||v_cible − v_prédit||²`,
moyennée sur tous les niveaux de bruit *t* du chemin. Elle mesure la
reconstruction du champ de vitesse, pas la qualité perçue. Et à NFE=1 le
déploiement n'utilise qu'une seule tranche de ce chemin, alors que la loss les
moyenne toutes : elle mesure quelque chose qu'on n'exécute pas.

## 7. Sélection du checkpoint

### 7.1 Protocole

Cinq candidats sauvegardés (étapes 17 500, 19 500, 20 000, 22 500, 25 000),
évalués sur le **split de validation** :

- 200 fichiers (et non les 20 de la boucle d'entraînement), tirés avec
  `selection_seed=42` — le même tirage pour les cinq candidats ;
- `seed=42` : le bruit du flow matching est **identique** pour tous, donc l'écart
  observé est une différence de modèle et rien d'autre ;
- NFE=1, pipeline offline, eager, fp32 — la configuration de déploiement.

### 7.2 Résultats

| Étape | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ | LSD ↓ |
|---|---|---|---|---|
| 17 500 | 1.9898 | 0.7551 | −9.68 | 15.48 |
| **19 500** | **2.0539** | **0.7609** | −8.18 | 15.42 |
| 20 000 | 2.0455 | 0.7607 | −8.61 | 15.56 |
| 22 500 | 2.0197 | 0.7553 | **−8.07** | 15.94 |
| 25 000 | 2.0242 | 0.7571 | −8.29 | **15.16** |

**Retenu : l'étape 19 500** — premier sur PESQ et sur ESTOI, jamais dernier.

La sélection **confirme le plateau** vu sur les courbes : entre 19 500 et 25 000
l'écart de PESQ est de 0.034, soit du bruit. Seul 17 500 décroche nettement.

### 7.3 Sélection sur métrique, pas sur loss

Comparaison directe, sur ces cinq mêmes checkpoints :

| Étape | `valid_loss` ↓ | PESQ (200 fichiers) ↑ |
|---|---|---|
| 17 500 | 395.5 | 1.9898 |
| 19 500 | 393.4 | **2.0539** |
| 20 000 | 393.4 | 2.0455 |
| 22 500 | 377.4 | 2.0197 |
| 25 000 | **374.7** | 2.0242 |

**Corrélation r = +0.07** — c'est-à-dire nulle, et de surcroît de signe contraire
à l'intuition (une loss plus basse va très légèrement avec une PESQ plus basse).
Une sélection sur la loss aurait désigné l'étape 25 000 ; la sélection sur la
qualité désigne 19 500.

La règle appliquée est donc : **on sélectionne sur la métrique qu'on cherche à
maximiser, mesurée dans la configuration qu'on va déployer.** La loss reste utile
pour diagnostiquer l'entraînement (divergence, explosion de gradient), pas pour
choisir un checkpoint.

Garde-fous contre le sur-ajustement de la sélection elle-même : 200 fichiers
plutôt que 20, bruit fixé, et **le split de test n'a joué aucun rôle dans le
choix**.

## 8. Résultat final sur le jeu de test

Mesure unique, réalisée après la sélection. Jeu de test standard
`EARS-Reverb_v2_16k/test.csv`, 50 fichiers tirés avec `selection_seed=42`,
`seed=42`, `crop_mode=full`, pipeline offline, eager, fp32, DistillMOS activé.
Le professeur a été **remesuré dans la même campagne** plutôt que cité depuis
l'ablation.

> **Contrôle de reproductibilité :** le professeur à NFE=1 retombe exactement sur
> les valeurs de la campagne d'ablation (1.6082 / 0.7270 / −14.714 / 15.885 /
> 3.2939), chiffre pour chiffre. Les deux campagnes sont donc cohérentes.

### NFE = 1 (configuration de déploiement)

| | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ | LSD ↓ | DistillMOS ↑ |
|---|---|---|---|---|---|
| k=0, professeur | 1.6082 | 0.7270 | −14.71 | 15.89 | 3.2939 |
| k=3, **sans** réparation | 1.0473 | 0.4339 | −21.44 | — | 1.7683 |
| k=3, **réparé** | **1.6195** | 0.7023 | **−14.34** | **15.70** | 3.2462 |

### NFE = 5

| | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ | LSD ↓ | DistillMOS ↑ |
|---|---|---|---|---|---|
| k=0, professeur | **1.9964** | **0.7890** | −15.25 | **12.61** | **3.6987** |
| k=3, réparé | 1.9910 | 0.7728 | **−14.60** | 13.01 | 3.6170 |

### Synthèse

| | PESQ professeur | PESQ réparé | Écart |
|---|---|---|---|
| NFE = 1 | 1.6082 | 1.6195 | +0.011 |
| NFE = 5 | 1.9964 | 1.9910 | −0.005 |

L'écart est de l'ordre du centième de PESQ, **dans un sens à NFE=1 et dans
l'autre à NFE=5** — c'est-à-dire nul aux incertitudes près. La réparation n'a
pas sur-spécialisé le modèle sur NFE=1, et ne lui a rien coûté à NFE=5.

## 9. Conclusion défendable

> L'élagage structuré en profondeur de trois blocs résiduels du décodeur, guidé
> par la Block Influence, réduit le backbone de 10.7 % de ses paramètres et la
> latence par trame de 8.8 % sur L4. Appliqué seul, il détruit le modèle
> (PESQ 1.61 → 1.05). Un fine-tuning de réparation de 25 000 étapes à partir des
> poids du professeur restaure intégralement la qualité, à NFE=1 comme à NFE=5,
> les écarts avec le modèle complet restant inférieurs à 0.02 PESQ dans les deux
> régimes.

Second résultat, méthodologique : la réparation converge en moins de 3 000 étapes
pour l'essentiel, et la loss de flow matching est décorrélée des métriques
perceptives (r = +0.07), ce qui impose une sélection de checkpoint sur métrique.

---

## 10. Réserves — à respecter dans la rédaction

1. **50 fichiers de test.** Ne revendiquer aucun des petits écarts, dans un sens
   ni dans l'autre. Écrire « qualité restaurée au niveau du modèle complet », et
   **jamais** « meilleur que le modèle complet ».
2. **Dire « checkpoint final » ou « checkpoint sélectionné sur validation »**,
   jamais « meilleur checkpoint ».
3. **Ne pas fusionner les campagnes de latence.** Le couple 1.1384 → 1.0378 ms
   est autonome ; il ne rejoint pas le chiffre de 1.097 ms de la campagne STFTPR.
4. **La PESQ interne à l'entraînement (≈ 2.6) n'est pas sur la même échelle que
   celle du test (≈ 1.62).** La boucle d'entraînement appelle `enhance()` avec le
   NFE par défaut de la config, pas NFE=1. Seuls les *mouvements relatifs* de
   cette courbe sont interprétables ; ne jamais citer sa valeur absolue à côté
   des chiffres du tableau final.
5. **Confusion non levée :** le modèle réparé a reçu 25 000 étapes de gradient
   supplémentaires que le professeur n'a pas reçues. Un contrôle propre
   (professeur non élagué, fine-tuné sur les mêmes données et le même nombre
   d'étapes) **n'a pas été réalisé**. On ne peut donc pas exclure qu'une partie
   du rattrapage vienne de l'entraînement supplémentaire plutôt que de la seule
   réparation. À mentionner en limite.
6. **Pool de RIR d'entraînement réduit :** 1 286 RIR (ARNI échantillonné depuis
   11 130) contre 132 037 dans l'article d'origine, en raison de liens morts dans
   les jeux de données sources. Cela concerne tout l'entraînement du projet, pas
   seulement cette partie.
7. **Trou dans la courbe W&B entre les étapes ≈ 14 400 et 20 000 :** sur cet
   intervalle, deux exécutions indépendantes ont écrit dans le même run W&B
   (incident de lancement en double). L'ordonnancement des étapes est propre à
   cinq régressions près, ce qui suggère qu'une seule des deux a effectivement
   enregistré, mais ce n'est pas garanti. **Cette portion de courbe ne doit pas
   être présentée comme une trajectoire d'entraînement unique.** Les métriques
   finales ne sont pas affectées : elles proviennent de checkpoints évalués
   directement, pas de la courbe.

## 11. Fichiers de référence

| Contenu | Chemin |
|---|---|
| Classement Block Influence | `results/pruning/block_influence_derev.json` |
| Ablation zero-shot k=0..3 | `results/pruning/prune_ablation_derev.json` |
| Latence par k | `results/pruning/prune_latency_derev.json` |
| Config du fine-tuning | `config/study_prune_streamfm_derev.yaml` |
| Calcul du BI | `experiments/pruning/block_influence.py` |
| Élagage en place | `sgmse/backbones/streaming_unet.py` (`prune_resblocks_`) |
| Évaluation des modèles réparés | `experiments/pruning/modal_eval_healed.py` |
| Checkpoint retenu | volume `streamfm-runs`, `training/derev-prune-k3-ft/checkpoints/sel-step19500.ckpt` |

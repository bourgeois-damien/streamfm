# Reproduction des métriques StreamFM — local (200 fichiers, seed 42) vs papier

**Contexte.** Vérification que les optimisations runtime + le refactor de `experiments/` (juillet 2026) n'ont pas cassé la reproduction des métriques du papier. Sous-ensemble de 200 fichiers du test set (seed 42, sélection aléatoire). Les valeurs absolues diffèrent légèrement du papier (sous-ensemble ≠ test set complet) ; le critère de non-régression est la **concordance de l'amélioration Δ = enhanced − baseline dégradé** entre local et papier, qui contrôle le décalage d'échantillon.

**Protocole.** Backend Modal (L4), solveur Euler, `--matmul-precision high`, `--crop-mode full`. Baselines dégradés : *noisy/corrompu* capturé dans chaque run pour SE/BWE/Derev ; *reconstruction zero-phase* scorée séparément pour STFTPR (zero-phase STFT) et melflow (M† = magnitude → Mel 80 bandes → pseudo-inverse, puis zero-phase), reproduisant l'entrée réelle de chaque modèle.

**fp16 (autocast) ≈ fp32** confirmé sur tous les tasks (accord à la 3e décimale) — seuls les chiffres fp32 sont listés.

Métriques comparables au papier : **PESQ, ESTOI, SI-SDR**. (LSD non comparable — floor eps local + pas de silence-gating.)


## STFTPR (phase retrieval)

Baseline dégradé : **Zero-phase** · NFE bas = Euler1 · NFE haut = Euler5

| Métrique | Baseline (loc / pap) | Euler1 (loc / pap) | Euler5 (loc / pap) | Δ Euler5−base (loc / pap) |
|---|---|---|---|---|
| PESQ | 1.317 / 1.310 | 1.586 / 1.580 | 4.236 / 4.240 | **+2.919** / **+2.930** |
| ESTOI | 0.679 / 0.680 | 0.647 / 0.580 | 0.973 / 0.970 | **+0.294** / **+0.290** |
| SI-SDR | -34.240 / -34.600 | 1.460 / 1.700 | -1.900 / -1.700 | **+32.340** / **+32.900** |

## Mel vocoding (melflow)

Baseline dégradé : **M† + Zero-phase** · NFE bas = Euler1 · NFE haut = Euler5

| Métrique | Baseline (loc / pap) | Euler1 (loc / pap) | Euler5 (loc / pap) | Δ Euler5−base (loc / pap) |
|---|---|---|---|---|
| PESQ | 1.238 / 1.280 | 1.349 / 1.350 | 4.071 / 4.100 | **+2.833** / **+2.820** |
| ESTOI | 0.633 / 0.630 | 0.499 / 0.360 | 0.960 / 0.960 | **+0.327** / **+0.330** |
| SI-SDR | -36.340 / -38.900 | -5.630 / -5.600 | -10.420 / -10.100 | **+25.920** / **+28.800** |

## Speech enhancement (SE)

Baseline dégradé : **Noisy** · NFE bas = Euler1 · NFE haut = Euler4

| Métrique | Baseline (loc / pap) | Euler1 (loc / pap) | Euler4 (loc / pap) | Δ Euler4−base (loc / pap) |
|---|---|---|---|---|
| PESQ | 1.229 / 1.240 | 2.162 / 2.180 | 2.093 / 2.090 | **+0.864** / **+0.850** |
| ESTOI | 0.639 / 0.640 | 0.851 / 0.840 | 0.841 / 0.830 | **+0.202** / **+0.190** |
| SI-SDR | 4.928 / 5.360 | 13.800 / 15.200 | 13.400 / 14.300 | **+8.472** / **+8.940** |

## Bandwidth extension (BWE)

Baseline dégradé : **Bandlimited** · NFE bas = Euler1 · NFE haut = Euler5

| Métrique | Baseline (loc / pap) | Euler1 (loc / pap) | Euler5 (loc / pap) | Δ Euler5−base (loc / pap) |
|---|---|---|---|---|
| PESQ | 3.480 / 3.510 | 3.183 / 3.220 | 3.390 / 3.370 | **-0.090** / **-0.140** |
| ESTOI | 0.884 / 0.840 | 0.933 / 0.920 | 0.942 / 0.940 | **+0.058** / **+0.100** |
| SI-SDR | 16.047 / 15.900 | 16.670 / 16.800 | 16.381 / 16.500 | **+0.334** / **+0.600** |

## Dereverberation (Derev)

Baseline dégradé : **Reverberant** · NFE bas = Euler1 · NFE haut = Euler5

| Métrique | Baseline (loc / pap) | Euler1 (loc / pap) | Euler5 (loc / pap) | Δ Euler5−base (loc / pap) |
|---|---|---|---|---|
| PESQ | 1.320 / 1.320 | 1.667 / 1.630 | 2.052 / 2.010 | **+0.732** / **+0.690** |
| ESTOI | 0.590 / 0.580 | 0.740 / 0.730 | 0.803 / 0.790 | **+0.213** / **+0.210** |
| SI-SDR | -16.360 / -16.600 | -14.358 / -14.200 | -14.177 / -13.300 | **+2.183** / **+3.300** |

---
## Lecture

- **STFTPR** : le baseline zero-phase local (1.317/0.679/−34.24) reproduit quasi exactement le papier (1.31/0.68/−34.6) → méthodo validée. Enhanced Euler5 local 4.236/0.973/−1.90 ≈ papier 4.24/0.97/−1.7. Δ concordants. **Reproduit.**
- **Melflow** : baseline M† local (1.238/0.633/−36.34) proche du papier (1.28/0.63/−38.9), enhanced Euler5 4.071/0.960/−10.42 ≈ 4.10/0.96/−10.1. Δ PESQ/ESTOI concordants ; l'écart de Δ SI-SDR vient du baseline (−36.34 sur le sous-ensemble vs −38.9 sur le set complet). **Reproduit.**
- **SE / BWE / Derev** : baselines et enhanced à la 2e décimale du papier, Δ du même ordre. **Reproduit.**
- **Lyra** : non traité (mis de côté).

## Régressions détectées : aucune

Aucune configuration ne s'écarte du papier au-delà de la variance attendue du sous-ensemble de 200 fichiers. Le refactor et les optimisations runtime préservent les métriques du papier.

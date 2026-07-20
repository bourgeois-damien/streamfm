# Profiling CPU macOS pour Stream.FM

Ce kit sépare quatre niveaux complémentaires :

1. `torch.profiler` attribue le coût aux stages, modules, formes et opérateurs PyTorch.
2. Instruments **Time Profiler** attribue le temps natif aux bibliothèques et stacks C/C++.
3. Instruments **CPU Counters** mesure les événements matériels disponibles sur la puce
   (cycles, instructions, caches et branches selon le modèle de Mac).
4. `powermetrics` suit fréquence, répartition P/E cores, IPC par processus, puissance et
   pression thermique. `/usr/bin/sample` fournit un fallback de stacks sans Xcode.

## Diagnostic

```bash
.venv/bin/python experiments/benchmarks/profiling/macos_cpu.py doctor
```

`xctrace` est livré avec Xcode complet, pas avec les seuls Command Line Tools. Après
installation de Xcode :

```bash
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch
xcrun xctrace list templates
```

Les noms de templates sont localisés/versionnés. Vérifier que `Time Profiler` et
`CPU Counters` figurent dans la liste avant une capture.

## Recette reproductible

Commencer par un profil PyTorch d'un checkpoint précis :

```bash
.venv/bin/python experiments/benchmarks/profiling/backbone.py \
  --ckpt compressed/streamfm_stftpr_k5.ckpt \
  --dtype fp32 --memory-format contiguous \
  --warmup 20 --iterations 100 \
  --out outputs/benchmark_profiles/k5_fp32_contiguous.json
```

Puis capturer une exécution assez longue pour les outils système :

```bash
.venv/bin/python experiments/benchmarks/profiling/macos_cpu.py time-profiler \
  --out outputs/benchmark_profiles/k5_time.trace -- \
  .venv/bin/python experiments/benchmarks/streamfm_benchmark.py \
  --hardware cpu --execution eager --dtype fp32 --pipeline audio \
  --ckpt compressed/streamfm_stftpr_k5.ckpt \
  --memory-format contiguous --preallocate-model-buffers \
  --warmup 50 --iterations 500
```

Remplacer `time-profiler` par `cpu-counters` pour les compteurs. Le fallback sans
Xcode est :

```bash
.venv/bin/python experiments/benchmarks/profiling/macos_cpu.py sample \
  --delay 5 --duration 10 --out outputs/benchmark_profiles/k5_sample.txt -- \
  .venv/bin/python experiments/benchmarks/streamfm_benchmark.py \
  --hardware cpu --execution eager --dtype fp32 --pipeline audio \
  --ckpt compressed/streamfm_stftpr_k5.ckpt \
  --memory-format contiguous --preallocate-model-buffers \
  --warmup 50 --iterations 500
```

`powermetrics` nécessite `sudo`; le wrapper authentifie avant de démarrer la cible :

```bash
.venv/bin/python experiments/benchmarks/profiling/macos_cpu.py powermetrics \
  --sample-rate-ms 100 \
  --out outputs/benchmark_profiles/k5_powermetrics.txt -- \
  .venv/bin/python experiments/benchmarks/streamfm_benchmark.py \
  --hardware cpu --execution eager --dtype fp32 --pipeline audio \
  --ckpt compressed/streamfm_stftpr_k5.ckpt \
  --memory-format contiguous --preallocate-model-buffers \
  --warmup 50 --iterations 500
```

Ne pas comparer la puissance absolue de deux machines à partir de `powermetrics` :
Apple la décrit comme une estimation. Sur une même machine, elle reste utile pour
observer fréquence, throttling, P/E placement et évolution énergétique d'une version.

## Lecture compute-bound / memory-bound

- IPC bas + forte proportion de stalls/cache misses + bande passante élevée : candidat
  memory-bound.
- IPC élevé + unités de calcul/cycles saturés + cache misses modestes : candidat
  compute-bound.
- Faible utilisation globale, wakeups, migrations P/E et nombreux petits opérateurs :
  overhead de dispatch/scheduling probable.
- La classification doit combiner compteurs, formes/MACs et stacks. Un seul compteur
  ou le nom `_slow_conv2d_forward` ne suffit pas.

Les événements Apple ne sont pas identiques aux événements Intel/AMD. Sous Linux,
la même méthode se transpose avec `perf stat`, `perf record`, VTune (Intel) ou uProf
(AMD); le profiler PyTorch et le rapport par formes restent portables.

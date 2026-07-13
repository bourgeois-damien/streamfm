# Experiments

This folder contains project-specific experiments and runnable helpers added on top of the original Stream.FM codebase.

Keep stable model/library code in `sgmse/`, official training and inference entrypoints at the repository root, and local exploration here.

## Layout

```text
experiments/
├── common.py
├── inference_local.py
├── baseline/
│   └── streamfm_se_baseline.py
├── streaming/
│   ├── pipeline.py
│   ├── stft.py
│   ├── eager.py
│   ├── cuda_graph.py
│   └── run_local.py
└── benchmarks/
    ├── streamfm_benchmark.py
    ├── local_streamfm_benchmark.py
    ├── modal_streamfm_benchmark.py
    ├── runner.py
    ├── options.py
    ├── paths.py
    ├── loading.py
    ├── model_loops.py
    ├── cuda_graph.py
    ├── results.py
    └── streamfm_benchmark_core.py
```

## Evaluation Entrypoint

Use `experiments/evaluation/streamfm_eval.py` as the main entrypoint for test-set inference.
When called with `--backend modal`, it delegates to `experiments/evaluation/modal_streamfm_eval.py` internally; you generally should not call the Modal wrapper directly.

By default, `streamfm_eval.py` evaluates complete files (`--crop-mode full`). `--limit 0` evaluates the full split; `--limit N` selects a reproducible random subset with `--selection random --selection-seed 42` unless overridden.

Modal runs automatically copy run metadata back to `outputs/evaluation_logs/<run-name>/`:

- `command.json`: user-facing command options.
- `summary.json`: runtime summary, timing, backend, output paths.
- `manifest.json`: selected files and produced WAV paths.
- `config.yaml`: Hydra config used for the run.

`experiments/evaluation/score_manifest.py` also saves local metric JSON copies by default. For a scored run, look in the same `outputs/evaluation_logs/<run-name>/` folder; for direct dataset/noisy scores, look in `outputs/evaluation_logs/dataset_scores/`.

## What Belongs Where

- `common.py`: shared device selection, timing, repo path, and streaming `forward_step` helpers.
- `inference_local.py`: local CPU/MPS/CUDA-friendly variant of root-level `inference.py`.
- `baseline/`: offline reference runs and speech-enhancement timing helpers.
- `streaming/pipeline.py`: public streaming API kept small for compatibility.
- `streaming/stft.py`: STFT framing, compression, real/imaginary conversion, and synthetic audio helpers.
- `streaming/eager.py`: eager frame-by-frame audio pipelines.
- `streaming/cuda_graph.py`: CUDA Graph audio pipeline variants.
- `benchmarks/streamfm_benchmark.py`: unified local/Modal CLI.
- `benchmarks/local_streamfm_benchmark.py`: compatibility wrapper that delegates to the unified CLI with `--local`.
- `benchmarks/modal_streamfm_benchmark.py`: Modal image/app definitions and remote entrypoint.
- `benchmarks/runner.py`: high-level benchmark orchestration.
- `benchmarks/options.py`, `paths.py`, `loading.py`: CLI normalization, paths, and checkpoint/model loading.
- `benchmarks/model_loops.py`, `cuda_graph.py`: benchmark kernels.
- `benchmarks/results.py`: JSON output and benchmark history writing.
- `benchmarks/streamfm_benchmark_core.py`: compatibility re-export module for older imports.
- `notebooks/`: exploratory notebooks that call into these scripts when possible.
- `inputs/`, `outputs/`, `checkpoints/`, `papers/`: local data/artifacts ignored by git.

Move code into `sgmse/` only when it becomes reusable model or library code rather than experiment glue.

## Read Order

For streaming audio, read `streaming/pipeline.py` first, then `streaming/stft.py`, then either `streaming/eager.py` or `streaming/cuda_graph.py`.

For benchmarks, read `benchmarks/streamfm_benchmark.py` first, then `benchmarks/runner.py`, then the specific implementation module you need: `model_loops.py`, `cuda_graph.py`, `loading.py`, or `results.py`.

## Commandes utiles (benchmark, éval et métriques)

Les trois scripts principaux sont :

- [experiments/benchmarks/streamfm_benchmark.py](benchmarks/streamfm_benchmark.py) pour mesurer un benchmark de modèle ou de pipeline audio.
- [experiments/evaluation/streamfm_eval.py](evaluation/streamfm_eval.py) pour lancer l’inférence sur la split de test.
- [experiments/evaluation/score_manifest.py](evaluation/score_manifest.py) pour calculer les métriques (SI-SDR, ESTOI, LSD, PSNR, PESQ, et éventuellement DistillMOS).

### 1. Lancer un benchmark

Exemple local :

```bash
python experiments/benchmarks/streamfm_benchmark.py \
  --local \
  --hardware auto \
  --task stftpr \
  --part model \
  --pipeline audio \
  --execution eager \
  --steps 1 \
  --iterations 10 \
  --warmup 2 \
  --output-json outputs/benchmark_demo.json
```

Exemple Modal :

```bash
python experiments/benchmarks/streamfm_benchmark.py \
  --backend modal \
  --hardware L4 \
  --task stftpr \
  --part model \
  --pipeline audio \
  --execution cuda_graph \
  --steps 1 \
  --iterations 100 \
  --warmup 10
```

Paramètres principaux :

- `--backend` / `--local` : exécuter localement ou via Modal.
- `--hardware` : `auto`, `cpu`, `mps`, `cuda` localement ; `cpu`, `t4`, `l4`, `l40s`, `a100` sur Modal.
- `--task` : `stftpr`, `bwe`, `derev`, `lyra`, `se`.
- `--part` : `model`, `predictor`, `flow`.
- `--pipeline` : `model_only` (bloc de modèle uniquement) ou `audio` (pipeline audio complet).
- `--execution` : `auto`, `eager`, `compiled`, `cuda_graph`.
- `--steps` : nombre de pas du flow, séparés par des virgules si besoin.
- `--iterations` : nombre de frames mesurées ; utiliser `--audio-duration-s` pour piloter la durée au lieu du nombre d’itérations.f
- `--warmup` : nombre de frames de chauffe.
- `--audio-duration-s` : remplace `--iterations` pour un run audio basé sur une durée en secondes.
- `--dtype` : `fp32`, `fp16`, `bf16`.
- `--num-threads` / `--num-interop-threads` : réglages CPU.
- `--memory-format` : `contiguous` ou `channels_last`.
- `--preallocate-model-buffers` : réutilise les tampons de modèle quand c’est possible.
- `--save-audio` / `--audio-output-dir` / `--input-audio` : enregistrer l’audio de sortie et choisir le fichier d’entrée.
- `--output-json` / `--history-json` : écrire les résultats au format JSON.

### 2. Lancer une évaluation sur la split de test

Exemple local :

```bash
python experiments/evaluation/streamfm_eval.py \
  --backend local \
  --hardware auto \
  --task stftpr \
  --config-name streamfm_stftpr \
  --ckpt checkpoints/streamfm_stftpr.ckpt \
  --split test \
  --limit 20 \
  --selection random \
  --selection-seed 42 \
  --output-dir outputs/eval_runs \
  --run-name demo_run \
  --score-after-run \
  --score-include-stats \
  --score-include-per-file
```

Exemple Modal :

```bash
python experiments/evaluation/streamfm_eval.py \
  --backend modal \
  --hardware L4 \
  --task stftpr \
  --config-name streamfm_stftpr \
  --ckpt checkpoints/streamfm_stftpr.ckpt \
  --split test \
  --limit 20 \
  --selection random \
  --selection-seed 42
```

Paramètres principaux :

- `--backend` / `--local` : local ou Modal.
- `--hardware` : même logique que pour les benchmarks.
- `--task` : modèle à charger.
- `--config-name` : override du nom de config Hydra.
- `--ckpt` : chemin ou nom du checkpoint.
- `--split` : `train`, `valid` ou `test`.
- `--data-path` / `--data-format` : override du dataset et du format de données.
- `--part` : `model` ou `predictor`.
- `--pipeline` : actuellement `offline`.
- `--execution` : `eager`, `compiled`, `cuda_graph`.
- `--solver` : solveur ODE, par exemple `euler` ou `5xeuler`.
- `--steps` : nombre de pas du solveur.
- `--limit` / `--offset` : limiter ou décaler le sous-ensemble traité.
- `--selection` : `first` (ordre du dataset) ou `random` (répétable via `--selection-seed`).
- `--seed` : graine globale.
- `--dtype` : `fp32`, `fp16`, `bf16`.
- `--crop-mode` : `full` pour des fichiers complets, ou `config` pour la durée définie dans la config.
- `--memory-format`, `--num-threads`, `--num-interop-threads` : réglages de calcul.
- `--output-dir` / `--run-name` : dossier de sortie et nom du run.
- `--overwrite` : écraser un run déjà existant.
- `--save-inputs` : sauvegarder les versions clean/noisy.
- `--continue-on-error` : continuer même si un fichier échoue.
- `--score-after-run` : déclencher automatiquement le scoring après l’évaluation.
- `--score-with-distillmos` / `--score-include-stats` / `--score-include-per-file` / `--score-target` : options de scoring.
- `--local-log-dir` / `--no-local-log` : contrôle des métadonnées locales copiées dans `outputs/evaluation_logs/`.

Les artefacts de sortie sont généralement stockés sous `outputs/eval_runs/<run-name>/` et les copies locales de métadonnées sous `outputs/evaluation_logs/<run-name>/`.

### 3. Calculer les métriques à partir d’un manifest ou d’un dataset

Exemple à partir d’un run d’évaluation :

```bash
python experiments/evaluation/score_manifest.py \
  outputs/eval_runs/demo_run/manifest.json \
  --backend local \
  --include-stats \
  --include-per-file \
  --output-json outputs/evaluation_logs/demo_run/metrics.json
```

Exemple à partir d’un dataset brut :

```bash
python experiments/evaluation/score_manifest.py \
  --backend local \
  --source dataset \
  --task stftpr \
  --split test \
  --data-path '' \
  --crop-mode full \
  --include-stats \
  --include-per-file \
  --output-json outputs/evaluation_logs/dataset_scores/stftpr_test.json
```

Paramètres principaux :

- `manifest` : chemin vers un `manifest.json` produit par l’évaluation, ou omis si vous passez `--run-name`.
- `--source` : `manifest` (par défaut) ou `dataset`.
- `--run-name` : nom du run sous `outputs/eval_runs/`.
- `--limit` / `--offset` : sous-échantillonnage des fichiers à scorer.
- `--selection` / `--selection-seed` : même logique que pour l’évaluation.
- `--task`, `--split`, `--data-path`, `--data-format` : contexte du dataset.
- `--crop-mode` : `full` ou `config`.
- `--with-distillmos` : active DistillMOS si le package est installé.
- `--output-json` : chemin du JSON de sortie.
- `--score-target` : `enhanced` (par défaut) ou `noisy`.
- `--include-stats` : ajoute les statistiques globales (`mean`, `min`, `median`, `max`).
- `--include-per-file` : ajoute la liste des métriques par fichier.
- `--local-log-dir` / `--no-local-log` : contrôle des copies de logs locales.

Les métriques calculées sont généralement : SI-SDR, ESTOI, LSD, PSNR, PESQ, et éventuellement DistillMOS si demandé.

### 4. Lancer une grille d'évaluations avec métriques

Le fichier `evaluation/sweeps/configs/sweep.yaml` décrit une grille locale avec les mêmes règles `exclude` que le sweep de benchmark. Le script est lancé depuis le Mac ; chaque essai peut néanmoins utiliser `backend: modal`. Il exécute l'inférence, le scoring, puis crée un run W&B par combinaison.

Toujours vérifier la grille avant de la lancer :

```bash
.venv/bin/python experiments/evaluation/sweeps/run_eval_sweep.py \
  --sweep-yaml experiments/evaluation/sweeps/configs/sweep.yaml \
  --dry-run
```

Puis lancer la grille avec un nom stable, qui permet aussi de reprendre les essais déjà scorés :

```bash
.venv/bin/python experiments/evaluation/sweeps/run_eval_sweep.py \
  --sweep-yaml experiments/evaluation/sweeps/configs/sweep.yaml \
  --group quality-ablation-v1 \
  --resume
```

Les variantes nommées (`baseline`, `quant_int8`, `svd_50`, etc.) se définissent dans `presets`. Chaque preset peut choisir son `config_name`, son `ckpt` et une liste de `config_overrides`. Les paramètres inconnus sont refusés afin d'éviter qu'une option de compression soit seulement journalisée sans être réellement appliquée au modèle.

### 5. Commandes de smoke test rapides

```bash
python experiments/inference_local.py --config-name streamfm_se_predgen +inpath=inputs/test_clips +outpath=outputs/local_inference +ckpt=checkpoints/streamfm_se_predgen.ckpt +solver=euler +device=auto
```

```bash
python experiments/streaming/run_local.py --input inputs/test_clips/audio_43m28_10s.wav --output outputs/streaming_audio_local_stftpr.wav --json outputs/streaming_audio_local_stftpr.json
```

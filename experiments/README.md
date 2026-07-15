# Experiments

Project-specific experiments and runnable helpers built on top of the original
Stream.FM codebase.

Keep stable model/library code in `sgmse/`, the official training and inference
entrypoints at the repository root, and local exploration here. Everything under
`experiments/` is organised as a real package: infrastructure shared by several
areas lives in `core/`, and each area (streaming, benchmarks, evaluation, …) is
its own subpackage.

## Layout

```text
experiments/
├── core/                     # shared infra, no experiment logic of its own
│   ├── repo.py               # repo root + import path (no torch)
│   ├── devices.py            # device selection, sync, matmul precision
│   ├── tensors.py            # memory format + tensor helpers
│   ├── streaming_state.py    # forward_step and streaming-state helpers
│   ├── timing.py             # ms summaries
│   ├── history.py            # atomic JSON writes + history file lock
│   ├── paths.py              # checkpoint/config path resolution
│   ├── options.py            # benchmark CLI option normalization
│   └── modal_cache.py        # shared Modal cache configuration
├── baseline/
│   └── streamfm_se_baseline.py   # from-scratch SE reference loop
├── streaming/
│   ├── pipeline.py           # public streaming API (re-exports)
│   ├── stft.py               # STFT framing / compression / synthetic audio
│   ├── eager.py              # eager frame-by-frame pipelines
│   ├── cuda_graph.py         # CUDA Graph pipeline variants
│   └── run_local.py          # local streaming CLI (STFTPR)
├── benchmarks/
│   ├── streamfm_benchmark.py     # unified local/Modal CLI (entry point)
│   ├── runner.py                 # in-process benchmark driver
│   ├── model_loops.py            # eager timing loops
│   ├── cuda_graph.py             # CUDA Graph timing loops
│   ├── loading.py                # checkpoint/backbone loading
│   ├── results.py                # history summaries + W&B logging
│   ├── modal_streamfm_benchmark.py   # Modal remote entrypoints
│   ├── upload_history_to_wandb.py    # backfill W&B from saved history
│   ├── cuda_profile_range.py         # profiler range markers
│   ├── profiling/                # backbone profiling (local + Modal + Nsight)
│   ├── tensorrt/                 # TensorRT streaming + INT8 probe
│   └── sweeps/                   # grid expansion and trial runners
├── evaluation/
│   ├── streamfm_eval.py          # test-set inference CLI (entry point)
│   ├── runner.py                 # in-process inference driver
│   ├── options.py                # task defaults + option normalization
│   ├── results.py                # history summaries + W&B logging
│   ├── modal_streamfm_eval.py    # Modal remote entrypoints
│   ├── modal_defaults.py         # default data paths on Modal
│   ├── upload_eval_history_to_wandb.py
│   ├── scoring/                  # metric scoring + subset convergence
│   └── sweeps/                   # eval grid runner
├── inference/
│   ├── local.py                  # minimal local inference entry point
│   └── compress_modal.py         # checkpoint compression on Modal
├── datasets/                     # dataset provisioning/inspection on Modal
└── training/
    └── modal_train.py            # durable Modal training launcher
```

Data and artifact directories (`audio/`, `inputs/`, `outputs/`, `checkpoints/`,
`papers/`) are local and ignored by git.

## What belongs where

- `core/` holds everything shared across areas and nothing task-specific.
  `repo.py`, `streaming_state.py`, `timing.py` and `history.py` are torch-free;
  `devices.py` and `tensors.py` import torch lazily so the infra modules stay
  cheap to import.
- `baseline/` is the plain from-scratch reference the optimized paths are
  compared against.
- `streaming/` is the simulated real-time audio path. `pipeline.py` is a thin
  public facade; the work lives in `stft.py`, `eager.py` and `cuda_graph.py`.
- `benchmarks/` measures latency/throughput of the model blocks and audio
  pipelines. `streamfm_benchmark.py` is the CLI, `runner.py` orchestrates, and
  the loops live in `model_loops.py` / `cuda_graph.py`.
- `evaluation/` runs test-set inference and scoring. `streamfm_eval.py` is the
  CLI, `runner.py` drives inference, and `scoring/` computes the metrics.
- `inference/` is minimal one-shot enhancement without the full eval harness.
- `datasets/` and `training/` are the Modal-side provisioning and training
  launchers.

Move code into `sgmse/` only when it becomes reusable model or library code
rather than experiment glue.

## Read order

For streaming audio: `streaming/pipeline.py`, then `streaming/stft.py`, then
`streaming/eager.py` or `streaming/cuda_graph.py`.

For benchmarks: `benchmarks/streamfm_benchmark.py`, then `benchmarks/runner.py`,
then the specific loop you need (`model_loops.py`, `cuda_graph.py`,
`loading.py`, `results.py`).

For evaluation: `evaluation/streamfm_eval.py`, then `evaluation/runner.py`, then
`evaluation/scoring/score_manifest.py`.

## Commandes utiles (benchmark, éval et métriques)

Les trois scripts principaux sont :

- [experiments/benchmarks/streamfm_benchmark.py](benchmarks/streamfm_benchmark.py) pour mesurer un benchmark de modèle ou de pipeline audio.
- [experiments/evaluation/streamfm_eval.py](evaluation/streamfm_eval.py) pour lancer l’inférence sur la split de test.
- [experiments/evaluation/scoring/score_manifest.py](evaluation/scoring/score_manifest.py) pour calculer les métriques (SI-SDR, ESTOI, LSD, PSNR, PESQ, et éventuellement DistillMOS).

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
- `--iterations` : nombre de frames mesurées ; utiliser `--audio-duration-s` pour piloter la durée au lieu du nombre d’itérations.
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

Par défaut, `streamfm_eval.py` évalue des fichiers complets (`--crop-mode full`).
`--limit 0` évalue toute la split ; `--limit N` sélectionne un sous-ensemble
aléatoire répétable via `--selection random --selection-seed 42`.

Quand `--backend modal` est utilisé, `streamfm_eval.py` délègue en interne à
`evaluation/modal_streamfm_eval.py` ; il n'y a pas besoin d'appeler le wrapper
Modal directement.

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

Les artefacts de sortie sont stockés sous `outputs/eval_runs/<run-name>/` et les
copies locales de métadonnées sous `outputs/evaluation_logs/<run-name>/`
(`command.json`, `summary.json`, `manifest.json`, `config.yaml`).

### 3. Calculer les métriques à partir d’un manifest ou d’un dataset

Exemple à partir d’un run d’évaluation :

```bash
python experiments/evaluation/scoring/score_manifest.py \
  outputs/eval_runs/demo_run/manifest.json \
  --backend local \
  --include-stats \
  --include-per-file \
  --output-json outputs/evaluation_logs/demo_run/metrics.json
```

Exemple à partir d’un dataset brut :

```bash
python experiments/evaluation/scoring/score_manifest.py \
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

Les métriques calculées sont : SI-SDR, ESTOI, LSD, PSNR, PESQ, et éventuellement
DistillMOS si demandé. Les scores par fichier (`--include-per-file`) alimentent
l'analyse de convergence [evaluation/scoring/subset_convergence.py](evaluation/scoring/subset_convergence.py).

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

### PTQ INT8 (benchmark CPU)

Post-training INT8 est branché sur les benchmarks via `--ptq-int8` (composants séparés par des virgules) :

- `linear` — dynamic INT8 sur les `nn.Linear`
- `conv` — static INT8 sur les `nn.Conv2d` plaines (ex. depthwise/pointwise après SVD)
- `causal_conv` — static INT8 sur `CausalConv2d` (wrapper streaming)
- `all` — les trois

Contraintes : **CPU**, **fp32**, **eager**. Exemple :

```bash
python experiments/benchmarks/streamfm_benchmark.py --local --hardware cpu \
  --task stftpr --pipeline audio --execution eager --dtype fp32 \
  --ptq-int8 causal_conv --ptq-calib-steps 32 \
  --num-threads 1 --num-interop-threads 1
```

Grille de smoke :

```bash
python experiments/benchmarks/sweeps/run_benchmark_sweep_batch.py \
  --sweep-yaml experiments/benchmarks/sweeps/configs/sweep_local_cpu_ptq_int8.yaml \
  --wandb-group local-cpu-ptq-int8
```

### 5. Commandes de smoke test rapides

```bash
python experiments/inference/local.py --config-name streamfm_se_predgen +inpath=inputs/test_clips +outpath=outputs/local_inference +ckpt=checkpoints/streamfm_se_predgen.ckpt +solver=euler +device=auto
```

```bash
python experiments/streaming/run_local.py --input inputs/test_clips/audio_43m28_10s.wav --output outputs/streaming_audio_local_stftpr.wav --json outputs/streaming_audio_local_stftpr.json
```

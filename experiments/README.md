# Experiments

Everything this fork adds on top of the upstream Stream.FM codebase: the harnesses that
measure streaming inference cost, the harness that scores restoration quality, and the
compression studies built on both.

Stable model and library code stays in `sgmse/`, the upstream training and inference entry
points stay at the repository root, and all exploration lives here. `experiments/` is a real
Python package: infrastructure shared by several areas lives in `core/`, and each area is its
own subpackage.

## Layout

```text
experiments/
├── core/                     # shared infrastructure, no experiment logic of its own
│   ├── repo.py               # repo root + import path (torch-free)
│   ├── devices.py            # device selection, sync, matmul precision
│   ├── tensors.py            # memory format + real/imag channel packing
│   ├── streaming_state.py    # forward_step and streaming-state helpers
│   ├── timing.py             # latency-sample summaries (mean / percentiles)
│   ├── history.py            # atomic, lock-protected JSON history writes
│   ├── paths.py              # checkpoint/config/output path resolution
│   ├── options.py            # benchmark CLI option normalization
│   └── modal_cache.py        # shared Modal volume/cache configuration
├── streaming/                # the simulated real-time audio path
│   ├── pipeline.py           # public streaming API (thin re-export facade)
│   ├── stft.py               # STFT framing / compression / synthetic audio
│   ├── eager.py              # eager frame-by-frame pipelines
│   ├── cuda_graph.py         # CUDA Graph pipeline variants
│   ├── enhance.py            # streaming enhancement + algorithmic delay
│   └── run_local.py          # local streaming CLI
├── benchmarks/               # latency and throughput measurement
│   ├── streamfm_benchmark.py     # ENTRY POINT: local/Modal benchmark CLI
│   ├── runner.py                 # in-process benchmark driver
│   ├── model_loops.py            # eager timing loops
│   ├── cuda_graph.py             # CUDA Graph timing loops
│   ├── loading.py                # checkpoint/backbone loading
│   ├── quality.py                # quality checks alongside timing
│   ├── results.py                # history summaries + W&B logging
│   ├── modal_streamfm_benchmark.py   # Modal image, volumes, remote entry points
│   ├── cuda_profile_range.py     # profiler range markers
│   ├── profiling/                # PyTorch profiler, Nsight Systems/Compute, macOS CPU
│   ├── tensorrt/                 # TensorRT streaming engines, INT8 calibration, engine cache
│   └── sweeps/                   # grid expansion and trial runners
├── evaluation/               # test-set inference and metric scoring
│   ├── streamfm_eval.py          # ENTRY POINT: test-set inference CLI
│   ├── runner.py                 # in-process inference driver
│   ├── options.py                # task defaults + option normalization
│   ├── results.py                # history summaries + W&B logging
│   ├── modal_streamfm_eval.py    # Modal remote entry points
│   ├── modal_defaults.py         # default dataset paths on Modal
│   ├── scoring/
│   │   ├── score_manifest.py     # ENTRY POINT: metric computation
│   │   ├── modal_score_manifest.py
│   │   └── subset_convergence.py # how many test files are needed for stable metrics
│   └── sweeps/                   # eval grid runner
├── pruning/                  # structured depth pruning study
│   ├── block_influence.py            # rank residual blocks by Block Influence
│   ├── modal_block_influence.py      # the same, on Modal
│   ├── modal_ablation.py             # zero-shot quality vs. blocks removed
│   ├── modal_eval_healed.py          # quality after the healing fine-tune
│   ├── modal_prune_latency.py        # measured latency vs. blocks removed
│   └── modal_true_1resblock_latency.py
├── inference/
│   ├── local.py                  # minimal local inference entry point
│   └── compress_modal.py         # checkpoint compression on Modal
├── datasets/                 # dataset provisioning/inspection on Modal
├── training/
│   └── modal_train.py        # durable Modal training launcher
├── baseline/
│   └── streamfm_se_baseline.py   # from-scratch reference loop the optimized paths beat
└── tools/                    # one-shot maintenance scripts, run by hand
```

Data and artifact directories (`inputs/`, `outputs/`, `checkpoints/`, `papers/`) are local and
git-ignored. Measurement dumps quoted by the write-ups live in `results/`.

## What belongs where

- `core/` holds what is shared across areas and nothing task-specific. `repo.py`,
  `streaming_state.py`, `timing.py` and `history.py` are torch-free; `devices.py` and
  `tensors.py` import torch lazily, so the infrastructure modules stay cheap to import.
- `streaming/` is the simulated real-time audio path. `pipeline.py` is a thin public facade;
  the work lives in `stft.py`, `eager.py` and `cuda_graph.py`.
- `benchmarks/` measures latency and throughput of the model blocks and audio pipelines.
- `evaluation/` runs test-set inference and scoring.
- `pruning/` is the structured depth-pruning study: rank blocks, ablate, heal, re-measure.
- `inference/` is minimal one-shot enhancement without the full eval harness.
- `datasets/` and `training/` are the Modal-side provisioning and training launchers.

Move code into `sgmse/` only when it becomes reusable model or library code rather than
experiment glue.

## Read order

- **Streaming audio**: `streaming/pipeline.py` → `streaming/stft.py` → `streaming/eager.py` or
  `streaming/cuda_graph.py`.
- **Benchmarks**: `benchmarks/streamfm_benchmark.py` → `benchmarks/runner.py` → the specific
  loop (`model_loops.py`, `cuda_graph.py`, `loading.py`, `results.py`).
- **Evaluation**: `evaluation/streamfm_eval.py` → `evaluation/runner.py` →
  `evaluation/scoring/score_manifest.py`.

---

# Command reference

The three scripts below are the ones you will actually run. They share the same
model-configuration flags, so a configuration measured for speed is the same configuration
scored for quality.

- [`benchmarks/streamfm_benchmark.py`](benchmarks/streamfm_benchmark.py) — measure latency.
- [`evaluation/streamfm_eval.py`](evaluation/streamfm_eval.py) — run test-set inference.
- [`evaluation/scoring/score_manifest.py`](evaluation/scoring/score_manifest.py) — compute the
  metrics.

## 1. Run a benchmark

Local:

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

On Modal:

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

Main options:

- `--backend` / `--local` — run locally or dispatch to Modal.
- `--hardware` — `auto`, `cpu`, `mps`, `cuda` locally; `cpu`, `t4`, `l4`, `l40s`, `a100` on Modal.
- `--task` — `stftpr`, `bwe`, `derev`, `lyra`, `se`.
- `--part` — `model`, `predictor`, `flow`.
- `--pipeline` — `model_only` (model block alone) or `audio` (full audio pipeline).
- `--execution` — `auto`, `eager`, `compiled`, `cuda_graph`, `tensorrt`, `tensorrt_cuda_graph`.
  Both TensorRT modes require CUDA and `--dtype fp32`/`fp16`; `tensorrt_cuda_graph` additionally
  replays the whole solver inside a CUDA Graph.
- `--steps` — number of flow steps (NFE), comma-separated for several values.
- `--iterations` — number of measured frames; use `--audio-duration-s` to drive the run by
  duration instead.
- `--warmup` — number of warm-up frames.
- `--dtype` — `fp32`, `fp16`, `bf16`.
- `--num-threads` / `--num-interop-threads` — CPU threading.
- `--memory-format` — `contiguous` or `channels_last`.
- `--preallocate-model-buffers` — reuse model buffers where possible.
- `--save-audio` / `--audio-output-dir` / `--input-audio` — save output audio, choose the input.
- `--output-json` / `--history-json` — write results as JSON.

## 2. Run a test-set evaluation

Local:

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

On Modal:

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

By default `streamfm_eval.py` evaluates whole files (`--crop-mode full`). `--limit 0` evaluates
the entire split; `--limit N` selects a repeatable random subset via
`--selection random --selection-seed 42`.

With `--backend modal`, `streamfm_eval.py` delegates internally to
`evaluation/modal_streamfm_eval.py` — there is no need to call the Modal wrapper directly.

Main options:

- `--backend` / `--local` — local or Modal.
- `--hardware` — same logic as for benchmarks.
- `--task` — which model to load (`stftpr`, `se`, `bwe`, `derev`, `lyra`, `melflow`).
- `--config-name` — override the Hydra config name.
- `--ckpt` — checkpoint path or name.
- `--split` — `train`, `valid` or `test`.
- `--data-path` / `--data-format` — override the dataset and its layout.
- `--part` — `model` or `predictor`.
- `--pipeline` — currently `offline`.
- `--execution` — `eager`, `compiled`, `cuda_graph`.
- `--solver` — ODE solver, e.g. `euler` or `5xeuler`.
- `--steps` — number of solver steps.
- `--limit` / `--offset` — limit or shift the processed subset.
- `--selection` — `first` (dataset order) or `random` (repeatable via `--selection-seed`).
- `--seed` — global seed.
- `--dtype` — `fp32`, `fp16`, `bf16`.
- `--crop-mode` — `full` for whole files, or `config` for the duration set in the config.
- `--memory-format`, `--num-threads`, `--num-interop-threads` — compute settings.
- `--output-dir` / `--run-name` — output directory and run name.
- `--overwrite` — overwrite an existing run.
- `--save-inputs` — also save the clean/noisy versions.
- `--continue-on-error` — keep going when a file fails.
- `--score-after-run` — trigger scoring automatically once inference finishes.
- `--score-with-distillmos` / `--score-include-stats` / `--score-include-per-file` /
  `--score-target` — scoring options.
- `--local-log-dir` / `--no-local-log` — control the local metadata copies.

Output artifacts land under `outputs/eval_runs/<run-name>/`, and local metadata copies under
`outputs/evaluation_logs/<run-name>/` (`command.json`, `summary.json`, `manifest.json`,
`config.yaml`).

## 3. Compute the metrics

From an evaluation run:

```bash
python experiments/evaluation/scoring/score_manifest.py \
  outputs/eval_runs/demo_run/manifest.json \
  --backend local \
  --include-stats \
  --include-per-file \
  --output-json outputs/evaluation_logs/demo_run/metrics.json
```

From a raw dataset (to score the degraded baseline itself):

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

Main options:

- `manifest` — path to a `manifest.json` produced by an evaluation run, or omitted if you pass
  `--run-name`.
- `--source` — `manifest` (default) or `dataset`.
- `--run-name` — run name under `outputs/eval_runs/`.
- `--limit` / `--offset` — subsample the files to score.
- `--selection` / `--selection-seed` — same logic as for evaluation.
- `--task`, `--split`, `--data-path`, `--data-format` — dataset context.
- `--crop-mode` — `full` or `config`.
- `--with-distillmos` — enable DistillMOS if the package is installed.
- `--output-json` — output JSON path.
- `--score-target` — `enhanced` (default) or `noisy`.
- `--include-stats` — add global statistics (`mean`, `min`, `median`, `max`).
- `--include-per-file` — add the per-file metric list.
- `--local-log-dir` / `--no-local-log` — control the local log copies.

Computed metrics: SI-SDR, ESTOI, LSD, PSNR, PESQ, and optionally DistillMOS. The per-file
scores (`--include-per-file`) feed the convergence analysis in
[evaluation/scoring/subset_convergence.py](evaluation/scoring/subset_convergence.py).

## 4. Run a grid of evaluations with metrics

`evaluation/sweeps/configs/sweep.yaml` describes a grid with the same `exclude` rules as the
benchmark sweep. The script is launched from the local machine; each trial can nevertheless use
`backend: modal`. It runs inference, then scoring, then creates one W&B run per combination.

Always inspect the grid before launching it:

```bash
python experiments/evaluation/sweeps/run_eval_sweep.py \
  --sweep-yaml experiments/evaluation/sweeps/configs/sweep.yaml \
  --dry-run
```

Then launch it with a stable group name, which also lets you resume trials that were already
scored:

```bash
python experiments/evaluation/sweeps/run_eval_sweep.py \
  --sweep-yaml experiments/evaluation/sweeps/configs/sweep.yaml \
  --group quality-ablation-v1 \
  --resume
```

Named variants (`baseline`, `quant_int8`, `svd_50`, …) are defined under `presets`. Each preset
picks its own `config_name`, `ckpt` and list of `config_overrides`. Unknown parameters are
rejected, so a compression option can never be merely logged without actually being applied to
the model.

## 5. INT8 post-training quantization

PTQ INT8 is wired into the benchmarks through `--ptq-int8`. On CPU (native PyTorch
quantization, components comma-separated):

- `linear` — dynamic INT8 on `nn.Linear`
- `conv` — static INT8 on plain `nn.Conv2d` (e.g. depthwise/pointwise after SVD)
- `causal_conv` — static INT8 on `CausalConv2d` (the streaming wrapper)
- `all` — all three

Constraints: **CPU**, **fp32**, **eager**.

```bash
python experiments/benchmarks/streamfm_benchmark.py --local --hardware cpu \
  --task stftpr --pipeline audio --execution eager --dtype fp32 \
  --ptq-int8 causal_conv --ptq-calib-steps 32 \
  --num-threads 1 --num-interop-threads 1
```

On GPU, INT8 goes through TensorRT with ModelOpt calibration: `--ptq-int8 tensorrt` together
with `--execution tensorrt` or `tensorrt_cuda_graph`, and `--dtype fp32` (the engine's inputs
and outputs stay fp32).

```bash
python experiments/benchmarks/streamfm_benchmark.py --hardware l4 \
  --task stftpr --pipeline model_only --execution tensorrt_cuda_graph \
  --dtype fp32 --ptq-int8 tensorrt --ptq-calib-steps 32
```

Smoke grid:

```bash
python experiments/benchmarks/sweeps/run_benchmark_sweep_batch.py \
  --sweep-yaml experiments/benchmarks/sweeps/configs/sweep_local_cpu_ptq_int8.yaml \
  --wandb-group local-cpu-ptq-int8
```

## 6. Quick smoke tests

No audio ships with the repository. `--pipeline audio` benchmarks generate synthetic audio when
`--input-audio` points at nothing — equivalent for timing, since the model spends the same
compute on every frame regardless of content:

```bash
python experiments/benchmarks/streamfm_benchmark.py --local --hardware auto --task stftpr --pipeline audio --execution eager --steps 1 --iterations 10 --warmup 2
```

The two scripts below need real speech: drop your own 16 kHz mono WAVs in `inputs/test_clips/`
(the directory is git-ignored) or point them elsewhere.

```bash
python experiments/inference/local.py --config-name streamfm_se_predgen +inpath=inputs/test_clips +outpath=outputs/local_inference +ckpt=checkpoints/streamfm_se_predgen.ckpt +solver=euler +device=auto
```

```bash
python experiments/streaming/run_local.py --input inputs/test_clips/benchmark_input_10s.wav --output outputs/streaming_audio_local_stftpr.wav --json outputs/streaming_audio_local_stftpr.json
```

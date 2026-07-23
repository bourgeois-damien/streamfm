# Stream.FM — inference-cost research fork

This is a research fork of [`sp-uhh/streamfm`](https://github.com/sp-uhh/streamfm), the
official implementation of *Real-Time Streamable Generative Speech Restoration with Flow
Matching* (Stream.FM) and *Real-Time Streaming Mel Vocoding with Generative Flow Matching*
(MelFlow).

The upstream repository provides the model, the training recipes and offline inference. This
fork adds the missing piece for studying **what streaming inference actually costs and how far
that cost can be pushed down**: a measurement harness that times the streaming pipeline under
controlled conditions, an evaluation harness that scores restoration quality on the test set,
and a set of model-compression paths whose speed/quality trade-off can be measured with both.

The original upstream README — installation, training recipes, checkpoints, citations — is
preserved verbatim in [UPSTREAM.md](UPSTREAM.md). **Read it first**: everything below assumes
the upstream setup is already working.

---

## What this fork adds

Stream.FM restores speech frame by frame: an ODE solver runs a causal U-Net (`CausalNCSNpp`)
once per STFT frame, so the per-frame wall time decides whether the model runs in real time.
The question this fork is built to answer is where that per-frame time goes and which
interventions remove it without destroying quality.

Three kinds of work were added:

**1. A benchmark harness** that measures per-frame latency of the streaming pipeline under a
controlled configuration — device, precision, execution mode, memory layout, thread counts,
number of solver steps (NFE) — locally or on rented GPUs.

**2. An evaluation harness** that runs the same model configurations over the test set and
scores them (PESQ, ESTOI, SI-SDR, LSD, PSNR, optionally DistillMOS), so any speed intervention
can be priced in quality terms rather than assumed harmless.

**3. Model-compression paths**, each usable from both harnesses:

| Path | Where | What it does |
|---|---|---|
| Execution modes | `--execution` | eager, `torch.compile`, CUDA Graphs, TensorRT, TensorRT + CUDA Graph |
| Reduced precision | `--dtype`, `--matmul-precision` | fp32 / fp16 / bf16, TF32 matmul control |
| PTQ INT8 (CPU) | `sgmse/util/ptq_int8.py` | dynamic INT8 on `Linear`, static INT8 on `Conv2d` / `CausalConv2d` |
| PTQ INT8 (GPU) | `experiments/benchmarks/tensorrt/` | TensorRT INT8 with ModelOpt calibration |
| Decoupled-SVD | `sgmse/util/model_compression.py` | low-rank factorisation of convolutions, baked into a reusable checkpoint |
| Structured depth pruning | `experiments/pruning/` | drop whole residual blocks by Block Influence, then heal by fine-tuning |

Alongside these, the fork adds a durable Modal training launcher, dataset provisioning on
Modal, GPU profiling (PyTorch profiler, Nsight Systems / Compute), and a W&B-backed history of
every run.

---

## Repository layout

```text
.
├── sgmse/                  # model and library code (upstream, plus fork additions)
│   ├── backbones/          #   streaming_unet.py: CausalNCSNpp, streaming API, pruning hooks
│   └── util/               #   + model_compression.py (SVD), ptq_int8.py (INT8 PTQ)
├── config/                 # Hydra configs: model, task, LRK solvers, ablation studies
├── experiments/            # everything this fork adds on top — see experiments/README.md
├── tests/                  # unit tests for the harness (options, caching, selection, PTQ)
├── results/                # measurement dumps referenced by the write-ups
├── train.py                # upstream training entry point
├── inference.py            # upstream offline inference entry point
├── fit_rk_scheme.py        # upstream learned-Runge-Kutta fitting
└── compress_checkpoint.py  # bake a decoupled-SVD compression into a checkpoint
```

`experiments/` is organised as a real package: shared infrastructure in `core/`, and one
subpackage per area (`benchmarks/`, `evaluation/`, `streaming/`, `pruning/`, `training/`,
`datasets/`, `inference/`). See [experiments/README.md](experiments/README.md) for the detailed
architecture and the full command reference.

Weights, datasets, run outputs and W&B logs are local and git-ignored.

---

## How the harness works

The three entry points chain into one another:

```text
                  ┌──────────────────────────────┐
   speed ────────▶│ benchmarks/streamfm_benchmark│──▶ latency JSON + W&B
                  └──────────────────────────────┘
                  ┌──────────────────────────────┐
 quality ────────▶│ evaluation/streamfm_eval     │──▶ enhanced audio + manifest.json
                  └──────────────┬───────────────┘
                                 ▼
                  ┌──────────────────────────────┐
                  │ evaluation/scoring/          │──▶ metrics JSON + W&B
                  │        score_manifest        │
                  └──────────────────────────────┘
```

Both harnesses take the same model-configuration flags, so a configuration measured for speed
is the same configuration scored for quality. Both accept `--backend local` or
`--backend modal`; the Modal path ships the repo to a container with the datasets and
checkpoints already mounted, so no code change is needed to move a run to a GPU.

Supported tasks: `stftpr` (phase retrieval), `se` (speech enhancement), `bwe` (bandwidth
extension), `derev` (dereverberation), `lyra` (codec artefact removal), and `melflow`
(Mel vocoding, evaluation only).

### 1. Measure latency

```bash
python experiments/benchmarks/streamfm_benchmark.py \
  --backend modal --hardware l4 \
  --task derev --pipeline audio --execution cuda_graph \
  --dtype fp16 --steps 1 --iterations 100 --warmup 10
```

### 2. Run test-set inference

```bash
python experiments/evaluation/streamfm_eval.py \
  --backend modal --hardware L4 \
  --task derev --ckpt checkpoints/streamfm_derev.ckpt \
  --split test --limit 200 --selection random --selection-seed 42 \
  --run-name derev-fp16-euler1 --score-after-run
```

This writes enhanced audio and a `manifest.json` under `outputs/eval_runs/<run-name>/`.
`--score-after-run` chains straight into scoring.

### 3. Score the metrics

```bash
python experiments/evaluation/scoring/score_manifest.py \
  outputs/eval_runs/derev-fp16-euler1/manifest.json \
  --backend local --include-stats --include-per-file
```

Grids of configurations are run with the sweep runners
(`experiments/benchmarks/sweeps/`, `experiments/evaluation/sweeps/`), which expand a YAML grid,
skip excluded combinations, and log one W&B run per trial.

Full option-by-option reference: [experiments/README.md](experiments/README.md).

---

## Setup

Follow the upstream [installation](UPSTREAM.md#installation) and
[pretrained checkpoints](UPSTREAM.md#pretrained-checkpoints) sections. Then, for the remote
runs, one-time Modal setup:

```bash
python -m pip install "modal>=1.0,<2"
modal setup
modal secret create wandb WANDB_API_KEY=YOUR_WANDB_API_KEY
```

Datasets are provisioned into the `streamfm-cache` Modal volume with
`experiments/datasets/modal_dataset_setup.py`; training checkpoints land in `streamfm-runs`.

Smoke test that the local install works. No audio ships with the repository, so the benchmark
falls back to generated audio — equivalent for timing, since the model spends the same compute
on every frame regardless of content:

```bash
python experiments/benchmarks/streamfm_benchmark.py \
  --local --hardware auto --task stftpr --pipeline audio \
  --execution eager --steps 1 --iterations 10 --warmup 2
```

To run the streaming pipeline on real speech, point it at your own 16 kHz mono WAV:

```bash
python experiments/streaming/run_local.py \
  --input path/to/your_clip.wav \
  --output outputs/streaming_local.wav
```

---

## Credits and license

All model architecture, training recipes and pretrained checkpoints are the work of the
Signal Processing group at Universität Hamburg (Welker, Lay, Hillemann, Peer, Gerkmann). If
you use this code, cite their papers — the citation block is at the end of
[UPSTREAM.md](UPSTREAM.md).

This fork keeps the upstream license, see [LICENSE](LICENSE).

# Stream.FM fork

Fork of https://github.com/sp-uhh/streamfm for my MSc project.

This project looks at the practical inference efficiency of Stream.FM under streaming constraints. The main goal is to better understand and improve the quality / latency trade-off for frame-by-frame inference.

## Main work

- reconstruct / extend the streaming evaluation pipeline
- measure the effect of runtime choices (FP16, torch.compile, CUDA Graphs, TensorRT, etc)
- look at NFE and hardware effects on the quality / latency point
- try model-side reductions, mainly pruning + fine-tuning on dereverberation
- check quality on a few restoration tasks

## Useful scripts

- `experiments/benchmarks/streamfm_benchmark.py` : latency / throughput
- `experiments/evaluation/streamfm_eval.py` : inference on test set
- `experiments/evaluation/scoring/score_manifest.py` : metrics (PESQ, ESTOI, ...)
- `experiments/streaming/run_local.py` : streaming on a wav
- `compress_checkpoint.py` : SVD compression

Tasks: `stftpr`, `se`, `bwe`, `derev`, `lyra` (+ `melflow` for eval)

## Modal setup

```bash
python -m pip install "modal>=1.0,<2"
modal setup
modal secret create wandb WANDB_API_KEY=YOUR_WANDB_API_KEY
```

Datasets / cache: `experiments/datasets/modal_dataset_setup.py`

## Benchmark

```bash
# local
python experiments/benchmarks/streamfm_benchmark.py \
  --local --hardware auto --task stftpr --pipeline audio \
  --execution eager --steps 1 --iterations 10 --warmup 2

# Modal
python experiments/benchmarks/streamfm_benchmark.py \
  --backend modal --hardware l4 \
  --task derev --pipeline audio --execution cuda_graph \
  --dtype fp16 --steps 1 --iterations 100 --warmup 10
```

## Eval + scoring

```bash
python experiments/evaluation/streamfm_eval.py \
  --backend modal --hardware L4 \
  --task derev --ckpt checkpoints/streamfm_derev.ckpt \
  --split test --limit 200 --selection random --selection-seed 42 \
  --run-name derev-fp16-euler1 --score-after-run

python experiments/evaluation/scoring/score_manifest.py \
  outputs/eval_runs/derev-fp16-euler1/manifest.json \
  --backend local --include-stats --include-per-file
```

## Streaming local

```bash
python experiments/streaming/run_local.py \
  --input path/to/clip.wav \
  --output outputs/streaming_local.wav
```

## Folders

- `sgmse/` : model
- `experiments/` : benches, eval, streaming, pruning, modal, etc
- `config/` : hydra configs
- `results/` : outputs

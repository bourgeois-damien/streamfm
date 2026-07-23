# Derev depth prune + heal (notes)

Drop decoder resblocks, then short fine-tune. Target: L4, streaming, NFE=1.
Teacher: `checkpoints/streamfm_derev.ckpt` - CausalNCSNpp nf=128 - 25 resblocks - ~27.9M params.

## What we cut

BI = 1 - cos(x_in, x_out) on the test set (`block_influence.py`).
Worst offenders are all on the up path:

| | block | BI |
|---|---|---|
| 1 | up lvl2_rnb1 | 0.137 |
| 2 | up lvl1_rnb1 | 0.173 |
| 3 | up lvl0_rnb1 | 0.187 |

Took **k=3** (one per up level) -> `StreamingIdentity`. Params 27.9M -> 24.9M (-10.7%).

Latency (own campaign - don't mix with STFTPR numbers): full audio CUDA graph, fp16, L4, NFE=1, 500 iters / 100 warmup:

| k | mean ms/frame | vs k=0 |
|---|---|---|
| 0 | 1.138 | - |
| 3 | 1.038 | -8.8% |

## Zero-shot is toast

Same teacher, identities, no train. 50 files, NFE=1:

| k | PESQ | ESTOI | SI-SDR | DistillMOS |
|---|---|---|---|---|
| 0 | 1.61 | 0.73 | -14.7 | 3.29 |
| 3 | 1.05 | 0.43 | -21.4 | 1.77 |

BI ranks what to cut. It does **not** mean you can cut without healing.

## Heal

Config `config/study_prune_streamfm_derev.yaml`: load teacher -> prune -> FT.
LR 5e-5 flat, 25k steps, batch 12, L40S.

Most of the PESQ climb is done by ~3k steps, then flat. FM loss doesn't track PESQ
(r≈0 on the 5 late ckpts) - pick the ckpt on metrics, not loss.

Picked **step 19500** on valid (200 files, seed 42, NFE=1): best PESQ/ESTOI among
17500 / 19500 / 20000 / 22500 / 25000.

## Test (after pick)

50 files, seed 42, same setup for teacher + healed:

**NFE=1**

| | PESQ | ESTOI | SI-SDR | DistillMOS |
|---|---|---|---|---|
| teacher | 1.608 | 0.727 | -14.71 | 3.294 |
| k=3 raw | 1.047 | 0.434 | -21.44 | 1.768 |
| k=3 healed | 1.620 | 0.702 | -14.34 | 3.246 |

**NFE=5** - teacher 1.996 vs healed 1.991. Gap is noise.

So: -10.7% params, -8.8% latency, quality basically back. Don't say "better than teacher".

## Don't screw up the write-up

- 50-file test -> no claiming tiny wins
- don't merge latency campaigns
- train-loop PESQ (~2.6) is not NFE=1 test PESQ (~1.6)
- no control FT on the unpruned teacher (limitation)
- smaller RIR pool than the paper (dead links)
- W&B double-run mess ~14k-20k - don't plot that stretch as one curve

Raw dumps: `results/pruning/*.json`. Selected ckpt: `derev-prune-k3-ft` step 19500 on the runs volume.

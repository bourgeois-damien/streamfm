# Structured depth pruning + healing fine-tune (derev task)

Working note for the write-up. Every number in this document is measured; the
uncertainties and the claims to avoid are flagged at the end.

---

## 1. Goal and positioning

Reduce the inference cost of the streaming Stream.FM dereverberation model,
beyond the execution-level optimisations already in place (CUDA graph,
`torch.compile`, buffer preallocation, `channels_last`), which had already cut
compute time roughly in half without touching the model.

The lever used here is **architectural**: remove whole residual blocks, then
repair the quality loss with a short fine-tune. This was chosen over a full
retrain for time-budget reasons.

Deployment target: L4 GPU, streaming pipeline, **NFE=1**.

## 2. Starting model

| | |
|---|---|
| Backbone | `CausalNCSNpp` (`sgmse/backbones/streaming_unet.py`) |
| `nf` | 128 |
| `ch_mult` | (1, 2, 2, 2) |
| `num_res_blocks` | 2 |
| `input_freqs` | 256 |
| Residual blocks | 25 |
| Backbone parameters | **27,895,944** |
| Teacher checkpoint | `checkpoints/streamfm_derev.ckpt` |
| Dataset | EARS-Reverb_v2_16k |

## 3. Pruning strategy

### 3.1 Block selection criterion: Block Influence

Each residual block is scored by its **Block Influence** (BI), defined as

```
BI = 1 - cos(x_in, x_out)
```

that is, the angle between the block's input and its output. A BI close to zero
means the block barely transforms its input: it already behaves almost like an
identity, so replacing it with one costs little.

Measured on the EARS-Reverb test set, code in
`experiments/pruning/block_influence.py`, full results in
`results/pruning/block_influence_derev.json`.

### 3.2 What the ranking reveals

The six least influential blocks are **all on the upward path** (decoder), with a
clear gap (BI < 0.20) before the first block of the downward path. That is an
interesting result in itself for the write-up: in this causal U-Net, the
redundancy is concentrated in the decoder.

Ranking of the blocks considered for removal:

| Rank | Block | BI | channels | frequencies |
|---|---|---|---|---|
| 1 | `up_modules.lvl2_rnb1` | 0.137 | 256 | 64 |
| 2 | `up_modules.lvl1_rnb1` | 0.173 | 256 | 128 |
| 3 | `up_modules.lvl0_rnb1` | 0.187 | 128 | 256 |
| 4 | `up_modules.lvl1_rnb0` | 0.191 | 256 | 128 |
| 5 | `up_modules.lvl2_rnb0` | 0.196 | 256 | 64 |
| 6 | `up_modules.lvl3_rnb1` | 0.199 | 256 | 32 |

### 3.3 Choosing k = 3

The **three** lowest-BI blocks are removed. A secondary argument supports the
choice: those three blocks fall **one per upward level** (lvl2, lvl1, lvl0), so
no level loses all of its residual capacity.

Mechanism: substituting the block with a `StreamingIdentity` module
(`sgmse.backbones.streaming_unet.prune_resblocks_`). The pruning is therefore
*structured along depth* — whole blocks are removed, not individual weights —
which guarantees a real latency gain without depending on hardware sparsity
support.

| | Backbone parameters | Δ |
|---|---|---|
| k = 0 | 27,895,944 | — |
| k = 3 | 24,901,896 | **−10.73 %** |

### 3.4 Latency gain

Dedicated campaign: full streaming audio pipeline, whole-pipeline CUDA graph,
fp16, `channels_last`, 500 iterations after 100 warm-up ones, **L4** GPU, NFE=1.
Results in `results/pruning/prune_latency_derev.json`.

| k | mean latency / frame | p50 | p90 | fraction of the 16 ms budget | Δ vs k=0 |
|---|---|---|---|---|---|
| 0 | 1.1384 ms | 1.1365 | 1.1430 | 7.12 % | — |
| 1 | 1.1028 ms | 1.0999 | 1.1080 | 6.89 % | −3.1 % |
| 2 | 1.0730 ms | 1.0686 | 1.0788 | 6.71 % | −5.7 % |
| **3** | **1.0378 ms** | 1.0358 | 1.0442 | 6.49 % | **−8.8 %** |

The latency gain (−8.8 %) is smaller than the parameter gain (−10.7 %), which is
expected: the removed blocks sit on high-frequency-resolution maps, but the
pipeline keeps its fixed costs (STFT, iSTFT, solver).

> **Do not do this:** the 1.1384 → 1.0378 ms pair comes from its own measurement
> campaign. It must **never** be put in the same table as the 1.097 ms figure
> from the STFTPR campaign, which was measured separately.

## 4. Why healing was indispensable

*Zero-shot* ablation: load the full teacher, replace the blocks with identities,
and evaluate **without any retraining**. 50 test files, NFE=1, offline pipeline,
eager execution. Results in `results/pruning/prune_ablation_derev.json`.

| k | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ | DistillMOS ↑ |
|---|---|---|---|---|
| 0 | 1.6082 | 0.7270 | −14.71 | 3.2939 |
| 1 | 1.4152 | 0.6626 | −16.99 | 2.9884 |
| 2 | 1.2241 | 0.5686 | −17.85 | 2.6894 |
| 3 | 1.0473 | 0.4339 | −21.44 | 1.7683 |

The degradation is **catastrophic and monotone**: at k=3 the model is essentially
destroyed (PESQ 1.05, DistillMOS 1.77). A low BI therefore indicates that a block
is *locally* close to the identity, but **does not predict** that the network
will survive its removal — errors compound along depth. That is the central
methodological point of this part: BI is a good *ranking* criterion, not a
certificate of harmlessness.

## 5. The healing fine-tune

Configuration: `config/study_prune_streamfm_derev.yaml`.

Scheme: build the full model → load the teacher strictly → prune in place →
fine-tune. The surviving weights therefore start exactly from the teacher's
weights (warm start), not from a random initialisation.

| Setting | Value | Rationale |
|---|---|---|
| Initialisation | `checkpoints/streamfm_derev.ckpt` | warm start from the teacher |
| Learning rate | 5e-5, constant | short repair, not a retrain |
| Scheduler | none | same |
| Gradient clipping | none | |
| `max_steps` | 25,000 | |
| Effective batch | 12 | same as the original protocol |
| Validation | every 1,000 steps, 50 batches | |
| Checkpointing | every 500 steps | for checkpoint selection |

Final run on **1× L40S**, 12 dataloader workers, about 0.87 it/s measured
end to end (throughput is dataloader-bound, not GPU-bound:
`it/s ≈ num_workers / 15`).

## 6. How training evolved

### 6.1 The quality metrics

Estimator internal to the training loop: 20 validation files.

| Step | PESQ | Reading |
|---|---|---|
| 0 | 1.533 | pruned model, unhealed |
| 999 | 2.394 | **most of the repair is already done** |
| 2,999 | 2.606 | |
| mean 2,999 – 7,999 | 2.546 | |
| mean 8,999 – 13,999 | 2.681 | last real step up (≈ 2.5 σ) |
| mean 14,999 – 20,999 | 2.642 | no measurable gain left |

ESTOI and SI-SDR follow the same shape. **Healing converges very fast**: ~90 % of
the recovery before step 3,000, a last step up around 9,000, then a plateau. The
last ~12,000 steps contributed nothing measurable.

### 6.2 The losses do not track the metrics

- `valid_loss`: stays between 369 and 396 over the whole run, with a slight
  **upward** drift (start ≈ 374, end ≈ 385).
- `train_loss`: falls from 338 to ≈ 290 up to step 8,000, then flattens.

This is not an anomaly. The flow-matching loss is **blind to perceptual
metrics**: `sgmse/model.py:673` only computes `0.5·||v_target − v_pred||²`,
averaged over all noise levels *t* along the path. It measures velocity-field
reconstruction, not perceived quality. And at NFE=1 deployment only uses a single
slice of that path, while the loss averages all of them: it measures something
that is not being run.

## 7. Checkpoint selection

### 7.1 Protocol

Five saved candidates (steps 17,500, 19,500, 20,000, 22,500, 25,000), evaluated
on the **validation split**:

- 200 files (not the 20 of the training loop), drawn with `selection_seed=42` —
  the same draw for all five candidates;
- `seed=42`: the flow-matching noise is **identical** for all of them, so any
  observed gap is a model difference and nothing else;
- NFE=1, offline pipeline, eager, fp32 — the deployment configuration.

### 7.2 Results

| Step | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ | LSD ↓ |
|---|---|---|---|---|
| 17,500 | 1.9898 | 0.7551 | −9.68 | 15.48 |
| **19,500** | **2.0539** | **0.7609** | −8.18 | 15.42 |
| 20,000 | 2.0455 | 0.7607 | −8.61 | 15.56 |
| 22,500 | 2.0197 | 0.7553 | **−8.07** | 15.94 |
| 25,000 | 2.0242 | 0.7571 | −8.29 | **15.16** |

**Selected: step 19,500** — first on PESQ and on ESTOI, never last.

The selection **confirms the plateau** seen on the curves: between 19,500 and
25,000 the PESQ gap is 0.034, i.e. noise. Only 17,500 clearly falls behind.

### 7.3 Selecting on a metric, not on the loss

Direct comparison, on those same five checkpoints:

| Step | `valid_loss` ↓ | PESQ (200 files) ↑ |
|---|---|---|
| 17,500 | 395.5 | 1.9898 |
| 19,500 | 393.4 | **2.0539** |
| 20,000 | 393.4 | 2.0455 |
| 22,500 | 377.4 | 2.0197 |
| 25,000 | **374.7** | 2.0242 |

**Correlation r = +0.07** — that is, none, and on top of that with the sign
opposite to intuition (a lower loss goes very slightly with a lower PESQ).
Selecting on the loss would have picked step 25,000; selecting on quality picks
19,500.

The rule applied is therefore: **select on the metric being maximised, measured
in the configuration that will be deployed.** The loss stays useful for
diagnosing training (divergence, gradient explosion), not for choosing a
checkpoint.

Safeguards against overfitting the selection itself: 200 files rather than 20,
fixed noise, and **the test split played no role in the choice**.

## 8. Final result on the test set

Single measurement, made after the selection. Standard test set
`EARS-Reverb_v2_16k/test.csv`, 50 files drawn with `selection_seed=42`,
`seed=42`, `crop_mode=full`, offline pipeline, eager, fp32, DistillMOS enabled.
The teacher was **re-measured in the same campaign** rather than quoted from the
ablation.

> **Reproducibility control:** the teacher at NFE=1 lands exactly on the ablation
> campaign's values (1.6082 / 0.7270 / −14.714 / 15.885 / 3.2939), figure for
> figure. The two campaigns are therefore consistent.

### NFE = 1 (deployment configuration)

| | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ | LSD ↓ | DistillMOS ↑ |
|---|---|---|---|---|---|
| k=0, teacher | 1.6082 | 0.7270 | −14.71 | 15.89 | 3.2939 |
| k=3, **unhealed** | 1.0473 | 0.4339 | −21.44 | — | 1.7683 |
| k=3, **healed** | **1.6195** | 0.7023 | **−14.34** | **15.70** | 3.2462 |

### NFE = 5

| | PESQ ↑ | ESTOI ↑ | SI-SDR ↑ | LSD ↓ | DistillMOS ↑ |
|---|---|---|---|---|---|
| k=0, teacher | **1.9964** | **0.7890** | −15.25 | **12.61** | **3.6987** |
| k=3, healed | 1.9910 | 0.7728 | **−14.60** | 13.01 | 3.6170 |

### Summary

| | Teacher PESQ | Healed PESQ | Gap |
|---|---|---|---|
| NFE = 1 | 1.6082 | 1.6195 | +0.011 |
| NFE = 5 | 1.9964 | 1.9910 | −0.005 |

The gap is on the order of a hundredth of a PESQ point, **in one direction at
NFE=1 and in the other at NFE=5** — that is, null within uncertainty. Healing did
not over-specialise the model on NFE=1, and cost it nothing at NFE=5.

## 9. Defensible conclusion

> Structured depth pruning of three decoder residual blocks, guided by Block
> Influence, shrinks the backbone by 10.7 % of its parameters and per-frame
> latency by 8.8 % on L4. Applied on its own, it destroys the model
> (PESQ 1.61 → 1.05). A 25,000-step healing fine-tune starting from the teacher's
> weights fully restores quality, at NFE=1 as well as at NFE=5, with the gaps to
> the full model staying below 0.02 PESQ in both regimes.

A second, methodological result: healing converges within fewer than 3,000 steps
for the most part, and the flow-matching loss is uncorrelated with the perceptual
metrics (r = +0.07), which forces metric-based checkpoint selection.

---

## 10. Caveats — to respect in the write-up

1. **50 test files.** Do not claim any of the small gaps, in either direction.
   Write "quality restored to the level of the full model", and **never** "better
   than the full model".
2. **Say "final checkpoint" or "checkpoint selected on validation"**, never "best
   checkpoint".
3. **Do not merge the latency campaigns.** The 1.1384 → 1.0378 ms pair is
   self-contained; it does not join the 1.097 ms figure from the STFTPR campaign.
4. **The training-loop PESQ (≈ 2.6) is not on the same scale as the test one
   (≈ 1.62).** The training loop calls `enhance()` with the config's default NFE,
   not NFE=1. Only the *relative movements* of that curve are interpretable;
   never quote its absolute value next to the final table's figures.
5. **Unresolved confound:** the healed model received 25,000 gradient steps that
   the teacher did not. A clean control (unpruned teacher, fine-tuned on the same
   data for the same number of steps) **was not run**. It therefore cannot be
   ruled out that part of the recovery comes from the extra training rather than
   from healing alone. To be mentioned as a limitation.
6. **Reduced training RIR pool:** 1,286 RIRs (ARNI sampled from 11,130) against
   132,037 in the original paper, because of dead links in the source datasets.
   This affects the whole project's training, not just this part.
7. **Gap in the W&B curve between steps ≈ 14,400 and 20,000:** over that
   interval, two independent runs wrote into the same W&B run (a double-launch
   incident). Step ordering is clean up to five regressions, which suggests only
   one of the two actually logged, but that is not guaranteed. **That portion of
   the curve must not be presented as a single training trajectory.** The final
   metrics are unaffected: they come from checkpoints evaluated directly, not
   from the curve.

## 11. Reference files

| Content | Path |
|---|---|
| Block Influence ranking | `results/pruning/block_influence_derev.json` |
| Zero-shot ablation k=0..3 | `results/pruning/prune_ablation_derev.json` |
| Latency per k | `results/pruning/prune_latency_derev.json` |
| Fine-tune config | `config/study_prune_streamfm_derev.yaml` |
| BI computation | `experiments/pruning/block_influence.py` |
| In-place pruning | `sgmse/backbones/streaming_unet.py` (`prune_resblocks_`) |
| Evaluation of healed models | `experiments/pruning/modal_eval_healed.py` |
| Selected checkpoint | `streamfm-runs` volume, `training/derev-prune-k3-ft/checkpoints/sel-step19500.ckpt` |

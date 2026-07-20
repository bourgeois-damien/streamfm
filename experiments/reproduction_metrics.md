# Metric check vs paper

200 random test files, seed 42. Modal L4, Euler, fp32 (`--matmul-precision high`).
Paper numbers on the full set, so absolutes won't match exactly - look at Δ (enhanced - degraded).

fp16 ≈ fp32 here. LSD skipped (not comparable to the paper setup).

Cells are local / paper.

## STFTPR (zero-phase base - Euler1 / Euler5)

| | base | E1 | E5 | Δ E5-base |
|---|---|---|---|---|
| PESQ | 1.317 / 1.310 | 1.586 / 1.580 | 4.236 / 4.240 | +2.919 / +2.930 |
| ESTOI | 0.679 / 0.680 | 0.647 / 0.580 | 0.973 / 0.970 | +0.294 / +0.290 |
| SI-SDR | -34.24 / -34.60 | 1.46 / 1.70 | -1.90 / -1.70 | +32.34 / +32.90 |

## MelFlow (M†+zero-phase - E1 / E5)

| | base | E1 | E5 | Δ E5-base |
|---|---|---|---|---|
| PESQ | 1.238 / 1.280 | 1.349 / 1.350 | 4.071 / 4.100 | +2.833 / +2.820 |
| ESTOI | 0.633 / 0.630 | 0.499 / 0.360 | 0.960 / 0.960 | +0.327 / +0.330 |
| SI-SDR | -36.34 / -38.90 | -5.63 / -5.60 | -10.42 / -10.10 | +25.92 / +28.80 |

## SE (noisy - E1 / E4)

| | base | E1 | E4 | Δ E4-base |
|---|---|---|---|---|
| PESQ | 1.229 / 1.240 | 2.162 / 2.180 | 2.093 / 2.090 | +0.864 / +0.850 |
| ESTOI | 0.639 / 0.640 | 0.851 / 0.840 | 0.841 / 0.830 | +0.202 / +0.190 |
| SI-SDR | 4.93 / 5.36 | 13.80 / 15.20 | 13.40 / 14.30 | +8.47 / +8.94 |

## BWE (bandlimited - E1 / E5)

| | base | E1 | E5 | Δ E5-base |
|---|---|---|---|---|
| PESQ | 3.480 / 3.510 | 3.183 / 3.220 | 3.390 / 3.370 | -0.090 / -0.140 |
| ESTOI | 0.884 / 0.840 | 0.933 / 0.920 | 0.942 / 0.940 | +0.058 / +0.100 |
| SI-SDR | 16.05 / 15.90 | 16.67 / 16.80 | 16.38 / 16.50 | +0.33 / +0.60 |

## Derev (reverb - E1 / E5)

| | base | E1 | E5 | Δ E5-base |
|---|---|---|---|---|
| PESQ | 1.320 / 1.320 | 1.667 / 1.630 | 2.052 / 2.010 | +0.732 / +0.690 |
| ESTOI | 0.590 / 0.580 | 0.740 / 0.730 | 0.803 / 0.790 | +0.213 / +0.210 |
| SI-SDR | -16.36 / -16.60 | -14.36 / -14.20 | -14.18 / -13.30 | +2.18 / +3.30 |

Looks fine vs paper on Δ. No Lyra. Nothing scary showed up.

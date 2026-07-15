"""Test-set subsampling convergence analysis.

Given per-file metric scores over the full test set (produced by
``score_dataset_noisy`` / ``score_dataset_phaseless`` with ``--include-per-file``),
this answers: how large a random subset of the test set do we need for the
subset-mean metric to land close to the full-set mean?

For a subset of size N drawn without replacement from M files, the standard
error of the subset mean is ``sigma/sqrt(N) * sqrt((M-N)/(M-1))`` (finite
population correction), where ``sigma`` is the per-file standard deviation.
We report both this analytic SE and an empirical bootstrap over many seeds.

Usage:
    python experiments/evaluation/scoring/subset_convergence.py \
        --scores outputs/dataset_scores/se_test_noisy_all_perfile.json \
                 outputs/dataset_scores/stftpr_test_phaseless_random_all_perfile.json \
        --out-dir outputs/convergence
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ("pesq", "estoi", "si_sdr", "lsd", "psnr")
DEFAULT_GRID = (10, 20, 50, 100, 150, 200, 300, 500)


def _label(payload: dict, path: Path) -> str:
    task = payload.get("task", "?")
    source = payload.get("source", "?")
    if source == "dataset_phaseless":
        return f"{task}/phaseless-{payload.get('phase_mode', '?')}"
    if source == "dataset":
        return f"{task}/noisy"
    return f"{task}/{source}"


def _metric_values(per_file: list[dict], metric: str) -> np.ndarray:
    vals = [row.get(metric) for row in per_file]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return np.asarray(vals, dtype=float)


def analyze(payload: dict, grid, n_boot: int, seed: int) -> dict:
    per_file = payload.get("per_file")
    if not per_file:
        raise ValueError("Score file has no 'per_file' list. Re-run scoring with --include-per-file.")
    total = len(per_file)
    rng = np.random.default_rng(seed)

    out: dict[str, dict] = {}
    for metric in METRICS:
        values = _metric_values(per_file, metric)
        m = len(values)
        if m == 0:
            continue
        full_mean = float(values.mean())
        full_std = float(values.std(ddof=1)) if m > 1 else 0.0

        rows = []
        for n in list(grid) + [m]:
            n = min(int(n), m)
            if n <= 0:
                continue
            # empirical bootstrap over subset selections
            means = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.choice(m, size=n, replace=False)
                means[b] = values[idx].mean()
            boot_mean = float(means.mean())
            boot_se = float(means.std(ddof=1)) if n_boot > 1 else 0.0
            lo, hi = np.percentile(means, [2.5, 97.5])
            # analytic SE with finite population correction
            if n < m:
                fpc = np.sqrt((m - n) / (m - 1))
            else:
                fpc = 0.0
            se_analytic = full_std / np.sqrt(n) * fpc
            rows.append(
                {
                    "n": n,
                    "boot_mean": boot_mean,
                    "boot_se": boot_se,
                    "ci95_lo": float(lo),
                    "ci95_hi": float(hi),
                    "se_analytic": float(se_analytic),
                    "rel_halfwidth_pct": float(1.96 * boot_se / abs(full_mean) * 100.0) if full_mean != 0 else float("nan"),
                    "abs_halfwidth": float(1.96 * boot_se),
                }
            )
        out[metric] = {
            "full_mean": full_mean,
            "full_std": full_std,
            "num_files": m,
            "rows": rows,
        }
    return {"label": None, "total_files": total, "metrics": out}


def print_report(label: str, result: dict) -> None:
    print("=" * 78)
    print(f"{label}   (files scored: {result['total_files']})")
    print("=" * 78)
    for metric, data in result["metrics"].items():
        fm, fs = data["full_mean"], data["full_std"]
        print(f"\n  {metric.upper():6s}  full-set mean = {fm:.4f}   per-file std = {fs:.4f}")
        print(f"    {'N':>5}  {'mean':>9}  {'±95% (abs)':>11}  {'±95% (%full)':>13}  {'analytic SE':>11}")
        for r in data["rows"]:
            print(
                f"    {r['n']:>5}  {r['boot_mean']:>9.4f}  {r['abs_halfwidth']:>11.4f}"
                f"  {r['rel_halfwidth_pct']:>12.2f}%  {r['se_analytic']:>11.4f}"
            )


def recommend(result: dict, rel_tol_pct: float) -> dict:
    """Smallest N whose 95% half-width is within rel_tol_pct of the full mean."""
    rec = {}
    for metric, data in result["metrics"].items():
        chosen = None
        for r in data["rows"]:
            if np.isfinite(r["rel_halfwidth_pct"]) and r["rel_halfwidth_pct"] <= rel_tol_pct:
                chosen = r["n"]
                break
        rec[metric] = chosen
    return rec


def make_plot(results: dict[str, dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.2 * n_metrics, 4.2))
    if n_metrics == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 2)))

    for ax, metric in zip(axes, METRICS):
        for (label, res), color in zip(results.items(), colors):
            data = res["metrics"].get(metric)
            if not data:
                continue
            ns = [r["n"] for r in data["rows"]]
            means = np.array([r["boot_mean"] for r in data["rows"]])
            hw = np.array([r["abs_halfwidth"] for r in data["rows"]])
            ax.plot(ns, means, "-o", color=color, ms=3, label=label)
            ax.fill_between(ns, means - hw, means + hw, color=color, alpha=0.18)
            ax.axhline(data["full_mean"], color=color, ls=":", lw=1, alpha=0.7)
        ax.set_title(metric.upper())
        ax.set_xlabel("subset size N")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("subset-mean metric (±95% over seeds)")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Test-set subsampling convergence (shaded = 95% band over random subsets)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nSaved plot -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", nargs="+", required=True, help="Per-file score JSON files.")
    parser.add_argument("--out-dir", default="outputs/convergence")
    parser.add_argument("--grid", type=int, nargs="+", default=list(DEFAULT_GRID))
    parser.add_argument("--n-boot", type=int, default=500, help="Random subsets per N.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rel-tol-pct", type=float, default=2.0, help="Target 95%% half-width as %% of full mean.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    summary_json: dict = {}
    for score_path in args.scores:
        path = Path(score_path)
        payload = json.loads(path.read_text())
        label = _label(payload, path)
        res = analyze(payload, args.grid, args.n_boot, args.seed)
        res["label"] = label
        results[label] = res
        print_report(label, res)
        rec = recommend(res, args.rel_tol_pct)
        print(f"\n  -> smallest N with 95% half-width <= {args.rel_tol_pct}% of full mean: {rec}")
        summary_json[label] = {
            "total_files": res["total_files"],
            "metrics": {
                m: {
                    "full_mean": d["full_mean"],
                    "full_std": d["full_std"],
                    "rows": d["rows"],
                }
                for m, d in res["metrics"].items()
            },
            "recommended_n": rec,
        }

    (out_dir / "convergence_summary.json").write_text(json.dumps(summary_json, indent=2))
    print(f"\nSaved summary -> {out_dir / 'convergence_summary.json'}")
    make_plot(results, out_dir / "convergence.png")


if __name__ == "__main__":
    main()

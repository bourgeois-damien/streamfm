"""
Test orchestrator for StreamFM pipeline validation against paper baselines.

This module handles:
- Test dataset configuration and validation
- Benchmark execution and result collection
- Comparison with paper baselines
- Progressive optimization tracking
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import logging

import torch
import numpy as np
import torchaudio
from scipy.io import wavfile


logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Configuration for a test run."""
    task: str  # "se", "stftpr", "bwe", "derev", "lyra", "melflow"
    part: str  # "predictor", "flow", "model" for SE; "model", "flow" for STFTPR
    solver_steps: tuple[int, ...] = (5,)  # ODE solver steps to test
    execution_modes: tuple[str, ...] = ("eager", "compiled", "cuda_graph")
    batch_size: int = 1
    num_warmup: int = 2
    num_iterations: int = 10
    dtype: str = "fp32"  # fp32, fp16, bf16
    use_streaming: bool = False
    model_only: bool = True  # False = full audio pipeline


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    task: str
    part: str
    execution_mode: str
    solver_steps: int
    dtype: str
    model_only: bool
    use_streaming: bool
    
    latency_ms: float  # mean
    latency_std_ms: float
    throughput_samples_per_sec: float
    memory_mb: float  # peak
    
    timestamp: str
    config_hash: str


@dataclass
class PaperBaseline:
    """Reference results from the Stream.FM paper."""
    task: str
    part: str
    metric_name: str
    value: float
    solver_steps: int
    notes: str


# Paper baselines (from StreamFM 2512.19442 v3)
PAPER_BASELINES = {
    "se_predictor_latency_5steps": PaperBaseline(
        task="se",
        part="predictor",
        metric_name="latency_ms",
        value=3.5,  # example - verify from paper
        solver_steps=5,
        notes="Predictor-only inference at 16kHz 1-sec chunks"
    ),
    "se_flow_latency_5steps": PaperBaseline(
        task="se",
        part="flow",
        metric_name="latency_ms",
        value=8.2,  # example - verify from paper
        solver_steps=5,
        notes="Flow-only inference"
    ),
    "se_full_latency_5steps": PaperBaseline(
        task="se",
        part="model",
        metric_name="latency_ms",
        value=12.0,  # example - verify from paper
        solver_steps=5,
        notes="Full SE model (predictor + flow)"
    ),
    "stftpr_latency_5steps": PaperBaseline(
        task="stftpr",
        part="model",
        metric_name="latency_ms",
        value=4.1,  # example - verify from paper
        solver_steps=5,
        notes="STFTPR full model"
    ),
}


class TestOrchestrator:
    """Orchestrates test execution and result tracking."""
    
    def __init__(
        self,
        repo_root: Path | str,
        output_dir: Path | str | None = None,
    ):
        self.repo_root = Path(repo_root)
        self.output_dir = Path(output_dir) if output_dir else self.repo_root / "test_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results tracking
        self.results: list[BenchmarkResult] = []
        self.comparisons: list[dict] = []
        
        logger.info(f"TestOrchestrator initialized. Output dir: {self.output_dir}")
    
    def get_checkpoint_path(self, task: str) -> Path:
        """Get checkpoint path for a task."""
        checkpoint_map = {
            "se_predictor": "streamfm_se_predictor_dnn_only.pt",
            "se_flow": "streamfm_se_predgen_dnn_only.pt",
            "se_full": "streamfm_se_predgen.ckpt",
            "stftpr": "streamfm_stftpr_dnn_only.pt",
            "bwe": "streamfm_bwe.ckpt",
            "derev": "streamfm_derev.ckpt",
            "lyra": "streamfm_lyra.ckpt",
            "melflow": "streamfm_melflow.ckpt",
        }
        ckpt_name = checkpoint_map.get(task)
        if not ckpt_name:
            raise ValueError(f"Unknown task: {task}")
        
        ckpt_path = self.repo_root / "checkpoints" / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        return ckpt_path
    
    def get_test_audio_path(self) -> Path:
        """Get a small test audio file."""
        test_audio = self.repo_root / "inputs" / "test_clips" / "audio_43m28_10s.wav"
        if test_audio.exists():
            return test_audio
        # Create a synthetic test audio
        return self._create_synthetic_audio()
    
    def _create_synthetic_audio(self, duration_sec: float = 1.0) -> Path:
        """Create synthetic test audio for quick validation."""
        sample_rate = 16000
        duration_samples = int(duration_sec * sample_rate)
        # Generate white noise
        audio = np.random.randn(duration_samples).astype(np.float32) * 0.1
        
        output_path = self.output_dir / "synthetic_test_audio.wav"
        torchaudio.save(output_path, torch.from_numpy(audio).unsqueeze(0), sample_rate)
        return output_path
    
    def record_result(self, result: BenchmarkResult) -> None:
        """Record a benchmark result."""
        self.results.append(result)
        logger.info(
            f"Recorded {result.task}/{result.part} "
            f"({result.execution_mode}, {result.solver_steps} steps): "
            f"{result.latency_ms:.2f}ms ± {result.latency_std_ms:.2f}ms"
        )
    
    def compare_with_baseline(
        self,
        result: BenchmarkResult,
        baseline: Optional[PaperBaseline] = None,
    ) -> dict:
        """Compare result with paper baseline."""
        if baseline is None:
            # Try to find matching baseline
            key = f"{result.task}_{result.part}_latency_{result.solver_steps}steps"
            baseline = PAPER_BASELINES.get(key)
        
        if baseline is None:
            logger.warning(f"No baseline found for {result.task}/{result.part}")
            return {"status": "no_baseline"}
        
        diff_pct = ((result.latency_ms - baseline.value) / baseline.value) * 100
        is_acceptable = diff_pct <= 10  # Within 10% of paper baseline
        
        comparison = {
            "task": result.task,
            "part": result.part,
            "solver_steps": result.solver_steps,
            "execution_mode": result.execution_mode,
            "paper_baseline_ms": baseline.value,
            "measured_ms": result.latency_ms,
            "diff_pct": diff_pct,
            "is_acceptable": is_acceptable,
            "baseline_notes": baseline.notes,
        }
        self.comparisons.append(comparison)
        return comparison
    
    def save_results(self) -> Path:
        """Save all results to JSON."""
        output_file = self.output_dir / "benchmark_results.json"
        data = {
            "results": [asdict(r) for r in self.results],
            "comparisons": self.comparisons,
            "summary": self._summarize_results(),
        }
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Results saved to {output_file}")
        return output_file
    
    def _summarize_results(self) -> dict:
        """Generate summary statistics."""
        if not self.results:
            return {}
        
        summary = {
            "total_runs": len(self.results),
            "total_comparisons": len(self.comparisons),
        }
        
        # Group by task
        by_task = {}
        for result in self.results:
            if result.task not in by_task:
                by_task[result.task] = []
            by_task[result.task].append(result)
        
        for task, results in by_task.items():
            avg_latency = np.mean([r.latency_ms for r in results])
            summary[f"{task}_avg_latency_ms"] = avg_latency
        
        # Comparison success rate
        if self.comparisons:
            acceptable_count = sum(1 for c in self.comparisons if c.get("is_acceptable", False))
            summary["baseline_compliance_pct"] = (acceptable_count / len(self.comparisons)) * 100
        
        return summary
    
    def generate_report(self) -> str:
        """Generate a human-readable report."""
        lines = ["# StreamFM Benchmark Report\n"]
        lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        lines.append(f"## Summary\n")
        summary = self._summarize_results()
        for key, value in summary.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.2f}")
            else:
                lines.append(f"- {key}: {value}")
        
        lines.append(f"\n## Detailed Results\n")
        for result in self.results:
            lines.append(f"### {result.task}/{result.part} ({result.execution_mode}, {result.solver_steps} steps)")
            lines.append(f"- Latency: {result.latency_ms:.2f}ms ± {result.latency_std_ms:.2f}ms")
            lines.append(f"- Throughput: {result.throughput_samples_per_sec:.0f} samples/sec")
            lines.append(f"- Memory: {result.memory_mb:.0f} MB")
            lines.append("")
        
        lines.append(f"\n## Baseline Comparisons\n")
        for comp in self.comparisons:
            status = "✓ PASS" if comp.get("is_acceptable") else "✗ FAIL"
            lines.append(
                f"### {status}: {comp['task']}/{comp['part']} ({comp['solver_steps']} steps)"
            )
            lines.append(f"- Paper: {comp['paper_baseline_ms']:.2f}ms")
            lines.append(f"- Measured: {comp['measured_ms']:.2f}ms")
            lines.append(f"- Diff: {comp['diff_pct']:+.1f}%")
            lines.append("")
        
        return "\n".join(lines)


def setup_logging(output_dir: Path) -> None:
    """Configure logging."""
    log_file = output_dir / "test.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )

"""Capture focused Nsight Systems traces for StreamFM on Modal GPUs.

The ordinary latency benchmark still measures 100 frames.  Only a few
post-warmup frames are exposed to CUPTI, which keeps the trace compact while
preserving the exact fixed-shape streaming execution being investigated.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import modal


REMOTE_ROOT = "/root/streamfm"
VOLUME_ROOT = "/data"
PROFILE_ROOT = f"{VOLUME_ROOT}/nsight_streamfm"
CACHE_VOLUME = modal.Volume.from_name("streamfm-cache")


def _find_repo_root() -> Path:
    remote_repo = Path(REMOTE_ROOT)
    if (remote_repo / "config").is_dir() and (remote_repo / "sgmse").is_dir():
        return remote_repo
    current_file = Path(__file__).resolve()
    for candidate in (current_file.parent, *current_file.parents):
        if (candidate / "config").is_dir() and (candidate / "sgmse").is_dir():
            return candidate
    raise RuntimeError("Could not locate the StreamFM repository root.")


LOCAL_ROOT = _find_repo_root()

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .env({"PYTHONPATH": REMOTE_ROOT})
    .run_commands(
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
        "--no-install-recommends ca-certificates curl gnupg libsndfile1",
        "curl -fsSL https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub "
        "-o /tmp/nvidia-devtools.pub && "
        "gpg --dearmor --yes -o /usr/share/keyrings/nvidia-devtools.gpg /tmp/nvidia-devtools.pub && "
        "echo 'deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] "
        "https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /' "
        "> /etc/apt/sources.list.d/nvidia-devtools.list",
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
        "--no-install-recommends nsight-systems-cli-2026.3.1 nsight-compute-2025.3.1 && "
        "rm -rf /var/lib/apt/lists/*",
    )
    .pip_install(
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "einops==0.8.1",
        "hydra-core==1.3.2",
        "numpy==1.26.4",
        "soundfile==0.12.1",
        "tensorrt==10.9.0.34",
        "torch-tensorrt==2.7.0",
        "requests",
        "nvidia-modelopt[torch]==0.17.0",
    )
    .add_local_dir(str(LOCAL_ROOT / "config"), remote_path=f"{REMOTE_ROOT}/config")
    .add_local_dir(
        str(LOCAL_ROOT / "experiments"),
        remote_path=f"{REMOTE_ROOT}/experiments",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(str(LOCAL_ROOT / "flow_autoparams"), remote_path=f"{REMOTE_ROOT}/flow_autoparams")
    .add_local_dir(
        str(LOCAL_ROOT / "sgmse"),
        remote_path=f"{REMOTE_ROOT}/sgmse",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
)

for checkpoint_name in (
    "streamfm_stftpr_dnn_only.pt",
    "streamfm_bwe_dnn_only.pt",
    "streamfm_derev_dnn_only.pt",
    "streamfm_lyra_dnn_only.pt",
):
    local_checkpoint = LOCAL_ROOT / "checkpoints" / checkpoint_name
    if local_checkpoint.exists():
        image = image.add_local_file(
            str(local_checkpoint),
            remote_path=f"{REMOTE_ROOT}/checkpoints/{checkpoint_name}",
        )


app = modal.App("streamfm-nsight", image=image)


def _run_checked(command: list[str], *, env: dict[str, str], cwd: str) -> subprocess.CompletedProcess:
    """Run a subprocess capturing merged stdout/stderr; never raises — exit codes land in the run metadata."""
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _run_diagnostic(command: list[str], *, env: dict[str, str], cwd: str) -> dict:
    """Run one profiler/diagnostic command and retain unambiguous process status.

    A positive return code is an ordinary application/tool exit code.  Only a
    negative return code means subprocess observed termination by a signal.
    Keeping stdout and stderr separate is important for Nsight, whose own
    diagnostics otherwise get mixed with the target application's output.
    """
    started_at = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        return {
            "command": command,
            "returncode": None,
            "exit_code": None,
            "signal_number": None,
            "signal_name": None,
            "duration_s": time.perf_counter() - started_at,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }

    returncode = result.returncode
    signal_number = -returncode if returncode < 0 else None
    return {
        "command": command,
        "returncode": returncode,
        "exit_code": returncode if returncode >= 0 else None,
        "signal_number": signal_number,
        "signal_name": (
            signal.Signals(signal_number).name if signal_number is not None else None
        ),
        "duration_s": time.perf_counter() - started_at,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _persist_command_result(output_dir: Path, name: str, result: dict) -> dict:
    """Write command streams separately and return compact JSON metadata."""
    stdout_path = output_dir / f"{name}.stdout.log"
    stderr_path = output_dir / f"{name}.stderr.log"
    stdout_path.write_text(result.get("stdout", ""), encoding="utf-8")
    stderr_path.write_text(result.get("stderr", ""), encoding="utf-8")
    return {
        key: value
        for key, value in result.items()
        if key not in {"stdout", "stderr"}
    } | {
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
        "stdout_bytes": stdout_path.stat().st_size,
        "stderr_bytes": stderr_path.stat().st_size,
    }


def _add_profiler_library_paths(env: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Expose CUPTI/NVPerf libraries that standalone profiler packages do not register."""
    candidates = []
    for root in (
        Path("/opt/nvidia/nsight-compute"),
        Path("/usr/local/cuda/extras/CUPTI"),
        Path("/usr/local/cuda/targets/x86_64-linux/lib"),
    ):
        if not root.exists():
            continue
        for pattern in ("libnvperf_host.so*", "libnvperf_target.so*", "libcupti.so*"):
            candidates.extend(path.parent for path in root.rglob(pattern) if path.is_file())
    directories = sorted({str(path) for path in candidates})
    updated = env.copy()
    if directories:
        existing = updated.get("LD_LIBRARY_PATH", "")
        updated["LD_LIBRARY_PATH"] = ":".join([*directories, existing] if existing else directories)
    return updated, directories


def _collect_environment_diagnostics(output_dir: Path, *, env: dict[str, str]) -> dict:
    """Capture enough environment detail to explain profiler compatibility failures."""
    commands = {
        "nsys_version": ["nsys", "--version"],
        "ncu_version": ["ncu", "--version"],
        "nvcc_version": ["nvcc", "--version"],
        "nvidia_smi": [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,compute_cap,memory.total",
            "--format=csv,noheader",
        ],
        "uname": ["uname", "-a"],
        "id": ["id"],
        "nsys_status": ["nsys", "status", "-e"],
        "ncu_location": ["bash", "-lc", "readlink -f \"$(command -v ncu)\""],
        "profiler_libraries": [
            "bash",
            "-lc",
            "find /opt/nvidia/nsight-compute /usr/local/cuda -type f "
            "\\( -name 'libnvperf*.so*' -o -name 'libcupti.so*' \\) 2>/dev/null | sort",
        ],
        "python_cuda": [
            sys.executable,
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({'python': __import__('sys').version, "
                "'torch': torch.__version__, 'torch_cuda': torch.version.cuda, "
                "'cuda_available': torch.cuda.is_available(), "
                "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
                "'properties': str(torch.cuda.get_device_properties(0)) if torch.cuda.is_available() else None}))"
            ),
        ],
    }
    diagnostic_runs = {}
    for name, command in commands.items():
        result = _run_diagnostic(command, env=env, cwd=REMOTE_ROOT)
        diagnostic_runs[name] = _persist_command_result(output_dir, name, result)

    paranoid_path = Path("/proc/sys/kernel/perf_event_paranoid")
    diagnostics = {
        "commands": diagnostic_runs,
        "perf_event_paranoid": (
            paranoid_path.read_text(encoding="utf-8").strip()
            if paranoid_path.exists()
            else None
        ),
        "environment": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
        },
    }
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    return diagnostics


def _target_command(
    *,
    execution: str,
    dtype: str,
    ptq_int8: bool,
    cuda_graph: bool,
    iterations: int,
    warmup: int,
    memory_format: str,
    tf32: str,
    trt_optimization_level: int,
    trt_avg_timing_iters: int,
    trt_workspace_size_mib: int,
) -> list[str]:
    """Build the streamfm_benchmark CLI invocation that every capture tool below profiles.

    --backend local because the benchmark process already runs inside the
    Modal container; the profiler wraps it there.
    """
    # cuda_graph is a capture-tool axis here; on the benchmark CLI the TensorRT
    # graph mode is spelled as its own execution value.
    if cuda_graph and execution == "tensorrt":
        execution = "tensorrt_cuda_graph"
    target = [
        sys.executable,
        "-m",
        "experiments.benchmarks.streamfm_benchmark",
        "--backend",
        "local",
        "--hardware",
        "cuda",
        "--task",
        "stftpr",
        "--part",
        "model",
        "--pipeline",
        "model_only",
        "--execution",
        execution,
        "--steps",
        "1",
        "--iterations",
        str(iterations),
        "--warmup",
        str(warmup),
        "--dtype",
        dtype,
        "--memory-format",
        memory_format,
        "--tf32",
        tf32,
        "--trt-optimization-level",
        str(trt_optimization_level),
        "--trt-avg-timing-iters",
        str(trt_avg_timing_iters),
        "--trt-workspace-size-mib",
        str(trt_workspace_size_mib),
        "--preallocate-model-buffers",
    ]
    if ptq_int8:
        target.extend(["--ptq-int8", "tensorrt", "--ptq-calib-steps", "32"])
    return target


@app.function(gpu="L4", timeout=3600, volumes={VOLUME_ROOT: CACHE_VOLUME})
def capture_nsys(
    *,
    execution: str,
    dtype: str,
    ptq_int8: bool,
    cuda_graph: bool,
    iterations: int,
    warmup: int,
    profile_frames: int,
    memory_format: str,
    tf32: str,
    trt_optimization_level: int,
    trt_avg_timing_iters: int,
    trt_workspace_size_mib: int,
    nsys_cuda_trace: str,
    require_full_compilation: bool,
    run_id: str,
) -> dict:
    """Run the benchmark under Nsight Systems on an L4 and store the artifacts on the volume.

    STREAMFM_CUDA_PROFILE_FRAMES opens the cudaProfilerApi capture window (see
    benchmarks/cuda_profile_range.py), so the trace covers only a few measured
    frames instead of model load, TensorRT compilation and warmup.
    """
    output_dir = Path(PROFILE_ROOT) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    report_base = output_dir / "streamfm"
    report_path = report_base.with_suffix(".nsys-rep")

    env, _ = _add_profiler_library_paths(os.environ.copy())
    env["PYTHONPATH"] = REMOTE_ROOT
    env["STREAMFM_CUDA_PROFILE_FRAMES"] = str(profile_frames)
    env["STREAMFM_TRT_REQUIRE_FULL_COMPILATION"] = "1" if require_full_compilation else "0"
    diagnostics = _collect_environment_diagnostics(output_dir, env=env)

    target = _target_command(
        execution=execution,
        dtype=dtype,
        ptq_int8=ptq_int8,
        cuda_graph=cuda_graph,
        iterations=iterations,
        warmup=warmup,
        memory_format=memory_format,
        tf32=tf32,
        trt_optimization_level=trt_optimization_level,
        trt_avg_timing_iters=trt_avg_timing_iters,
        trt_workspace_size_mib=trt_workspace_size_mib,
    )

    nsys_command = [
        "nsys",
        "profile",
        "--force-overwrite=true",
        "--sample=none",
        "--cpuctxsw=none",
        # Modal's gVisor kernel does not expose perf_event_open.  CUDA and
        # NVTX tracing use CUPTI and remain useful; OS-runtime/CPU sampling is
        # intentionally disabled rather than requested and then silently lost.
        f"--trace={nsys_cuda_trace},nvtx",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--cuda-graph-trace=node",
        f"--output={report_base}",
        *target,
    ]
    run = _run_diagnostic(nsys_command, env=env, cwd=REMOTE_ROOT)
    run_metadata = _persist_command_result(output_dir, "nsys_profile", run)

    # Pre-render the standard stats tables to text so the results are readable
    # without downloading the report into the Nsight GUI.
    reports = (
        "cuda_api_sum",
        "cuda_gpu_kern_sum",
        "cuda_gpu_mem_time_sum",
        "cuda_gpu_mem_size_sum",
        "cuda_gpu_trace",
        "nvtx_sum",
    )
    stats_chunks = []
    if report_path.exists():
        for report in reports:
            result = _run_diagnostic(
                ["nsys", "stats", "--report", report, str(report_path)],
                env=env,
                cwd=REMOTE_ROOT,
            )
            stats_chunks.append(
                f"\n===== {report} (exit {result['returncode']}) =====\n"
                f"{result['stdout']}\n{result['stderr']}"
            )
    stats_text = "".join(stats_chunks)
    (output_dir / "nsys_stats.txt").write_text(stats_text, encoding="utf-8")

    ncu_path = subprocess.run(
        ["bash", "-lc", "command -v ncu || true"],
        text=True,
        stdout=subprocess.PIPE,
        check=False,
    ).stdout.strip()
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution": execution,
        "dtype": dtype,
        "ptq_int8": ptq_int8,
        "cuda_graph": cuda_graph,
        "iterations": iterations,
        "warmup": warmup,
        "profile_frames": profile_frames,
        "memory_format": memory_format,
        "tf32": tf32,
        "trt_optimization_level": trt_optimization_level,
        "trt_avg_timing_iters": trt_avg_timing_iters,
        "trt_workspace_size_mib": trt_workspace_size_mib,
        "require_full_compilation": require_full_compilation,
        "nsys_cuda_trace": nsys_cuda_trace,
        "nsys_exit_code": run["returncode"],
        "nsys_run": run_metadata,
        "report_exists": report_path.exists(),
        "report_bytes": report_path.stat().st_size if report_path.exists() else 0,
        "capture_valid": (
            run["returncode"] == 0
            and report_path.exists()
            and report_path.stat().st_size > 0
        ),
        "ncu_path": ncu_path,
        "diagnostics_file": "diagnostics.json",
        "perf_event_paranoid": diagnostics["perf_event_paranoid"],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    CACHE_VOLUME.commit()
    return metadata


@app.function(gpu="L4", timeout=3600, volumes={VOLUME_ROOT: CACHE_VOLUME})
def capture_ncu(
    *,
    execution: str,
    dtype: str,
    ptq_int8: bool,
    cuda_graph: bool,
    iterations: int,
    warmup: int,
    profile_frames: int,
    memory_format: str,
    tf32: str,
    trt_optimization_level: int,
    trt_avg_timing_iters: int,
    trt_workspace_size_mib: int,
    require_full_compilation: bool,
    run_id: str,
    ncu_set: str,
    launch_skip: int,
    launch_count: int,
) -> dict:
    """Run the benchmark under Nsight Compute for per-kernel hardware metrics.

    --profile-from-start off defers to the same cudaProfilerApi window as nsys;
    launch-skip/launch-count then select which kernel launches inside that
    window get replayed and measured.
    """
    output_dir = Path(PROFILE_ROOT) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    report_base = output_dir / "streamfm"
    report_path = report_base.with_suffix(".ncu-rep")
    env, _ = _add_profiler_library_paths(os.environ.copy())
    env["PYTHONPATH"] = REMOTE_ROOT
    env["STREAMFM_CUDA_PROFILE_FRAMES"] = str(profile_frames)
    env["STREAMFM_TRT_REQUIRE_FULL_COMPILATION"] = "1" if require_full_compilation else "0"
    diagnostics = _collect_environment_diagnostics(output_dir, env=env)
    target = _target_command(
        execution=execution,
        dtype=dtype,
        ptq_int8=ptq_int8,
        cuda_graph=cuda_graph,
        iterations=iterations,
        warmup=warmup,
        memory_format=memory_format,
        tf32=tf32,
        trt_optimization_level=trt_optimization_level,
        trt_avg_timing_iters=trt_avg_timing_iters,
        trt_workspace_size_mib=trt_workspace_size_mib,
    )
    command = [
        "ncu",
        "--target-processes",
        "all",
        "--profile-from-start",
        "off",
        "--graph-profiling",
        "node",
        "--set",
        ncu_set,
        "--launch-skip",
        str(launch_skip),
        "--launch-count",
        str(launch_count),
        "--force-overwrite",
        "--export",
        str(report_base),
        *target,
    ]
    run = _run_diagnostic(command, env=env, cwd=REMOTE_ROOT)
    run_metadata = _persist_command_result(output_dir, "ncu_profile", run)
    details = ""
    import_metadata = None
    if report_path.exists():
        imported = _run_diagnostic(
            ["ncu", "--import", str(report_path), "--page", "details"],
            env=env,
            cwd=REMOTE_ROOT,
        )
        import_metadata = _persist_command_result(output_dir, "ncu_import", imported)
        details = f"{imported['stdout']}\n{imported['stderr']}"
    (output_dir / "ncu_details.txt").write_text(details, encoding="utf-8")
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool": "ncu",
        "execution": execution,
        "dtype": dtype,
        "ptq_int8": ptq_int8,
        "cuda_graph": cuda_graph,
        "iterations": iterations,
        "warmup": warmup,
        "profile_frames": profile_frames,
        "memory_format": memory_format,
        "tf32": tf32,
        "trt_optimization_level": trt_optimization_level,
        "trt_avg_timing_iters": trt_avg_timing_iters,
        "trt_workspace_size_mib": trt_workspace_size_mib,
        "require_full_compilation": require_full_compilation,
        "ncu_set": ncu_set,
        "launch_skip": launch_skip,
        "launch_count": launch_count,
        "ncu_exit_code": run["returncode"],
        "ncu_run": run_metadata,
        "ncu_import": import_metadata,
        "report_exists": report_path.exists(),
        "report_bytes": report_path.stat().st_size if report_path.exists() else 0,
        "capture_valid": (
            run["returncode"] == 0
            and report_path.exists()
            and report_path.stat().st_size > 0
        ),
        "diagnostics_file": "diagnostics.json",
        "perf_event_paranoid": diagnostics["perf_event_paranoid"],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    CACHE_VOLUME.commit()
    return metadata


def _classify_ncu_probe(*, baseline: dict, profiled: dict, report_path: Path) -> str:
    """Classify a minimal NCU probe without confusing target and profiler failures."""
    if baseline["returncode"] != 0:
        return "target_failed"
    combined = f"{profiled.get('stdout', '')}\n{profiled.get('stderr', '')}".lower()
    if "err_nvgpuctrperm" in combined or (
        "permission" in combined and "performance counter" in combined
    ):
        return "permission_denied"
    if "librarynotloaded" in combined or "compatible driver library" in combined:
        return "counter_library_unavailable"
    if "metric" in combined and any(
        marker in combined for marker in ("not found", "not available", "unsupported")
    ):
        return "metric_unavailable"
    if (
        profiled["returncode"] == 0
        and report_path.exists()
        and report_path.stat().st_size > 0
    ):
        return "allowed"
    return "profiler_failed"


@app.function(gpu="L4", timeout=900, volumes={VOLUME_ROOT: CACHE_VOLUME})
def probe_ncu_permissions(*, run_id: str) -> dict:
    """Test one hardware counter on one CUDA kernel, independently of StreamFM.

    This intentionally excludes PyTorch, TensorRT, CUDA Graph and
    cudaProfilerStart/Stop so the result answers only whether NCU can collect a
    protected GPU metric in the Modal container.
    """
    output_dir = Path(PROFILE_ROOT) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    env, added_library_paths = _add_profiler_library_paths(os.environ.copy())
    env["PYTHONPATH"] = REMOTE_ROOT
    diagnostics = _collect_environment_diagnostics(output_dir, env=env)

    source = Path(REMOTE_ROOT) / "experiments/benchmarks/profiling/ncu_probe.cu"
    binary = output_dir / "ncu_probe"
    report_base = output_dir / "ncu_probe_report"
    report_path = report_base.with_suffix(".ncu-rep")

    compile_result = _run_diagnostic(
        ["nvcc", "-O2", "-lineinfo", str(source), "-o", str(binary)],
        env=env,
        cwd=REMOTE_ROOT,
    )
    compile_metadata = _persist_command_result(output_dir, "probe_compile", compile_result)

    if compile_result["returncode"] == 0:
        baseline = _run_diagnostic([str(binary)], env=env, cwd=REMOTE_ROOT)
    else:
        baseline = {
            "command": [str(binary)],
            "returncode": None,
            "exit_code": None,
            "signal_number": None,
            "signal_name": None,
            "duration_s": 0.0,
            "stdout": "",
            "stderr": "probe was not executed because nvcc compilation failed",
        }
    baseline_metadata = _persist_command_result(output_dir, "probe_baseline", baseline)

    metric_query = _run_diagnostic(
        ["ncu", "--query-metrics", "--query-metrics-mode", "all"],
        env=env,
        cwd=REMOTE_ROOT,
    )
    metric_query_metadata = _persist_command_result(
        output_dir, "probe_metric_query", metric_query
    )

    if baseline["returncode"] == 0:
        profiled = _run_diagnostic(
            [
                "ncu",
                "--target-processes",
                "all",
                "--metrics",
                "sm__cycles_elapsed.avg",
                "--launch-count",
                "1",
                "--clock-control",
                "none",
                "--force-overwrite",
                "--export",
                str(report_base),
                str(binary),
            ],
            env=env,
            cwd=REMOTE_ROOT,
        )
    else:
        profiled = {
            "command": ["ncu", str(binary)],
            "returncode": None,
            "exit_code": None,
            "signal_number": None,
            "signal_name": None,
            "duration_s": 0.0,
            "stdout": "",
            "stderr": "NCU was not executed because the baseline target failed",
        }
    profiled_metadata = _persist_command_result(output_dir, "probe_ncu", profiled)
    classification = _classify_ncu_probe(
        baseline=baseline, profiled=profiled, report_path=report_path
    )

    imported_metadata = None
    if report_path.exists():
        imported = _run_diagnostic(
            ["ncu", "--import", str(report_path), "--page", "details"],
            env=env,
            cwd=REMOTE_ROOT,
        )
        imported_metadata = _persist_command_result(output_dir, "probe_import", imported)

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool": "ncu_probe",
        "classification": classification,
        "counter_access_allowed": classification == "allowed",
        "metric": "sm__cycles_elapsed.avg",
        "added_profiler_library_paths": added_library_paths,
        "compile": compile_metadata,
        "baseline": baseline_metadata,
        "metric_query": metric_query_metadata,
        "profiled": profiled_metadata,
        "imported": imported_metadata,
        "report_exists": report_path.exists(),
        "report_bytes": report_path.stat().st_size if report_path.exists() else 0,
        "diagnostics_file": "diagnostics.json",
        "perf_event_paranoid": diagnostics["perf_event_paranoid"],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    CACHE_VOLUME.commit()
    return metadata


@app.function(gpu="L4", timeout=3600, volumes={VOLUME_ROOT: CACHE_VOLUME})
def capture_torch_trace(
    *,
    execution: str,
    dtype: str,
    ptq_int8: bool,
    cuda_graph: bool,
    iterations: int,
    warmup: int,
    profile_frames: int,
    memory_format: str,
    tf32: str,
    trt_optimization_level: int,
    trt_avg_timing_iters: int,
    trt_workspace_size_mib: int,
    require_full_compilation: bool,
    run_id: str,
) -> dict:
    """Run the benchmark with the torch profiler exporting a Perfetto/Chrome trace; no Nsight involved.

    STREAMFM_CUDA_PROFILE_FRAMES stays 0 so the CUDA profiler window is never
    opened — only the torch-profiler switch is armed.
    """
    output_dir = Path(PROFILE_ROOT) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    trace_path = output_dir / "streamfm_perfetto_trace.json.gz"
    env = os.environ.copy()
    env["PYTHONPATH"] = REMOTE_ROOT
    env["STREAMFM_CUDA_PROFILE_FRAMES"] = "0"
    env["STREAMFM_TORCH_PROFILE_FRAMES"] = str(profile_frames)
    env["STREAMFM_TORCH_TRACE_PATH"] = str(trace_path)
    env["STREAMFM_TRT_REQUIRE_FULL_COMPILATION"] = "1" if require_full_compilation else "0"
    if execution == "tensorrt":
        env["STREAMFM_TRT_LAYER_INFO_PATH"] = str(output_dir / "tensorrt_engine_layers.json")
        env["STREAMFM_TRT_LAYER_PROFILE_DIR"] = str(output_dir / "tensorrt_layer_profile")
        env["STREAMFM_TRT_EXPORTED_OPS_PATH"] = str(output_dir / "tensorrt_exported_ops.json")
    target = _target_command(
        execution=execution,
        dtype=dtype,
        ptq_int8=ptq_int8,
        cuda_graph=cuda_graph,
        iterations=iterations,
        warmup=warmup,
        memory_format=memory_format,
        tf32=tf32,
        trt_optimization_level=trt_optimization_level,
        trt_avg_timing_iters=trt_avg_timing_iters,
        trt_workspace_size_mib=trt_workspace_size_mib,
    )
    run = _run_checked(target, env=env, cwd=REMOTE_ROOT)
    (output_dir / "torch_profile.log").write_text(run.stdout, encoding="utf-8")
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool": "torch",
        "execution": execution,
        "dtype": dtype,
        "ptq_int8": ptq_int8,
        "cuda_graph": cuda_graph,
        "iterations": iterations,
        "warmup": warmup,
        "profile_frames": profile_frames,
        "memory_format": memory_format,
        "tf32": tf32,
        "trt_optimization_level": trt_optimization_level,
        "trt_avg_timing_iters": trt_avg_timing_iters,
        "trt_workspace_size_mib": trt_workspace_size_mib,
        "require_full_compilation": require_full_compilation,
        "target_exit_code": run.returncode,
        "trace_exists": trace_path.exists(),
        "trace_bytes": trace_path.stat().st_size if trace_path.exists() else 0,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    CACHE_VOLUME.commit()
    return metadata


@app.local_entrypoint()
def main(
    tool: str = "nsys",
    execution: str = "tensorrt",
    dtype: str = "fp16",
    ptq_int8: bool = False,
    cuda_graph: bool = True,
    iterations: int = 100,
    warmup: int = 10,
    profile_frames: int = 3,
    memory_format: str = "contiguous",
    tf32: str = "auto",
    trt_optimization_level: int = 3,
    trt_avg_timing_iters: int = 1,
    trt_workspace_size_mib: int = 0,
    nsys_cuda_trace: str = "cuda-sw",
    require_full_compilation: bool = True,
    output_dir: str = "outputs/nsight_streamfm",
    ncu_set: str = "basic",
    launch_skip: int = 0,
    launch_count: int = 1,
):
    """Dispatch one capture to the right Modal function, then download its artifacts locally."""
    if tool not in {"nsys", "ncu", "ncu_probe", "torch"}:
        raise ValueError("tool must be nsys, ncu, ncu_probe, or torch.")
    if nsys_cuda_trace not in {"cuda", "cuda-sw"}:
        raise ValueError("nsys_cuda_trace must be cuda or cuda-sw.")

    if tool == "ncu_probe":
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_ncu_probe_{uuid.uuid4().hex[:8]}"
        metadata = probe_ncu_permissions.remote(run_id=run_id)
        local_root = Path(output_dir)
        local_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "volume",
                "get",
                "--force",
                "streamfm-cache",
                f"nsight_streamfm/{run_id}",
                str(local_root),
            ],
            check=True,
        )
        print(json.dumps(metadata, indent=2))
        print(f"Downloaded NCU probe artifacts to {local_root / run_id}")
        return
    if execution == "tensorrt_cuda_graph":
        # Same run as execution=tensorrt + cuda_graph=True; keep the split
        # form internally so run ids and metadata stay consistent.
        execution = "tensorrt"
        cuda_graph = True
    if execution not in {"eager", "compiled", "tensorrt", "cuda_graph"}:
        raise ValueError(
            "Profiling supports eager, compiled, cuda_graph, tensorrt, or tensorrt_cuda_graph."
        )
    if dtype not in {"fp32", "fp16"}:
        raise ValueError("Nsight capture supports dtype=fp32 or fp16.")
    if ptq_int8 and execution != "tensorrt":
        raise ValueError("INT8 profiling requires execution=tensorrt.")
    if ptq_int8 and dtype not in {"fp32", "fp16"}:
        raise ValueError("TensorRT INT8 profiling supports FP32 or FP16 fallback.")
    if tf32 not in {"auto", "on", "off"}:
        raise ValueError("tf32 must be auto, on, or off.")
    if not 0 <= trt_optimization_level <= 5:
        raise ValueError("trt_optimization_level must be between 0 and 5.")
    if trt_avg_timing_iters < 1:
        raise ValueError("trt_avg_timing_iters must be at least 1.")
    if trt_workspace_size_mib < 0:
        raise ValueError("trt_workspace_size_mib must be non-negative.")
    if execution == "cuda_graph" and cuda_graph:
        # Native PyTorch CUDA Graph is selected by execution itself; the
        # cuda_graph axis only applies to TensorRT runs, where the target
        # command maps it to --execution tensorrt_cuda_graph.
        cuda_graph = False

    precision = "int8" if ptq_int8 else dtype
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tool}_{execution}_{precision}_{uuid.uuid4().hex[:8]}"
    common = {
        "execution": execution,
        "dtype": dtype,
        "ptq_int8": ptq_int8,
        "cuda_graph": cuda_graph,
        "iterations": iterations,
        "warmup": warmup,
        "profile_frames": profile_frames,
        "memory_format": memory_format,
        "tf32": tf32,
        "trt_optimization_level": trt_optimization_level,
        "trt_avg_timing_iters": trt_avg_timing_iters,
        "trt_workspace_size_mib": trt_workspace_size_mib,
        "require_full_compilation": require_full_compilation,
        "run_id": run_id,
    }
    if tool == "ncu":
        metadata = capture_ncu.remote(
            **common,
            ncu_set=ncu_set,
            launch_skip=launch_skip,
            launch_count=launch_count,
        )
    elif tool == "torch":
        metadata = capture_torch_trace.remote(**common)
    else:
        metadata = capture_nsys.remote(
            **common,
            nsys_cuda_trace=nsys_cuda_trace,
        )
    local_root = Path(output_dir)
    local_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "volume",
            "get",
            "--force",
            "streamfm-cache",
            f"nsight_streamfm/{run_id}",
            str(local_root),
        ],
        check=True,
    )
    run_dir = local_root / run_id
    if tool == "torch":
        from experiments.benchmarks.profiling.analyze_trace import analyze_trace

        trace_path = run_dir / "streamfm_perfetto_trace.json.gz"
        metadata_path = run_dir / "metadata.json"
        if trace_path.exists():
            analysis = analyze_trace(trace_path, metadata_path)
            (run_dir / "trace_analysis.json").write_text(
                json.dumps(analysis, indent=2), encoding="utf-8"
            )
    print(json.dumps(metadata, indent=2))
    print(f"Downloaded Nsight artifacts to {run_dir}")

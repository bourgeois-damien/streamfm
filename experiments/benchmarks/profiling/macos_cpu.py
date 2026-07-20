#!/usr/bin/env python3
"""Capture Stream.FM CPU profiles with the native macOS profiling tools.

The wrapper deliberately keeps each profiler opt-in: ``doctor`` is read-only,
while the capture commands run exactly the target command supplied after ``--``.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import time
from pathlib import Path


_SENSITIVE_ENV_FRAGMENTS = (
    "ACCESS_KEY",
    "API_KEY",
    "AUTH",
    "CREDENTIAL",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
    "SECRET",
    "TOKEN",
)


def _profiling_env() -> dict[str, str]:
    """Copy the environment without credentials that Instruments would persist."""
    return {
        key: value
        for key, value in os.environ.items()
        if not any(fragment in key.upper() for fragment in _SENSITIVE_ENV_FRAGMENTS)
    }


def _restrict_trace_permissions(output: Path) -> None:
    if output.exists():
        output.chmod(0o700 if output.is_dir() else 0o600)


def _run_text(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        return 127, str(exc)
    output = (result.stdout or result.stderr).strip()
    return result.returncode, output


def _target_command(values: list[str]) -> list[str]:
    command = list(values)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("A target command is required after --.")
    return command


def _prepare_output(path: str) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def doctor(*, as_json: bool = False) -> int:
    xcode_rc, developer_dir = _run_text(["xcode-select", "-p"])
    xctrace_rc, xctrace_path = _run_text(["xcrun", "--find", "xctrace"])
    chip_rc, chip = _run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": chip if chip_rc == 0 else platform.processor(),
        "developer_dir": developer_dir if xcode_rc == 0 else "",
        "full_xcode_active": xcode_rc == 0 and developer_dir.endswith(".app/Contents/Developer"),
        "xctrace": xctrace_path if xctrace_rc == 0 else "",
        "powermetrics": shutil.which("powermetrics") or "",
        "sample": shutil.which("sample") or "",
    }
    report["ready"] = {
        "time_profiler": bool(report["xctrace"]),
        "cpu_counters": bool(report["xctrace"]),
        "power_frequency_thermal": bool(report["powermetrics"]),
        "sampling_fallback": bool(report["sample"]),
    }
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Platform:       {report['platform']}")
        print(f"Chip:           {report['chip'] or 'unknown'}")
        print(f"Developer dir:  {report['developer_dir'] or 'not configured'}")
        print(f"xctrace:        {report['xctrace'] or 'MISSING (full Xcode required)'}")
        print(f"powermetrics:   {report['powermetrics'] or 'missing'}")
        print(f"sample:         {report['sample'] or 'missing'}")
        if not report["xctrace"]:
            print("\nTo enable Instruments/xctrace after installing full Xcode:")
            print("  sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer")
            print("  sudo xcodebuild -runFirstLaunch")
            print("  xcrun xctrace list templates")
    return 0 if platform.system() == "Darwin" else 1


def capture_xctrace(
    *,
    template: str,
    output: str,
    command: list[str],
    attach_delay: float = 0.0,
    time_limit: float = 0.0,
    ready_timeout: float = 0.0,
) -> int:
    target = _target_command(command)
    if _run_text(["xcrun", "--find", "xctrace"])[0] != 0:
        raise SystemExit("xctrace is unavailable. Run doctor and install/select full Xcode first.")
    out = _prepare_output(output)
    base = ["xcrun", "xctrace", "record", "--template", template, "--output", str(out)]
    if time_limit > 0:
        limit = f"{max(1, round(time_limit * 1000))}ms" if time_limit < 1 else f"{time_limit:g}s"
        base.extend(["--time-limit", limit])

    if attach_delay <= 0 and ready_timeout <= 0:
        invocation = [*base, "--launch", "--", *target]
        print("Running:", " ".join(invocation))
        result = subprocess.run(invocation, check=False, env=_profiling_env())
        _restrict_trace_permissions(out)
        return result.returncode

    child_env = _profiling_env()
    ready_file = None
    if ready_timeout > 0:
        ready_file = out.parent / f".{out.name}.ready"
        ready_file.unlink(missing_ok=True)
        child_env["STREAMFM_PROFILE_READY_FILE"] = str(ready_file)
    process = subprocess.Popen(target, env=child_env)
    try:
        if ready_file is not None:
            deadline = time.monotonic() + ready_timeout
            while not ready_file.exists():
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Target exited with code {process.returncode} before signalling readiness."
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Target did not signal profiling readiness within {ready_timeout:g}s."
                    )
                time.sleep(0.1)
        if attach_delay > 0:
            time.sleep(attach_delay)
        if process.poll() is not None:
            raise RuntimeError(
                f"Target exited with code {process.returncode} before the attach delay elapsed."
            )
        invocation = [*base, "--attach", str(process.pid)]
        print("Running:", " ".join(invocation))
        trace_rc = subprocess.run(invocation, check=False).returncode
        _restrict_trace_permissions(out)
        target_rc = process.wait()
    finally:
        if ready_file is not None:
            ready_file.unlink(missing_ok=True)
        if process.poll() is None:
            process.terminate()
            process.wait()
    return target_rc if target_rc else trace_rc


def capture_sample(
    *, duration: int, interval_ms: int, delay: float, output: str, command: list[str]
) -> int:
    target = _target_command(command)
    sample_path = shutil.which("sample")
    if not sample_path:
        raise SystemExit("/usr/bin/sample is unavailable on this host.")
    out = _prepare_output(output)
    process = subprocess.Popen(target)
    try:
        if delay > 0:
            time.sleep(delay)
        if process.poll() is not None:
            raise RuntimeError(
                f"Target exited with code {process.returncode} before the sampling delay elapsed."
            )
        invocation = [
            sample_path,
            str(process.pid),
            str(duration),
            str(interval_ms),
            "-mayDie",
            "-fullPaths",
            "-file",
            str(out),
        ]
        sample_rc = subprocess.run(invocation, check=False).returncode
        target_rc = process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
    print(f"Wrote sampling stacks to {out}")
    return target_rc if target_rc else sample_rc


def capture_powermetrics(
    *,
    sample_rate_ms: int,
    output: str,
    command: list[str],
    ready_timeout: float = 0.0,
    duration: float = 0.0,
) -> int:
    target = _target_command(command)
    power_path = shutil.which("powermetrics")
    if not power_path:
        raise SystemExit("powermetrics is unavailable on this host.")
    out = _prepare_output(output)

    # Authenticate before launching the benchmark, otherwise the password prompt
    # can consume the entire short profiling window.
    if subprocess.run(["sudo", "-v"], check=False).returncode != 0:
        raise SystemExit("sudo authentication failed; powermetrics requires root privileges.")

    invocation = [
        "sudo",
        power_path,
        "--sample-rate",
        str(sample_rate_ms),
        "--sample-count",
        "-1",
        "--samplers",
        "tasks,cpu_power,thermal",
        "--show-process-amp",
        "--show-process-ipc",
        "--show-process-energy",
        "--show-plimits",
        "--show-cpu-qos",
        "--show-usage-summary",
    ]
    child_env = _profiling_env()
    ready_file = None
    if ready_timeout > 0:
        ready_file = out.parent / f".{out.name}.ready"
        ready_file.unlink(missing_ok=True)
        child_env["STREAMFM_PROFILE_READY_FILE"] = str(ready_file)
    target_process = subprocess.Popen(target, env=child_env)
    try:
        if ready_file is not None:
            deadline = time.monotonic() + ready_timeout
            while not ready_file.exists():
                if target_process.poll() is not None:
                    raise RuntimeError(
                        f"Target exited with code {target_process.returncode} before signalling readiness."
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Target did not signal profiling readiness within {ready_timeout:g}s."
                    )
                time.sleep(0.1)
        if target_process.poll() is not None:
            raise RuntimeError(f"Target exited with code {target_process.returncode} before capture.")

        with out.open("w", encoding="utf-8") as stream:
            profiler = subprocess.Popen(
                invocation,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            try:
                if duration > 0:
                    time.sleep(duration)
                else:
                    target_process.wait()
            finally:
                if profiler.poll() is None:
                    os.killpg(profiler.pid, signal.SIGINT)
                try:
                    profiler.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(profiler.pid, signal.SIGTERM)
                    profiler.wait()
        target_rc = target_process.wait()
    finally:
        if ready_file is not None:
            ready_file.unlink(missing_ok=True)
        if target_process.poll() is None:
            target_process.terminate()
            target_process.wait()
    print(f"Wrote power/frequency/IPC/thermal samples to {out}")
    return target_rc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Report profiler availability.")
    doctor_parser.add_argument("--json", action="store_true")

    for name, template, default_out in (
        ("time-profiler", "Time Profiler", "outputs/benchmark_profiles/macos_time.trace"),
        ("cpu-counters", "CPU Counters", "outputs/benchmark_profiles/macos_counters.trace"),
    ):
        capture = subparsers.add_parser(name, help=f"Capture the Instruments {template} template.")
        capture.add_argument("--out", default=default_out)
        capture.add_argument(
            "--attach-delay",
            type=float,
            default=0.0,
            help="Launch normally, then attach xctrace after this many seconds to exclude startup.",
        )
        capture.add_argument(
            "--time-limit",
            type=float,
            default=0.0,
            help="Stop recording after this many seconds (0 records until the target exits).",
        )
        capture.add_argument(
            "--ready-timeout",
            type=float,
            default=0.0,
            help=(
                "Wait up to this many seconds for a Stream.FM warmup-complete signal "
                "before attaching (0 disables readiness signalling)."
            ),
        )
        capture.add_argument("command", nargs=argparse.REMAINDER)

    sample_parser = subparsers.add_parser("sample", help="Capture native stacks without Xcode.")
    sample_parser.add_argument("--duration", type=int, default=10)
    sample_parser.add_argument("--interval-ms", type=int, default=1)
    sample_parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait after launching the target, useful for excluding load/compile/warmup.",
    )
    sample_parser.add_argument("--out", default="outputs/benchmark_profiles/macos_sample.txt")
    sample_parser.add_argument("command", nargs=argparse.REMAINDER)

    power_parser = subparsers.add_parser(
        "powermetrics", help="Capture process CPU, IPC, power, frequency and thermal data."
    )
    power_parser.add_argument("--sample-rate-ms", type=int, default=100)
    power_parser.add_argument("--ready-timeout", type=float, default=0.0)
    power_parser.add_argument("--duration", type=float, default=0.0)
    power_parser.add_argument("--out", default="outputs/benchmark_profiles/macos_powermetrics.txt")
    power_parser.add_argument("command", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if args.action == "doctor":
        raise SystemExit(doctor(as_json=args.json))
    if args.action in {"time-profiler", "cpu-counters"}:
        template = "Time Profiler" if args.action == "time-profiler" else "CPU Counters"
        raise SystemExit(
            capture_xctrace(
                template=template,
                output=args.out,
                command=args.command,
                attach_delay=args.attach_delay,
                time_limit=args.time_limit,
                ready_timeout=args.ready_timeout,
            )
        )
    if args.action == "sample":
        raise SystemExit(
            capture_sample(
                duration=args.duration,
                interval_ms=args.interval_ms,
                delay=args.delay,
                output=args.out,
                command=args.command,
            )
        )
    raise SystemExit(
        capture_powermetrics(
            sample_rate_ms=args.sample_rate_ms,
            output=args.out,
            command=args.command,
            ready_timeout=args.ready_timeout,
            duration=args.duration,
        )
    )


if __name__ == "__main__":
    main()

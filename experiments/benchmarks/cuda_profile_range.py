"""Small opt-in CUDA profiler range used by Nsight launchers.

Normal benchmarks are unaffected.  When ``STREAMFM_CUDA_PROFILE_FRAMES`` is
positive, the first measured frames are enclosed by the CUDA Profiler API so
Nsight Systems/Compute can ignore model loading, TensorRT compilation and
warm-up.  Per-frame NVTX ranges make the resulting timeline readable.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import os


class CudaProfileRange:
    """Bracket the first N measured frames for external profilers; a no-op otherwise.

    Two independent env-var switches: STREAMFM_CUDA_PROFILE_FRAMES opens a
    cudaProfilerStart/Stop window (Nsight launched with --capture-range
    cudaProfilerApi records only that window), and STREAMFM_TORCH_TRACE_PATH
    + STREAMFM_TORCH_PROFILE_FRAMES export a Chrome trace of those frames.
    """

    def __init__(self, torch, *, label: str):
        self.torch = torch
        self.label = label
        self.frame_limit = max(0, int(os.environ.get("STREAMFM_CUDA_PROFILE_FRAMES", "0")))
        self.torch_trace_path = os.environ.get("STREAMFM_TORCH_TRACE_PATH", "")
        self.torch_frame_limit = max(
            0, int(os.environ.get("STREAMFM_TORCH_PROFILE_FRAMES", "0"))
        )
        self.active = False
        self.stopped = False
        self.torch_profiler = None
        self.torch_profiler_active = False

    def start(self) -> None:
        """Arm the requested profilers just before the measured loop begins."""
        if self.active or self.stopped:
            return
        if self.torch_trace_path and self.torch_frame_limit > 0:
            from torch.profiler import ProfilerActivity, profile

            self.torch_profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            )
            self.torch_profiler.start()
            self.torch_profiler_active = True
        if self.frame_limit > 0:
            # Drain the queue first so leftover warmup kernels don't leak
            # into the Nsight capture window.
            self.torch.cuda.synchronize()
            status = self.torch.cuda.cudart().cudaProfilerStart()
            if status != 0:
                raise RuntimeError(f"cudaProfilerStart failed with CUDA status {status}.")
        if self.frame_limit > 0 or self.torch_profiler_active:
            self.torch.cuda.nvtx.range_push(f"streamfm/profile/{self.label}")
            self.active = True

    @contextmanager
    def frame(self, measured_index: int):
        """Wrap one measured frame in an NVTX range so it is findable on the timeline."""
        active_limit = max(self.frame_limit, self.torch_frame_limit)
        profile_this_frame = self.active and measured_index < active_limit
        if profile_this_frame:
            self.torch.cuda.nvtx.range_push(
                f"streamfm/frame/{self.label}/{measured_index:03d}"
            )
        record_context = (
            self.torch.profiler.record_function(
                f"streamfm/frame/{self.label}/{measured_index:03d}"
            )
            if profile_this_frame and self.torch_profiler_active
            else nullcontext()
        )
        try:
            with record_context:
                yield
        finally:
            if profile_this_frame:
                self.torch.cuda.nvtx.range_pop()

    @contextmanager
    def section(self, name: str):
        """Create matching NVTX and torch-profiler ranges for a profiling-only stage."""
        if not self.active:
            yield
            return
        label = f"streamfm/section/{self.label}/{name}"
        self.torch.cuda.nvtx.range_push(label)
        record_context = (
            self.torch.profiler.record_function(label)
            if self.torch_profiler_active
            else nullcontext()
        )
        try:
            with record_context:
                yield
        finally:
            self.torch.cuda.nvtx.range_pop()

    def finish_frame(self, measured_index: int) -> None:
        """Advance the torch profiler and shut both profilers down once the frame limit is hit.

        Synchronizes before stopping so trailing GPU work still lands inside
        the capture window / trace.
        """
        if self.torch_profiler_active:
            self.torch_profiler.step()
            if measured_index + 1 >= self.torch_frame_limit:
                self.torch.cuda.synchronize()
                self.torch_profiler.stop()
                self.torch_profiler.export_chrome_trace(self.torch_trace_path)
                self.torch_profiler_active = False
        active_limit = max(self.frame_limit, self.torch_frame_limit)
        if not self.active or measured_index + 1 < active_limit:
            return
        self.torch.cuda.synchronize()
        self.torch.cuda.nvtx.range_pop()
        if self.frame_limit > 0:
            status = self.torch.cuda.cudart().cudaProfilerStop()
            if status != 0:
                raise RuntimeError(f"cudaProfilerStop failed with CUDA status {status}.")
        self.active = False
        self.stopped = True

    def close(self) -> None:
        """Safety net for early loop exits: stop whatever is still recording."""
        if not self.active:
            return
        self.torch.cuda.synchronize()
        if self.torch_profiler_active:
            self.torch_profiler.stop()
            self.torch_profiler.export_chrome_trace(self.torch_trace_path)
            self.torch_profiler_active = False
        self.torch.cuda.nvtx.range_pop()
        if self.frame_limit > 0:
            self.torch.cuda.cudart().cudaProfilerStop()
        self.active = False
        self.stopped = True

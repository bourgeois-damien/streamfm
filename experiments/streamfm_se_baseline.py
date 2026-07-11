# Source Generated with Decompyle++
# File: streamfm_se_baseline.cpython-310.pyc (Python 3.10)

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Iterable
import torch
import torchaudio
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

def select_device(name = None):
    if name == 'auto':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        if None.cuda.is_available():
            return torch.device('cuda')
        return None.device('cpu')
    return None.device(name)


def sync_device(device = None):
    if device.type == 'mps':
        torch.mps.synchronize()
        return None
    if None.type == 'cuda':
        torch.cuda.synchronize()
        return None


def load_streamfm_se_model(device = None, config_name = None, ckpt_path = None):
    repo_root = Path(__file__).resolve().parents[1]
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = repo_root / ckpt_path
    with initialize_config_dir(str(repo_root / 'config'), '1.3', **('config_dir', 'version_base')):
        cfg = compose(config_name, **('config_name',))
    model = instantiate(cfg.model)
    ckpt = torch.load(ckpt_path, 'cpu', False, **('map_location', 'weights_only'))
    model.load_state_dict(ckpt['state_dict'])
    model = model.eval().to(device)
    return (model, cfg)


def load_mono_audio(path = None, device = None, target_sr = None):
    (wav, sr) = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav[:1]
    if target_sr is not None and sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr, 64, **('lowpass_filter_width',))
        sr = target_sr
    return (wav.to(device), sr)


def run_offline_inference(model, wav, sr = None, device = None, solver = None, steps = ('euler', 1, 0), seed = ('wav', 'torch.Tensor', 'sr', 'int', 'device', 'torch.device', 'solver', 'str', 'steps', 'int', 'seed', 'int')):
    torch.manual_seed(seed)
    sync_device(device)
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.enhance(wav, sr, solver, int(steps), **('solver', 'N'))
    sync_device(device)
    elapsed_s = time.perf_counter() - start
    duration_s = wav.shape[-1] / sr
    return {
        'mode': 'offline_enhance',
        'device': device.type,
        'solver': solver,
        'steps': int(steps),
        'audio_duration_s': duration_s,
        'elapsed_s': elapsed_s,
        'rtf': elapsed_s / duration_s,
        'output': out.detach().cpu() }


def _forward_step(module = None, x = None, *, state, time_cond, use_compiled):
    if use_compiled:
        return module.forward_step(x, time_cond, state, **('time_cond', 'state'))
    fn = None.forward_step.__wrapped__
    return fn(module, x, time_cond, state, **('time_cond', 'state'))


def _summarize_ms(times_ms = None):
    values = sorted(times_ms)
    return {
        'mean_ms': sum(values) / len(values),
        'p50_ms': values[len(values) // 2],
        'p90_ms': values[int(0.9 * (len(values) - 1))],
        'p99_ms': values[int(0.99 * (len(values) - 1))] }


def benchmark_frame_steps(model, device = None, steps_list = None, iterations = None, warmup = ((1,), 100, 10, False), use_compiled = ('device', 'torch.device', 'steps_list', 'Iterable[int]', 'iterations', 'int', 'warmup', 'int', 'use_compiled', 'bool', 'return', 'list[dict[str, float | int | str | bool]]')):
    predictor = model.initial_predictor.dnn.eval()
    flow = model.dnn.eval()
    results = []
    for steps in steps_list:
        steps = int(steps)
        y_frame = torch.randn(1, 2, 256, 1, device, **('device',))
        predictor_state = predictor.init_state()
        flow_states = [flow.init_state() for _ in range(steps)]
        times_ms = []
        with torch.inference_mode():
            for frame_idx in range(warmup + iterations):
                sync_device(device)
                start = time.perf_counter()
                (e_frame, predictor_state) = _forward_step(predictor, y_frame, predictor_state, use_compiled, **('state', 'use_compiled'))
                x_t = e_frame
                for step_idx in range(steps):
                    t = torch.full((1,), step_idx / max(steps, 1), device, **('device',))
                    dnn_input = torch.cat([
                        x_t,
                        e_frame,
                        y_frame], 1, **('dim',))
                    (v, flow_states[step_idx]) = _forward_step(flow, dnn_input, flow_states[step_idx], t, use_compiled, **('state', 'time_cond', 'use_compiled'))
                    x_t = x_t + v / steps
                sync_device(device)
                if frame_idx >= warmup:
                    times_ms.append((time.perf_counter() - start) * 1000)
        summary = _summarize_ms(times_ms)
        summary.update({
            'mode': 'frame_step_predictor_plus_flow',
            'device': device.type,
            'steps': steps,
            'iterations': iterations,
            'warmup': warmup,
            'compiled': use_compiled,
            'frame_budget_ms': 16,
            'budget_ratio_mean': summary['mean_ms'] / 16 })
        results.append(summary)
    return results


def write_results(path = None, rows = None):
    path = Path(path)
    path.parent.mkdir(True, True, **('parents', 'exist_ok'))
    path.write_text(json.dumps(rows, 2, **('indent',)), 'utf-8', **('encoding',))


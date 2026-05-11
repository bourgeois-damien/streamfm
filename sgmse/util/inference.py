import warnings
import torch
from torch.utils.flop_counter import FlopCounterMode
from torchaudio import functional
from sgmse.util.other import si_sdr
from pesq import pesq
from pystoi import stoi
import distillmos
import numpy as np

# Plotting settings
MAX_VIS_SAMPLES = 10


def my_stft(x, sr):
    n_fft = int(32e-3 * sr)
    hop = int(8e-3 * sr)
    return torch.stft(
        x, n_fft=n_fft, hop_length=hop, center=False, normalized=False, return_complex=True,
        window=torch.hann_window(n_fft).to(x.device)
    )


class EvaluateModel:
    def __init__(self, model, num_eval_files, spec=False, audio=False):
        self.model = model
        self.num_eval_files = num_eval_files
        self.spec = spec
        self.audio = audio
        self.distillmos_models = {}
        self.get_distillmos_model(model.device)

    def get_distillmos_model(self, device):
        if device not in self.distillmos_models:
            distillmos_model = distillmos.ConvTransformerSQAModel()
            distillmos_model.eval().to(device)
            self.distillmos_models[device] = distillmos_model
        return self.distillmos_models[device]

    def __call__(self):
        return evaluate_model(self.model, self.num_eval_files, spec=self.spec, audio=self.audio,
                              distillmos_model=self.get_distillmos_model(self.model.device))

@torch.inference_mode()
def evaluate_model(model, num_eval_files, spec=False, audio=False, distillmos_model=None):
    _pesq, _si_sdr, _estoi, _distillmos = 0., 0., 0., 0.
    _pesq_successful_count = 0
    if spec:
        noisy_spec_list, estimate_spec_list, clean_spec_list = [], [], []
    if audio:
        noisy_audio_list, estimate_audio_list, clean_audio_list = [], [], []

    for i in range(num_eval_files):
        # Load wavs
        x, y, info = model.data_module.valid_set.__getitem__(i, no_crop=True) #d,t

        # pass y through preprocess/postprocess

        x_path = info['x_path']
        y_path = info['y_path']
        sr = info['sr']

        # we overwrite y here with the model.enhance output
        # since model may have modified y for 'simulation' e.g. STFT phase retrieval via post_Y_fn
        result = model.enhance(y, y_sr=sr, return_all=True)

        if len(result) == 5:
            x_hat, _, y, _, _ = result
        else:
            x_hat, _, y, _, _, _, _ = result

        if x_hat.ndim == 1:
            x_hat = x_hat.unsqueeze(0)

        if x.ndim == 1:
            x = x.unsqueeze(0).cpu()
            x_hat = x_hat.unsqueeze(0).cpu()
            y = y.unsqueeze(0).cpu()
        else: #eval only first channel
            x = x[0].unsqueeze(0).cpu()
            x_hat = x_hat[0].unsqueeze(0).cpu()
            y = y[0].unsqueeze(0).cpu()

        # PESQ, ESTOI, SISDR -- first resample to 16kHz for PESQ and ESTOI
        if model.sampling_rate != 16000:
            x_16k = functional.resample(x, model.sampling_rate, 16000, lowpass_filter_width=64)
            x_hat_16k = functional.resample(x_hat, model.sampling_rate, 16000, lowpass_filter_width=64)
        else:
            x_16k, x_hat_16k = x, x_hat
        _si_sdr += si_sdr(x[0].numpy(), x_hat[0].numpy())
        # PESQ with error handling since it may throw
        try:
            _pesq += pesq(16000, x_16k[0].numpy(), x_hat_16k[0].numpy(), 'wb')
        except Exception as e:
            warnings.warn(f'PESQ failed with exception ({e}) for file pair x {x_path} / y {y_path}')
        else:
            _pesq_successful_count += 1

        # STOI
        _estoi += stoi(x_16k[0].numpy(), x_hat_16k[0].numpy(), 16000, extended=True)

        # DistillMOS
        distillmos_sqa = distillmos_model(x_hat_16k.to(model.device))
        _distillmos += distillmos_sqa.item()

        if spec and i < MAX_VIS_SAMPLES:
            y_stft, x_hat_stft, x_stft = my_stft(y[0], sr=sr), my_stft(x_hat[0], sr=sr), my_stft(x[0], sr=sr)
            noisy_spec_list.append(y_stft)
            estimate_spec_list.append(x_hat_stft)
            clean_spec_list.append(x_stft)

        if audio and i < MAX_VIS_SAMPLES:
            noisy_audio_list.append(y[0])
            estimate_audio_list.append(x_hat[0])
            clean_audio_list.append(x[0])

    mean_pesq = _pesq/_pesq_successful_count if _pesq_successful_count > 0 else -1

    if spec:
        if audio:
            return mean_pesq, _si_sdr/num_eval_files, _estoi/num_eval_files, _distillmos/num_eval_files, [noisy_spec_list, estimate_spec_list, clean_spec_list], [noisy_audio_list, estimate_audio_list, clean_audio_list]
        else:
            return mean_pesq, _si_sdr/num_eval_files, _estoi/num_eval_files, _distillmos/num_eval_files, [noisy_spec_list, estimate_spec_list, clean_spec_list], None
    elif audio and not spec:
            return mean_pesq, _si_sdr/num_eval_files, _estoi/num_eval_files, _distillmos/num_eval_files, None, [noisy_audio_list, estimate_audio_list, clean_audio_list]
    else:
        return mean_pesq, _si_sdr/num_eval_files, _estoi/num_eval_files, _distillmos/num_eval_files, None, None



def cuda_timeit(func, *args, n_runs=50, warmup=10, **kwargs):
    """
    Times a function using torch.cuda.Event for GPU operations, similar to %timeit.

    Args:
        func: callable — function to time
        *args, **kwargs: arguments to pass to func
        n_runs: number of timed runs
        warmup: number of warmup runs (not timed)

    Returns:
        avg_time_ms: average execution time per run in milliseconds
    """
    # Warm-up runs (to stabilize GPU performance)
    for _ in range(warmup):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()  # make sure previous kernels finished
        start_event.record()

        func(*args, **kwargs)

        end_event.record()
        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event)  # milliseconds
        times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"Average execution time over {n_runs} runs: {avg_time:.3f} ms +/- {std_time:.3f} ms")
    return avg_time, std_time


def get_flops(model, *args, **kwargs):
    """
    Uses torch FlopCounterMode to count FLOPs of the call `model(*args, **kwargs)`.
    Returns a tuple: (total FLOPs, the `flop_counter` object from FlopCounterMode).
    """
    flop_counter = FlopCounterMode(display=False, depth=None)
    with flop_counter:
        model(*args, **kwargs)
    total_flops = flop_counter.get_total_flops()
    return total_flops, flop_counter

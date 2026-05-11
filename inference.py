import os
import glob
import tqdm
import multiprocessing as mp
from math import ceil

import hydra
from hydra.utils import instantiate
import omegaconf

import numpy as np
import torchaudio
import torch
import torchinfo

from sgmse.model import CustomRKSolverEnhancementModel, DiscriminativeModel


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def list_to_lower_triangular(entries, q):
    expected_len = q * (q - 1) // 2
    if len(entries) != expected_len:
        raise ValueError(f"Expected {expected_len} entries for a {q}x{q} lower triangular matrix, got {len(entries)}")

    mat = np.zeros((q, q))
    idx = 0
    for i in range(1, q):
        for j in range(i):
            mat[i, j] = entries[idx]
            idx += 1
    print(f"Converted {len(entries)} entries to a {q}x{q} lower triangular matrix.")
    print(f"Input entries: {entries}")
    print(f"Lower triangular matrix:\n{mat}")
    return mat


@hydra.main(config_path="./config/", version_base="1.3")
def main(cfg: omegaconf.DictConfig) -> None:
    assert hasattr(cfg, "ckpt"), "Must pass a model checkpoint! Use +ckpt=... to specify a checkpoint."
    assert hasattr(cfg, "inpath"), "Must pass an inpath! Use +inpath=... to specify the path."
    assert hasattr(cfg, "outpath"), "Must pass an inpath! Use +outpath=... to specify the path."
    nested_dirs = any(os.path.isdir(os.path.join(cfg.inpath, d)) for d in os.listdir(cfg.inpath))
    print(f"Detected nested dirs: {nested_dirs}")

    init_pred_only = cfg.get("init_pred_only", False)
    apply_corruption = cfg.get("apply_corruption", True)

    assert torch.cuda.is_available(), "Must have a GPU available for inference."
    ngpu = getattr(cfg, "gpus", 1)

    # Create output dir
    os.makedirs(cfg.outpath, exist_ok=True)

    # Find all files
    if nested_dirs:
        allfiles = glob.glob(os.path.join(cfg.inpath, "**", "*.wav"))
    else:
        allfiles = glob.glob(os.path.join(cfg.inpath, "*.wav"))
    print("Found {} files to process.".format(len(allfiles)))

    if ngpu > 1:
        # Split list
        gpu_lists = []
        n_files_per_gpu = ceil(len(allfiles) / ngpu)

        chunker = chunks(allfiles, n_files_per_gpu)
        for i in range(ngpu):
            gpu_lists.append(next(chunker))

        # Start processes
        processes = [mp.Process(target=enhance_on_gpu, args=(cfg, i, gpu_lists[i], nested_dirs, init_pred_only, apply_corruption)) for i in range(ngpu)]
        for process in processes:
            process.start()

        for process in processes:
            process.join()
    else:
        enhance_on_gpu(cfg, 0, allfiles, nested_dirs, init_pred_only, apply_corruption)


# This runs on separate subprocess per GPU
@torch.inference_mode()
def enhance_on_gpu(cfg, device_id, noisy_files, nested_dirs, init_pred_only=False, apply_corruption=True):
    device = f"cuda:{device_id}"

    # Initialize model from config
    if hasattr(cfg, 'solver_model'):
        wrapped_model = instantiate(cfg.model)
        model = instantiate(cfg.solver_model, wrapped_model=wrapped_model).to(device)
    else:
        model = instantiate(cfg.model)
    # import torchinfo; print(torchinfo.summary(model, depth=2))
    # import sys; sys.exit(0)

    if hasattr(cfg, 'transform_model_backbone_fn'):
        print("Applying transform_model_backbone_fn to model backbone...")
        transform_fn = instantiate(cfg.transform_model_backbone_fn)
        model.transform_backbone_(transform_fn)

    # Set up some global torch options
    torch.autograd.set_detect_anomaly(False)
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    torchaudio.set_audio_backend("ffmpeg")

    # Load checkpoint
    ckpt = torch.load(cfg.solver_ckpt if hasattr(cfg, 'solver_ckpt') else cfg.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    if (ip_ckpt_file := getattr(cfg, "override_ip_ckpt", None)) is not None:
        print("Using overridden IP checkpoint from", ip_ckpt_file)
        ip_ckpt = torch.load(ip_ckpt_file, map_location="cpu", weights_only=False)
        if isinstance(model, CustomRKSolverEnhancementModel):
            ip = model.wrapped_model.initial_predictor
        else:
            ip = model.initial_predictor
        ip.load_state_dict(ip_ckpt["state_dict"])

    model = model.eval().to(device)

    print(torchinfo.summary(model, depth=0))

    if init_pred_only:
        print("Using initial predictor only.")
        assert hasattr(model, 'initial_predictor')

    if isinstance(model, CustomRKSolverEnhancementModel) and not hasattr(cfg, "solver"):
        print("Using custom RK solver stored on model config")
        solver = 'rk'
        a = model.a.detach().cpu().numpy()
        b = model.b.detach().cpu().numpy()
        c = model.c.detach().cpu().numpy()
        solver_args = {
            'a': np.array(a),
            'b': np.array(b),
            'c': np.array(c),
            'euler_last': False
        }
        N = 1
        print(solver_args)
    else:
        if isinstance(model, CustomRKSolverEnhancementModel) and hasattr(cfg, "solver"):
            print("Overriding learned custom RK solver from checkpoint, enhancing via wrapped_model instead...")
            model = model.wrapped_model

        solver = cfg.solver
        N = cfg.N if hasattr(cfg, "N") else 1
        solver_b = cfg.solver_b if hasattr(cfg, "solver_b") else None
        if solver_b is not None:
            q = len(solver_b)
            solver_args = {
                'a': list_to_lower_triangular(cfg.solver_a, q=q),
                'b': np.array(cfg.solver_b),
                'c': np.array(cfg.solver_c),
                'euler_last': cfg.get('euler_last', False)
            }
        else:
            solver_args = {}

        if hasattr(cfg, "alpha"):  # for generic 2nd order method
            assert solver == 'generic2', 'alpha option only valid for generic2 solver'
            q = 2
            solver_args['alpha'] = cfg.alpha

    np.random.seed(0)
    torch.manual_seed(0)
    torch.random.manual_seed(0)

    def get_outdir(outpath, file):
        if nested_dirs:
            outdir = os.path.join(outpath, file.split('/')[-2])
        else:
            outdir = outpath
        return outdir

    for file in tqdm.tqdm(list(sorted(noisy_files)), position=device_id+1):
        outdir = get_outdir(cfg.outpath, file)
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, os.path.basename(file))

        if os.path.isfile(outpath):
            continue

        y, y_sr = torchaudio.load(file)

        if y.shape[0] > 1:
            print("[!] Audio file has more than one channel, using only the first one")
            y = y[0:1]

        if (target_sr := getattr(cfg, "force_lowpass_resample_to", None)) is not None:
            y = torchaudio.functional.resample(y, y_sr, target_sr, lowpass_filter_width=64)
            y = torchaudio.functional.resample(y, target_sr, y_sr, lowpass_filter_width=64)  # embed back into original SR

        if y_sr != model.sampling_rate and getattr(cfg, "resample", True):
            y = torchaudio.functional.resample(y, y_sr, model.sampling_rate, lowpass_filter_width=64)
            y_sr = model.sampling_rate

        if apply_corruption and (y_corr := getattr(model.data_module, 'y_corruption')) is not None:
            y = y_corr(y)
            outdir_corrupted = get_outdir(cfg.outpath + "_corrupted", file)
            outpath_corrupted = os.path.join(outdir_corrupted, os.path.basename(file))
            os.makedirs(outdir_corrupted, exist_ok=True)
            torchaudio.save(outpath_corrupted, y, sample_rate=y_sr)

        if init_pred_only:
            print("Used initial predictor only for file {}".format(file))
            xhat = model.initial_predictor.enhance(y, y_sr)
            torchaudio.save(outpath, xhat.cpu(), sample_rate=y_sr)
        elif isinstance(model, DiscriminativeModel):
            xhat = model.enhance(y, y_sr)
        else:
            xhat = model.enhance(
                y, y_sr, solver=solver, N=N, solver_args=solver_args, return_all=False,
            )
        torchaudio.save(outpath, xhat, sample_rate=y_sr)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()

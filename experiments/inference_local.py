import glob
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import omegaconf
import torch
import torchaudio
import tqdm

from experiments.core.devices import select_torch_device
from sgmse.model import CustomRKSolverEnhancementModel, DiscriminativeModel


def select_device(name: str | None) -> torch.device:
    """Resolve a user device name to the best available torch device."""
    return select_torch_device(name)


def load_config_from_cli(argv: list[str]) -> omegaconf.DictConfig:
    """Parse Hydra-like CLI arguments and compose the selected config."""
    config_name = "streamfm_stftpr"
    config_path = REPO_ROOT / "config"
    overrides = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--config-name":
            i += 1
            config_name = argv[i]
        elif arg.startswith("--config-name="):
            config_name = arg.split("=", 1)[1]
        elif arg == "--config-path":
            i += 1
            config_path = Path(argv[i]).expanduser()
        elif arg.startswith("--config-path="):
            config_path = Path(arg.split("=", 1)[1]).expanduser()
        else:
            overrides.append(arg)
        i += 1

    config_dir = config_path if config_path.is_absolute() else (Path.cwd() / config_path).resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name=config_name, overrides=overrides)


def run(cfg: omegaconf.DictConfig) -> None:
    """Load a checkpoint and enhance all wav files from the input folder."""
    assert hasattr(cfg, "ckpt"), "Pass +ckpt=... with the checkpoint path."
    assert hasattr(cfg, "inpath"), "Pass +inpath=... with a folder containing .wav files."
    assert hasattr(cfg, "outpath"), "Pass +outpath=... with the output folder."

    device = select_device(cfg.get("device", "auto"))
    print(f"Using device: {device}")

    nested_dirs = any(os.path.isdir(os.path.join(cfg.inpath, d)) for d in os.listdir(cfg.inpath))
    pattern = os.path.join(cfg.inpath, "**", "*.wav") if nested_dirs else os.path.join(cfg.inpath, "*.wav")
    allfiles = sorted(glob.glob(pattern, recursive=nested_dirs))
    print(f"Detected nested dirs: {nested_dirs}")
    print(f"Found {len(allfiles)} files to process.")

    os.makedirs(cfg.outpath, exist_ok=True)

    if hasattr(cfg, "solver_model"):
        wrapped_model = instantiate(cfg.model)
        model = instantiate(cfg.solver_model, wrapped_model=wrapped_model)
    else:
        model = instantiate(cfg.model)

    if hasattr(cfg, "transform_model_backbone_fn"):
        transform_fn = instantiate(cfg.transform_model_backbone_fn)
        model.transform_backbone_(transform_fn)

    ckpt_path = cfg.solver_ckpt if hasattr(cfg, "solver_ckpt") else cfg.ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.eval().to(device)

    if isinstance(model, CustomRKSolverEnhancementModel) and not hasattr(cfg, "solver"):
        solver = "rk"
        solver_args = {
            "a": model.a.detach().cpu().numpy(),
            "b": model.b.detach().cpu().numpy(),
            "c": model.c.detach().cpu().numpy(),
            "euler_last": False,
        }
        n_steps = 1
    else:
        if isinstance(model, CustomRKSolverEnhancementModel) and hasattr(cfg, "solver"):
            model = model.wrapped_model
        solver = cfg.solver
        n_steps = cfg.N if hasattr(cfg, "N") else 1
        solver_args = {}

    torch.manual_seed(cfg.get("seed", 0))

    def output_dir_for(path: str) -> str:
        """Mirror one nested input level in the output folder when needed."""
        if nested_dirs:
            return os.path.join(cfg.outpath, os.path.basename(os.path.dirname(path)))
        return cfg.outpath

    with torch.inference_mode():
        for wav_path in tqdm.tqdm(allfiles):
            outdir = output_dir_for(wav_path)
            os.makedirs(outdir, exist_ok=True)
            outpath = os.path.join(outdir, os.path.basename(wav_path))
            if os.path.isfile(outpath):
                continue

            y, y_sr = torchaudio.load(wav_path)
            if y.shape[0] > 1:
                print(f"[!] {wav_path} has more than one channel; using the first one.")
                y = y[0:1]

            y = y.to(device)
            if y_sr != model.sampling_rate and cfg.get("resample", True):
                y = torchaudio.functional.resample(y, y_sr, model.sampling_rate, lowpass_filter_width=64)
                y_sr = model.sampling_rate

            if isinstance(model, DiscriminativeModel):
                xhat = model.enhance(y, y_sr)
            else:
                xhat = model.enhance(y, y_sr, solver=solver, N=n_steps, solver_args=solver_args)

            torchaudio.save(outpath, xhat.detach().cpu(), sample_rate=y_sr)


if __name__ == "__main__":
    run(load_config_from_cli(sys.argv[1:]))

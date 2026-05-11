import os
import sys
import argparse
import math
import numpy as np
import torch
from hydra import initialize, compose
from hydra.utils import instantiate
import tqdm

from sgmse.feature_extractors import CompressedAmplitudeComplexSTFT

def rreplace(s, old, new, occurrence=1):
    li = s.rsplit(old, occurrence)
    return new.join(li)

def t2n(x):
    return x.detach().cpu().numpy()

def abs_quantile(x, q):
    return np.quantile(np.abs(x).reshape(-1), q)

def get_feats(model, batch_x, batch_y, is_storm=False):
    Y, X, y, x, preprocess_info = model.preprocess(batch_y, y_sr=model.data_module.sampling_rate, x=batch_x)

    if is_storm:
        E = model.initial_predictor(Y)
        return X, E

    return X, Y


@torch.inference_mode
def main():
    try:
        initialize(config_path="config", version_base="1.3")  # Adjust path as needed
    except ValueError:
        print("Yeah hydra is already initialized probably")

    parser = argparse.ArgumentParser(description="Estimate parameters for flow")
    parser.add_argument("--config-name", type=str, required=True, help="Name of the config file to use")
    parser.add_argument("--train-path", type=str, required=False, help="(optional) override for the training data path in the config file")
    parser.add_argument("--n-samples", type=int, default=2500)
    parser.add_argument("--beta1", action='store_true', help="Force beta to 1.0 instead of estimating it. Still estimates sigma_y.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--qx", type=float, default=0.997,
        help="Quantile of the global distribution of clean audio x that we will rescale to 1.0 (via *beta). 0.997 by default."
    )
    parser.add_argument(
        "--qrmse", type=float, default=0.997,
        help="Quantile of the RMSEs of clean x vs coded y that can be used as a reasonable default sigma_y. 0.997 by default."
    )
    parser.add_argument("--per-band", action='store_true', help="Pass to calculate frequency-dependent sigma_y. The --qrmse value is then used for each frequency band separately to determine a per-band sigma_y.")
    parser.add_argument("--outfile-suffix", type=str, required=False, help="Suffix to append to the results filename")
    parser.add_argument("--overwrite", action='store_true', help="If passed, will recalculate and overwrite results file even if it exists")
    parser.add_argument("--device", type=int, default=0, help="Index of CUDA device to use for feature extraction")
    args = parser.parse_args()

    # Prepare config and overrides
    cfg = compose(config_name=args.config_name)
    cfg['model']['feature_extractor']['beta'] = 1.0  # Set beta to 1.0 for feature extraction
    if args.train_path is not None:
        cfg['model']['data_module']['train_path'] = args.train_path

    is_storm = False
    if cfg['model']['_target_'] == 'sgmse.model.StoRMOnTheFlyFlowModel':
        # StoRM mode
        is_storm = True
        ip_cfg_path = cfg['model']['initial_predictor_config_path']
        ip_cfg_path = os.path.join(os.path.dirname(__file__), "config", os.path.basename(ip_cfg_path))
        cfg['model']['initial_predictor_config_path'] = ip_cfg_path

    # Initialize model and datamodule
    model = instantiate(cfg.model)
    model = model.eval()
    model = model.to(args.device)
    datamodule = model.data_module
    datamodule.setup(stage='_train_only')
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed + 1)
    dataloader = datamodule.train_dataloader()
    print("DataLoader batch size:", dataloader.batch_size, file=sys.stderr)

    # Prep output file
    outfile_suffix = ''
    outfile_suffix += f'_n{args.n_samples}' if args.n_samples != 2500 else ''
    outfile_suffix += f'_qx{args.qx:.3f}' if args.qx != 0.997 else ''
    outfile_suffix += f'_qrmse{args.qrmse:.3f}' if args.qrmse != 0.997 else ''
    outfile_suffix += '_perband' if args.per_band else ''
    outfile_suffix += f'_{args.outfile_suffix}' if args.outfile_suffix is not None else ''
    outfile_dir = os.path.join("flow_autoparams", args.config_name)
    os.makedirs(outfile_dir, exist_ok=True)
    outfile_path = os.path.join(outfile_dir, f"autoparams{outfile_suffix}.txt")
    if os.path.isfile(outfile_path) and not args.overwrite:
        print("Output file exists, printing its contents:", file=sys.stderr)
        with open(outfile_path, 'r') as f:
            for line in f:
                print(line.rstrip("\n"))
        sys.exit(0)

    # Run the feature extraction and quantile calculations
    print("Running...", file=sys.stderr)
    all_bins_x = []
    all_feats_x = []
    all_feats_y = []
    post_Y_fn = getattr(model, 'post_Y_fn', None)

    drawn_samples = 0
    for batch in tqdm.tqdm(dataloader, desc="Processing batches", total=int(math.ceil(args.n_samples / dataloader.batch_size))):
        batch_x, batch_y, _ = batch  # ignore extra info dict
        if not isinstance(batch_x, torch.Tensor) or not isinstance(batch_y, torch.Tensor):
            raise TypeError("Expected batch_x and batch_y to be torch tensors, got: {}, {}".format(type(batch_x), type(batch_y)))

        batch_x_feats, batch_y_feats = get_feats(model, batch_x, batch_y, is_storm=is_storm)
        batch_y_feats = batch_y_feats if post_Y_fn is None else post_Y_fn(batch_y_feats)
        all_bins_x.append(batch_x_feats.reshape(-1))
        all_feats_x.append(batch_x_feats)
        all_feats_y.append(batch_y_feats)

        drawn_samples += batch_x_feats.shape[0]
        if drawn_samples >= args.n_samples:
            print(f"Reached {drawn_samples} samples, stopping feature extraction.", file=sys.stderr)
            break

    all_bins_x = torch.cat(all_bins_x, dim=0)
    all_feats_x = torch.cat(all_feats_x, dim=0)
    all_feats_y = torch.cat(all_feats_y, dim=0)

    abs_quantile_x = abs_quantile(t2n(all_bins_x), args.qx)

    if args.beta1:
        print("Beta=1.0 forced by argument, skipping beta estimation and using beta=1.0 for sigma_y calculation.", file=sys.stderr)
        suggested_beta = 1.0
    else:
        suggested_beta = 1/abs_quantile_x  # need this here since it affects sigma_y

    spec_diffs = [(afy_ - afx_) for (afy_, afx_) in zip(all_feats_y, all_feats_x)]

    # Per-band RMSE calculation
    rmses_per_band = np.array([
        torch.linalg.norm(diff.squeeze(), ord=2, dim=-1).cpu().numpy() / diff.shape[-2]**0.5
        for diff in spec_diffs
    ])
    rmse_quantile_per_band = np.quantile(rmses_per_band, args.qrmse, axis=0)
    per_band_outfile_path = rreplace(outfile_path, '.txt', 'sigy_perband.npy')

    suggested_sigma_y_per_band = suggested_beta * rmse_quantile_per_band / 3
    print(f"Writing resulting per_band sigma_y to {per_band_outfile_path}", file=sys.stderr)
    np.save(per_band_outfile_path, suggested_sigma_y_per_band)
    # Global RMSE calculation
    rmses = np.array([
        torch.linalg.norm(diff.reshape(-1), ord=2).cpu().item() / diff.numel()**0.5
        for diff in spec_diffs
    ])
    rmse_quantile = np.quantile(rmses, args.qrmse)
    suggested_sigma_y = suggested_beta * rmse_quantile / 3

    # Output results
    print(f"Writing results to {outfile_path}", file=sys.stderr)
    with open(outfile_path, "w") as f:
        for stream in (sys.stdout, f):
            print(f"Config name: {args.config_name}", file=stream)
            if isinstance(model.feature_extractor, CompressedAmplitudeComplexSTFT):
                print(f"Config alpha = {model.feature_extractor.alpha}")
            print(f"Args: {args}", file=stream)
            print(f"=== Results ===", file=stream)
            print(f"   \tq{args.qx}( |x|  ) = {abs_quantile_x:.3f}, max( |x|  ) = {all_bins_x.abs().max():.3f}", file=stream)
            print(f"   \tq{args.qrmse}( RMSE ) = {rmse_quantile:.3f}, max( RMSE ) = {np.max(rmses):.3f}", file=stream)
            print(f"-->\tbeta={suggested_beta:.2f}, sigma_y={suggested_sigma_y:.2f}", file=stream)
            print(f"-->\tsigma_y_perband written to {per_band_outfile_path}", file=stream)



if __name__ == '__main__':
    main()

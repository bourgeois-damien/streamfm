import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

def sigma_y_from_file(file, smoothing: None | int | float = None, factor: float = 1.0):
    sigy = np.load(file)
    if sigy.ndim == 1:
        # always broadcast along time
        sigy = sigy[:, np.newaxis]

    assert sigy.shape[-1] == 1, f"expected sigma_y to be constant along time but found T={sigy.shape[-1]}"
    assert sigy.ndim >= 2, f"{file} has ndim={sigy.ndim}"

    if smoothing is not None and smoothing > 0:
        # Apply Gaussian smoothing along second-to-last dimension
        axis = -2
        sigy = gaussian_filter1d(sigy, sigma=smoothing, axis=axis, mode='nearest')

    sigy = factor * torch.from_numpy(sigy).float()
    return sigy

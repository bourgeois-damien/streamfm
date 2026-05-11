import numpy as np
import torch


# for lsd
stft_kwargs = {"n_fft": 510, "hop_length": 128, "window": torch.hann_window(510), "return_complex": True}

def lsd(s_hat, s, eps=1e-10):
    S_hat, S = torch.stft(torch.from_numpy(s_hat), **stft_kwargs), torch.stft(torch.from_numpy(s), **stft_kwargs)
    logPowerS_hat, logPowerS = 2*torch.log(eps + torch.abs(S_hat)), 2*torch.log(eps + torch.abs(S))
    return torch.mean( torch.sqrt(torch.mean(torch.abs( logPowerS_hat - logPowerS ))) ).item()


def si_sdr_components(s_hat, s, n, eps=1e-10):
    # s_target
    alpha_s = np.dot(s_hat, s) / (eps + np.linalg.norm(s)**2)
    s_target = alpha_s * s

    # e_noise
    alpha_n = np.dot(s_hat, n) / (eps + np.linalg.norm(n)**2)
    e_noise = alpha_n * n

    # e_art
    e_art = s_hat - s_target - e_noise

    return s_target, e_noise, e_art


def energy_ratios(s_hat, s, n, eps=1e-10):
    """
    """
    s_target, e_noise, e_art = si_sdr_components(s_hat, s, n)

    si_sdr = 10*np.log10(eps + np.linalg.norm(s_target)**2 / (eps + np.linalg.norm(e_noise + e_art)**2))
    si_sir = 10*np.log10(eps + np.linalg.norm(s_target)**2 / (eps + np.linalg.norm(e_noise)**2))
    si_sar = 10*np.log10(eps + np.linalg.norm(s_target)**2 / (eps + np.linalg.norm(e_art)**2))

    return si_sdr, si_sir, si_sar


def si_sdr(s, s_hat):
    alpha = np.dot(s_hat, s)/np.linalg.norm(s)**2
    sdr = 10*np.log10(np.linalg.norm(alpha*s)**2/np.linalg.norm(
        alpha*s - s_hat)**2)
    return sdr

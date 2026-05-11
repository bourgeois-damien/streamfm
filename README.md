# Stream.FM & MelFlow: Real-Time Streamable Generative Speech Restoration with Flow Matching

This repository contains the official PyTorch implementation for our works

- [1] Stream.FM aka [*Real-Time Streamable Generative Speech Restoration with Flow Matching*](https://arxiv.org/abs/2512.19442), preprint, arXiv, 2026.
- [2] MelFlow aka [*Real-Time Streaming Mel Vocoding with Generative Flow Matching*](https://arxiv.org/abs/2509.15085), IEEE ICASSP 2026.

📖 For the full citations, see the end of this README.

🔊 On our [Stream.FM project page](https://sp-uhh.github.io/streamfm_examples/), you can find audio examples and our [demo video](https://www.youtube.com/watch?v=ezXzPia3EVs).


## Note on MelFlow [2]

The Stream.FM paper [1] expands upon the MelFlow paper [2], see the description of additional contributions in [1].
Since the underlying methodology (model, inference, ...) is the same, we provide code for both works in this single codebase.

The training configuration described in the MelFlow paper [2] can be found in `config/melflow_original.yaml`.
However, we recommend to instead use `config/streamfm_melflow.yaml` matching the description in [1]. The differences are that `streamfm_melflow.yaml` adds gradient clipping, and uses only 2 instead of 4 GPUs, and `streamfm_melflow.yaml` trains for 150k steps to be aligned with the other restoration task configs. The Mel vocoding checkpoint we provide was trained using `streamfm_melflow.yaml`, matching [1].


## Installation

- Create a new virtual environment with Python 3.10 or newer (we have not tested other Python versions, but they may work). Example with Conda (miniforge3):
  ```bash
  conda create -n streamfm python=3.10 -c conda-forge
  ```
- Activate your virtual environment, e.g. `conda activate streamfm`
- Install the dependencies via
  ```bash
  pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
  ```
  where you should replace the `--extra-index-url` parameter to match your local CUDA installation, see [the PyTorch instructions](https://pytorch.org/get-started/locally/).


## Pretrained checkpoints

You can find the Stream.FM model checkpoints in our [Google Drive folder](https://drive.google.com/drive/folders/1u2QKjGAdxblQVV8-qmifSM9AwhT86LER?usp=sharing).
To download all provided checkpoints to the `checkpoints/` folder, you can use:
```bash
pip install gdown  # if not already installed in your virtual environment
gdown https://drive.google.com/drive/folders/1u2QKjGAdxblQVV8-qmifSM9AwhT86LER?usp=sharing -O checkpoints --folder
```


## Training

* Training is done by executing `train.py` which uses [Hydra](https://hydra.cc/) configs, see the `config/` folder.
* First, plase check and modify `config/_paths.yaml` to refer to your local dataset paths.
* A minimal running example with default settings, e.g., for bandwidth extension, can then be run with:
  ```bash
  python train.py --config-name streamfm_bwe
  ```
  which refers to the `config/streamfm_bwe.yaml` config file.
* Note that for the config files starting with `LRK`, for fitting a learned Runge-Kutta scheme as described in [1], you must use the `fit_rk_scheme.py` script instead, e.g.,
  ```bash
  python fit_rk_scheme.py --config-name LRK5_streamfm_bwe
  ```


## Inference

To run (offline) inference on your data, see `inference.py`. This needs at least the extra Hydra config keys `inpath`, `outpath`, `solver`, and `ckpt`.
The script expects `inpath` to be a folder containing .wav files, and will reproduce the same input filenames for the enhanced files in the `outpath` folder.

An example call for speech enhancement (make sure to modify the `inpath`):
```bash
python inference.py --config-name streamfm_se_predgen +inpath=EARS-WHAM_v2_16k/test/noisy/ +outpath=enhancement_results/test-se/ +ckpt=checkpoints/streamfm_se_predgen.ckpt +solver=5xeuler +gpus=2
```
* You can parallelize the inference over your available GPUs using e.g. `+gpus=2`.
* For solver variants and settings, see the file `sgmse/util/solvers.py`.

### Mel Vocoding inference
Typically in this repo we "simulate" the process of Mel vocoding by taking a clean input audio, mapping it to a Mel spectrogram, and then using our model to map this back to the time domain.
To actually run inference from Mel spectra directly, you should do something like:
```python
model = ... # some Mel vocoder sgmse.model.FlowModel instance
from sgmse.util.diffphase import PhaselessMelAndBack; assert isinstance(model.post_Y_fn, PhaselessMelAndBack)  # sanity check for model instance
input_mel = ... # your Mel spectrogram, **must** match the configuration that `model` was trained with
# Project input_mel back to STFT space, using Mel pseudoinverse and zero-phase real-in-complex embedding, see the paper
Y = torch.matmul(model.post_Y_fn.pseudoinverse.T, input_mel).abs()**model.post_Y_fn.alpha + 0j
# Run model directly on Y
Xhat = model.enhance_from_features(Y, solver='euler', N=5)
```
which can be adapted to streaming inference (see below) by applying the pseudoinverse on each incoming frame.

### Streaming inference

* To perform streaming (frame-by-frame) inference, refer to the `init_state()` and `forward_step(x, state)` functions of the `sgmse.backbones.streaming_unet.CausalNCSNpp` class. For improved speed, consult the supplementary material of [the Stream.FM paper](https://arxiv.org/abs/2512.19442), particularly the section "MODEL IMPLEMENTATION AND OPTIMIZATION". We especially recommend the use of CUDA graphs as described there.
* Note that the default `forward_step()` function of our backbone is already decorated with the recommended `torch.compile` wrapper:
  ```python
  @torch.compile(fullgraph=True, options={
      'max_autotune': True, 'epilogue_fusion': True, 'shape_padding': True,
  })
  ```
  but you may want to modify or remove this decorator, depending on your goals and your hardware.
* If you wish to adapt your own backbone DNN for this streaming type of inference within this repo, see the abstract class `sgmse.backbones.streaming_unet.CausalStreamingModule` that your DNN should implement.


## Citations / References

We kindly ask you to cite our papers in your publication when using any of our research or code:

```bib
@article{welker2025streamfm,
  title={Real-Time Streamable Generative Speech Restoration with Flow Matching},
  author={Simon Welker and Bunlong Lay and Maris Hillemann and Tal Peer and Timo Gerkmann},
  year={2025},
  journal={arXiv preprint arXiv:2512.19442},
  doi={10.48550/arXiv.2512.19442}
}

@inproceedings{welker2026melflow,
  title={Real-Time Streaming Mel Vocoding with Generative Flow Matching},
  author={Welker, Simon and Peer, Tal and Gerkmann, Timo},
  booktitle={IEEE Int. Conf. on Acoustics, Speech and Sig. Process. (ICASSP)},
  year={2026},
  organization={IEEE}
}
```

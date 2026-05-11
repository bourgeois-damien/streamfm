"""
Little helper script to print the learned Runge-Kutta scheme coefficients from a trained checkpoint
in readable/copyable formats.
"""

import hydra
from hydra.utils import instantiate
import omegaconf
import torch


def nospacelist(l):
    return '[' + ','.join([f"{x:.3f}" for x in l]) + ']'


def display_tex_float(x):
    if x != 0:
        return f'{x:.3f}'
    else:
        return '0'

def vector_as_tex(vec):
    return ', & '.join(f'{display_tex_float(val)}' for val in vec)

def matrix_as_tex(mat):
    lines = (' & '.join(f'{display_tex_float(val)}' for val in row) for row in mat)
    return ' \\\\\n'.join(lines)


@hydra.main(config_path="./config/", version_base="1.3")
def main(cfg: omegaconf.DictConfig) -> None:
    #assert hasattr(cfg, 'ckpt'), "cfg must contain 'ckpt' key"
    assert hasattr(cfg, 'solver_ckpt'), "cfg must contain 'solver_ckpt' key"

    wrapped_model = instantiate(cfg.model)
    # skip wrapped_model ckpt loading, as we don't need to instantiate the wrapped model weights for this

    solver_model = instantiate(cfg.solver_model, wrapped_model=wrapped_model)
    ckpt = torch.load(cfg.solver_ckpt, map_location="cpu", weights_only=False)
    solver_model.load_state_dict(ckpt["state_dict"])

    a_raw = solver_model.a_raw.detach().cpu().numpy()
    b_raw = solver_model.b_raw.detach().cpu().numpy()
    c_raw = solver_model.c_raw.detach().cpu().numpy()
    a = solver_model.a.detach().cpu().numpy()
    b = solver_model.b.detach().cpu().numpy()
    c = solver_model.c.detach().cpu().numpy()

    # print lower triangular entries of `a` as a list
    a_list = []
    for i in range(a.shape[0]):
        for j in range(i):
            a_list.append(float(a[i, j]))
    print("-"*80)
    print(f"RK scheme from {cfg.solver_ckpt}:")
    print(f"  a_raw: {a_raw}")
    print(f"  b_raw: {b_raw}")
    print(f"  c_raw: {c_raw}")
    print(f"  a: {a}")
    print(f"  b: {b}")
    print(f"  c: {c}")
    print("-"*80)
    print(f"+solver_a={nospacelist(a_list)}", end=' ')
    print(f"+solver_b={nospacelist(b.tolist())}", end=' ')
    print(f"+solver_c={nospacelist(c.tolist())}")
    print("-"*80)
    print(r"\mathbf A &= \begin{bmatrix}")
    print(matrix_as_tex(a))
    print(r"\end{bmatrix} \\")
    print(r"\mathbf b &= \begin{bmatrix}")
    print(vector_as_tex(b))
    print(r"\end{bmatrix} \\")
    print(r"\mathbf c &= \begin{bmatrix}")
    print(vector_as_tex(c))
    print(r"\end{bmatrix} \\")



if __name__ == "__main__":
    main()
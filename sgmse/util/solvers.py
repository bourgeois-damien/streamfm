import torch
from typing import Optional, Sequence
import numpy as np


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


def get_butcher_tableau(method):
    # ------------------------
    # Composite DSL
    # ------------------------
    if '+' in method or 'x' in method:
        return _get_composite_tableau(method)

    # 3-tuples of (a,b,c)
    elif method == 'euler': return (
        torch.tensor([[0]], dtype=torch.float32),
        torch.tensor([1], dtype=torch.float32),
        torch.tensor([0], dtype=torch.float32),
    )
    elif method == 'midpoint': return (
        torch.tensor([[0,    0],
                      [1/2,  0]], dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.float32),
        torch.tensor([0, 1/2], dtype=torch.float32),
    )
    elif method == 'ralston2': return (
        torch.tensor([[0,    0],
                      [2/3,  0]], dtype=torch.float32),
        torch.tensor([1/4, 3/4], dtype=torch.float32),
        torch.tensor([0, 2/3], dtype=torch.float32),
    )
    elif method.startswith('generic2-'):
        alpha = float(method.split('-')[1])
        assert 0 < alpha <= 1
        return (
            torch.tensor([[0,     0],
                          [alpha, 0]], dtype=torch.float32),
            torch.tensor([1-1/(2*alpha), 1/(2*alpha)], dtype=torch.float32),
            torch.tensor([0, alpha], dtype=torch.float32),
        )
    elif method == 'heun3': return (
        torch.tensor([[0,     0,  0],
                      [1/3,   0,  0],
                      [0,   2/3,  0]], dtype=torch.float32),
        torch.tensor([1/4, 0, 3/4], dtype=torch.float32),
        torch.tensor([0, 1/3, 2/3], dtype=torch.float32),
    )
    elif method == 'ralston3': return (
        torch.tensor([[0,     0,  0],
                      [1/2,   0,  0],
                      [0,   3/4,  0]], dtype=torch.float32),
        torch.tensor([2/9, 1/3, 4/9], dtype=torch.float32),
        torch.tensor([0, 1/2, 3/4], dtype=torch.float32),
    )
    elif method == 'ssprk3': return (
        torch.tensor([[0,     0,  0],
                      [1,     0,  0],
                      [1/4, 1/4,  0]], dtype=torch.float32),
        torch.tensor([1/6, 1/6, 2/3], dtype=torch.float32),
        torch.tensor([0, 1, 1/2], dtype=torch.float32),
    )
    elif method == 'rk4_3over8': return (
        torch.tensor([[0,      0,   0, 0],
                      [1/3,    0,   0, 0],
                      [-1/3,   1,   0, 0],
                      [1,     -1,   1, 0]], dtype=torch.float32),
        torch.tensor([1/8, 3/8, 3/8, 1/8], dtype=torch.float32),
        torch.tensor([0, 1/3, 2/3, 1], dtype=torch.float32),
    )
    elif method == 'rk4': return (
        torch.tensor([[0,    0,    0,   0],
                      [1/2,  0,    0,   0],
                      [0,    1/2,  0,   0],
                      [0,    0,    1,   0]], dtype=torch.float32),
        torch.tensor([1/6, 1/3, 1/3, 1/6], dtype=torch.float32),
        torch.tensor([0, 1/2, 1/2, 1], dtype=torch.float32),
    )
    # this one is a nonstandard stage-4 method that avoids c_i = 1 for all i
    # it should have order 3 and better stability than e.g. heun3
    elif method == 'rk4_order3_custom': return (
        torch.tensor([[0,     0,    0,   0],
                      [1/3,   0,    0,   0],
                      [1/3, 1/3,    0,   0],
                      [0,     0,  2/3,   0]], dtype=torch.float32),
        torch.tensor([1/4, 0, 1/2, 1/4], dtype=torch.float32),
        torch.tensor([0, 1/3, 2/3, 2/3], dtype=torch.float32),
    )
    elif method == 'rk5_ralston3_ralston2':
        a = 0.55
        return (
            torch.tensor([[0,       0, 0, 0, 0],
                          [a/2,     0, 0, 0, 0],
                          [0,   a*3/4, 0, 0, 0],
                          [a*2/9, a*1/3, a*4/9, 0, 0],
                          [a*2/9, a*1/3, a*4/9, (1-a)*2/3, 0]], dtype=torch.float32),
            torch.tensor([a*2/9, a*1/3, a*4/9, (1-a)/4, 3*(1-a)/4], dtype=torch.float32),
            torch.tensor([0, a/2, a*3/4, a, 2/3 + a/3], dtype=torch.float32),
        )
    raise ValueError(f"Unknown Runge-Kutta method {method}.")


def _get_composite_tableau(dsl):
    """
    DSL grammar:
        block := [N 'x'] method ['@' tau]
        dsl   := block ('+' block)*

    Semantics:
        - tau is fraction of total dt
        - if tau omitted, remaining time is shared equally
    """

    blocks = dsl.split('+')

    parsed = []
    specified_tau = 0.0
    unspecified = []

    for blk in blocks:
        main, *tau_part = blk.split('@', 1)
        tau = float(tau_part[0]) if tau_part else None

        if 'x' in main:
            N_str, base = main.split('x', 1)
            N = int(N_str)
        else:
            N = 1
            base = main

        parsed.append((base, N, tau))
        if tau is None:
            unspecified.append(len(parsed) - 1)
        else:
            specified_tau += tau

    if specified_tau > 1.0 + 1e-12:
        raise ValueError("Time fractions exceed 1.")

    remaining = 1.0 - specified_tau
    if unspecified:
        share = remaining / len(unspecified)
        for i in unspecified:
            base, N, _ = parsed[i]
            parsed[i] = (base, N, share)

    # Flatten into individual substeps
    substeps = []  # list of (A_base, b_base, c_base, h)
    for base, N, tau in parsed:
        A_t, b_t, c_t = get_butcher_tableau(base)
        A_base = A_t.numpy()
        b_base = b_t.numpy()
        c_base = c_t.numpy()

        h = tau / N
        for _ in range(N):
            substeps.append((A_base, b_base, c_base, h))

    # Build global tableau
    S = sum(len(b) for (_, b, _, _) in substeps)
    A = np.zeros((S, S), dtype=np.float32)
    b = np.zeros(S, dtype=np.float32)
    c = np.zeros(S, dtype=np.float32)

    offset = 0
    t0 = 0.0

    for (A_base, b_base, c_base, h) in substeps:
        s = len(b_base)

        for i in range(s):
            gi = offset + i
            c[gi] = t0 + h * c_base[i]
            # intra-step coupling
            A[gi, offset:offset+s] = h * A_base[i]
            # coupling from all previous completed steps
            if offset > 0:
                A[gi, :offset] = b[:offset]

        b[offset:offset+s] = h * b_base
        offset += s
        t0 += h

    return (
        torch.tensor(A, dtype=torch.float32),
        torch.tensor(b, dtype=torch.float32),
        torch.tensor(c, dtype=torch.float32),
    )


def _solver_step_euler(X_t, t, v_theta, dt):
    return X_t + dt * v_theta(X_t, t)


def _solver_step_rk(X_t, t, v_theta, dt, a, b, c):
    """
    Implements a generic arbitrary-stage explicit Runge-Kutta method
    Assumes that `a` is lower-triangular. Any elements above and on the diagonal are ignored.
    """
    q = a.shape[0]
    k_list = []
    for i in range(q):
        if i == 0:
            stage_input = X_t
        else:
            k_prev = torch.stack(k_list[:i], dim=0)  # (q, B, C, F, T)
            weighted_sum = torch.sum(a[i, :i][:, None, None, None, None] * k_prev, dim=0)
            stage_input = X_t + dt * weighted_sum
        t_i = t + c[i:i+1] * dt
        k_i = v_theta(stage_input, t_i)
        k_list.append(k_i)
    k_all = torch.stack(k_list, dim=0)  # (q, B, C, F, T)
    k_total = torch.sum(b[:, None, None, None, None] * k_all, dim=0)
    out = X_t + dt * k_total
    return out


def solve_ode(X_0, v_theta, solver, N, solver_args: Optional[dict] = None, grad_for_steps: Sequence[int] = ()):
    euler_last = solver_args.get('euler_last', False) if solver_args is not None else False

    if solver == 'rk':
        assert solver_args is not None and all(k in solver_args for k in ('a', 'b', 'c')), \
            "For 'rk' solver, solver_args must contain 'a', 'b', and 'c' keys."
        a = solver_args['a']
        b = solver_args['b']
        c = solver_args['c']
        a, b, c = [torch.tensor(v, dtype=torch.float32, device=X_0.device) if isinstance(v, np.ndarray) else v for v in (a, b, c)]
    elif solver != 'euler':
        # get butcher tableau for other RK methods
        a, b, c = get_butcher_tableau(solver)
        a, b, c = [v.to(X_0.device) for v in (a, b, c)]

    dt = 1/N
    t = torch.zeros(1, dtype=torch.float32, device=X_0.device)
    X_t = X_0

    euler_step = lambda X_t, t: _solver_step_euler(X_t, t, v_theta, dt)
    if solver == 'euler':
        solver_step = euler_step
    else:
        solver_step = lambda X_t, t: _solver_step_rk(X_t, t, v_theta, dt, a, b, c)

    if len(grad_for_steps):
        for i in range(N):
            with torch.set_grad_enabled(i in grad_for_steps):
                if euler_last and i == N-1:
                    X_t = euler_step(X_t, t)
                else:
                    X_t = solver_step(X_t, t)
                t += dt
    else:
        with torch.inference_mode():
            for i in range(N):
                if euler_last and i == N-1:
                    X_t = euler_step(X_t, t)
                else:
                    X_t = solver_step(X_t, t)
                t += dt

    return X_t

"""In-sample LSMC for gas storage. Eqs. (15)–(21)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gas_storage.config import DEFAULT, Params
from gas_storage.contract import all_feasible_k, node_index, penalty_grid, volume_grid


@dataclass(frozen=True)
class LSMCResult:
    mean: float
    se: float
    y0: np.ndarray
    policy: np.ndarray  # (T, M, N) int8 action k


def phi(S: np.ndarray, params: Params = DEFAULT) -> np.ndarray:
    x = S / params.basis_scale
    return np.column_stack([np.ones_like(x), x, x**2, x**3])


def value(S: np.ndarray, params: Params = DEFAULT) -> LSMCResult:
    """S shape (M, T). In-sample mean is Eq. (21)."""
    n_paths, T = S.shape
    if T != params.T:
        raise ValueError("S.shape[1] must equal params.T")

    N = params.n_nodes
    alpha = params.alpha
    ks = all_feasible_k(params)
    n0 = node_index(params.v0, params)
    idx = np.arange(n_paths)

    Y = np.broadcast_to(penalty_grid(params), (n_paths, N)).copy()
    policy = np.zeros((T, n_paths, N), dtype=np.int8)

    for t in range(T - 1, -1, -1):
        St = S[:, t]
        Phi = phi(St, params)
        C = np.empty((n_paths, N))
        for n in range(N):
            beta, *_ = np.linalg.lstsq(Phi, Y[:, n], rcond=None)
            C[:, n] = Phi @ beta

        Y_next = Y
        Y = np.empty_like(Y_next)
        for n in range(N):
            kn = ks[n]
            Hs = -St[:, None] * (kn[None, :] * alpha)
            Cs = np.column_stack([C[:, n + k] for k in kn])
            obj = Hs + Cs - 1e-8 * np.abs(kn)[None, :]
            j = np.argmax(obj, axis=1)
            k_star = kn[j]
            policy[t, :, n] = k_star
            Y[:, n] = -St * (k_star * alpha) + Y_next[idx, n + k_star]

    y0 = Y[:, n0]
    return LSMCResult(
        mean=float(y0.mean()),
        se=float(y0.std(ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else 0.0,
        y0=y0,
        policy=policy,
    )


def volume_paths(policy: np.ndarray, params: Params = DEFAULT) -> np.ndarray:
    """(M, T+1) inventories; column 0 is v0."""
    T, n_paths, _ = policy.shape
    n = np.full(n_paths, node_index(params.v0, params), dtype=int)
    grid = volume_grid(params)
    v = np.empty((n_paths, T + 1))
    v[:, 0] = grid[n]
    rows = np.arange(n_paths)
    for t in range(T):
        k = policy[t, rows, n].astype(int)
        n = n + k
        v[:, t + 1] = grid[n]
    return v
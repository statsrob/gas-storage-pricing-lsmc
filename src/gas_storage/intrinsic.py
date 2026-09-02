"""Deterministic storage DP on the forward curve (intrinsic value)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gas_storage.config import DEFAULT, Params
from gas_storage.contract import all_feasible_k, node_index, penalty_grid, volume_grid
from gas_storage.forwards import build_forward_curve


@dataclass(frozen=True)
class IntrinsicResult:
    value: float
    volume: np.ndarray  # length T+1; [0] = v0, [-1] = v after last action
    dv: np.ndarray  # length T
    cash: np.ndarray  # length T, h on each day


def intrinsic(
    F: pd.Series | None = None,
    params: Params = DEFAULT,
) -> IntrinsicResult:
    if F is None:
        F = build_forward_curve(params)
    if len(F) != params.T:
        raise ValueError("F length must equal params.T")

    prices = F.to_numpy(dtype=float)
    N = params.n_nodes
    alpha = params.alpha
    ks = all_feasible_k(params)
    n0 = node_index(params.v0, params)

    U = np.zeros((params.T + 2, N))
    policy = np.zeros((params.T + 1, N), dtype=int)
    U[params.T + 1] = penalty_grid(params)

    for t in range(params.T, 0, -1):
        Ft = prices[t - 1]
        for n in range(N):
            best, best_k = -np.inf, 0
            for k in ks[n]:
                val = -Ft * (k * alpha) + U[t + 1, n + k]
                if val > best + 1e-12:
                    best, best_k = val, int(k)
                elif abs(val - best) <= 1e-12 and abs(int(k)) < abs(best_k):
                    best_k = int(k)
            U[t, n] = best
            policy[t, n] = best_k

    n = n0
    volume = np.empty(params.T + 1)
    dv = np.empty(params.T)
    cash = np.empty(params.T)
    grid = volume_grid(params)
    volume[0] = grid[n]
    for t in range(1, params.T + 1):
        k = int(policy[t, n])
        dv[t - 1] = k * alpha
        cash[t - 1] = -prices[t - 1] * dv[t - 1]
        n = n + k
        volume[t] = grid[n]

    q = float(U[params.T + 1, n])
    if not np.isclose(cash.sum() + q, U[1, n0]):
        raise RuntimeError("forward cash-flows do not match U(1, v0)")

    return IntrinsicResult(value=float(U[1, n0]), volume=volume, dv=dv, cash=cash)
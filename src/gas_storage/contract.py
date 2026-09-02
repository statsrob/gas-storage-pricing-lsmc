"""Exhibit 1 storage constraints, payoff (2)–(4), terminal penalty."""

from __future__ import annotations

import numpy as np

from gas_storage.config import DEFAULT, Params


def volume_grid(params: Params = DEFAULT) -> np.ndarray:
    return np.linspace(params.v_min, params.v_max, params.n_nodes)


def node_index(v: float, params: Params = DEFAULT) -> int:
    n = int(round(v / params.alpha))
    if abs(n * params.alpha - v) > 1e-8:
        raise ValueError(f"{v} is not on the volume grid")
    return n


def feasible_k(n: int, params: Params = DEFAULT) -> np.ndarray:
    """Integer action steps k with Δv = k * alpha, from node n."""
    v = n * params.alpha
    lo = max(params.v_min - v, params.i_min_rate)
    hi = min(params.v_max - v, params.i_max_rate)
    k_lo = int(np.ceil(lo / params.alpha - 1e-12))
    k_hi = int(np.floor(hi / params.alpha + 1e-12))
    ks = np.arange(k_lo, k_hi + 1)
    nxt = n + ks
    ks = ks[(nxt >= 0) & (nxt < params.n_nodes)]
    return ks[np.argsort(np.abs(ks), kind="stable")]


def all_feasible_k(params: Params = DEFAULT) -> list[np.ndarray]:
    return [feasible_k(n, params) for n in range(params.n_nodes)]


def payoff(S: float, dv: float, params: Params = DEFAULT) -> float:
    """Eq. (2) with (3)–(4). With a_i=b_i=0 this is -S * dv."""
    if dv > 0:
        return -((1.0 + params.a1) * S + params.b1) * dv
    if dv < 0:
        return -((1.0 - params.a2) * S - params.b2) * dv
    return 0.0


def penalty(v: float, params: Params = DEFAULT) -> float:
    return -params.penalty_per_mwh * abs(v - params.v_end)


def penalty_grid(params: Params = DEFAULT) -> np.ndarray:
    return -params.penalty_per_mwh * np.abs(volume_grid(params) - params.v_end)
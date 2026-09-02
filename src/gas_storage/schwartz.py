"""One-factor Schwartz (1997) with E[S(t)] = F(t). Eqs. (22)–(23)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gas_storage.config import DEFAULT, Params
from gas_storage.forwards import build_forward_curve


def simulate(
    F: pd.Series | None = None,
    sigma: float | None = None,
    params: Params = DEFAULT,
    seed: int | None = None,
) -> np.ndarray:
    """
    Return S with shape (n_paths, T).
    sigma defaults to params.sigma_low.
    """
    if F is None:
        F = build_forward_curve(params)
    if sigma is None:
        sigma = params.sigma_low
    if seed is None:
        seed = params.seed
    if params.n_paths % 2:
        raise ValueError("n_paths must be even for antithetic sampling")

    T = len(F)
    half = params.n_paths // 2
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((half, T))
    Z = np.concatenate([Z, -Z], axis=0)

    kappa = params.kappa
    rho = np.exp(-kappa)
    shock_sd = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa)) / (2.0 * kappa))

    eps = np.empty((params.n_paths, T))
    eps[:, 0] = shock_sd * Z[:, 0]
    for n in range(1, T):
        eps[:, n] = rho * eps[:, n - 1] + shock_sd * Z[:, n]

    steps = np.arange(1, T + 1, dtype=float)
    V = (sigma**2) / (2.0 * kappa) * (1.0 - np.exp(-2.0 * kappa * steps))
    S = F.to_numpy()[None, :] * np.exp(eps - 0.5 * V)
    return S
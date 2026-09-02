"""Exhibit 1 and numerical settings. Paper vs prototype assumptions labelled."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Params:
    # --- Exhibit 1, p. 9 (paper) ---
    start: date = date(2005, 7, 1)
    end: date = date(2006, 6, 30)
    T: int = 365
    v_min: float = 0.0
    v_max: float = 250_000.0
    v0: float = 100_000.0
    v_end: float = 100_000.0
    i_min_rate: float = -7_500.0  # MWh/day, withdrawal
    i_max_rate: float = 2_500.0  # MWh/day, injection
    alpha: float = 2_500.0
    n_nodes: int = 101

    # Eqs. (3)–(4), example on p. 9
    a1: float = 0.0
    a2: float = 0.0
    b1: float = 0.0
    b2: float = 0.0
    delta: float = 0.0

    # TTF quotes p. 9–10 (paper)
    july_2005: float = 14.88
    feb_2006: float = 25.44

    # Schwartz p. 9–10 (paper)
    kappa: float = 0.05
    sigma_low: float = 0.0315
    sigma_high: float = 0.0945
    n_paths: int = 500

    # --- prototype assumptions (not in the paper) ---
    peak: date = date(2006, 2, 14)  # A3: cosine peak
    weekend_discount: float = 1.0  # A5: €/MWh
    penalty_per_mwh: float = 1e6  # B3
    seed: int = 42  # D7
    basis_scale: float = 20.0  # E2: x = S / scale

    def __post_init__(self) -> None:
        if (self.end - self.start).days + 1 != self.T:
            raise ValueError("T must equal inclusive day count of [start, end]")
        if abs((self.v_max - self.v_min) / self.alpha + 1 - self.n_nodes) > 1e-9:
            raise ValueError("n_nodes must match (v_max - v_min) / alpha + 1")
        if self.n_paths % 2:
            raise ValueError("n_paths must be even (antithetic pairs)")


DEFAULT = Params()

# Paper pp. 10–11, for the README comparison (not used in pricing)
PAPER_INTRINSIC = 2.7e6
PAPER_LOW = 3.1e6
PAPER_HIGH = 5.4e6
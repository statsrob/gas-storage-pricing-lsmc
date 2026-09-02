"""CLI: reproduce Boogert & de Jong (2008) Exhibit 1 numbers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gas_storage.config import (
    DEFAULT,
    PAPER_HIGH,
    PAPER_INTRINSIC,
    PAPER_LOW,
    Params,
)
from gas_storage.forwards import build_forward_curve
from gas_storage.intrinsic import intrinsic
from gas_storage.lsmc import value, volume_paths
from gas_storage.schwartz import simulate


def _figures_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    out = root / "figures"
    out.mkdir(exist_ok=True)
    return out


def run(params: Params = DEFAULT) -> pd.DataFrame:
    F = build_forward_curve(params)
    intr = intrinsic(F, params)

    S_low = simulate(F, sigma=params.sigma_low, params=params)
    S_high = simulate(F, sigma=params.sigma_high, params=params)
    low = value(S_low, params)
    high = value(S_high, params)

    table = pd.DataFrame(
        {
            "paper_eur": [PAPER_INTRINSIC, PAPER_LOW, PAPER_HIGH],
            "repo_eur": [intr.value, low.mean, high.mean],
            "se_eur": [float("nan"), low.se, high.se],
            "ratio_to_intrinsic": [1.0, low.mean / intr.value, high.mean / intr.value],
        },
        index=["intrinsic", "LSMC low vol", "LSMC high vol"],
    )
    _write_figures(F, S_low, S_high, intr, low, high, params)
    return table


def _write_figures(F, S_low, S_high, intr, low, high, params: Params) -> None:
    out = _figures_dir()
    idx = F.index

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, F.values, lw=1)
    ax.set_ylabel("€/MWh")
    ax.set_title("Stylized TTF forwards (Exhibit 1 tenor)")
    fig.tight_layout()
    fig.savefig(out / "forwards.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, S, title in (
        (axes[0], S_low, r"Schwartz paths, $\sigma=3.15\%$"),
        (axes[1], S_high, r"Schwartz paths, $\sigma=9.45\%$"),
    ):
        ax.plot(idx, S[:40].T, color="C0", alpha=0.25, lw=0.6)
        ax.plot(idx, S.mean(0), color="C1", lw=1.3, label="MC mean")
        ax.plot(idx, F.values, color="k", lw=1.1, label=r"$F(t)$")
        ax.set_ylabel("€/MWh")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "schwartz_paths.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, intr.volume[1:], lw=1.2)
    ax.axhline(params.v_max, ls="--", lw=0.6, color="grey")
    ax.axhline(params.v_end, ls=":", lw=0.6, color="grey")
    ax.set_ylabel("MWh")
    ax.set_title("Intrinsic inventory")
    fig.tight_layout()
    fig.savefig(out / "intrinsic_volume.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, res, title in (
        (axes[0], low, r"LSMC inventory, low $\sigma$"),
        (axes[1], high, r"LSMC inventory, high $\sigma$"),
    ):
        v = volume_paths(res.policy, params)
        ax.fill_between(idx, v[:, 1:].min(0), v[:, 1:].max(0), alpha=0.35, label="min–max")
        ax.plot(idx, v[:, 1:].mean(0), lw=1.2, label="mean")
        ax.axhline(params.v_max, ls="--", lw=0.6, color="grey")
        ax.set_ylabel("MWh")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "lsmc_volume.png", dpi=120)
    plt.close(fig)


def main() -> None:
    print("Running intrinsic + LSMC (M=500, T=365). This can take a couple of minutes.")
    table = run()
    pd.set_option("display.float_format", lambda x: f"{x:,.0f}")
    print(table.to_string())
    print(f"\nFigures written to {_figures_dir()}")


if __name__ == "__main__":
    main()
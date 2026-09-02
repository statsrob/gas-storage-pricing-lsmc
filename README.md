# Gas storage valuation (LSMC)

Reproduction of Boogert and de Jong (2008), Gas Storage Valuation Using a Monte Carlo Method, Journal of Derivatives.

In-sample Least Squares Monte Carlo on their Exhibit 1 salt-cavern contract, with a stylized TTF forward curve (the paper does not publish the strip).

## Results (seed 42)

| | Paper | This repo |
| --- | --- | --- |
| Intrinsic | EUR 2.7m | about EUR 2.79m |
| LSMC, sigma = 3.15 percent per day | EUR 3.1m (+15 percent) | about EUR 3.15m (+13 percent) |
| LSMC, sigma = 9.45 percent per day | EUR 5.4m (about 2x) | about EUR 5.43m (about 1.95x) |

Ratios matter more than ticks: the curve is a cosine fitted to the two published monthly quotes (14.88 July 2005, 25.44 February 2006), not the 10 Jun 2005 TTF close.

## Run

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    pytest -q
    python -m gas_storage.reproduce

The last command writes figures/forwards.png, figures/schwartz_paths.png, figures/intrinsic_volume.png, and figures/lsmc_volume.png.

## Method

1. Daily forwards F(t): annual cosine plus week-weekend overlay, fitted to the two quotes.
2. Contract: Exhibit 1 rates and volume grid (alpha = 2500, N = 101).
3. Intrinsic: deterministic DP on F.
4. Spots: one-factor Schwartz (1997) with E[S_t] = F_t, antithetic, kappa = 0.05.
5. LSMC: per day and next-volume node, OLS of remaining cash on 1, S, S^2, S^3. Longstaff-Schwartz: decide with C-hat, roll realised Y.

## Not in this repo (on purpose)

- Live TTF strip and a stochastic forward curve (one-factor spot only).
- Bid-ask, fuel gas, inventory-dependent rates.
- Out-of-sample / dual bounds.

This is a paper reproduction and teaching implementation, not a production TTF pricer. A trading book would typically run rolling intrinsic on the live curve and a multi-factor LSMC for unhedged optionality.
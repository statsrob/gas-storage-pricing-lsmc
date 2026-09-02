"""Stylized daily TTF forwards. Spec: cosine + weekend overlay fitted to July/Feb quotes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gas_storage.config import DEFAULT, Params


def delivery_index(params: Params = DEFAULT) -> pd.DatetimeIndex:
    """Inclusive Exhibit 1 tenor."""
    return pd.date_range(params.start, params.end, freq="D")


def _seasonal_cosine(dates: pd.DatetimeIndex, params: Params) -> np.ndarray:
    peak = pd.Timestamp(params.peak)
    dt = (dates - peak).days.to_numpy(dtype=float)
    return np.cos(2.0 * np.pi * dt / 365.0)


def _weekend_overlay(dates: pd.DatetimeIndex, params: Params) -> np.ndarray:
    is_we = dates.dayofweek.to_numpy() >= 5
    n_we = int(is_we.sum())
    n_wd = len(dates) - n_we
    w = np.empty(len(dates), dtype=float)
    w[is_we] = -params.weekend_discount
    w[~is_we] = params.weekend_discount * n_we / n_wd
    return w


def build_forward_curve(params: Params = DEFAULT) -> pd.Series:
    """
    F(t) = a + b s(t) + w(t) with (a, b) such that
    mean(July 2005) and mean(Feb 2006) match the paper quotes.
    """
    dates = delivery_index(params)
    s = _seasonal_cosine(dates, params)
    w = _weekend_overlay(dates, params)

    jul = (dates.year == 2005) & (dates.month == 7)
    feb = (dates.year == 2006) & (dates.month == 2)
    A = np.array([[1.0, float(s[jul].mean())], [1.0, float(s[feb].mean())]])
    rhs = np.array(
        [
            params.july_2005 - float(w[jul].mean()),
            params.feb_2006 - float(w[feb].mean()),
        ]
    )
    a, b = np.linalg.solve(A, rhs)
    F = pd.Series(a + b * s + w, index=dates, name="F")
    F.index.name = "delivery"
    F.attrs["a"] = float(a)
    F.attrs["b"] = float(b)
    return F
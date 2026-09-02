import numpy as np

from gas_storage.config import DEFAULT
from gas_storage.forwards import build_forward_curve, delivery_index


def test_tenor_length():
    F = build_forward_curve()
    assert len(F) == DEFAULT.T == 365
    assert F.index[0].date() == DEFAULT.start
    assert F.index[-1].date() == DEFAULT.end


def test_monthly_quotes():
    F = build_forward_curve()
    assert np.isclose(F.loc["2005-07"].mean(), DEFAULT.july_2005)
    assert np.isclose(F.loc["2006-02"].mean(), DEFAULT.feb_2006)


def test_weekend_cheaper_and_overlay_mean_zero():
    F = build_forward_curve()
    is_we = F.index.dayofweek >= 5
    assert F[is_we].mean() < F[~is_we].mean()
    from gas_storage.forwards import _weekend_overlay

    w = _weekend_overlay(delivery_index(), DEFAULT)
    assert abs(float(w.mean())) < 1e-12


def test_seasonal_shape():
    F = build_forward_curve()
    assert F.idxmin().month in (6, 7, 8)
    assert F.idxmax().month in (1, 2, 3)
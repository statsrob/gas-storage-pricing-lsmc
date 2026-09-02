import numpy as np

from gas_storage.config import DEFAULT
from gas_storage.intrinsic import intrinsic


def test_accounting_and_end_volume():
    r = intrinsic()
    assert np.isclose(r.volume[0], DEFAULT.v0)
    assert np.isclose(r.volume[-1], DEFAULT.v_end)
    assert np.isclose(r.cash.sum(), r.value)  # q = 0 if we hit v_end


def test_value_in_paper_ballpark():
    r = intrinsic()
    assert 2.0e6 < r.value < 4.0e6


def test_uses_working_gas():
    r = intrinsic()
    assert r.volume.max() > 200_000.0
    assert r.volume.min() < 50_000.0


def test_inject_in_summer_withdraw_in_winter():
    r = intrinsic()
    # first 60 days ~ Jul–Aug; days 180–240 ~ Dec–Feb
    assert r.dv[:60].sum() > 0
    assert r.dv[180:240].sum() < 0
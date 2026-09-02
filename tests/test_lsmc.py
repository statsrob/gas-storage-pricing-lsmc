import numpy as np

from gas_storage.forwards import build_forward_curve
from gas_storage.intrinsic import intrinsic
from gas_storage.lsmc import value, volume_paths
from gas_storage.schwartz import simulate


def test_constant_paths_match_intrinsic():
    """If every path is F, LSMC collapses to the intrinsic DP."""
    F = build_forward_curve()
    S = np.tile(F.to_numpy(), (4, 1))
    r = value(S)
    assert np.isclose(r.mean, intrinsic(F).value, rtol=1e-8)
    v = volume_paths(r.policy)
    assert np.allclose(v[:, -1], 100_000.0)


def test_low_vol_smoke():
    from dataclasses import replace
    from gas_storage.config import DEFAULT

    p = replace(DEFAULT, n_paths=20)
    S = simulate(sigma=p.sigma_low, params=p, seed=0)
    r = value(S, params=p)
    assert np.isfinite(r.mean)
    assert r.mean > 0
    assert volume_paths(r.policy, p)[:, -1].min() == 100_000.0
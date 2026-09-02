import numpy as np

from gas_storage.config import DEFAULT
from gas_storage.forwards import build_forward_curve
from gas_storage.schwartz import simulate


def test_shape_positive_seed():
    F = build_forward_curve()
    S = simulate(F, sigma=DEFAULT.sigma_low)
    assert S.shape == (DEFAULT.n_paths, DEFAULT.T)
    assert (S > 0).all()
    S2 = simulate(F, sigma=DEFAULT.sigma_low, seed=DEFAULT.seed)
    assert np.allclose(S, S2)


def test_mean_tracks_forward_low_vol():
    F = build_forward_curve()
    S = simulate(F, sigma=DEFAULT.sigma_low)
    rel = np.abs(S.mean(axis=0) / F.to_numpy() - 1.0)
    assert rel.max() < 0.08


def test_high_vol_wider_than_low():
    F = build_forward_curve()
    low = simulate(F, sigma=DEFAULT.sigma_low)
    high = simulate(F, sigma=DEFAULT.sigma_high)
    assert high.std() > low.std()
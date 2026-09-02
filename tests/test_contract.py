import numpy as np
import pytest

from gas_storage.config import DEFAULT
from gas_storage.contract import (
    feasible_k,
    node_index,
    payoff,
    penalty,
    volume_grid,
)


def test_grid():
    g = volume_grid()
    assert len(g) == DEFAULT.n_nodes == 101
    assert np.isclose(g[0], 0.0)
    assert np.isclose(g[-1], 250_000.0)
    assert node_index(DEFAULT.v0) == 40


def test_actions_empty_full_start():
    assert set(feasible_k(0).tolist()) == {0, 1}
    assert set(feasible_k(100).tolist()) == {0, -1, -2, -3}
    assert set(feasible_k(40).tolist()) == {0, 1, -1, -2, -3}


def test_payoff_signs():
    assert np.isclose(payoff(20.0, 2_500.0), -50_000.0)
    assert np.isclose(payoff(20.0, -7_500.0), 150_000.0)
    assert payoff(20.0, 0.0) == 0.0


def test_penalty():
    assert penalty(100_000.0) == 0.0
    assert np.isclose(penalty(102_500.0), -DEFAULT.penalty_per_mwh * 2_500.0)


def test_off_grid_raises():
    with pytest.raises(ValueError):
        node_index(100_001.0)
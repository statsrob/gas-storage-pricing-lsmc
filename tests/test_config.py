from datetime import date

from gas_storage.config import DEFAULT, PAPER_HIGH, PAPER_INTRINSIC, PAPER_LOW, Params


def test_exhibit1_tenor():
    p = DEFAULT
    assert p.start == date(2005, 7, 1)
    assert p.end == date(2006, 6, 30)
    assert p.T == 365
    assert p.n_nodes == 101
    assert p.v0 == p.v_end == 100_000.0


def test_paper_quotes_and_vols():
    p = DEFAULT
    assert p.july_2005 == 14.88
    assert p.feb_2006 == 25.44
    assert p.kappa == 0.05
    assert p.sigma_low == 0.0315
    assert p.sigma_high == 0.0945
    assert p.n_paths == 500


def test_frozen():
    try:
        DEFAULT.T = 1
    except AttributeError:
        return
    raise AssertionError("Params should be frozen")


def test_paper_benchmarks_are_labels():
    assert PAPER_INTRINSIC < PAPER_LOW < PAPER_HIGH
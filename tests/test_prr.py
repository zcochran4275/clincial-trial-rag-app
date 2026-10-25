import math

import pytest

from trial_signal.analysis.prr import contingency, prr


def test_contingency_cells_sum_to_n():
    a, b, c, d = contingency(a=20, drug_total=100, event_total=500, n_total=10_000)
    assert (a, b, c, d) == (20, 80, 480, 9420)
    assert a + b + c + d == 10_000


def test_prr_matches_hand_calculation():
    r = prr(a=20, drug_total=100, event_total=500, n_total=10_000)
    expected = (20 / 100) / (480 / 9900)
    assert r.prr == pytest.approx(expected)
    assert r.ci_low < r.prr < r.ci_high


def test_no_disproportionality_gives_prr_near_one():
    # event is 5% of drug reports and 5% of all other reports
    r = prr(a=50, drug_total=1_000, event_total=50 + 4_950, n_total=100_000)
    assert r.prr == pytest.approx(1.0)
    assert not r.is_signal


@pytest.mark.parametrize(
    "a, drug_total, event_total, n_total, signal",
    [
        (30, 200, 1_000, 100_000, True),   # strong disproportionality
        (2, 10, 50, 100_000, False),       # PRR high but a < 3
        (50, 1_000, 5_000, 100_000, False) # PRR ~1
    ],
)
def test_signal_screening_rule(a, drug_total, event_total, n_total, signal):
    assert prr(a, drug_total, event_total, n_total).is_signal is signal


@pytest.mark.parametrize("bad", [dict(a=10, drug_total=5, event_total=50, n_total=1000),
                                 dict(a=0, drug_total=5, event_total=50, n_total=1000)])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        prr(**bad)


def test_chi2_is_finite():
    assert math.isfinite(prr(30, 200, 1_000, 100_000).chi2)

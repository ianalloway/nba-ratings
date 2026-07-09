"""Invariant tests for fair-probability distributions produced by kelly.py.

These guard the contract that `remove_vig` and `parlay_odds` always return
valid probability distributions (entries in [0, 1], summing to 1) so callers
can feed the output straight into Kelly sizing without re-validating bounds.
"""

import pytest

from nba_edge.kelly import parlay_odds, remove_vig

TRADEABLE = [-110, -150, -200, -300, 100, 150, 200, 300]


def test_remove_vig_proportional_is_valid_distribution() -> None:
    # Every proportional fair-prob pair is a proper distribution: non-negative,
    # sum-to-1, across a grid of real book lines.
    for a in TRADEABLE:
        for b in TRADEABLE:
            pa, pb = remove_vig(a, b, method="proportional")
            assert 0.0 <= pa <= 1.0
            assert 0.0 <= pb <= 1.0
            assert pa + pb == pytest.approx(1.0)


def test_remove_vig_equal_is_valid_distribution() -> None:
    # Equal-margin fair probs also stay a valid distribution for ordinary lines
    # (the documented negative-probability case is mathematically unreachable
    # because fair_a >= (p_a - p_b + 1) / 2 > 0 for any two valid implied probs).
    for a in TRADEABLE:
        for b in TRADEABLE:
            fa, fb = remove_vig(a, b, method="equal")
            assert 0.0 <= fa <= 1.0
            assert 0.0 <= fb <= 1.0
            assert fa + fb == pytest.approx(1.0)


def test_remove_vig_equal_matches_implied_when_no_vig() -> None:
    # A fair market (e.g. +100 / +100) has zero overround, so both methods
    # collapse to 0.5 / 0.5.
    pa, pb = remove_vig(100, 100, method="equal")
    assert pa == pytest.approx(0.5)
    assert pb == pytest.approx(0.5)
    prop_a, prop_b = remove_vig(100, 100, method="proportional")
    assert prop_a == pytest.approx(pa)
    assert prop_b == pytest.approx(pb)


def test_parlay_implied_prob_equals_product_of_legs() -> None:
    # The parlay's combined implied probability must equal the product of the
    # per-leg implied probabilities (independence assumption of parlay_odds).
    for legs in ([-110, -110], [-110, -110, -110], [150, -200], [-105, -115, 200]):
        res = parlay_odds(list(legs))
        product = 1.0
        for odds in legs:
            product *= (
                100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)
            )
        assert res["implied_prob"] == pytest.approx(product)


def test_parlay_is_valid_distribution_over_single_outcome() -> None:
    # A parlay is a single Bernoulli outcome, so its implied prob is a valid
    # probability in [0, 1] (independent legs always have combined prob <= 1).
    for legs in ([-110, -110], [-150, 130, 200], [-200, -200, -200]):
        prob = parlay_odds(list(legs))["implied_prob"]
        assert 0.0 <= prob <= 1.0


def test_remove_vig_fair_market_is_unchanged() -> None:
    # Feeding a no-vig market through proportional removal returns the inputs.
    pa, pb = remove_vig(-110, 110, method="proportional")
    assert pa == pytest.approx(110.0 / 210.0, abs=1e-9)
    assert pb == pytest.approx(100.0 / 210.0, abs=1e-9)

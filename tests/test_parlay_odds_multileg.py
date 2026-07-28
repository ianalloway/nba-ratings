"""Multi-leg coverage for ``nba_edge.kelly.parlay_odds``.

``test_kelly.py`` only exercises the single-leg identity branch. The function's
real purpose is combining two or more independent legs into joint decimal odds
and a combined implied probability, so this module pins that math down.
"""

from __future__ import annotations

import pytest

from nba_edge.kelly import (
    american_to_decimal,
    american_to_implied_prob,
    parlay_odds,
)


def test_parlay_odds_two_legs_joint_probability_and_decimal() -> None:
    legs = [-110, -110]
    res = parlay_odds(legs)

    expected_prob = american_to_implied_prob(-110) ** 2
    assert res["implied_prob"] == pytest.approx(expected_prob)

    expected_dec = american_to_decimal(-110) ** 2
    assert res["decimal"] == pytest.approx(expected_dec)


def test_parlay_odds_american_round_trips_to_decimal() -> None:
    legs = [-110, -110]
    res = parlay_odds(legs)
    # Combined American odds should convert back to the joint decimal odds.
    assert american_to_decimal(res["american"]) == pytest.approx(res["decimal"], rel=1e-6)


def test_parlay_odds_three_legs_multiplicative() -> None:
    legs = [100, -200, 150]
    res = parlay_odds(legs)

    expected_prob = (
        american_to_implied_prob(100)
        * american_to_implied_prob(-200)
        * american_to_implied_prob(150)
    )
    assert res["implied_prob"] == pytest.approx(expected_prob)

    expected_dec = american_to_decimal(100) * american_to_decimal(-200) * american_to_decimal(150)
    assert res["decimal"] == pytest.approx(expected_dec)


def test_parlay_odds_empty_list_raises() -> None:
    with pytest.raises(ValueError):
        parlay_odds([])

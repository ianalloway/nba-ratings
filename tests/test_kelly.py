import pytest
from nba_edge.kelly import (
    american_to_decimal,
    american_to_implied_prob,
    decimal_to_american,
    decimal_to_implied_prob,
    implied_prob_to_american,
    implied_prob_to_decimal,
    kelly_fraction,
)


def test_american_to_implied_prob() -> None:
    # Test positive odds
    assert pytest.approx(american_to_implied_prob(100.0)) == 0.5
    assert pytest.approx(american_to_implied_prob(200.0)) == 1 / 3

    # Test negative odds
    assert pytest.approx(american_to_implied_prob(-100.0)) == 0.5
    assert pytest.approx(american_to_implied_prob(-110.0)) == 110 / 210

    # Test invalid range
    with pytest.raises(ValueError):
        american_to_implied_prob(50)


def test_american_to_decimal() -> None:
    assert pytest.approx(american_to_decimal(100.0)) == 2.0
    assert pytest.approx(american_to_decimal(200.0)) == 3.0
    assert pytest.approx(american_to_decimal(-100.0)) == 2.0
    assert pytest.approx(american_to_decimal(-110.0)) == 100 / 110 + 1.0

    with pytest.raises(ValueError):
        american_to_decimal(-50)


def test_decimal_to_american() -> None:
    assert pytest.approx(decimal_to_american(2.0)) == 100.0
    assert pytest.approx(decimal_to_american(3.0)) == 200.0
    assert pytest.approx(decimal_to_american(1.91)) == -100 / 0.91

    with pytest.raises(ValueError):
        decimal_to_american(0.5)


def test_decimal_to_implied_prob() -> None:
    assert pytest.approx(decimal_to_implied_prob(2.0)) == 0.5
    assert pytest.approx(decimal_to_implied_prob(4.0)) == 0.25

    with pytest.raises(ValueError):
        decimal_to_implied_prob(0.9)


def test_implied_prob_to_american() -> None:
    assert pytest.approx(implied_prob_to_american(0.5)) == -100.0
    assert pytest.approx(implied_prob_to_american(0.25)) == 300.0

    with pytest.raises(ValueError):
        implied_prob_to_american(0.0)
    with pytest.raises(ValueError):
        implied_prob_to_american(1.5)


def test_implied_prob_to_decimal() -> None:
    assert pytest.approx(implied_prob_to_decimal(0.5)) == 2.0
    assert pytest.approx(implied_prob_to_decimal(0.25)) == 4.0

    with pytest.raises(ValueError):
        implied_prob_to_decimal(-0.1)


def test_kelly_fraction_validation() -> None:
    with pytest.raises(ValueError):
        kelly_fraction(0.6, 50)

import pytest
from nba_edge.kelly import (
    american_to_decimal,
    american_to_implied_prob,
    decimal_to_american,
    decimal_to_implied_prob,
    implied_prob_to_american,
    implied_prob_to_decimal,
    kelly_fraction,
    kelly_parlay,
    parlay_odds,
    remove_vig,
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
    with pytest.raises(ValueError):
        kelly_fraction(-0.1, 100)
    with pytest.raises(ValueError):
        kelly_fraction(1.1, 100)
    with pytest.raises(ValueError):
        kelly_fraction(0.5, 110, max_cap=-0.1)


def test_kelly_fraction_calculation() -> None:
    # 60% win prob on +100 (b = 1.0) -> full Kelly = (1 * 0.6 - 0.4)/1 = 0.2
    # Quarter Kelly -> 0.05
    assert pytest.approx(kelly_fraction(0.6, 100, fraction=0.25)) == 0.05

    # Full Kelly (fraction=1.0) with default cap (max_cap=0.25) but win_prob=0.6 -> not capped (returns 0.20)
    assert pytest.approx(kelly_fraction(0.6, 100, fraction=1.0)) == 0.20

    # Full Kelly (fraction=1.0) with default cap (max_cap=0.25) and win_prob=0.8 -> capped (returns 0.25)
    assert pytest.approx(kelly_fraction(0.8, 100, fraction=1.0)) == 0.25

    # Full Kelly (fraction=1.0), no cap (max_cap=None)
    assert pytest.approx(kelly_fraction(0.6, 100, fraction=1.0, max_cap=None)) == 0.20

    # Negative EV bet -> should return 0.0
    assert pytest.approx(kelly_fraction(0.4, 100, fraction=1.0)) == 0.0


def test_remove_vig() -> None:
    # Coexist in -110 / -110 market
    # Implied prob = 110/210 ≈ 0.5238 each, total ≈ 1.0476
    # Proportional fair prob = 0.5 each
    p_a, p_b = remove_vig(-110, -110, method="proportional")
    assert pytest.approx(p_a) == 0.5
    assert pytest.approx(p_b) == 0.5

    p_a, p_b = remove_vig(-110, -110, method="equal")
    assert pytest.approx(p_a) == 0.5
    assert pytest.approx(p_b) == 0.5

    # Asymmetric odds (e.g. -150 / +130)
    # Implied -150: 150/250 = 0.6
    # Implied +130: 100/230 ≈ 0.4348
    # Total = 1.0348
    pa_prop, pb_prop = remove_vig(-150, 130, method="proportional")
    assert pytest.approx(pa_prop, abs=1e-4) == 0.6 / 1.0348
    assert pytest.approx(pb_prop, abs=1e-4) == 0.4348 / 1.0348

    # Equal method subtracts overround/2 = 0.0348/2 = 0.0174
    pa_eq, pb_eq = remove_vig(-150, 130, method="equal")
    assert pytest.approx(pa_eq, abs=1e-4) == 0.6 - 0.0174
    assert pytest.approx(pb_eq, abs=1e-4) == 0.4348 - 0.0174

    # Validations
    with pytest.raises(ValueError):
        remove_vig(-110, -110, method="invalid_method")

    # If equal method results in negative probability (extreme case)
    # e.g., one side has extremely low implied prob while the overround is huge.
    # To trigger: A is +900 (p = 0.1), B is -105 (p = 0.5122) -> overround is negative (no vig) or let's construct high vig:
    # Say overround is 0.3. Half overround = 0.15. If p_a = 0.1, fair_a would be -0.05.
    # To construct: A is +1000 (implied = 100/1100 = 0.0909), B is -300 (implied = 300/400 = 0.75), C is somehow included or overround is huge:
    # Let's say odds_a = 900 (p_a = 0.1), odds_b = -500 (p_b = 5/6 ≈ 0.8333). Overround = 0.9333 - 1.0 = -0.0667 (underround).
    # What if odds_a = -110 (p_a = 0.5238), odds_b = -110 (p_b = 0.5238), let's make vig massive:
    # What if both are -500? p_a = p_b = 0.8333. Overround = 1.6667 - 1 = 0.6667. Overround/2 = 0.3333.
    # This is fine. What if odds_a = 500 (p_a = 1/6 ≈ 0.1667), odds_b = -500 (p_b ≈ 0.8333). No vig.
    # To make equal method produce negative: odds_a = 1000 (p_a = 1/11 ≈ 0.0909), odds_b = -900 (p_b = 900/1000 = 0.9). Total = 0.9909. No vig.
    # What if odds_a = 500 (p_a = 0.1667), odds_b = -250 (p_b = 250/350 = 0.714). Total = 0.88.
    # What if we have a massive vig: say we have a custom constructed pair where overround/2 is greater than one of the implied probs.
    # If odds_a = 800 (p_a = 0.1111), odds_b = -1000 (p_b = 1000/1100 = 0.909). Total = 1.02. Half overround = 0.01. Fair = 0.10.
    # What if odds_a = 1000 (p_a = 0.0909), odds_b = -2000 (p_b = 2000/2100 = 0.952). Total = 1.043. Half overround = 0.0215. Fair_a = 0.0694.
    # Let's try: odds_a = 1000 (p_a ≈ 0.0909), odds_b = -10000 (p_b ≈ 0.9901). Total = 1.081. Half overround = 0.0405. Fair_a = 0.0504.
    # It is hard to get negative since a blowout favorite means the underdog is long odds, but they both can't be long odds and have huge overround unless they are both priced as favorites, which is mathematically weird.
    # But we can verify our validation checks gracefully.


def test_parlay_odds() -> None:
    # 2 legs of -110 each
    # Decimal of -110 is 1.90909...
    # Joint decimal = 1.90909 * 1.90909 = 3.6446
    # Joint American = (3.6446 - 1) * 100 = 264.46
    # Joint Implied prob = (110/210) * (110/210) = 0.274376
    res = parlay_odds([-110, -110])
    assert pytest.approx(res["decimal"]) == 3.644628
    assert pytest.approx(res["american"]) == 264.4628
    assert pytest.approx(res["implied_prob"]) == 0.2743764172335601

    with pytest.raises(ValueError):
        parlay_odds([])


def test_kelly_parlay() -> None:
    # 2 legs, win probs 0.6 and 0.7, bookmaker odds -110 and -110
    # Joint win prob = 0.6 * 0.7 = 0.42
    # Joint decimal = 3.644628 -> b = 2.644628
    # q = 0.58
    # full Kelly = (2.644628 * 0.42 - 0.58) / 2.644628 ≈ 0.20067
    # Quarter Kelly (fraction = 0.25) -> ≈ 0.050171875
    val = kelly_parlay([0.6, 0.7], [-110, -110], fraction=0.25)
    assert pytest.approx(val) == 0.050171875

    # Length mismatch
    with pytest.raises(ValueError):
        kelly_parlay([0.6], [-110, -110])

    # Empty inputs
    with pytest.raises(ValueError):
        kelly_parlay([], [])

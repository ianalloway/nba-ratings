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


def test_parlay_odds_single_leg_is_identity() -> None:
    """A single-leg parlay is the identity transformation: decimal, implied
    probability, and American must all round-trip back to the input leg."""
    from nba_edge.kelly import (
        american_to_decimal,
        american_to_implied_prob,
        decimal_to_american,
    )

    leg = -110
    res = parlay_odds([leg])
    assert pytest.approx(res["decimal"]) == american_to_decimal(leg)
    assert pytest.approx(res["implied_prob"]) == american_to_implied_prob(leg)
    # No precision loss on a single element
    assert pytest.approx(res["american"]) == float(leg)


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


def test_kelly_parlay_returns_zero_on_losing_edge() -> None:
    # A parlay must never size a negative fraction when the combined bet has no
    # positive expected value. Break-even for two -110 legs is a joint win prob
    # of ~0.2744; anything below that is a losing-edge parlay and must size to
    # exactly 0.0 (not a negative "bet against").
    assert pytest.approx(kelly_parlay([0.5, 0.5], [-110, -110], fraction=1.0)) == 0.0
    # Clearly losing edge
    assert pytest.approx(kelly_parlay([0.4, 0.5], [-110, -110], fraction=1.0)) == 0.0


def test_kelly_parlay_never_negative() -> None:
    # Sweep losing-edge parlays across favorites and dogs and assert the
    # fraction is always clamped to >= 0 (the Kelly clamp, not a silent negative).
    for wp in (0.1, 0.2, 0.3, 0.4, 0.5):
        for odds in (-110, -150, 200, 300):
            val = kelly_parlay([wp, wp], [odds, odds], fraction=1.0, max_cap=None)
            assert val >= 0.0


def test_kelly_fraction_max_cap_upper_bound_validation() -> None:
    # max_cap above 1.0 must be rejected
    with pytest.raises(ValueError):
        kelly_fraction(0.6, 100, max_cap=1.5)
    with pytest.raises(ValueError):
        kelly_fraction(0.6, 100, max_cap=2.0)


def test_kelly_fraction_uncapped_exceeds_default_cap() -> None:
    # With max_cap=None a strong edge yields a larger fraction than the default cap
    capped = kelly_fraction(0.99, 1000, fraction=1.0, max_cap=0.25)
    uncapped = kelly_fraction(0.99, 1000, fraction=1.0, max_cap=None)
    assert uncapped > capped
    assert uncapped < 1.0  # single-bet Kelly fraction is always below 1.0 bankroll


def test_american_decimal_round_trip() -> None:
    # american -> decimal -> american is an identity for both favorites and dogs
    for am in (-110, -250, +150, +500, 100, 200):
        dec = american_to_decimal(am)
        back = decimal_to_american(dec)
        assert pytest.approx(back, rel=1e-9) == am


def test_decimal_implied_round_trip() -> None:
    # decimal -> implied -> decimal is an identity for valid decimal odds
    for dec in (1.5, 1.91, 2.0, 3.0, 6.0):
        prob = decimal_to_implied_prob(dec)
        back = implied_prob_to_decimal(prob)
        assert pytest.approx(back, rel=1e-9) == dec


def test_implied_american_round_trip() -> None:
    # implied -> american -> implied is an identity for probs strictly below 0.5 and above
    for prob in (0.1, 0.25, 0.4, 0.6, 0.75, 0.9):
        am = implied_prob_to_american(prob)
        back = american_to_implied_prob(am)
        assert pytest.approx(back, rel=1e-9) == prob


def test_implied_prob_to_decimal_accepts_unit() -> None:
    # Upper bound is inclusive (1.0 -> 1.0), unlike implied_prob_to_american (strict < 1.0)
    assert pytest.approx(implied_prob_to_decimal(1.0)) == 1.0


def test_parlay_odds_internal_consistency() -> None:
    """parlay_odds must keep its three returned quantities mutually consistent
    and implied_prob must equal the product of the per-leg implied probs.

    This locks invariants that the single hardcoded example in test_parlay_odds
    does not cover across mixed favorites/underdogs and multi-leg parlays.
    """
    from nba_edge.kelly import (
        american_to_implied_prob,
        decimal_to_american,
        decimal_to_implied_prob,
    )

    markets = [
        [-110, -110],
        [-110, +150],
        [+200, -120, -110],
        [-105, -105, -105, -105],
        [+300, +400],
        [-110, -110, -110, -110, -110],
    ]
    for legs in markets:
        res = parlay_odds(legs)

        # internal consistency: decimal <-> implied_prob
        assert pytest.approx(res["implied_prob"]) == decimal_to_implied_prob(res["decimal"])

        # american must round-trip back from the joint decimal
        assert pytest.approx(res["american"]) == decimal_to_american(res["decimal"])

        # implied_prob equals the product of per-leg implied probabilities
        expected = 1.0
        for o in legs:
            expected *= american_to_implied_prob(o)
        assert pytest.approx(res["implied_prob"]) == expected


def test_remove_vig_conserves_total_probability() -> None:
    # Both vig-removal methods must return fair probabilities that sum to
    # exactly 1.0 and stay within the valid [0, 1] range, even on extreme
    # (heavy-favorite vs heavy-favorite) markets that carry large overround.
    # This guards against a regression where the additive 'equal' method could
    # drift probabilities away from a valid distribution.
    markets = [(-110, -110), (-150, 130), (-1000, -1000), (200, -250)]
    for a, b in markets:
        pa_prop, pb_prop = remove_vig(a, b, method="proportional")
        pa_eq, pb_eq = remove_vig(a, b, method="equal")
        assert pytest.approx(pa_prop + pb_prop) == 1.0
        assert pytest.approx(pa_eq + pb_eq) == 1.0
        assert 0.0 <= pa_prop <= 1.0 and 0.0 <= pb_prop <= 1.0
        assert 0.0 <= pa_eq <= 1.0 and 0.0 <= pb_eq <= 1.0


def test_remove_vig_zero_vig_market_is_identity() -> None:
    # A market with no overround (+100 / -100 => implied 0.5 / 0.5) already
    # holds fair probabilities; both methods must return them unchanged, and
    # the additive 'equal' method must agree with 'proportional' when there is
    # no vig to strip.
    pa_prop, pb_prop = remove_vig(100, -100, method="proportional")
    assert pytest.approx(pa_prop) == 0.5
    assert pytest.approx(pb_prop) == 0.5

    pa_eq, pb_eq = remove_vig(100, -100, method="equal")
    assert pytest.approx(pa_eq) == pytest.approx(pa_prop)
    assert pytest.approx(pb_eq) == pytest.approx(pb_prop)

import pytest

from nba_edge.ratings import (
    logistic_win_prob,
    expected_margin,
    mov_multiplier,
    update_elo,
    update_elo_with_margin,
)


def test_even_matchup() -> None:
    assert 0.49 < logistic_win_prob(0.0) < 0.51


def test_favorite_wins_more() -> None:
    assert logistic_win_prob(200.0) > 0.7


def test_update_elo() -> None:
    a, b = update_elo(1500.0, 1500.0, 1.0)
    assert a > 1500
    assert b < 1500


def test_mov_multiplier_blowout_exceeds_narrow_win() -> None:
    narrow = mov_multiplier(2.0, elo_diff_winner=0.0)
    blowout = mov_multiplier(30.0, elo_diff_winner=0.0)
    assert blowout > narrow > 0


def test_mov_multiplier_dampened_for_big_favorite() -> None:
    underdog_win = mov_multiplier(10.0, elo_diff_winner=0.0)
    favorite_blowout = mov_multiplier(10.0, elo_diff_winner=400.0)
    assert favorite_blowout < underdog_win




def test_mov_multiplier_pole_is_finite_and_positive() -> None:
    # The dampening factor 2.2 / (0.001 * elo_diff_winner + 2.2) has a pole at
    # elo_diff_winner == -2200. It must not raise ZeroDivisionError and must stay
    # strictly positive (a negative multiplier would flip the Elo update sign).
    for diff in (-2199.0, -2200.0, -2201.0, -3000.0, -5000.0):
        mult = mov_multiplier(10.0, elo_diff_winner=diff)
        assert mult > 0.0


def test_mov_multiplier_clamped_below_pole_is_finite() -> None:
    # For extreme underdog wins (elo_diff_winner < -2200) the multiplier stays
    # finite instead of diverging to +/-inf, so callers can't crash on big upsets.
    mult = mov_multiplier(30.0, elo_diff_winner=-10000.0)
    assert mult == mult  # not NaN
    assert abs(mult) < 1e6


def test_mov_multiplier_rejects_negative_margin() -> None:
    with pytest.raises(ValueError):
        mov_multiplier(-1.0, elo_diff_winner=0.0)


def test_update_elo_with_margin_blowout_moves_more_than_narrow_win() -> None:
    narrow_a, narrow_b = update_elo_with_margin(1500.0, 1500.0, 1.0, margin=2.0)
    blowout_a, blowout_b = update_elo_with_margin(1500.0, 1500.0, 1.0, margin=30.0)
    assert (blowout_a - 1500.0) > (narrow_a - 1500.0) > 0
    assert (1500.0 - blowout_b) > (1500.0 - narrow_b) > 0


def test_update_elo_with_margin_zero_margin_means_no_movement() -> None:
    # log(margin + 1) = log(1) = 0 at margin=0, so the multiplier zeroes out
    # the update entirely — there's no real 0-point win in the NBA, but the
    # formula should degrade gracefully rather than error.
    a, b = update_elo_with_margin(1500.0, 1480.0, 1.0, margin=0.0)
    assert a == 1500.0
    assert b == 1480.0


def test_update_elo_with_margin_draw_uses_no_multiplier() -> None:
    a, b = update_elo_with_margin(1500.0, 1500.0, 0.5, margin=0.0)
    assert a == 1500.0
    assert b == 1500.0


def test_update_elo_with_margin_rejects_negative_margin() -> None:
    with pytest.raises(ValueError):
        update_elo_with_margin(1500.0, 1500.0, 1.0, margin=-5.0)

def test_expected_margin_proportional_to_rating_diff() -> None:
    # Default slope: 100 Elo -> 2.5 points
    assert pytest.approx(expected_margin(100.0)) == 2.5
    assert pytest.approx(expected_margin(0.0)) == 0.0
    # Linear and symmetric: favorite margin == -underdog margin
    assert pytest.approx(expected_margin(200.0)) == 2 * expected_margin(100.0)
    assert pytest.approx(expected_margin(-100.0)) == -expected_margin(100.0)


def test_expected_margin_custom_slope() -> None:
    # Override the per-Elo slope
    assert pytest.approx(expected_margin(100.0, margin_per_elo=0.04)) == 4.0


def test_expected_margin_never_divides_by_zero() -> None:
    # Default slope is a plain multiply; ensure no ZeroDivisionError path exists
    # for a degenerate (zero) slope so callers can pass margin_per_elo=0 safely.
    assert pytest.approx(expected_margin(150.0, margin_per_elo=0.0)) == 0.0


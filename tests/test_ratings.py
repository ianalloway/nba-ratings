import math

import pytest

from nba_edge.ratings import (
    expected_margin,
    logistic_win_prob,
    update_elo,
    update_elo_with_margin,
)


def test_expected_margin_proportional_to_elo_diff() -> None:
    assert expected_margin(0.0) == 0.0
    assert expected_margin(100.0) > 0
    assert expected_margin(-100.0) < 0
    assert expected_margin(200.0) == pytest.approx(2.0 * expected_margin(100.0))


def test_expected_margin_custom_slope() -> None:
    assert expected_margin(100.0, margin_per_elo=0.05) == pytest.approx(5.0)
    assert expected_margin(100.0) == pytest.approx(2.5)


def test_logistic_win_prob_symmetry() -> None:
    for diff in (-300.0, -50.0, 0.0, 50.0, 300.0):
        p_home = logistic_win_prob(diff)
        p_away = logistic_win_prob(-diff)
        assert p_home + p_away == pytest.approx(1.0)


def test_logistic_win_prob_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError, match="scale must be positive"):
        logistic_win_prob(100.0, scale=0.0)
    with pytest.raises(ValueError, match="scale must be positive"):
        logistic_win_prob(100.0, scale=-400.0)


def test_logistic_win_prob_extreme_underdog_no_overflow() -> None:
    # Very large negative diff used to raise OverflowError; now clamps to ~0.
    assert logistic_win_prob(-1e10) == pytest.approx(0.0)
    assert logistic_win_prob(-1e6) == pytest.approx(0.0, abs=1e-10)
    # Very large positive diff still returns ~1.0 (no overflow on that branch).
    assert logistic_win_prob(1e10) == pytest.approx(1.0)


def test_update_elo_conserves_rating_mass() -> None:
    a, b = update_elo(1500.0, 1450.0, 1.0, k=32.0)
    assert (a + b) == pytest.approx(1500.0 + 1450.0)


def test_update_elo_draw_moves_both_halfway_to_expected() -> None:
    a, b = update_elo(1600.0, 1400.0, 0.5)
    drop = 1600.0 - a
    rise = b - 1400.0
    assert drop == pytest.approx(rise)
    assert (a + b) == pytest.approx(1600.0 + 1400.0)


def test_update_elo_rejects_invalid_score() -> None:
    with pytest.raises(ValueError, match="score_a must be 0, 0.5, or 1"):
        update_elo(1500.0, 1500.0, 0.3)


def test_update_elo_rejects_nonpositive_k_and_scale() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        update_elo(1500.0, 1500.0, 1.0, k=0.0)
    with pytest.raises(ValueError, match="scale must be positive"):
        update_elo(1500.0, 1500.0, 1.0, scale=-100.0)


def test_update_elo_with_margin_conserves_rating_mass() -> None:
    a, b = update_elo_with_margin(1500.0, 1450.0, 1.0, 30.0, k=20.0)
    assert (a + b) == pytest.approx(1500.0 + 1450.0)


def test_update_elo_with_margin_draw_ignores_margin() -> None:
    # On a draw the margin-of-victory multiplier is set to 1.0, so the
    # result should match plain update_elo regardless of margin.
    a1, b1 = update_elo_with_margin(1500.0, 1450.0, 0.5, 10.0, k=20.0)
    a2, b2 = update_elo(1500.0, 1450.0, 0.5, k=20.0)
    assert a1 == pytest.approx(a2)
    assert b1 == pytest.approx(b2)


def test_update_elo_with_margin_larger_margin_moves_more() -> None:
    # A blowout win should move ratings further than a narrow win.
    a_wide, b_wide = update_elo_with_margin(1500.0, 1450.0, 1.0, 35.0, k=20.0)
    a_narrow, b_narrow = update_elo_with_margin(1500.0, 1450.0, 1.0, 2.0, k=20.0)
    wide_move = abs(a_wide - 1500.0) + abs(b_wide - 1450.0)
    narrow_move = abs(a_narrow - 1500.0) + abs(b_narrow - 1450.0)
    assert wide_move > narrow_move


def test_update_elo_with_margin_underdog_upset_moves_more_than_fav_blowout() -> None:
    # FiveThirtyEight-style MOV dampening makes underdog upsets swing harder
    # than equal-margin favorite wins.
    a_fav, b_fav = update_elo_with_margin(1500.0, 1450.0, 1.0, 30.0, k=20.0)
    a_dog, b_dog = update_elo_with_margin(1400.0, 1600.0, 1.0, 25.0, k=20.0)
    fav_move = abs(a_fav - 1500.0) + abs(b_fav - 1450.0)
    dog_move = abs(a_dog - 1400.0) + abs(b_dog - 1600.0)
    assert dog_move > fav_move


def test_update_elo_with_margin_rejects_negative_margin() -> None:
    with pytest.raises(ValueError, match="margin must be non-negative"):
        update_elo_with_margin(1500.0, 1450.0, 1.0, -5.0)


def test_update_elo_with_margin_rejects_invalid_score() -> None:
    with pytest.raises(ValueError, match="score_a must be 0, 0.5, or 1"):
        update_elo_with_margin(1500.0, 1450.0, 0.3, 10.0)


def test_update_elo_with_margin_pole_underdog_conserves_mass() -> None:
    # The MOV multiplier has a mathematical pole at elo_diff_winner == -2200
    # that is guarded by a clamp inside mov_multiplier.  End-to-end, the
    # update must not crash, must conserve rating mass, and the extreme
    # underdog winner must receive a strictly positive rating gain.
    base_a, base_b = 1400.0, 3600.0  # elo_diff_winner = -2200 exactly
    a, b = update_elo_with_margin(base_a, base_b, 1.0, 30.0, k=20.0)
    assert pytest.approx(a + b) == pytest.approx(base_a + base_b)
    assert a > base_a  # underdog winner gains ratings
    assert b < base_b  # favorite loser sheds ratings


def test_update_elo_with_margin_beyond_pole_stays_stable() -> None:
    # For elo_diff_winner well below the pole (-10 000), the clamp in
    # mov_multiplier keeps the denominator at 1e-3 so the multiplier stays
    # finite and the Elo update completes without crashing or returning nan/inf.
    a, b = update_elo_with_margin(1400.0, 11400.0, 1.0, 30.0, k=20.0)
    assert math.isfinite(a)
    assert math.isfinite(b)
    assert pytest.approx(a + b) == pytest.approx(1400.0 + 11400.0)

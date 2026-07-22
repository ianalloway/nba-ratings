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


def test_mov_multiplier_rejects_negative_margin() -> None:
    with pytest.raises(ValueError):
        mov_multiplier(-1.0, elo_diff_winner=0.0)


def test_update_elo_with_margin_blowout_moves_more_than_narrow_win() -> None:
    narrow_a, narrow_b = update_elo_with_margin(1500.0, 1500.0, 1.0, margin=2.0)
    blowout_a, blowout_b = update_elo_with_margin(1500.0, 1500.0, 1.0, margin=30.0)
    assert (blowout_a - 1500.0) > (narrow_a - 1500.0) > 0
    assert (1500.0 - blowout_b) > (1500.0 - narrow_b) > 0


def test_update_elo_with_margin_zero_margin_means_no_movement() -> None:
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
    assert pytest.approx(expected_margin(100.0)) == 2.5
    assert pytest.approx(expected_margin(0.0)) == 0.0
    assert pytest.approx(expected_margin(200.0)) == 2 * expected_margin(100.0)
    assert pytest.approx(expected_margin(-100.0)) == -expected_margin(100.0)


def test_expected_margin_custom_slope() -> None:
    assert pytest.approx(expected_margin(100.0, margin_per_elo=0.04)) == 4.0


def test_expected_margin_never_divides_by_zero() -> None:
    assert pytest.approx(expected_margin(150.0, margin_per_elo=0.0)) == 0.0


# --- Added coverage: symmetry, conservation, and input validation ---


def test_logistic_win_prob_symmetric_about_even_matchup() -> None:
    for d in (50.0, 150.0, 200.0, -100.0, 400.0):
        assert pytest.approx(logistic_win_prob(d) + logistic_win_prob(-d)) == 1.0


def test_logistic_win_prob_bounds_and_endpoints() -> None:
    assert 0.0 < logistic_win_prob(0.0) < 1.0
    assert logistic_win_prob(10_000.0) > 0.999
    assert logistic_win_prob(-10_000.0) < 0.001


def test_logistic_win_prob_rejects_nonpositive_scale() -> None:
    for scale in (0.0, -1.0, -400.0):
        with pytest.raises(ValueError):
            logistic_win_prob(100.0, scale=scale)


def test_logistic_win_prob_scale_changes_spread() -> None:
    # Smaller scale exaggerates win-probability differences; larger scale
    # flattens the curve toward 0.5.
    diff = 100.0
    narrow = logistic_win_prob(diff, scale=200.0)
    wide = logistic_win_prob(diff, scale=400.0)
    assert 0.5 < narrow < 1.0
    assert 0.5 < wide < 1.0
    assert narrow > wide  # smaller scale → stronger favorite signal


def test_update_elo_zero_sum_conservation() -> None:
    cases = [
        (1500.0, 1500.0, 1.0),
        (1500.0, 1480.0, 0.0),
        (1600.0, 1400.0, 0.5),
    ]
    for ra, rb, score in cases:
        a, b = update_elo(ra, rb, score)
        assert pytest.approx(a + b) == ra + rb


def test_update_elo_with_margin_zero_sum_conservation() -> None:
    cases = [
        (1500.0, 1480.0, 1.0, 30.0),
        (1600.0, 1400.0, 0.0, 15.0),
        (1550.0, 1550.0, 0.5, 5.0),
    ]
    for ra, rb, score, margin in cases:
        a, b = update_elo_with_margin(ra, rb, score, margin)
        assert pytest.approx(a + b) == ra + rb


def test_update_elo_rejects_invalid_score() -> None:
    for bad in (2.0, -1.0, 0.3):
        with pytest.raises(ValueError):
            update_elo(1500.0, 1500.0, bad)


def test_update_elo_rejects_invalid_k_and_scale() -> None:
    for k in (0.0, -5.0):
        with pytest.raises(ValueError):
            update_elo(1500.0, 1500.0, 1.0, k=k)
    for scale in (0.0, -100.0):
        with pytest.raises(ValueError):
            update_elo(1500.0, 1500.0, 1.0, scale=scale)


def test_update_elo_with_margin_rejects_invalid_score_and_params() -> None:
    for bad in (2.0, -1.0, 0.3):
        with pytest.raises(ValueError):
            update_elo_with_margin(1500.0, 1500.0, bad, 10.0)
    for k in (0.0, -5.0):
        with pytest.raises(ValueError):
            update_elo_with_margin(1500.0, 1500.0, 1.0, 10.0, k=k)
    for scale in (0.0, -100.0):
        with pytest.raises(ValueError):
            update_elo_with_margin(1500.0, 1500.0, 1.0, 10.0, scale=scale)


def test_update_elo_winner_always_gains_loser_always_loses() -> None:
    for ra, rb in ((1500.0, 1500.0), (1600.0, 1400.0), (1400.0, 1600.0)):
        a_win, b_win = update_elo(ra, rb, 1.0)
        a_loss, b_loss = update_elo(ra, rb, 0.0)
        assert a_win > ra and b_win < rb
        assert a_loss < ra and b_loss > rb

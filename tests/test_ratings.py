import pytest

from nba_edge.ratings import (
    expected_margin,
    logistic_win_prob,
    update_elo,
)


def test_expected_margin_proportional_to_elo_diff() -> None:
    # Synonym-team edge: 0 Elo diff -> 0 predicted margin
    assert expected_margin(0.0) == 0.0
    # Positive diff predicts a home-favorite margin; sign is preserved
    assert expected_margin(100.0) > 0
    assert expected_margin(-100.0) < 0
    # Linear in the rating difference
    assert expected_margin(200.0) == pytest.approx(2.0 * expected_margin(100.0))


def test_expected_margin_custom_slope() -> None:
    # Default slope is 0.025; a custom slope scales accordingly
    assert expected_margin(100.0, margin_per_elo=0.05) == pytest.approx(5.0)
    assert expected_margin(100.0) == pytest.approx(2.5)


def test_logistic_win_prob_symmetry() -> None:
    # P(home) + P(away) == 1 at any rating diff (away = -diff)
    for diff in (-300.0, -50.0, 0.0, 50.0, 300.0):
        p_home = logistic_win_prob(diff)
        p_away = logistic_win_prob(-diff)
        assert p_home + p_away == pytest.approx(1.0)


def test_logistic_win_prob_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError, match="scale must be positive"):
        logistic_win_prob(100.0, scale=0.0)
    with pytest.raises(ValueError, match="scale must be positive"):
        logistic_win_prob(100.0, scale=-400.0)


def test_update_elo_conserves_rating_mass() -> None:
    # With k applied symmetrically, the sum of the two ratings is preserved
    a, b = update_elo(1500.0, 1450.0, 1.0, k=32.0)
    assert (a + b) == pytest.approx(1500.0 + 1450.0)


def test_update_elo_draw_moves_both_halfway_to_expected() -> None:
    # On a draw the expected winner (higher-rated team) still drifts down and
    # the underdog drifts up by the same magnitude, conserving rating mass.
    a, b = update_elo(1600.0, 1400.0, 0.5)
    drop = 1600.0 - a
    rise = b - 1400.0
    assert drop == pytest.approx(rise)  # symmetric drift
    assert (a + b) == pytest.approx(1600.0 + 1400.0)  # mass conserved


def test_update_elo_rejects_invalid_score() -> None:
    with pytest.raises(ValueError, match="score_a must be 0, 0.5, or 1"):
        update_elo(1500.0, 1500.0, 0.3)


def test_update_elo_rejects_nonpositive_k_and_scale() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        update_elo(1500.0, 1500.0, 1.0, k=0.0)
    with pytest.raises(ValueError, match="scale must be positive"):
        update_elo(1500.0, 1500.0, 1.0, scale=-100.0)

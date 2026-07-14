"""Regression coverage for mov_multiplier's pole / extreme-underdog behavior.

The ratings-validation-symmetry merge refactored test_elo.py and dropped the
tests that guard the -2200 denominator pole inside mov_multiplier. That clamp
keeps the Elo margin-of-victory multiplier finite and strictly positive for
extreme underdog wins; without it a huge upset would raise ZeroDivisionError or
flip the sign of the rating update. This module restores that coverage.
"""

from nba_edge.ratings import mov_multiplier


def test_mov_multiplier_pole_is_finite_and_positive() -> None:
    # The dampening factor 2.2 / (0.001 * elo_diff_winner + 2.2) has a pole at
    # elo_diff_winner == -2200. The clamp in mov_multiplier must keep the
    # multiplier strictly positive and finite (a negative multiplier would flip
    # the sign of the Elo update).
    for diff in (-2199.0, -2200.0, -2201.0, -3000.0, -5000.0):
        mult = mov_multiplier(10.0, elo_diff_winner=diff)
        assert mult > 0.0


def test_mov_multiplier_clamped_below_pole_is_finite() -> None:
    # For extreme underdog wins (elo_diff_winner < -2200) the multiplier stays
    # finite and bounded instead of diverging to +/-inf, so callers can't crash
    # on big upsets.
    mult = mov_multiplier(30.0, elo_diff_winner=-10000.0)
    assert mult > 0.0
    assert abs(mult) < 1e6

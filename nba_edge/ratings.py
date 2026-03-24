"""Elo-style updates and logistic win probability from power difference."""

from __future__ import annotations

import math


def logistic_win_prob(rating_diff: float, scale: float = 400.0) -> float:
    """P(home beats away) given rating_home - rating_away (Elo logistic)."""
    return 1.0 / (1.0 + math.pow(10.0, -rating_diff / scale))


def expected_margin(rating_diff: float, margin_per_elo: float = 0.025) -> float:
    """Toy mapping from Elo diff to predicted point margin (calibrate externally)."""
    return rating_diff * margin_per_elo


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    *,
    k: float = 20.0,
    scale: float = 400.0,
) -> tuple[float, float]:
    """Elo update after match; score_a in {0, 0.5, 1} for loss/draw/win."""
    exp_a = logistic_win_prob(rating_a - rating_b, scale=scale)
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1 - score_a) - (1 - exp_a))
    return new_a, new_b

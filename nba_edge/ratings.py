"""Elo-style updates and logistic win probability from power difference."""

from __future__ import annotations

import math

# Minimum value for the mov_multiplier denominator. Without this clamp the
# formula has a pole at elo_diff_winner == -2200 (0.001 * -2200 + 2.2 == 0)
# and would raise ZeroDivisionError for any more extreme underdog win.
_MIN_MOV_DENOM: float = 1e-3


def logistic_win_prob(rating_diff: float, scale: float = 400.0) -> float:
    """P(home beats away) given rating_home - rating_away (Elo logistic)."""
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    try:
        return 1.0 / (1.0 + math.pow(10.0, -rating_diff / scale))
    except OverflowError:
        # Extreme underdog: 10^(large positive) overflows → probability ≈ 0
        return 0.0 if rating_diff < 0 else 1.0


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
    if score_a not in (0, 0.5, 1):
        raise ValueError(f"score_a must be 0, 0.5, or 1, got {score_a}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    exp_a = logistic_win_prob(rating_a - rating_b, scale=scale)
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1 - score_a) - (1 - exp_a))
    return new_a, new_b


def mov_multiplier(margin: float, elo_diff_winner: float) -> float:
    """FiveThirtyEight-style margin-of-victory multiplier for Elo updates.

    Blowouts move ratings further than narrow wins, but the effect is
    dampened when the winner already held a large rating edge (otherwise
    a heavy favorite blowing out a weak team would gain ratings forever).

    Args:
        margin: Absolute point differential of the game (>= 0).
        elo_diff_winner: rating_winner - rating_loser, before this game.
    """
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")

    # The dampening factor 2.2 / (0.001 * elo_diff_winner + 2.2) has a pole at
    # elo_diff_winner == -2200 (raw ZeroDivisionError) and goes negative for any
    # more extreme underdog win, which would flip the sign of the Elo update.
    # Clamp the denominator so the multiplier stays strictly positive and finite
    # for every float input (behavior for elo_diff_winner > -2200 is unchanged).
    denom = 0.001 * elo_diff_winner + 2.2
    denom = max(denom, _MIN_MOV_DENOM)
    return math.log(margin + 1) * (2.2 / denom)


def update_elo_with_margin(
    rating_a: float,
    rating_b: float,
    score_a: float,
    margin: float,
    *,
    k: float = 20.0,
    scale: float = 400.0,
) -> tuple[float, float]:
    """Elo update scaled by margin of victory (FiveThirtyEight-style).

    Same semantics as `update_elo`, plus `margin`: the game's absolute point
    differential. A 30-point win moves ratings more than a 2-point win.
    """
    if score_a not in (0, 0.5, 1):
        raise ValueError(f"score_a must be 0, 0.5, or 1, got {score_a}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")

    exp_a = logistic_win_prob(rating_a - rating_b, scale=scale)

    if score_a == 0.5:
        mult = 1.0
    else:
        winner_diff = (rating_a - rating_b) if score_a == 1 else (rating_b - rating_a)
        mult = mov_multiplier(margin, winner_diff)

    k_adj = k * mult
    new_a = rating_a + k_adj * (score_a - exp_a)
    new_b = rating_b + k_adj * ((1 - score_a) - (1 - exp_a))
    return new_a, new_b

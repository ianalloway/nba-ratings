"""Installable primitives for NBA-style edge models."""

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
from nba_edge.metrics import brier_score, calibration_curve, log_loss
from nba_edge.ratings import (
    expected_margin,
    logistic_win_prob,
    mov_multiplier,
    update_elo,
    update_elo_with_margin,
)

__all__ = [
    "american_to_decimal",
    "american_to_implied_prob",
    "decimal_to_american",
    "decimal_to_implied_prob",
    "implied_prob_to_american",
    "implied_prob_to_decimal",
    "kelly_fraction",
    "kelly_parlay",
    "parlay_odds",
    "remove_vig",
    "brier_score",
    "calibration_curve",
    "log_loss",
    "expected_margin",
    "logistic_win_prob",
    "mov_multiplier",
    "update_elo",
    "update_elo_with_margin",
]
__version__ = "0.1.0"

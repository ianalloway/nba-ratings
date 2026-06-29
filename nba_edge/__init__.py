"""Installable primitives for NBA-style edge models."""

from nba_edge.kelly import (
    american_to_decimal,
    american_to_implied_prob,
    decimal_to_american,
    decimal_to_implied_prob,
    implied_prob_to_american,
    implied_prob_to_decimal,
    kelly_fraction,
)
from nba_edge.metrics import brier_score, log_loss
from nba_edge.ratings import expected_margin, logistic_win_prob, update_elo

__all__ = [
    "american_to_decimal",
    "american_to_implied_prob",
    "decimal_to_american",
    "decimal_to_implied_prob",
    "implied_prob_to_american",
    "implied_prob_to_decimal",
    "kelly_fraction",
    "brier_score",
    "log_loss",
    "expected_margin",
    "logistic_win_prob",
    "update_elo",
]
__version__ = "0.1.0"

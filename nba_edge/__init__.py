"""Installable primitives for NBA-style edge models."""

from nba_edge.kelly import kelly_fraction, american_to_implied_prob
from nba_edge.ratings import (
    expected_margin,
    logistic_win_prob,
    mov_multiplier,
    update_elo,
    update_elo_with_margin,
)

__all__ = [
    "american_to_implied_prob",
    "kelly_fraction",
    "expected_margin",
    "logistic_win_prob",
    "mov_multiplier",
    "update_elo",
    "update_elo_with_margin",
]
__version__ = "0.1.0"

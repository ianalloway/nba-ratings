"""Installable primitives for NBA-style edge models."""

from nba_edge.clv import (
    ClvBookSummary,
    RollingClvSummary,
    clv_book_summary,
    clv_edge,
    clv_pts,
    rolling_clv,
)
from nba_edge.kelly import (
    american_to_decimal,
    american_to_implied_prob,
    decimal_to_american,
    decimal_to_implied_prob,
    fair_american_odds,
    implied_prob_to_american,
    implied_prob_to_decimal,
    kelly_fraction,
    kelly_parlay,
    parlay_odds,
    remove_vig,
)
from nba_edge.metrics import CalibrationBin, brier_score, calibration_curve, log_loss
from nba_edge.ratings import (
    expected_margin,
    logistic_win_prob,
    mov_multiplier,
    update_elo,
    update_elo_with_margin,
)

__all__ = [
    "CalibrationBin",
    "ClvBookSummary",
    "RollingClvSummary",
    "american_to_decimal",
    "american_to_implied_prob",
    "brier_score",
    "calibration_curve",
    "clv_book_summary",
    "clv_edge",
    "clv_pts",
    "decimal_to_american",
    "decimal_to_implied_prob",
    "expected_margin",
    "fair_american_odds",
    "implied_prob_to_american",
    "implied_prob_to_decimal",
    "kelly_fraction",
    "kelly_parlay",
    "log_loss",
    "logistic_win_prob",
    "mov_multiplier",
    "parlay_odds",
    "remove_vig",
    "rolling_clv",
    "update_elo",
    "update_elo_with_margin",
]
__version__ = "0.2.0"

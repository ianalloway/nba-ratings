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
    expected_margin_with_rest,
    logistic_win_prob,
    mov_multiplier,
    update_elo,
    update_elo_with_margin,
)
from nba_edge.rest import (
    DEFAULT_REST_COEFFS,
    FatigueFeatures,
    RestAdjustmentCoeffs,
    ScheduleGame,
    compute_fatigue_features,
    haversine_miles,
    rest_spread_adjustment,
)

__all__ = [
    "DEFAULT_REST_COEFFS",
    "CalibrationBin",
    "ClvBookSummary",
    "FatigueFeatures",
    "RestAdjustmentCoeffs",
    "RollingClvSummary",
    "ScheduleGame",
    "american_to_decimal",
    "american_to_implied_prob",
    "brier_score",
    "calibration_curve",
    "clv_book_summary",
    "clv_edge",
    "clv_pts",
    "compute_fatigue_features",
    "decimal_to_american",
    "decimal_to_implied_prob",
    "expected_margin",
    "expected_margin_with_rest",
    "fair_american_odds",
    "haversine_miles",
    "implied_prob_to_american",
    "implied_prob_to_decimal",
    "kelly_fraction",
    "kelly_parlay",
    "log_loss",
    "logistic_win_prob",
    "mov_multiplier",
    "parlay_odds",
    "remove_vig",
    "rest_spread_adjustment",
    "rolling_clv",
    "update_elo",
    "update_elo_with_margin",
]
__version__ = "0.2.0"

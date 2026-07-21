"""Evaluation metrics for ratings and win-probability forecasts."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TypedDict


class CalibrationBin(TypedDict):
    bin_low: float
    bin_high: float
    mean_predicted: float
    mean_actual: float
    count: int


def brier_score(predictions: float | Iterable[float], outcomes: float | Iterable[float]) -> float:
    """Calculate the Brier Score for probability forecasts.

    The Brier Score is the mean squared error of the probability forecasts:
        BS = (1/N) * sum((p_i - y_i)^2)

    A lower Brier Score (closer to 0) indicates more accurate probability forecasts.

    Args:
        predictions: Predicted probabilities, either a single float or an iterable of floats.
        outcomes: Actual outcomes (1.0 for a win/event, 0.0 for a loss/no-event),
                 either a single float or an iterable of floats.

    Returns:
        The Brier Score as a float.
    """
    if isinstance(predictions, (int, float)):
        preds = [float(predictions)]
    else:
        preds = [float(p) for p in predictions]

    if isinstance(outcomes, (int, float)):
        outs = [float(outcomes)]
    else:
        outs = [float(o) for o in outcomes]

    if len(preds) != len(outs):
        raise ValueError(
            f"Lengths of predictions ({len(preds)}) and outcomes ({len(outs)}) must match"
        )

    if not preds:
        raise ValueError("Predictions and outcomes lists cannot be empty")

    total_se = 0.0
    for p, o in zip(preds, outs):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Predicted probability must be between 0 and 1, got {p}")
        if o not in (0.0, 1.0):
            raise ValueError(f"Outcome must be 0 or 1, got {o}")
        total_se += (p - o) ** 2

    return total_se / len(preds)


def log_loss(
    predictions: float | Iterable[float], outcomes: float | Iterable[float], eps: float = 1e-15
) -> float:
    """Calculate binary cross-entropy (log loss) for probability forecasts.

    Log loss penalizes highly confident incorrect predictions:
        LL = - (1/N) * sum(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))

    Args:
        predictions: Predicted probabilities, either a single float or an iterable of floats.
        outcomes: Actual outcomes (1.0 for a win, 0.0 for a loss),
                 either a single float or an iterable of floats.
        eps: Small epsilon to clip predictions to avoid log(0).

    Returns:
        The mean log loss as a float.
    """
    if isinstance(predictions, (int, float)):
        preds = [float(predictions)]
    else:
        preds = [float(p) for p in predictions]

    if isinstance(outcomes, (int, float)):
        outs = [float(outcomes)]
    else:
        outs = [float(o) for o in outcomes]

    if len(preds) != len(outs):
        raise ValueError(
            f"Lengths of predictions ({len(preds)}) and outcomes ({len(outs)}) must match"
        )

    if not preds:
        raise ValueError("Predictions and outcomes lists cannot be empty")

    total_loss = 0.0
    for p, o in zip(preds, outs):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Predicted probability must be between 0 and 1, got {p}")
        if o not in (0.0, 1.0):
            raise ValueError(f"Outcome must be 0 or 1, got {o}")
        # Clip probability to avoid log(0) or log(1)
        p = max(eps, min(1.0 - eps, p))
        total_loss += o * math.log(p) + (1.0 - o) * math.log(1.0 - p)

    return -total_loss / len(preds)


def calibration_curve(
    predictions: Iterable[float],
    outcomes: Iterable[float],
    bins: int = 10,
) -> list[CalibrationBin]:
    """Partition probability predictions into equal-width bins and compare
    mean predicted probability against mean actual win rate per bin.

    Returns the data needed for a reliability diagram (calibration plot).
    A perfectly calibrated model has mean_predicted ≈ mean_actual in every bin.
    Overconfident models show mean_predicted > mean_actual at high probabilities.

    Empty bins (no predictions in that range) are omitted from the result.

    Args:
        predictions: Predicted win probabilities (each in [0, 1]).
        outcomes:    Binary outcomes — 1.0 for wins, 0.0 for losses.
        bins:        Number of equal-width probability bins (default 10 → 0-10%, 10-20%, …).

    Returns:
        List of CalibrationBin dicts sorted by bin_low, one per non-empty bin.

    Example::

        curve = calibration_curve([0.6, 0.7, 0.4, 0.55], [1.0, 1.0, 0.0, 1.0])
        # [{'bin_low': 0.4, 'bin_high': 0.5, 'mean_predicted': 0.4, 'mean_actual': 0.0, 'count': 1},
        #  {'bin_low': 0.5, 'bin_high': 0.6,
        #   'mean_predicted': 0.55, 'mean_actual': 1.0, 'count': 1},
        #  {'bin_low': 0.6, 'bin_high': 0.7, 'mean_predicted': 0.6, 'mean_actual': 1.0, 'count': 1},
        #  {'bin_low': 0.7, 'bin_high': 0.8, 'mean_predicted': 0.7, 'mean_actual': 1.0, 'count': 1}]
    """
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")

    preds = [float(p) for p in predictions]
    outs = [float(o) for o in outcomes]

    if len(preds) != len(outs):
        raise ValueError(
            f"Lengths of predictions ({len(preds)}) and outcomes ({len(outs)}) must match"
        )
    if not preds:
        raise ValueError("predictions and outcomes cannot be empty")

    for p in preds:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Predicted probability must be in [0, 1], got {p}")
    for o in outs:
        if o not in (0.0, 1.0):
            raise ValueError(f"Outcome must be 0 or 1, got {o}")

    width = 1.0 / bins
    # bucket[i] holds (sum_predicted, sum_actual, count) for bin i
    buckets: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(bins)]

    for p, o in zip(preds, outs):
        # Edge case: p == 1.0 goes into the last bin
        idx = min(int(p / width), bins - 1)
        buckets[idx][0] += p
        buckets[idx][1] += o
        buckets[idx][2] += 1.0

    result: list[CalibrationBin] = []
    for i, (sum_p, sum_o, count) in enumerate(buckets):
        if count == 0:
            continue
        result.append(
            CalibrationBin(
                bin_low=round(i * width, 10),
                bin_high=round((i + 1) * width, 10),
                mean_predicted=round(sum_p / count, 10),
                mean_actual=round(sum_o / count, 10),
                count=int(count),
            )
        )

    return result

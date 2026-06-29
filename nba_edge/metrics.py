"""Evaluation metrics for ratings and win-probability forecasts."""

from __future__ import annotations

import math
from collections.abc import Iterable


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
        raise ValueError(f"Lengths of predictions ({len(preds)}) and outcomes ({len(outs)}) must match")

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


def log_loss(predictions: float | Iterable[float], outcomes: float | Iterable[float], eps: float = 1e-15) -> float:
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
        raise ValueError(f"Lengths of predictions ({len(preds)}) and outcomes ({len(outs)}) must match")

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

import pytest
import math
from nba_edge.metrics import brier_score, log_loss


def test_brier_score_single() -> None:
    # Perfect predictions
    assert brier_score(1.0, 1.0) == 0.0
    assert brier_score(0.0, 0.0) == 0.0

    # Completely wrong predictions
    assert brier_score(1.0, 0.0) == 1.0
    assert brier_score(0.0, 1.0) == 1.0

    # 50-50 guess
    assert brier_score(0.5, 1.0) == 0.25


def test_brier_score_multiple() -> None:
    preds = [0.8, 0.1, 0.5]
    outs = [1.0, 0.0, 1.0]
    # Expected: ((0.8-1)^2 + (0.1-0)^2 + (0.5-1)^2) / 3 = (0.04 + 0.01 + 0.25) / 3 = 0.3 / 3 = 0.1
    assert pytest.approx(brier_score(preds, outs)) == 0.1


def test_brier_score_invalid() -> None:
    with pytest.raises(ValueError, match="Lengths.*must match"):
        brier_score([0.5, 0.6], [1.0])

    with pytest.raises(ValueError, match="cannot be empty"):
        brier_score([], [])

    with pytest.raises(ValueError, match="must be between 0 and 1"):
        brier_score(1.5, 1.0)

    with pytest.raises(ValueError, match="Outcome must be 0 or 1"):
        brier_score(0.5, 0.5)


def test_log_loss_single() -> None:
    # Perfect predictions should have low loss
    assert pytest.approx(log_loss(0.999999999, 1.0), abs=1e-5) == 0.0

    # 50-50 prediction: -ln(0.5) = 0.693147...
    assert pytest.approx(log_loss(0.5, 1.0)) == math.log(2)


def test_log_loss_multiple() -> None:
    preds = [0.8, 0.2, 0.5]
    outs = [1.0, 0.0, 1.0]
    # Expected: -(1 * ln(0.8) + 1 * ln(0.8) + 1 * ln(0.5)) / 3
    # ln(0.8) = -0.22314355
    # ln(0.5) = -0.69314718
    # Total loss = -(-0.22314355 - 0.22314355 - 0.69314718) = 1.13943428
    # Average = 0.3798114
    expected = -(math.log(0.8) + math.log(0.8) + math.log(0.5)) / 3.0
    assert pytest.approx(log_loss(preds, outs)) == expected


def test_log_loss_invalid() -> None:
    with pytest.raises(ValueError, match="Lengths.*must match"):
        log_loss([0.5, 0.6], [1.0])

    with pytest.raises(ValueError, match="cannot be empty"):
        log_loss([], [])

    with pytest.raises(ValueError, match="must be between 0 and 1"):
        log_loss(1.5, 1.0)

    with pytest.raises(ValueError, match="Outcome must be 0 or 1"):
        log_loss(0.5, 0.5)

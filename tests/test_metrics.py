import math

import pytest

from nba_edge import CalibrationBin
from nba_edge.metrics import brier_score, calibration_curve, log_loss


def test_calibration_bin_importable_from_package() -> None:
    """CalibrationBin must be reachable from the top-level nba_edge namespace
    so callers can annotate variables holding calibration_curve results."""
    expected_keys = {"bin_low", "bin_high", "mean_predicted", "mean_actual", "count"}
    assert set(CalibrationBin.__annotations__) == expected_keys


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


def test_log_loss_eps_clipping_at_boundaries() -> None:
    # log_loss clips predictions into [eps, 1-eps] before taking the log so
    # boundary values p=0.0 and p=1.0 do not raise log(0). Default eps is 1e-15.
    assert log_loss(1.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert log_loss(0.0, 0.0) == pytest.approx(0.0, abs=1e-12)

    # A wider eps changes the clipping threshold: p=1.0 is clipped to 1-eps,
    # so a larger eps produces a slightly larger (but still tiny) penalty.
    default_eps = log_loss(1.0, 1.0)
    wider_eps = log_loss(1.0, 1.0, eps=1e-6)
    assert wider_eps > default_eps


def test_log_loss_boundary_wrong_prediction_clipped() -> None:
    # Confident wrong predictions at p=0.0 and p=1.0 must stay finite
    # (thanks to the [eps, 1-eps] clip inside log_loss). Each direction
    # is tested independently because the float representation of 1-eps
    # is not bitwise symmetric with eps, so the two paths land on
    # slightly different but equally valid finite values.
    for p, o in ((1.0, 0.0), (0.0, 1.0)):
        val = log_loss(p, o)
        assert math.isfinite(val)
        assert val > 10.0  # strongly penalized, not a no-op

    # A wider eps makes the penalty slightly smaller than the default.
    default_wrong = log_loss(1.0, 0.0)
    wider_wrong = log_loss(1.0, 0.0, eps=1e-6)
    assert wider_wrong < default_wrong


def test_log_loss_rejects_non_positive_eps() -> None:
    # eps <= 0 defeats the [eps, 1-eps] clipping and allows log(0) to crash.
    for bad in (0.0, -0.01, -1e-15):
        with pytest.raises(ValueError, match="eps must be a finite value in"):
            log_loss(1.0, 1.0, eps=bad)


def test_log_loss_rejects_eps_at_or_above_half() -> None:
    # eps >= 0.5 collapses the clipping interval [eps, 1-eps] to a point or
    # an empty range, silently flattening every prediction to a single value.
    for bad in (0.5, 0.6, 0.999999, 1.0, 2.0):
        with pytest.raises(ValueError, match="eps must be a finite value in"):
            log_loss(0.5, 1.0, eps=bad)


def test_log_loss_rejects_nonfinite_eps() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="eps must be a finite value in"):
            log_loss(1.0, 1.0, eps=bad)


def test_log_loss_default_eps_does_not_raise() -> None:
    # The documented default (1e-15) must still work for boundary predictions.
    assert log_loss(1.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert log_loss(0.0, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_calibration_curve_basic_binning() -> None:
    # Two predictions in 0.6-0.7 bin, one in 0.8-0.9 bin
    preds = [0.65, 0.62, 0.85]
    outs = [1.0, 0.0, 1.0]
    curve = calibration_curve(preds, outs, bins=10)
    # Only bins 6 and 8 are occupied
    assert len(curve) == 2
    b6, b8 = curve[0], curve[1]
    assert b6["bin_low"] == pytest.approx(0.6)
    assert b6["bin_high"] == pytest.approx(0.7)
    assert b6["count"] == 2
    assert b6["mean_predicted"] == pytest.approx(0.635)
    assert b6["mean_actual"] == pytest.approx(0.5)
    assert b8["count"] == 1
    assert b8["mean_actual"] == pytest.approx(1.0)


def test_calibration_curve_perfect_model() -> None:
    # Predictions perfectly track actual rates — each bin maps to its midpoint
    preds = [0.25, 0.25, 0.75, 0.75]
    outs = [1.0, 0.0, 1.0, 1.0]
    curve = calibration_curve(preds, outs, bins=4)
    assert len(curve) == 2
    low_bin = next(b for b in curve if b["bin_low"] == pytest.approx(0.25))
    assert low_bin["mean_predicted"] == pytest.approx(0.25)


def test_calibration_curve_p1_lands_in_last_bin() -> None:
    curve = calibration_curve([1.0, 1.0], [1.0, 1.0], bins=10)
    assert len(curve) == 1
    assert curve[0]["bin_low"] == pytest.approx(0.9)
    assert curve[0]["count"] == 2


def test_calibration_curve_empty_bins_omitted() -> None:
    # All predictions in one bin → only one entry returned
    curve = calibration_curve([0.55, 0.58], [1.0, 0.0], bins=10)
    assert len(curve) == 1


def test_calibration_curve_single_bin() -> None:
    curve = calibration_curve([0.3, 0.7], [0.0, 1.0], bins=1)
    assert len(curve) == 1
    assert curve[0]["bin_low"] == pytest.approx(0.0)
    assert curve[0]["bin_high"] == pytest.approx(1.0)
    assert curve[0]["count"] == 2


def test_calibration_curve_returns_valid_calibration_bins() -> None:
    """Every non-empty bin returned by calibration_curve must be a valid
    CalibrationBin dict containing all five required keys, so callers can
    destructure results without KeyError."""
    required_keys = set(CalibrationBin.__annotations__)
    curve = calibration_curve(
        [0.15, 0.25, 0.35, 0.75, 0.85, 0.95],
        [0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        bins=5,
    )
    assert curve  # non-empty
    for bin_ in curve:
        assert set(bin_.keys()) == required_keys


def test_calibration_curve_validation() -> None:
    with pytest.raises(ValueError, match="bins must be at least 1"):
        calibration_curve([0.5], [1.0], bins=0)

    with pytest.raises(ValueError, match="Lengths.*must match"):
        calibration_curve([0.5, 0.6], [1.0])

    with pytest.raises(ValueError, match="cannot be empty"):
        calibration_curve([], [])

    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        calibration_curve([1.5], [1.0])

    with pytest.raises(ValueError, match="Outcome must be 0 or 1"):
        calibration_curve([0.5], [0.5])

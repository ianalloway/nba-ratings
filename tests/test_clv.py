"""Tests for closing-line value (CLV) helpers."""

from __future__ import annotations

import math

import pytest

from nba_edge import (
    ClvBookSummary,
    RollingClvSummary,
    clv_book_summary,
    clv_edge,
    clv_pts,
    rolling_clv,
)
from nba_edge.kelly import american_to_implied_prob


def test_rolling_clv_summary_keys() -> None:
    expected = {
        "edges",
        "hit_rate",
        "mean_clv",
        "current_streak",
        "rolling_hit_rate",
        "rolling_mean_clv",
    }
    assert set(RollingClvSummary.__annotations__) == expected


def test_clv_book_summary_keys() -> None:
    expected = {
        "edges_pts",
        "hit_rate",
        "mean_clv_pts",
        "beats",
        "misses",
        "pushes",
        "current_streak",
    }
    assert set(ClvBookSummary.__annotations__) == expected


def test_clv_edge_beat_the_close_favorite() -> None:
    # Took -110, closed -120 → open was better (lower implied).
    edge = clv_edge(-110, -120)
    assert edge == pytest.approx(american_to_implied_prob(-120) - american_to_implied_prob(-110))
    assert edge > 0.0


def test_clv_pts_matches_ai_advantage_convention() -> None:
    """clv_pts is clv_edge * 100 — the ai-advantage closeLineValuePts scale."""
    edge = clv_edge(-110, -120)
    pts = clv_pts(-110, -120)
    assert pts == pytest.approx(edge * 100.0)
    assert pts > 0.0
    # Unchanged line → 0 pts
    assert clv_pts(-110, -110) == pytest.approx(0.0)


def test_clv_edge_missed_close_dog() -> None:
    # Took +150, closed +180 → close got longer; open was worse.
    edge = clv_edge(150, 180)
    assert edge < 0.0


def test_clv_edge_unchanged_line() -> None:
    assert clv_edge(-110, -110) == pytest.approx(0.0)
    assert clv_edge(140, 140) == pytest.approx(0.0)


def test_clv_edge_rejects_invalid_american() -> None:
    with pytest.raises(ValueError, match="between -100 and 100"):
        clv_edge(-50, -110)
    with pytest.raises(ValueError, match="finite"):
        clv_edge(float("nan"), -110)


def test_rolling_clv_from_odds_pairs() -> None:
    # Bet1: -110 → -120 beat; Bet2: -105 → -100 miss; Bet3: +140 → +130 beat
    summary = rolling_clv([-110, -105, 140], [-120, -100, 130])
    assert len(summary["edges"]) == 3
    assert summary["edges"][0] > 0.0
    assert summary["edges"][1] < 0.0
    assert summary["edges"][2] > 0.0
    assert summary["hit_rate"] == pytest.approx(2 / 3)
    assert summary["mean_clv"] == pytest.approx(sum(summary["edges"]) / 3)
    assert summary["current_streak"] == 1  # last bet is a beat
    assert len(summary["rolling_hit_rate"]) == 3
    assert len(summary["rolling_mean_clv"]) == 3
    # Expanding window: first position hit rate is 1.0
    assert summary["rolling_hit_rate"][0] == pytest.approx(1.0)
    assert summary["rolling_hit_rate"][-1] == pytest.approx(2 / 3)


def test_rolling_clv_from_precomputed_edges() -> None:
    summary = rolling_clv([0.02, -0.01, 0.03, 0.01])
    assert summary["edges"] == [0.02, -0.01, 0.03, 0.01]
    assert summary["hit_rate"] == pytest.approx(0.75)
    assert summary["mean_clv"] == pytest.approx(0.0125)
    assert summary["current_streak"] == 2  # last two are beats


def test_rolling_clv_negative_streak() -> None:
    summary = rolling_clv([0.01, -0.02, -0.03])
    assert summary["current_streak"] == -2


def test_rolling_clv_zero_edge_counts_as_miss() -> None:
    summary = rolling_clv([0.0, 0.0])
    assert summary["hit_rate"] == pytest.approx(0.0)
    assert summary["current_streak"] == -2


def test_rolling_clv_fixed_window() -> None:
    edges = [0.01, -0.02, 0.03, -0.04, 0.05]
    summary = rolling_clv(edges, window=2)
    assert summary["rolling_hit_rate"][0] == pytest.approx(1.0)  # [0.01]
    assert summary["rolling_hit_rate"][1] == pytest.approx(0.5)  # [0.01, -0.02]
    assert summary["rolling_hit_rate"][2] == pytest.approx(0.5)  # [-0.02, 0.03]
    assert summary["rolling_hit_rate"][3] == pytest.approx(0.5)  # [0.03, -0.04]
    assert summary["rolling_hit_rate"][4] == pytest.approx(0.5)  # [-0.04, 0.05]
    assert summary["rolling_mean_clv"][1] == pytest.approx((-0.01) / 2)
    assert summary["rolling_mean_clv"][4] == pytest.approx(0.01 / 2)


def test_rolling_clv_length_mismatch() -> None:
    with pytest.raises(ValueError, match=r"Lengths.*must match"):
        rolling_clv([-110, -105], [-120])


def test_rolling_clv_empty() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        rolling_clv([])


def test_rolling_clv_rejects_bad_window() -> None:
    with pytest.raises(ValueError, match="window must be an int"):
        rolling_clv([0.01], window=0)
    with pytest.raises(ValueError, match="window must be an int"):
        rolling_clv([0.01], window=-1)
    with pytest.raises(ValueError, match="window must be an int"):
        rolling_clv([0.01], window=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="window must be an int"):
        rolling_clv([0.01], window=True)  # type: ignore[arg-type]


def test_rolling_clv_rejects_nonfinite_precomputed_edges() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="CLV edge must be finite"):
            rolling_clv([0.01, bad])


def test_rolling_clv_single_beat() -> None:
    summary = rolling_clv([0.05])
    assert summary["hit_rate"] == 1.0
    assert summary["mean_clv"] == pytest.approx(0.05)
    assert summary["current_streak"] == 1
    assert summary["rolling_hit_rate"] == [1.0]
    assert summary["rolling_mean_clv"] == [pytest.approx(0.05)]


def test_clv_edge_symmetric_vig_neutral_move() -> None:
    # Same magnitude move both ways should be finite and antisymmetric.
    up = clv_edge(-110, -120)
    down = clv_edge(-120, -110)
    assert math.isfinite(up) and math.isfinite(down)
    assert up == pytest.approx(-down)


def test_clv_book_summary_from_odds_pairs() -> None:
    # Clear beat, clear miss, near-zero push under the 0.05 pts default band.
    # -110 → -120 is a real beat (~+2.16 pts); -105 → -100 is a miss;
    # identical open/close is a push.
    book = clv_book_summary([-110, -105, -110], [-120, -100, -110])
    assert len(book["edges_pts"]) == 3
    assert book["edges_pts"][0] == pytest.approx(clv_pts(-110, -120))
    assert book["beats"] == 1
    assert book["misses"] == 1
    assert book["pushes"] == 1
    assert book["hit_rate"] == pytest.approx(1 / 3)
    assert book["mean_clv_pts"] == pytest.approx(sum(book["edges_pts"]) / 3)
    assert book["beats"] + book["misses"] + book["pushes"] == 3


def test_clv_book_summary_push_threshold_matches_ai_advantage() -> None:
    # ai-advantage closeVerdict: |clvPts| < 0.05 → push.
    # A microscopic move should count as push at the default threshold and as
    # a beat when the threshold is lowered to 0.
    tiny_pts = 0.02  # within default 0.05 band
    book = clv_book_summary([tiny_pts])
    assert book["pushes"] == 1
    assert book["beats"] == 0
    assert book["hit_rate"] == pytest.approx(0.0)

    strict = clv_book_summary([tiny_pts], push_threshold_pts=0.0)
    assert strict["beats"] == 1
    assert strict["pushes"] == 0
    assert strict["hit_rate"] == pytest.approx(1.0)


def test_clv_book_summary_from_precomputed_pts() -> None:
    book = clv_book_summary([2.0, -1.5, 0.01, 3.0])
    assert book["edges_pts"] == [2.0, -1.5, 0.01, 3.0]
    assert book["beats"] == 2
    assert book["misses"] == 1
    assert book["pushes"] == 1  # 0.01 within 0.05
    assert book["current_streak"] == 1  # last is a beat


def test_clv_book_summary_boundary_at_push_threshold_is_beat() -> None:
    # ai-advantage uses strict |clv| < 0.05 for push, so exactly 0.05 is a beat.
    book = clv_book_summary([0.05, -0.05, 0.049])
    assert book["beats"] == 1
    assert book["misses"] == 1
    assert book["pushes"] == 1


def test_clv_book_summary_streak_treats_push_as_miss() -> None:
    book = clv_book_summary([2.0, 0.01, 0.0])
    assert book["current_streak"] == -2  # trailing push + zero


def test_clv_book_summary_validation() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        clv_book_summary([])
    with pytest.raises(ValueError, match=r"Lengths.*must match"):
        clv_book_summary([-110], [-120, -115])
    with pytest.raises(ValueError, match="push_threshold_pts"):
        clv_book_summary([1.0], push_threshold_pts=-0.1)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="push_threshold_pts"):
            clv_book_summary([1.0], push_threshold_pts=bad)
        with pytest.raises(ValueError, match="CLV pts must be finite"):
            clv_book_summary([1.0, bad])


def test_clv_book_summary_consistent_with_clv_pts_mean() -> None:
    opens = [-110, -105, 140, -108]
    closes = [-115, -100, 130, -112]
    book = clv_book_summary(opens, closes)
    expected_mean = sum(clv_pts(o, c) for o, c in zip(opens, closes, strict=True)) / len(opens)
    assert book["mean_clv_pts"] == pytest.approx(expected_mean)
    # Probability-point rolling mean * 100 equals book mean pts.
    rolling = rolling_clv(opens, closes)
    assert book["mean_clv_pts"] == pytest.approx(rolling["mean_clv"] * 100.0)

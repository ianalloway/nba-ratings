"""Closing-line value (CLV) helpers for open vs close American odds."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypedDict

from nba_edge.kelly import american_to_implied_prob

__all__ = [
    "ClvBookSummary",
    "RollingClvSummary",
    "clv_book_summary",
    "clv_edge",
    "clv_pts",
    "rolling_clv",
]

# Default push band in percentage points — matches ai-advantage closeVerdict
# (|clvPts| < 0.05 → "push" / matched close).
_DEFAULT_PUSH_THRESHOLD_PTS: float = 0.05


class RollingClvSummary(TypedDict):
    """Aggregate and rolling CLV stats for a sequence of bets.

    Attributes:
        edges: Per-bet CLV in probability points (close implied - open implied).
            Positive means the open price beat the close.
        hit_rate: Fraction of bets with strictly positive CLV.
        mean_clv: Arithmetic mean of ``edges``.
        current_streak: Trailing streak ending at the last bet. Positive when
            the last bet beat the close (and counts consecutive beats);
            negative when it missed (consecutive non-positive edges); ``0``
            when there are no bets.
        rolling_hit_rate: Per-position hit rate over the trailing window
            (or expanding history when ``window`` is ``None``).
        rolling_mean_clv: Per-position mean CLV over the same window.

    """

    edges: list[float]
    hit_rate: float
    mean_clv: float
    current_streak: int
    rolling_hit_rate: list[float]
    rolling_mean_clv: list[float]


class ClvBookSummary(TypedDict):
    """Book-level CLV summary using the ai-advantage ``clvPts`` convention.

    Attributes:
        edges_pts: Per-bet CLV in percentage points
            (``(implied(close) - implied(open)) * 100``).
        hit_rate: Fraction of bets classified as beats
            (``edge_pts > push_threshold_pts``).
        mean_clv_pts: Arithmetic mean of ``edges_pts``.
        beats: Count of bets with ``edge_pts > push_threshold_pts``.
        misses: Count of bets with ``edge_pts < -push_threshold_pts``.
        pushes: Count of bets with ``|edge_pts| < push_threshold_pts``
            (strict inequality, matching ai-advantage ``closeVerdict``).
        current_streak: Trailing beat/miss streak (pushes count as misses),
            same sign convention as :class:`RollingClvSummary`.

    """

    edges_pts: list[float]
    hit_rate: float
    mean_clv_pts: float
    beats: int
    misses: int
    pushes: int
    current_streak: int


def clv_edge(open_american: float, close_american: float) -> float:
    """Closing-line value in probability points for one side of a market.

    ``clv = implied_prob(close) - implied_prob(open)``.

    A positive value means the open American price was better than the close
    (lower implied probability at the open), i.e. you beat the closing line.
    Zero means the line did not move; negative means the close was better than
    the price you took.

    Args:
        open_american: American odds at bet placement (e.g. ``-110`` or ``+150``).
        close_american: American odds at market close for the same side.

    Returns:
        CLV in probability points (e.g. ``0.02`` ≈ +2 pp).

    """
    p_open = american_to_implied_prob(open_american)
    p_close = american_to_implied_prob(close_american)
    return p_close - p_open


def clv_pts(open_american: float, close_american: float) -> float:
    """Closing-line value in percentage points (ai-advantage ``clvPts``).

    ``clv_pts = (implied_prob(close) - implied_prob(open)) * 100``.

    Identical to :func:`clv_edge` scaled by 100 so display units match
    ai-advantage's ``closeLineValuePts`` / ``formatCloseBadge`` (e.g. ``+1.5 pts``).

    Args:
        open_american: American odds at bet placement.
        close_american: American odds at market close for the same side.

    Returns:
        CLV in percentage points (e.g. ``2.0`` for a +2 pp edge).

    """
    return clv_edge(open_american, close_american) * 100.0


def _current_streak(edges: Sequence[float], *, hit_threshold: float = 0.0) -> int:
    """Trailing streak of hits (edge > threshold) or misses (edge <= threshold)."""
    if not edges:
        return 0
    last_hit = edges[-1] > hit_threshold
    streak = 0
    for edge in reversed(edges):
        if (edge > hit_threshold) != last_hit:
            break
        streak += 1
    return streak if last_hit else -streak


def _is_clv_beat(edge_pts: float, push_threshold_pts: float) -> bool:
    """True when edge clears the push band on the positive side (ai-advantage)."""
    return abs(edge_pts) >= push_threshold_pts and edge_pts > 0.0


def _current_streak_with_push(edges_pts: Sequence[float], push_threshold_pts: float) -> int:
    """Beat/miss streak where pushes (and misses) break a beat run."""
    if not edges_pts:
        return 0
    last_hit = _is_clv_beat(edges_pts[-1], push_threshold_pts)
    streak = 0
    for edge in reversed(edges_pts):
        if _is_clv_beat(edge, push_threshold_pts) != last_hit:
            break
        streak += 1
    return streak if last_hit else -streak


def _window_slice(edges: Sequence[float], index: int, window: int | None) -> Sequence[float]:
    if window is None:
        return edges[: index + 1]
    start = max(0, index + 1 - window)
    return edges[start : index + 1]


def rolling_clv(
    opens: Sequence[float],
    closes: Sequence[float] | None = None,
    *,
    window: int | None = None,
) -> RollingClvSummary:
    """Rolling hit rate, mean CLV, and current beat-the-close streak.

    Pass parallel American open/close odds, or pass precomputed CLV edges as
    ``opens`` with ``closes=None``.

    Args:
        opens: Open American odds for each bet, or already-computed CLV edges
            when ``closes`` is omitted.
        closes: Close American odds paired with ``opens``. When ``None``,
            ``opens`` is treated as a sequence of CLV edges (probability
            points from :func:`clv_edge`).
        window: Fixed trailing window length for ``rolling_hit_rate`` /
            ``rolling_mean_clv``. ``None`` uses an expanding window from the
            first bet through each position.

    Returns:
        A :class:`RollingClvSummary` with per-bet edges and summary stats.

    Example::

        summary = rolling_clv([-110, -105, +140], [-115, -110, +130], window=2)
        # summary["hit_rate"], summary["mean_clv"], summary["current_streak"]

        from_edges = rolling_clv([0.01, -0.02, 0.03])  # precomputed edges

    """
    if window is not None:
        if not isinstance(window, int) or isinstance(window, bool):
            raise ValueError(f"window must be an int >= 1 (or None), got {window!r}")
        if window < 1:
            raise ValueError(f"window must be an int >= 1 (or None), got {window}")

    open_list = [float(x) for x in opens]

    if closes is None:
        edges = open_list
        for edge in edges:
            if not math.isfinite(edge):
                raise ValueError(f"CLV edge must be finite, got {edge}")
    else:
        close_list = [float(x) for x in closes]
        if len(open_list) != len(close_list):
            raise ValueError(
                f"Lengths of opens ({len(open_list)}) and closes ({len(close_list)}) must match"
            )
        edges = [clv_edge(o, c) for o, c in zip(open_list, close_list, strict=True)]

    if not edges:
        raise ValueError("opens/edges cannot be empty")

    hits = [e > 0.0 for e in edges]
    n = len(edges)
    hit_rate = sum(hits) / n
    mean_clv = sum(edges) / n

    rolling_hit_rate: list[float] = []
    rolling_mean_clv: list[float] = []
    for i in range(n):
        slice_ = _window_slice(edges, i, window)
        rolling_hit_rate.append(sum(1 for e in slice_ if e > 0.0) / len(slice_))
        rolling_mean_clv.append(sum(slice_) / len(slice_))

    return RollingClvSummary(
        edges=edges,
        hit_rate=hit_rate,
        mean_clv=mean_clv,
        current_streak=_current_streak(edges),
        rolling_hit_rate=rolling_hit_rate,
        rolling_mean_clv=rolling_mean_clv,
    )


def clv_book_summary(
    opens: Sequence[float],
    closes: Sequence[float] | None = None,
    *,
    push_threshold_pts: float = _DEFAULT_PUSH_THRESHOLD_PTS,
) -> ClvBookSummary:
    """Summarize a book of bets using ai-advantage ``clvPts`` units.

    Classifies each ticket as beat / miss / push with the same default
    push band as ai-advantage ``closeVerdict`` (``|clvPts| < 0.05`` → push).

    Args:
        opens: Open American odds for each bet, or already-computed CLV edges
            in percentage points when ``closes`` is omitted.
        closes: Close American odds paired with ``opens``. When ``None``,
            ``opens`` is treated as a sequence of ``clvPts`` values (from
            :func:`clv_pts`, not probability-point :func:`clv_edge`).
        push_threshold_pts: Absolute percentage-point band treated as a push.
            Must be finite and ``>= 0``. Default ``0.05`` matches ai-advantage.

    Returns:
        A :class:`ClvBookSummary` with per-bet ``edges_pts`` and book stats.

    Example::

        book = clv_book_summary([-110, -105, +140], [-115, -100, +130])
        # book["mean_clv_pts"], book["beats"], book["pushes"]

        from_pts = clv_book_summary([1.5, -0.02, 2.0])  # precomputed clvPts

    """
    if not math.isfinite(push_threshold_pts) or push_threshold_pts < 0.0:
        raise ValueError(
            f"push_threshold_pts must be a finite value >= 0, got {push_threshold_pts}"
        )

    open_list = [float(x) for x in opens]

    if closes is None:
        edges_pts = open_list
        for edge in edges_pts:
            if not math.isfinite(edge):
                raise ValueError(f"CLV pts must be finite, got {edge}")
    else:
        close_list = [float(x) for x in closes]
        if len(open_list) != len(close_list):
            raise ValueError(
                f"Lengths of opens ({len(open_list)}) and closes ({len(close_list)}) must match"
            )
        edges_pts = [clv_pts(o, c) for o, c in zip(open_list, close_list, strict=True)]

    if not edges_pts:
        raise ValueError("opens/edges cannot be empty")

    # Match ai-advantage closeVerdict: |clvPts| < threshold → push;
    # otherwise sign(clvPts) decides beat vs lost.
    beats = sum(1 for e in edges_pts if _is_clv_beat(e, push_threshold_pts))
    misses = sum(1 for e in edges_pts if abs(e) >= push_threshold_pts and e <= 0.0)
    pushes = sum(1 for e in edges_pts if abs(e) < push_threshold_pts)
    n = len(edges_pts)

    return ClvBookSummary(
        edges_pts=edges_pts,
        hit_rate=beats / n,
        mean_clv_pts=sum(edges_pts) / n,
        beats=beats,
        misses=misses,
        pushes=pushes,
        current_streak=_current_streak_with_push(edges_pts, push_threshold_pts),
    )

"""Kelly and implied probability (shared with line-shop-cli idea)."""

from __future__ import annotations


def american_to_implied_prob(american: float) -> float:
    if american > 0:
        return 100 / (american + 100)
    return abs(american) / (abs(american) + 100)


def kelly_fraction(win_prob: float, american_odds: float, *, fraction: float = 0.25) -> float:
    if american_odds > 0:
        b = american_odds / 100
    else:
        b = 100 / abs(american_odds)
    q = 1.0 - win_prob
    full = (b * win_prob - q) / b if b > 0 else 0.0
    return max(0.0, min(0.25, fraction * full))

"""Kelly and implied probability (shared with line-shop-cli idea)."""

from __future__ import annotations


def american_to_implied_prob(american: float) -> float:
    """Convert American odds to implied probability."""
    if -100 < american < 100:
        raise ValueError(f"American odds cannot be between -100 and 100, got {american}")
    if american > 0:
        return 100.0 / (american + 100.0)
    return abs(american) / (abs(american) + 100.0)


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds (return >= 1.0)."""
    if -100 < american < 100:
        raise ValueError(f"American odds cannot be between -100 and 100, got {american}")
    if american > 0:
        return (american / 100.0) + 1.0
    return (100.0 / abs(american)) + 1.0


def decimal_to_american(decimal: float) -> float:
    """Convert decimal odds to American odds."""
    if decimal <= 1.0:
        raise ValueError(f"Decimal odds must be strictly greater than 1.0, got {decimal}")
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    return -100.0 / (decimal - 1.0)


def decimal_to_implied_prob(decimal: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal <= 1.0:
        raise ValueError(f"Decimal odds must be strictly greater than 1.0, got {decimal}")
    return 1.0 / decimal


def implied_prob_to_american(prob: float) -> float:
    """Convert implied probability to American odds."""
    if not (0.0 < prob < 1.0):
        raise ValueError(f"Probability must be strictly between 0 and 1, got {prob}")
    if prob >= 0.5:
        return -100.0 * (prob / (1.0 - prob))
    return 100.0 * ((1.0 - prob) / prob)


def implied_prob_to_decimal(prob: float) -> float:
    """Convert implied probability to decimal odds."""
    if not (0.0 < prob <= 1.0):
        raise ValueError(f"Probability must be strictly between 0 and 1, got {prob}")
    return 1.0 / prob


def kelly_fraction(win_prob: float, american_odds: float, *, fraction: float = 0.25) -> float:
    """Calculate Kelly fraction sizing given win probability and American odds."""
    if -100 < american_odds < 100:
        raise ValueError(f"American odds cannot be between -100 and 100, got {american_odds}")
    if american_odds > 0:
        b = american_odds / 100.0
    else:
        b = 100.0 / abs(american_odds)
    q = 1.0 - win_prob
    full = (b * win_prob - q) / b if b > 0 else 0.0
    return max(0.0, min(0.25, fraction * full))

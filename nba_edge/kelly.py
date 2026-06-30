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


def kelly_fraction(
    win_prob: float,
    american_odds: float,
    *,
    fraction: float = 0.25,
    max_cap: float | None = 0.25,
) -> float:
    """Calculate Kelly fraction sizing given win probability and American odds.

    Args:
        win_prob: Model estimated win probability (0.0 to 1.0).
        american_odds: Oddsmaker American odds (e.g., -110 or +150).
        fraction: Kelly multiplier/fraction (e.g., 0.25 for quarter-Kelly).
        max_cap: Maximum fraction of bankroll to risk. Set to None for no cap.
    """
    if not (0.0 <= win_prob <= 1.0):
        raise ValueError(f"Win probability must be between 0.0 and 1.0, got {win_prob}")
    if -100 < american_odds < 100:
        raise ValueError(f"American odds cannot be between -100 and 100, got {american_odds}")
    if american_odds > 0:
        b = american_odds / 100.0
    else:
        b = 100.0 / abs(american_odds)
    q = 1.0 - win_prob
    full = (b * win_prob - q) / b if b > 0 else 0.0
    val = fraction * full
    if max_cap is not None:
        if max_cap < 0.0 or max_cap > 1.0:
            raise ValueError(f"max_cap must be between 0.0 and 1.0 (or None), got {max_cap}")
        val = min(max_cap, val)
    return max(0.0, val)


def remove_vig(odds_a: float, odds_b: float, method: str = "proportional") -> tuple[float, float]:
    """Remove the bookmaker's vigorish (vig) to find the fair probabilities of two outcomes.

    Args:
        odds_a: American odds for outcome A.
        odds_b: American odds for outcome B.
        method: Method used to remove vig. Supported: 'proportional' or 'equal'.

    Returns:
        A tuple containing the fair probabilities of outcome A and B (summing to 1.0).
    """
    if method not in ("proportional", "equal"):
        raise ValueError(f"Unsupported method: {method}. Must be 'proportional' or 'equal'.")

    p_a = american_to_implied_prob(odds_a)
    p_b = american_to_implied_prob(odds_b)
    total = p_a + p_b

    if total <= 0:
        raise ValueError("Combined implied probability must be positive")

    if method == "proportional":
        return p_a / total, p_b / total
    else:  # equal (additive)
        overround = total - 1.0
        fair_a = p_a - (overround / 2.0)
        fair_b = p_b - (overround / 2.0)
        if fair_a < 0.0 or fair_b < 0.0:
            raise ValueError(
                "Equal margin method produced negative probability. Use 'proportional' instead."
            )
        return fair_a, fair_b


def parlay_odds(odds_list: list[float]) -> dict[str, float]:
    """Calculate the combined decimal odds, American odds, and implied probability of a parlay.

    Assumes all legs are independent.

    Args:
        odds_list: A list of American odds for each leg of the parlay.

    Returns:
        A dictionary with keys:
            'decimal': Combined decimal odds.
            'american': Combined American odds.
            'implied_prob': Combined implied probability of the parlay.
    """
    if not odds_list:
        raise ValueError("odds_list cannot be empty")

    joint_decimal = 1.0
    joint_prob = 1.0
    for odds in odds_list:
        dec = american_to_decimal(odds)
        p = american_to_implied_prob(odds)
        joint_decimal *= dec
        joint_prob *= p

    combined_american = decimal_to_american(joint_decimal)

    return {
        "decimal": joint_decimal,
        "american": combined_american,
        "implied_prob": joint_prob,
    }


def kelly_parlay(
    win_probs: list[float],
    odds_list: list[float],
    *,
    fraction: float = 0.25,
    max_cap: float | None = 0.25,
) -> float:
    """Calculate combined Kelly Criterion sizing for a multi-leg parlay of independent events.

    Args:
        win_probs: List of model-estimated win probabilities for each leg (0.0 to 1.0).
        odds_list: List of bookmaker American odds for each leg of the parlay.
        fraction: Kelly multiplier/fraction (e.g., 0.25 for quarter-Kelly).
        max_cap: Maximum fraction of bankroll to risk. Set to None for no cap.

    Returns:
        The fraction of bankroll to wager on the parlay.
    """
    if len(win_probs) != len(odds_list):
        raise ValueError(
            f"Lengths of win_probs ({len(win_probs)}) and odds_list ({len(odds_list)}) must match"
        )
    if not win_probs:
        raise ValueError("Input lists cannot be empty")

    # Combined win probability (assuming independence)
    joint_prob = 1.0
    for p in win_probs:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Win probability must be between 0.0 and 1.0, got {p}")
        joint_prob *= p

    # Combined decimal odds
    joint_decimal = 1.0
    for odds in odds_list:
        joint_decimal *= american_to_decimal(odds)

    b = joint_decimal - 1.0  # Net decimal odds
    q = 1.0 - joint_prob

    full = (b * joint_prob - q) / b if b > 0.0 else 0.0
    val = fraction * full
    if max_cap is not None:
        if max_cap < 0.0 or max_cap > 1.0:
            raise ValueError(f"max_cap must be between 0.0 and 1.0 (or None), got {max_cap}")
        val = min(max_cap, val)
    return max(0.0, val)

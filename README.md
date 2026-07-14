# nba-edge

> Repo: `nba-ratings` · Package: `nba-edge`

[![CI](https://github.com/ianalloway/nba-ratings/actions/workflows/ci.yml/badge.svg)](https://github.com/ianalloway/nba-ratings/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Reusable Elo, win-probability, and Kelly-sizing primitives for NBA-style models.

## Why This Repo Matters

This is the library layer of the sports ML stack:

- reusable rating logic instead of notebook snippets
- portable win-probability helpers for downstream services
- Kelly and implied-probability helpers for decision support
- small, dependency-light package design

Pairs well with [`nba-clv-dashboard`](https://github.com/ianalloway/nba-clv-dashboard) for evaluation UI. Employer one-pager: [case study](https://ianalloway.xyz/papers/sports-ml-evaluation-case-study.html).

## What It Includes

- Elo updates, including a margin-of-victory-weighted variant
- Logistic win probability
- Kelly fraction sizing & multi-leg parlay sizing
- Bidirectional odds format conversions (American, Decimal, Implied Probability)
- Bookmaker vig-removal tools (Proportional and Equal Margin methods)
- Model evaluation metrics (Brier Score, Log Loss)

## Install

```bash
pip install -e .  # local; not yet on PyPI
```

The library has zero runtime dependencies (see `pyproject.toml`). The
root-level `requirements.txt` is unrelated to the library — it only exists so
Streamlit Community Cloud can find the demo app's dependencies
(`demo/requirements.txt`, i.e. `streamlit` + `pandas`).

## Example

```python
from nba_edge import (
    logistic_win_prob,
    update_elo,
    kelly_fraction,
    american_to_decimal,
    american_to_implied_prob,
    remove_vig,
    parlay_odds,
    kelly_parlay,
    brier_score,
    log_loss,
)

# 1. Ratings & win probability
p = logistic_win_prob(rating_diff=120)
new_h, new_a = update_elo(1600, 1580, 1.0)

# 2. Odds conversions & Vig Removal
dec = american_to_decimal(-110)      # Convert American to Decimal
p_implied = american_to_implied_prob(-110)  # Convert American to Implied Probability
p_fair_h, p_fair_a = remove_vig(-110, -110, method="proportional")  # Remove vig

# 3. Bet sizing & Parlays
stake = kelly_fraction(p, -110, fraction=0.25)
# Compute combined parlay odds/joint probability for independent legs
parlay = parlay_odds([-110, +130])
# Parlay Kelly sizing (fraction = 0.25 for quarter-Kelly)
parlay_stake = kelly_parlay([0.60, 0.55], [-110, +130], fraction=0.25)

# 4. Model evaluation
predictions = [0.75, 0.40, 0.65]
outcomes = [1.0, 0.0, 1.0]
bs = brier_score(predictions, outcomes)
ll = log_loss(predictions, outcomes)
```

### Margin-of-victory Elo

Plain Elo treats every win the same, but a 30-point blowout is a stronger
signal than a 2-point nail-biter. `update_elo_with_margin` applies a
FiveThirtyEight-style multiplier so ratings move further on lopsided games
and less when a team that was already heavily favored piles on:

```python
from nba_edge import update_elo_with_margin

new_h, new_a = update_elo_with_margin(1600, 1580, 1.0, margin=22)
```

## Publish

```bash
pip install build twine
python -m build
twine upload dist/*
```

## Non-goals

- No bundled NBA database or scrapers
- Not a tipster product
- Not a full modeling workflow by itself

## CI

`pytest` + `ruff` on Python 3.10-3.12.

## Related Repos

- [`sports-betting-ml`](https://github.com/ianalloway/sports-betting-ml): applied modeling demo
- [`nba-clv-dashboard`](https://github.com/ianalloway/nba-clv-dashboard): evaluation dashboard

## License

MIT

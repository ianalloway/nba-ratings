# nba-ratings

> Published to PyPI as [`nba-edge`](https://pypi.org/project/nba-edge/).

[![CI](https://github.com/ianalloway/nba-ratings/actions/workflows/ci.yml/badge.svg)](https://github.com/ianalloway/nba-ratings/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nba-edge.svg)](https://pypi.org/project/nba-edge/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nba-edge.svg)](https://pypi.org/project/nba-edge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Reusable Elo, win-probability, and Kelly-sizing primitives for NBA-style moneylines.

## Why This Repo Matters

Installable library layer for the sports ML stack:

- reusable rating logic instead of notebook snippets
- portable win-probability helpers for downstream services
- Kelly and implied-probability helpers for decision support
- small, dependency-light package design

Pairs well with [`nba-clv-dashboard`](https://github.com/ianalloway/nba-clv-dashboard) for evaluation UI. Employer one-pager: [case study](https://ianalloway.xyz/papers/sports-ml-evaluation-case-study.html).

## What It Includes

- Elo updates, including a margin-of-victory-weighted variant
- Logistic win probability
- Optional rest / schedule-density adjustment for point spreads
- Kelly fraction sizing & multi-leg parlay sizing
- Bidirectional odds format conversions (American, Decimal, Implied Probability)
- Bookmaker vig-removal tools (Proportional and Equal Margin methods) plus `fair_american_odds`
- Closing-line value helpers (`clv_edge`, `clv_pts`, `rolling_clv`, `clv_book_summary`) for beat-the-close streaks
- Model evaluation metrics (Brier Score, Log Loss, Calibration Curve)

## Install

```bash
pip install nba-edge
```

Package page: [pypi.org/project/nba-edge/](https://pypi.org/project/nba-edge/).

The library has **zero runtime dependencies** (see `pyproject.toml`). The
root-level `requirements.txt` is unrelated to the library — it only exists so
Streamlit Community Cloud can find the demo app's dependencies
(`demo/requirements.txt`, i.e. `streamlit` + `pandas`).

## Example

```python
from nba_edge import (
    logistic_win_prob,
    update_elo,
    expected_margin,
    expected_margin_with_rest,
    mov_multiplier,
    compute_fatigue_features,
    rest_spread_adjustment,
    kelly_fraction,
    american_to_decimal,
    american_to_implied_prob,
    remove_vig,
    parlay_odds,
    kelly_parlay,
    clv_edge,
    rolling_clv,
    brier_score,
    calibration_curve,
    log_loss,
)

# 1. Ratings & win probability
p = logistic_win_prob(rating_diff=120)
margin = expected_margin(rating_diff=120)  # Predicted point spread
mov = mov_multiplier(margin=22, elo_diff_winner=120)  # MOV weighting factor
new_h, new_a = update_elo(1600, 1580, 1.0)

# 2. Odds conversions & Vig Removal
dec = american_to_decimal(-110)  # Convert American to Decimal
p_implied = american_to_implied_prob(-110)  # Convert American to Implied Probability
p_fair_h, p_fair_a = remove_vig(-110, -110, method="proportional")  # Remove vig

# 3. Bet sizing & Parlays
stake = kelly_fraction(p, -110, fraction=0.25)
# Compute combined parlay odds/joint probability for independent legs
parlay = parlay_odds([-110, +130])
# Parlay Kelly sizing (fraction = 0.25 for quarter-Kelly)
parlay_stake = kelly_parlay([0.60, 0.55], [-110, +130], fraction=0.25)

# 4. Closing-line value (beat the close)
edge = clv_edge(-110, -120)  # +pp when open beat close
clv = rolling_clv([-110, -105, +140], [-115, -100, +130], window=2)
# clv["hit_rate"], clv["mean_clv"], clv["current_streak"]

# 5. Model evaluation
predictions = [0.75, 0.40, 0.65]
outcomes = [1.0, 0.0, 1.0]
bs = brier_score(predictions, outcomes)
ll = log_loss(predictions, outcomes)
curve = calibration_curve(predictions, outcomes, bins=10)
```

### Rolling CLV

Track whether your ticket prices beat the close — the market-aware cousin of
raw win rate. `clv_edge` returns probability-point CLV
(`implied(close) - implied(open)`); `clv_pts` scales that by 100 to match the
ai-advantage `clvPts` display units. `rolling_clv` turns a sequence of
open/close American odds (or precomputed edges) into hit rate, mean CLV, and
the current beat/miss streak; `clv_book_summary` adds beat/miss/push counts
with the same `0.05` pts push band as ai-advantage:

```python
from nba_edge import clv_edge, clv_pts, clv_book_summary, rolling_clv

assert clv_edge(-110, -120) > 0  # got a better number than close
assert clv_pts(-110, -120) == clv_edge(-110, -120) * 100

summary = rolling_clv(
    [-110, -105, +140, -108],
    [-115, -100, +130, -112],
    window=3,
)
summary["hit_rate"]         # fraction of bets that beat the close
summary["mean_clv"]         # average CLV in probability points
summary["current_streak"]   # +n trailing beats, -n trailing misses
summary["rolling_hit_rate"] # per-bet hit rate over the trailing window

book = clv_book_summary(
    [-110, -105, +140, -108],
    [-115, -100, +130, -112],
)
book["mean_clv_pts"]  # average CLV in percentage points
book["beats"], book["misses"], book["pushes"]
```

### Margin-of-victory Elo

Plain Elo treats every win the same, but a 30-point blowout is a stronger
signal than a 2-point nail-biter. `update_elo_with_margin` applies a
FiveThirtyEight-style multiplier so ratings move further on lopsided games
and less when a team that was already heavily favored piles on:

```python
from nba_edge import update_elo_with_margin, expected_margin, mov_multiplier

predicted_spread = expected_margin(rating_diff=120)  # toy mapping
mov_factor = mov_multiplier(margin=22, elo_diff_winner=120)  # weighting factor
new_h, new_a = update_elo_with_margin(1600, 1580, 1.0, margin=22)
```

### Rest and schedule density

NBA schedules create uneven rest: back-to-backs, 3-in-4s, long road trips, and
cross-country travel. `compute_fatigue_features` turns one team's ordered
schedule into per-game flags and counts; `rest_spread_adjustment` maps home vs
away features into points on the spread using documented toy defaults
(`DEFAULT_REST_COEFFS`) that you can override.

Rest is **opt-in** on the prediction path. `expected_margin(...)` is unchanged
unless you pass `rest_points`, or call `expected_margin_with_rest`:

```python
from nba_edge import (
    DEFAULT_REST_COEFFS,
    compute_fatigue_features,
    expected_margin,
    expected_margin_with_rest,
    rest_spread_adjustment,
)

home_sked = [
    {"date": "2025-11-01", "location": "home"},
    {"date": "2025-11-05", "location": "home"},  # 3 days rest
]
away_sked = [
    {"date": "2025-11-04", "location": "away"},
    {
        "date": "2025-11-05",
        "location": "away",  # back-to-back
        "arena_lat": 34.043,
        "arena_lon": -118.267,
    },
]
# Optional: prior away game coords → haversine travel_miles on the B2B night
away_sked[0]["arena_lat"] = 40.7505
away_sked[0]["arena_lon"] = -73.9934

home_fatigue = compute_fatigue_features(home_sked)[-1]
away_fatigue = compute_fatigue_features(away_sked)[-1]

adj = rest_spread_adjustment(home_fatigue, away_fatigue)
# Or override a single coefficient:
adj = rest_spread_adjustment(
    home_fatigue, away_fatigue, coeffs={**DEFAULT_REST_COEFFS, "back_to_back": 2.0}
)

# Elo-only (default) vs Elo + rest
elo_only = expected_margin(rating_diff=80)  # rest_points defaults to 0
with_rest = expected_margin_with_rest(80, home_fatigue, away_fatigue)
assert with_rest == expected_margin(80, rest_points=adj)
```

Feature conventions: season opener → `days_rest is None` (treated as fully
rested); neutral sites continue a road trip (only `home` resets it); travel
miles come from an explicit `travel_miles` field or haversine between consecutive
`arena_lat` / `arena_lon` values.

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

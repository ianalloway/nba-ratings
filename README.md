# nba-edge

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

- Elo updates
- logistic win probability
- Kelly fraction sizing
- American odds to implied probability

## Install

```bash
pip install nba-edge
```

## Example

```python
from nba_edge import logistic_win_prob, update_elo, kelly_fraction, american_to_implied_prob

p = logistic_win_prob(rating_diff=120)
new_h, new_a = update_elo(1600, 1580, 1.0)
stake = kelly_fraction(p, -110, fraction=0.25)
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

# nba-edge

[![CI](https://github.com/ianalloway/nba-ratings/actions/workflows/ci.yml/badge.svg)](https://github.com/ianalloway/nba-ratings/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Part of the sports ML flagship:** pairs with **[nba-clv-dashboard](https://github.com/ianalloway/nba-clv-dashboard)** (eval UI) and **[line-shop-cli](https://github.com/ianalloway/line-shop-cli)**. Employer one-pager: [case study](https://ianalloway.xyz/papers/sports-ml-evaluation-case-study.html).

**Problem:** Notebook odds and Elo snippets are **not reusable** across services.

**Solution:** A small **`nba-edge`** package — **Elo updates**, **logistic win probability**, **Kelly / implied prob** — with **NumPy-free** pure Python (NumPy listed for future vectorized extensions; core is stdlib + `math`).

```bash
pip install nba-edge  # PyPI name; repo: **nba-ratings** (distinct from legacy `nba-edge` CLI repo)
```

```python
from nba_edge import logistic_win_prob, update_elo, kelly_fraction, american_to_implied_prob

p = logistic_win_prob(rating_diff=120)  # home - away
new_h, new_a = update_elo(1600, 1580, 1.0)  # home won
stake = kelly_fraction(p, -110, fraction=0.25)
```

## Publish

```bash
pip install build twine
python -m build
twine upload dist/*
```

## Non-goals

- **No** bundled NBA database or scrapers — bring your own features.
- **Not** a tipster product — primitives for **your** models.

## CI

pytest + ruff on Python 3.10–3.12.

## Suggested GitHub topics

`python` · `nba` · `elo` · `kelly-criterion` · `sports-analytics`

## License

MIT

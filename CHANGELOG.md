# Changelog

Notable changes to nba-edge (the `nba_edge` package in this repo). Format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Rest / schedule-density** module `nba_edge.rest`:
  - `compute_fatigue_features(schedule)`: per-game days of rest, back-to-back,
    3-in-4, 4-in-6, road-trip length, and optional travel miles (explicit or
    haversine from arena coordinates).
  - `haversine_miles(...)` and `rest_spread_adjustment(...)` with documented
    `DEFAULT_REST_COEFFS` that callers can override.
  - `expected_margin(..., rest_points=0.0)` and `expected_margin_with_rest(...)`
    so rest can optionally adjust the Elo→spread mapping; default remains
    Elo-only (no behavior change for existing callers).
  - TypedDicts: `ScheduleGame`, `FatigueFeatures`, `RestAdjustmentCoeffs`.

## [0.2.0] - 2026-09-29

First release since `0.1.0`. `main` briefly declared `0.1.1` (#31), but that
version was never tagged or published, so it is folded into this release.

### Added

- **Closing-line value (CLV)** module `nba_edge.clv`:
  - `clv_edge(open_american, close_american)`: CLV in probability points.
  - `rolling_clv(...)` returning `RollingClvSummary`: hit rate, mean CLV,
    current beat-the-close streak, and an optional fixed-window rolling
    series. Accepts open/close odds or precomputed edges (#30).
  - `clv_pts(...)` and `clv_book_summary(...)` returning `ClvBookSummary`:
    CLV in percentage points, with beat/miss/push classification using a
    configurable push band (#31).
- **Odds conversions** (#9): `american_to_decimal`, `decimal_to_american`,
  `decimal_to_implied_prob`, `implied_prob_to_american`,
  `implied_prob_to_decimal`.
- **Vig and parlays** (#10): `remove_vig()` (proportional and equal-margin
  methods), `parlay_odds()`, and `kelly_parlay()` for sizing independent
  multi-leg parlays. `fair_american_odds()` returns no-vig American odds for
  a two-way market (#31).
- `kelly_fraction(..., max_cap=0.25)`: the bankroll cap is now a parameter
  (previously a hardcoded 0.25); pass `None` to disable it (#10).
- **Margin-of-victory Elo** (#8): `mov_multiplier()` and
  `update_elo_with_margin()`.
- **Model evaluation metrics** in `nba_edge.metrics`: `brier_score()`,
  `log_loss()` (#9), and `calibration_curve()` with the `CalibrationBin`
  TypedDict for reliability diagrams (#18).
- All of the above are exported from the top-level `nba_edge` package.
  Submodules now declare `__all__`.
- PEP 561 `py.typed` marker, so type checkers see the package annotations.
- `demo` optional extra (`pip install "nba-edge[demo]"`) mirroring
  `demo/requirements.txt` for the Streamlit demo (#23).

### Changed

- Public functions now validate their inputs and raise `ValueError` instead
  of returning silently wrong numbers. This covers non-finite (NaN/inf)
  ratings, K-factors, scales, margins, odds, probabilities and fractions;
  American odds strictly between -100 and 100; and probabilities outside
  their valid range. Code that relied on garbage-in/garbage-out behaviour in
  `0.1.0` may now see exceptions.
- `kelly_fraction()` rejects a negative `fraction`, which previously sized
  losing bets. `calibration_curve()` rejects non-integer (and bool) `bins`
  (#31).
- `expected_margin()` rejects a negative `margin_per_elo`.
- `parlay_odds()` and `kelly_parlay()` accept any `Sequence[float]`, for
  example tuples.

### Fixed

- `mov_multiplier()` / `update_elo_with_margin()` no longer divide by zero
  or flip sign at `elo_diff_winner <= -2200` (#12).
- `logistic_win_prob()` no longer raises `OverflowError` on extreme rating
  differences. It returns the 0.0 / 1.0 limit instead.
- `log_loss()` validates its `eps` clipping parameter.
- Streamlit demo: Kelly tab guards against invalid American odds input.
- Lint and typing cleanups (ruff SIM/RET/RUF/I rules enabled, mypy fixes,
  `zip(strict=True)` in metrics).

### Maintenance

- Much larger test suite: public API surface contract, version sync between
  `__init__.py` and `pyproject.toml`, and invariants for vig removal,
  parlays, Elo and metrics.
- README documents the new APIs. Added contributing guide, code of conduct,
  issue/PR templates, Dependabot with auto-merge for patch/minor bumps, and
  dependency bumps (streamlit, actions/setup-python).

## [0.1.0] - 2026-07-25

Initial release on PyPI as `nba-edge`: `logistic_win_prob`, `expected_margin`,
`update_elo`, `american_to_implied_prob`, `kelly_fraction`.

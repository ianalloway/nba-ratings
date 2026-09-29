"""Tests for rest / schedule-density fatigue features and spread adjustment."""

from __future__ import annotations

from datetime import date

import pytest

from nba_edge.ratings import expected_margin, expected_margin_with_rest
from nba_edge.rest import (
    DEFAULT_REST_COEFFS,
    compute_fatigue_features,
    haversine_miles,
    rest_spread_adjustment,
)


def test_season_opener_fully_rested() -> None:
    feats = compute_fatigue_features(
        [{"date": "2025-10-22", "location": "home"}],
    )
    assert len(feats) == 1
    opener = feats[0]
    assert opener["days_rest"] is None
    assert opener["back_to_back"] is False
    assert opener["three_in_four"] is False
    assert opener["four_in_six"] is False
    assert opener["road_trip_length"] == 0
    assert opener["travel_miles"] == 0.0


def test_consecutive_back_to_backs() -> None:
    # Three nights in a row: game 2 and 3 are B2Bs; game 3 is also 3-in-4.
    schedule = [
        {"date": date(2025, 11, 1), "location": "home"},
        {"date": date(2025, 11, 2), "location": "away"},
        {"date": date(2025, 11, 3), "location": "away"},
    ]
    feats = compute_fatigue_features(schedule)
    assert feats[0]["back_to_back"] is False
    assert feats[0]["days_rest"] is None

    assert feats[1]["days_rest"] == 0
    assert feats[1]["back_to_back"] is True
    assert feats[1]["three_in_four"] is False  # only 2 games in window
    assert feats[1]["road_trip_length"] == 1

    assert feats[2]["days_rest"] == 0
    assert feats[2]["back_to_back"] is True
    assert feats[2]["three_in_four"] is True  # 3 games in 3 days ⊂ 4-day window
    assert feats[2]["road_trip_length"] == 2


def test_neutral_site_continues_road_trip() -> None:
    schedule = [
        {"date": "2025-12-01", "location": "away"},
        {"date": "2025-12-03", "location": "neutral"},
        {"date": "2025-12-05", "location": "home"},
        {"date": "2025-12-08", "location": "neutral"},
    ]
    feats = compute_fatigue_features(schedule)
    assert feats[0]["road_trip_length"] == 1
    assert feats[0]["location"] == "away"
    # Neutral continues the non-home stretch.
    assert feats[1]["road_trip_length"] == 2
    assert feats[1]["location"] == "neutral"
    assert feats[1]["days_rest"] == 1
    # Home resets the trip.
    assert feats[2]["road_trip_length"] == 0
    # Neutral after home starts a new non-home stretch at 1.
    assert feats[3]["road_trip_length"] == 1


def test_three_in_four_and_four_in_six_windows() -> None:
    # Games on days 1, 2, 4 → third game is 3-in-4.
    # Games on 1, 2, 4, 6 → fourth game is 4-in-6; its 4-day window only
    # covers games on 4 and 6, so three_in_four is False there.
    schedule = [
        {"date": "2026-01-01", "location": "home"},
        {"date": "2026-01-02", "location": "home"},
        {"date": "2026-01-04", "location": "away"},
        {"date": "2026-01-06", "location": "away"},
    ]
    feats = compute_fatigue_features(schedule)
    assert feats[2]["three_in_four"] is True
    assert feats[2]["four_in_six"] is False
    assert feats[3]["three_in_four"] is False  # only 2 games in [Jan 3, Jan 6]
    assert feats[3]["four_in_six"] is True  # all four in [Jan 1, Jan 6]


def test_haversine_and_auto_travel_miles() -> None:
    # Madison Square Garden ≈ 40.7505, -73.9934
    # Crypto.com Arena ≈ 34.0430, -118.2673
    miles = haversine_miles(40.7505, -73.9934, 34.0430, -118.2673)
    assert 2400 < miles < 2500  # ~2451 mi

    schedule = [
        {
            "date": "2025-11-10",
            "location": "home",
            "arena_lat": 40.7505,
            "arena_lon": -73.9934,
        },
        {
            "date": "2025-11-12",
            "location": "away",
            "arena_lat": 34.0430,
            "arena_lon": -118.2673,
        },
    ]
    feats = compute_fatigue_features(schedule)
    assert feats[0]["travel_miles"] == 0.0
    assert feats[1]["travel_miles"] == pytest.approx(miles)


def test_explicit_travel_miles_overrides_coords() -> None:
    schedule = [
        {
            "date": "2025-11-10",
            "location": "home",
            "arena_lat": 40.0,
            "arena_lon": -74.0,
        },
        {
            "date": "2025-11-12",
            "location": "away",
            "travel_miles": 100.0,
            "arena_lat": 34.0,
            "arena_lon": -118.0,
        },
    ]
    feats = compute_fatigue_features(schedule)
    assert feats[1]["travel_miles"] == 100.0


def test_rest_spread_adjustment_favors_rested_home() -> None:
    home_schedule = [
        {"date": "2025-11-01", "location": "home"},
        {"date": "2025-11-05", "location": "home"},  # 3 days rest
    ]
    away_schedule = [
        {"date": "2025-11-04", "location": "away"},
        {"date": "2025-11-05", "location": "away"},  # B2B
    ]
    home = compute_fatigue_features(home_schedule)[-1]
    away = compute_fatigue_features(away_schedule)[-1]
    adj = rest_spread_adjustment(home, away)
    # Away on B2B with short rest → positive adjustment (home more favored).
    assert adj > 0


def test_rest_spread_adjustment_coeff_override() -> None:
    home = compute_fatigue_features(
        [
            {"date": "2025-11-01", "location": "home"},
            {"date": "2025-11-02", "location": "home"},
        ]
    )[-1]
    away = compute_fatigue_features(
        [
            {"date": "2025-11-01", "location": "away"},
            {"date": "2025-11-02", "location": "away"},
        ]
    )[-1]
    # Both on B2B → B2B terms cancel; zero other coeffs → 0.
    zeroed = {k: 0.0 for k in DEFAULT_REST_COEFFS}
    assert rest_spread_adjustment(home, away, coeffs=zeroed) == 0.0

    only_b2b = {**zeroed, "back_to_back": 2.0}
    # Same B2B status → still 0.
    assert rest_spread_adjustment(home, away, coeffs=only_b2b) == 0.0


def test_expected_margin_rest_points_default_unchanged() -> None:
    # Off by default: rest_points=0 leaves the Elo-only mapping alone.
    assert expected_margin(100.0) == pytest.approx(2.5)
    assert expected_margin(100.0, rest_points=0.0) == pytest.approx(2.5)
    assert expected_margin(100.0, rest_points=1.5) == pytest.approx(4.0)


def test_expected_margin_with_rest_matches_manual() -> None:
    home = compute_fatigue_features(
        [
            {"date": "2025-11-01", "location": "home"},
            {"date": "2025-11-05", "location": "home"},
        ]
    )[-1]
    away = compute_fatigue_features(
        [
            {"date": "2025-11-04", "location": "away"},
            {"date": "2025-11-05", "location": "away"},
        ]
    )[-1]
    adj = rest_spread_adjustment(home, away)
    combined = expected_margin_with_rest(80.0, home, away)
    assert combined == pytest.approx(expected_margin(80.0, rest_points=adj))


def test_compute_fatigue_rejects_out_of_order_and_bad_location() -> None:
    with pytest.raises(ValueError, match="chronological"):
        compute_fatigue_features(
            [
                {"date": "2025-11-05", "location": "home"},
                {"date": "2025-11-01", "location": "home"},
            ]
        )
    with pytest.raises(ValueError, match="location"):
        compute_fatigue_features([{"date": "2025-11-01", "location": "road"}])


def test_haversine_rejects_bad_coords() -> None:
    with pytest.raises(ValueError, match="latitude"):
        haversine_miles(100.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        haversine_miles(float("nan"), 0.0, 0.0, 0.0)


def test_unknown_coeff_rejected() -> None:
    home = compute_fatigue_features([{"date": "2025-11-01", "location": "home"}])[0]
    away = compute_fatigue_features([{"date": "2025-11-01", "location": "away"}])[0]
    with pytest.raises(ValueError, match="unknown rest coefficient"):
        rest_spread_adjustment(home, away, coeffs={"not_a_coeff": 1.0})

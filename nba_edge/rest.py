"""Rest and schedule-density fatigue features for point-spread adjustments.

Given a team's ordered schedule (game dates, home/away/neutral, and optionally
travel miles or arena coordinates), this module computes per-game fatigue
features and turns them into a point-spread adjustment.

Default coefficients are documented toy priors — not calibrated NBA estimates.
Override via :class:`RestAdjustmentCoeffs` (or a plain mapping with the same
keys) when you have your own fit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence
from datetime import date, datetime, timedelta
from typing import Literal, TypedDict, cast

__all__ = [
    "DEFAULT_REST_COEFFS",
    "FatigueFeatures",
    "RestAdjustmentCoeffs",
    "ScheduleGame",
    "compute_fatigue_features",
    "haversine_miles",
    "rest_spread_adjustment",
]

# Mean Earth radius in miles (WGS-84 authalic approximation).
_EARTH_RADIUS_MILES: float = 3958.7613

Location = Literal["home", "away", "neutral"]


class ScheduleGame(TypedDict, total=False):
    """One entry in a team's ordered schedule.

    Required keys (enforced at runtime, not by the TypedDict ``total`` flag):

    - ``date``: ``datetime.date``, ISO ``YYYY-MM-DD`` string, or ``datetime``.
    - ``location``: ``"home"``, ``"away"``, or ``"neutral"``.

    Optional:

    - ``travel_miles``: Miles traveled into this game. When omitted, miles are
      derived via haversine if both this game and the previous game provide
      ``arena_lat`` / ``arena_lon``.
    - ``arena_lat`` / ``arena_lon``: Arena coordinates in decimal degrees.
    """

    date: date | datetime | str
    location: Location
    travel_miles: float
    arena_lat: float
    arena_lon: float


class FatigueFeatures(TypedDict):
    """Per-game rest / schedule-density features for one team.

    Attributes:
        date: Game date.
        location: ``home`` / ``away`` / ``neutral``.
        days_rest: Whole days since the previous game. ``None`` for the season
            opener (treated as fully rested — no B2B / density flags).
        back_to_back: ``True`` when ``days_rest == 0`` (played yesterday).
        three_in_four: ``True`` when this game is the 3rd (or more) in the
            inclusive 4-day window ending today.
        four_in_six: ``True`` when this game is the 4th (or more) in the
            inclusive 6-day window ending today.
        road_trip_length: Consecutive non-home games ending at this game
            (away and neutral both continue a trip; home resets to 0).
        travel_miles: Miles into this game, or ``None`` when unknown.
    """

    date: date
    location: Location
    days_rest: int | None
    back_to_back: bool
    three_in_four: bool
    four_in_six: bool
    road_trip_length: int
    travel_miles: float | None


class RestAdjustmentCoeffs(TypedDict, total=False):
    """Coefficients mapping fatigue features to a home-perspective point spread.

    All values are in **points**. The signed adjustment added to the home
    team's expected margin is::

        adj = (
            back_to_back      * (away_b2b - home_b2b)
          + three_in_four    * (away_3in4 - home_3in4)
          + four_in_six      * (away_4in6 - home_4in6)
          + rest_day         * (clip(home_rest) - clip(away_rest))
          + road_trip        * (away_trip - home_trip)
          + travel_per_1000  * (away_miles - home_miles) / 1000
        )

    where ``clip(rest) = min(days_rest if known else rest_day_cap, rest_day_cap)``
    and unknown travel miles contribute 0 (skipped on that side).

    Positive ``adj`` favors the home team (widens a home favorite / shrinks
    an away favorite). Zero every coefficient you do not want.
    """

    back_to_back: float
    three_in_four: float
    four_in_six: float
    rest_day: float
    rest_day_cap: float
    road_trip: float
    travel_per_1000: float


# Sensible toy defaults. Rationale (order-of-magnitude priors, not a fit):
# - B2B is the strongest single-night effect (~1.5 pts).
# - 3-in-4 / 4-in-6 stack smaller density penalties on top of rest days.
# - Each extra day of rest (capped at 3) is worth ~0.4 pts vs the opponent.
# - Long road trips add a mild per-game tax beyond the first away night.
# - Cross-country travel (~1000-2500 mi) is a fraction of a point.
DEFAULT_REST_COEFFS: RestAdjustmentCoeffs = {
    "back_to_back": 1.5,
    "three_in_four": 0.75,
    "four_in_six": 0.5,
    "rest_day": 0.4,
    "rest_day_cap": 3.0,
    "road_trip": 0.15,
    "travel_per_1000": 0.25,
}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two WGS-84 lat/lon points."""
    for name, value in (
        ("lat1", lat1),
        ("lon1", lon1),
        ("lat2", lat2),
        ("lon2", lon2),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if not (-90.0 <= lat1 <= 90.0 and -90.0 <= lat2 <= 90.0):
        raise ValueError(f"latitude must be in [-90, 90], got lat1={lat1}, lat2={lat2}")
    if not (-180.0 <= lon1 <= 180.0 and -180.0 <= lon2 <= 180.0):
        raise ValueError(f"longitude must be in [-180, 180], got lon1={lon1}, lon2={lon2}")

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * _EARTH_RADIUS_MILES * math.asin(min(1.0, math.sqrt(a)))


def _parse_date(value: date | datetime | str, *, index: int) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"schedule[{index}].date must be YYYY-MM-DD, got {value!r}"
            ) from exc
    raise ValueError(
        f"schedule[{index}].date must be date, datetime, or ISO str, got {type(value).__name__}"
    )


def _parse_location(value: object, *, index: int) -> Location:
    if value not in ("home", "away", "neutral"):
        raise ValueError(
            f"schedule[{index}].location must be 'home', 'away', or 'neutral', got {value!r}"
        )
    return cast(Location, value)


def _optional_finite(name: str, value: object, *, index: int) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"schedule[{index}].{name} must be a finite number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"schedule[{index}].{name} must be a finite number, got {value!r}")
    return number


def _games_in_window(dates: Sequence[date], end: date, window_days: int) -> int:
    """Count games with dates in ``[end - (window_days - 1), end]`` inclusive."""
    start = end - timedelta(days=window_days - 1)
    return sum(1 for d in dates if start <= d <= end)


def _resolve_travel_miles(
    game: Mapping[str, object],
    prev: Mapping[str, object] | None,
    *,
    index: int,
) -> float | None:
    explicit = _optional_finite("travel_miles", game.get("travel_miles"), index=index)
    if explicit is not None:
        if explicit < 0:
            raise ValueError(
                f"schedule[{index}].travel_miles must be non-negative, got {explicit}"
            )
        return explicit
    if prev is None:
        return 0.0  # season opener: no inbound travel for feature purposes

    lat = _optional_finite("arena_lat", game.get("arena_lat"), index=index)
    lon = _optional_finite("arena_lon", game.get("arena_lon"), index=index)
    prev_lat = _optional_finite("arena_lat", prev.get("arena_lat"), index=index - 1)
    prev_lon = _optional_finite("arena_lon", prev.get("arena_lon"), index=index - 1)
    if None in (lat, lon, prev_lat, prev_lon):
        return None
    # Narrowed above: all four are floats.
    return haversine_miles(
        cast(float, prev_lat),
        cast(float, prev_lon),
        cast(float, lat),
        cast(float, lon),
    )


def compute_fatigue_features(
    schedule: Sequence[Mapping[str, object]],
) -> list[FatigueFeatures]:
    """Compute per-game rest / density features for one team's schedule.

    The schedule must be ordered chronologically (non-decreasing dates). Each
    mapping needs ``date`` and ``location``; see :class:`ScheduleGame`.

    Edge-case conventions:

    - **Season opener**: ``days_rest`` is ``None``; B2B / 3-in-4 / 4-in-6 are
      ``False``; ``travel_miles`` is ``0.0`` unless an explicit value is given.
    - **Consecutive back-to-backs**: a third game on three straight nights has
      ``days_rest == 0`` and both density flags set once the windows fill.
    - **Neutral site**: does not reset a road trip (counts like away for
      ``road_trip_length``); home is the only location that resets the trip to 0.
    """
    if not isinstance(schedule, Sequence) or isinstance(schedule, (str, bytes)):
        raise ValueError("schedule must be a sequence of game mappings")

    parsed: list[tuple[date, Location, Mapping[str, object]]] = []
    for i, raw in enumerate(schedule):
        if not isinstance(raw, Mapping):
            raise ValueError(f"schedule[{i}] must be a mapping, got {type(raw).__name__}")
        if "date" not in raw:
            raise ValueError(f"schedule[{i}] missing required key 'date'")
        if "location" not in raw:
            raise ValueError(f"schedule[{i}] missing required key 'location'")
        game_date = _parse_date(raw["date"], index=i)  # type: ignore[arg-type]
        location = _parse_location(raw["location"], index=i)
        if parsed and game_date < parsed[-1][0]:
            raise ValueError(
                f"schedule must be chronological; schedule[{i}].date={game_date} "
                f"is before schedule[{i - 1}].date={parsed[-1][0]}"
            )
        parsed.append((game_date, location, raw))

    dates = [g[0] for g in parsed]
    features: list[FatigueFeatures] = []
    road_trip = 0

    for i, (game_date, location, raw) in enumerate(parsed):
        if i == 0:
            days_rest: int | None = None
            back_to_back = False
        else:
            days_rest = (game_date - dates[i - 1]).days - 1
            if days_rest < 0:
                # Same calendar day as previous game — treat as 0 days rest
                # (doubleheader / same-day scheduling anomaly).
                days_rest = 0
            back_to_back = days_rest == 0

        # Inclusive windows ending on game_date; only prior+current dates matter.
        prior_and_current = dates[: i + 1]
        three_in_four = _games_in_window(prior_and_current, game_date, 4) >= 3
        four_in_six = _games_in_window(prior_and_current, game_date, 6) >= 4

        if location == "home":
            road_trip = 0
        else:
            # away and neutral both continue (or start) a non-home stretch.
            road_trip += 1

        prev_raw = parsed[i - 1][2] if i > 0 else None
        travel = _resolve_travel_miles(raw, prev_raw, index=i)

        features.append(
            {
                "date": game_date,
                "location": location,
                "days_rest": days_rest,
                "back_to_back": back_to_back,
                "three_in_four": three_in_four,
                "four_in_six": four_in_six,
                "road_trip_length": road_trip,
                "travel_miles": travel,
            }
        )

    return features


def _merge_coeffs(overrides: Mapping[str, float] | None) -> RestAdjustmentCoeffs:
    merged: MutableMapping[str, float] = dict(DEFAULT_REST_COEFFS)
    if overrides is None:
        return cast(RestAdjustmentCoeffs, dict(merged))
    known = set(DEFAULT_REST_COEFFS)
    for key, value in overrides.items():
        if key not in known:
            raise ValueError(
                f"unknown rest coefficient {key!r}; expected one of {sorted(known)}"
            )
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"coefficient {key!r} must be a finite number, got {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"coefficient {key!r} must be a finite number, got {value!r}")
        if key == "rest_day_cap" and number < 0:
            raise ValueError(f"rest_day_cap must be non-negative, got {number}")
        merged[key] = number
    return cast(RestAdjustmentCoeffs, dict(merged))


def _clipped_rest_days(features: FatigueFeatures, cap: float) -> float:
    days = features["days_rest"]
    if days is None:
        # Season opener: treat as fully rested at the cap.
        return cap
    return float(min(max(days, 0), cap))


def _travel_or_zero(features: FatigueFeatures) -> float:
    miles = features["travel_miles"]
    if miles is None:
        return 0.0
    return float(miles)


def rest_spread_adjustment(
    home: FatigueFeatures,
    away: FatigueFeatures,
    coeffs: Mapping[str, float] | None = None,
) -> float:
    """Point-spread adjustment from home vs away fatigue (home perspective).

    Returns points to **add** to the home team's expected margin. Positive
    values favor home. Uses :data:`DEFAULT_REST_COEFFS` when ``coeffs`` is
    omitted; pass a partial mapping to override individual terms.

    Args:
        home: Fatigue features for the home (or designated home) side.
        away: Fatigue features for the away side.
        coeffs: Optional coefficient overrides; see :class:`RestAdjustmentCoeffs`.

    """
    c = _merge_coeffs(coeffs)
    cap = c["rest_day_cap"]

    home_b2b = 1.0 if home["back_to_back"] else 0.0
    away_b2b = 1.0 if away["back_to_back"] else 0.0
    home_3 = 1.0 if home["three_in_four"] else 0.0
    away_3 = 1.0 if away["three_in_four"] else 0.0
    home_4 = 1.0 if home["four_in_six"] else 0.0
    away_4 = 1.0 if away["four_in_six"] else 0.0

    adj = 0.0
    adj += c["back_to_back"] * (away_b2b - home_b2b)
    adj += c["three_in_four"] * (away_3 - home_3)
    adj += c["four_in_six"] * (away_4 - home_4)
    adj += c["rest_day"] * (_clipped_rest_days(home, cap) - _clipped_rest_days(away, cap))
    adj += c["road_trip"] * float(away["road_trip_length"] - home["road_trip_length"])
    adj += c["travel_per_1000"] * (_travel_or_zero(away) - _travel_or_zero(home)) / 1000.0
    return adj

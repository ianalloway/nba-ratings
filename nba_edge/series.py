"""Series forecasts from independent, game-specific win probabilities."""

from collections.abc import Iterable
from typing import TypedDict


class SeriesForecast(TypedDict):
    win_probability: float
    length_probabilities: dict[int, float]
    expected_games: float


def forecast_series(game_win_probabilities: Iterable[float]) -> SeriesForecast:
    """Forecast one team's best-of series. Supply a probability for every possible game.

    Games are independent and the series stops when either team has a majority.
    Input order is the schedule order, from the same team's perspective throughout.
    """
    schedule = list(game_win_probabilities)
    if len(schedule) % 2 == 0:
        raise ValueError("supply a positive odd number of game probabilities")
    if any(not 0 <= p <= 1 for p in schedule):
        raise ValueError("game win probabilities must be in [0, 1]")

    clinch = len(schedule) // 2 + 1
    length_probabilities = dict.fromkeys(range(clinch, len(schedule) + 1), 0.0)
    # Only still-live paths enter the next game, indexed by the team's wins.
    active = {0: 1.0}
    win_probability = 0.0
    for game, p in enumerate(schedule, start=1):
        remaining: dict[int, float] = {}
        for wins, mass in active.items():
            for new_wins, chance in ((wins + 1, p), (wins, 1 - p)):
                weight = mass * chance
                if new_wins == clinch:
                    win_probability += weight
                    length_probabilities[game] += weight
                elif game - new_wins == clinch:
                    length_probabilities[game] += weight
                else:
                    remaining[new_wins] = remaining.get(new_wins, 0.0) + weight
        active = remaining
    return {
        "win_probability": win_probability,
        "length_probabilities": length_probabilities,
        "expected_games": sum(n * mass for n, mass in length_probabilities.items()),
    }

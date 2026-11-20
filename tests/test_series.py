import pytest

from nba_edge import forecast_series


def test_evenly_matched_best_of_seven():
    report = forecast_series([0.5] * 7)
    assert report["win_probability"] == 0.5
    assert report["length_probabilities"] == {4: 0.125, 5: 0.25, 6: 0.3125, 7: 0.3125}
    assert report["expected_games"] == 5.8125


def test_order_controls_series_length():
    for schedule, length in (([1, 1, 0], 2), ([1, 0, 1], 3)):
        report = forecast_series(iter(schedule))
        assert report["win_probability"] == 1
        assert report["length_probabilities"][length] == 1
        assert report["expected_games"] == length


def test_different_game_probabilities():
    report = forecast_series([0.6, 0.7, 0.2])
    assert report["win_probability"] == pytest.approx(0.42 + (0.18 + 0.28) * 0.2)
    assert report["length_probabilities"] == pytest.approx({2: 0.54, 3: 0.46})
    assert forecast_series([0.25])["win_probability"] == 0.25


@pytest.mark.parametrize("schedule", [[], [0.5, 0.5], [-0.1], [1.1], [float("nan")]])
def test_invalid_schedule(schedule):
    with pytest.raises(ValueError):
        forecast_series(schedule)

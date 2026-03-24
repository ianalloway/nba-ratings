from nba_edge.ratings import logistic_win_prob, update_elo


def test_even_matchup() -> None:
    assert 0.49 < logistic_win_prob(0.0) < 0.51


def test_favorite_wins_more() -> None:
    assert logistic_win_prob(200.0) > 0.7


def test_update_elo() -> None:
    a, b = update_elo(1500.0, 1500.0, 1.0)
    assert a > 1500
    assert b < 1500

"""Interactive demo of the nba_edge package (Elo, win probability, Kelly).

Run locally:  streamlit run demo/streamlit_app.py
Deployed on Streamlit Community Cloud; imports the real package from this repo.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Import the actual package from the repo root (no pip install needed).
sys.path.insert(0, str(Path(__file__).parent.parent))

from nba_edge import (
    american_to_implied_prob,
    expected_margin,
    kelly_fraction,
    logistic_win_prob,
    update_elo,
)

st.set_page_config(page_title="nba-edge demo", page_icon="📈", layout="wide")

st.title("📈 nba-edge — live demo")
st.caption(
    "Every number on this page is computed by the real `nba_edge` package "
    "([source](https://github.com/ianalloway/nba-ratings)) — not a re-implementation."
)

tab_prob, tab_elo, tab_kelly, tab_season = st.tabs(
    ["Win probability", "Elo update", "Kelly sizing", "Season simulation"]
)

# ---------------------------------------------------------------- win prob
with tab_prob:
    st.subheader("Rating difference → win probability")
    col1, col2 = st.columns([1, 2])
    with col1:
        diff = st.slider("Home Elo − Away Elo", -400, 400, 80, step=10)
        scale = st.select_slider("Logistic scale", [200, 300, 400, 500], value=400)
        p = logistic_win_prob(diff, scale=scale)
        st.metric("P(home wins)", f"{p:.1%}")
        st.metric("Expected margin", f"{expected_margin(diff):+.1f} pts")
    with col2:
        xs = list(range(-400, 401, 10))
        curve = pd.DataFrame(
            {"rating diff": xs, "P(home wins)": [logistic_win_prob(x, scale=scale) for x in xs]}
        ).set_index("rating diff")
        st.line_chart(curve, height=320)
    st.caption("`logistic_win_prob(diff, scale)` — the classic Elo logistic curve.")

# ---------------------------------------------------------------- elo update
with tab_elo:
    st.subheader("One-game Elo update")
    c1, c2, c3 = st.columns(3)
    ra = c1.number_input("Team A rating", 1000.0, 2000.0, 1550.0, step=10.0)
    rb = c2.number_input("Team B rating", 1000.0, 2000.0, 1480.0, step=10.0)
    k = c3.slider("K-factor", 5, 50, 20)
    result = st.radio("Result", ["A wins", "Draw", "B wins"], horizontal=True)
    score = {"A wins": 1, "Draw": 0.5, "B wins": 0}[result]
    new_a, new_b = update_elo(ra, rb, score, k=k)
    pre = logistic_win_prob(ra - rb)
    c1.metric("New A rating", f"{new_a:.1f}", f"{new_a - ra:+.1f}")
    c2.metric("New B rating", f"{new_b:.1f}", f"{new_b - rb:+.1f}")
    c3.metric("Pre-game P(A wins)", f"{pre:.1%}")
    st.caption(
        "Upsets move ratings more: the update is K × (actual − expected). "
        "`update_elo(ra, rb, score_a, k=K)`"
    )

# ---------------------------------------------------------------- kelly
with tab_kelly:
    st.subheader("Edge → stake size (fractional Kelly)")
    c1, c2 = st.columns(2)
    with c1:
        win_prob = st.slider("Your model's win probability", 0.30, 0.80, 0.58, 0.01)
        odds = st.number_input("American odds offered", -500, 500, -110, step=5)
        frac = st.select_slider("Kelly fraction", [0.1, 0.25, 0.5, 1.0], value=0.25)
        bankroll = st.number_input("Bankroll ($)", 100, 100000, 1000, step=100)
    with c2:
        implied = american_to_implied_prob(odds)
        stake = kelly_fraction(win_prob, odds, fraction=frac)
        edge = win_prob - implied
        st.metric("Market implied probability", f"{implied:.1%}")
        st.metric("Your edge", f"{edge:+.1%}")
        st.metric("Recommended stake", f"${bankroll * stake:,.2f} ({stake:.2%} of bankroll)")
        if stake == 0:
            st.info("No bet: the market price already exceeds your estimated probability.")
        st.caption(
            "`kelly_fraction` caps stakes at 25% of bankroll and returns 0 on negative edge — "
            "guardrails matter more than the formula."
        )

# ---------------------------------------------------------------- season sim
with tab_season:
    st.subheader("Elo convergence over a simulated season")
    import random

    n_games = st.slider("Games per team", 20, 200, 82, step=2)
    seed = st.number_input("Random seed", 0, 9999, 42)
    rng = random.Random(int(seed))

    true_strength = {"Sharks": 1650, "Wolves": 1550, "Comets": 1450, "Drifters": 1350}
    ratings = {t: 1500.0 for t in true_strength}
    history = {t: [1500.0] for t in true_strength}
    teams = list(true_strength)

    for _ in range(n_games * len(teams) // 2):
        a, b = rng.sample(teams, 2)
        p_true = logistic_win_prob(true_strength[a] - true_strength[b])
        score = 1 if rng.random() < p_true else 0
        ratings[a], ratings[b] = update_elo(ratings[a], ratings[b], score)
        for t in teams:
            history[t].append(ratings[t])

    chart = pd.DataFrame(history)
    chart.index.name = "game"
    st.line_chart(chart, height=360)
    st.caption(
        "Four teams start at 1500; games are simulated from hidden true strengths "
        "(1650/1550/1450/1350). Watch `update_elo` recover the truth from results alone."
    )

st.divider()
st.markdown(
    "Built from [`ianalloway/nba-ratings`](https://github.com/ianalloway/nba-ratings) · "
    "part of the demo suite at [ianalloway.xyz/demos](https://ianalloway.xyz/demos)"
)

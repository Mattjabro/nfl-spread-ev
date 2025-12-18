import numpy as np
import pandas as pd
from math import erf, sqrt

from load_data import load_games, attach_qbs
from model_margin_decay_weighted import fit_margin_decay_model

TEMPERATURE = 1.75
EV_THRESHOLD = 0.05   # strong only
SPREAD_PAYOUT = 0.9091  # -110 odds


def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def cover_prob_home(mu, sigma, spread_home):
    z = (mu + spread_home) / sigma
    return norm_cdf(z / TEMPERATURE)


def backtest_week(season, week):
    # -------------------------------
    # Load historical games
    # -------------------------------
    df, team_to_idx = load_games()
    df = attach_qbs(df)

    # Train ONLY on past data
    train = df[
        (df["season"] < season) |
        ((df["season"] == season) & (df["week"] < week))
    ]

    trace = fit_margin_decay_model(train, len(team_to_idx))

    post = trace.posterior
    team_s = post["team_strength"].values
    hfa_s = post["hfa"].values
    sigma_s = post["sigma"].values

    # Actual games this week
    week_games = df[
        (df["season"] == season) &
        (df["week"] == week)
    ].copy()

    rows = []

    for _, g in week_games.iterrows():
        mu = (
            hfa_s
            + team_s[..., g.home_idx]
            - team_s[..., g.away_idx]
            + g.home_qb_val
            - g.away_qb_val
        )

        mu = mu.reshape(-1)
        sigma = sigma_s.reshape(-1).mean()

        vegas_spread = g.get("closing_spread_home", None)
        if pd.isna(vegas_spread):
            continue

        p_home = cover_prob_home(mu.mean(), sigma, vegas_spread)
        p_away = 1 - p_home

        ev_home = p_home * SPREAD_PAYOUT - (1 - p_home)
        ev_away = p_away * SPREAD_PAYOUT - (1 - p_away)

        if ev_home >= EV_THRESHOLD:
            bet = "HOME"
            ev = ev_home
            won = (g.margin + vegas_spread) > 0
        elif ev_away >= EV_THRESHOLD:
            bet = "AWAY"
            ev = ev_away
            won = (g.margin + vegas_spread) < 0
        else:
            continue

        rows.append({
            "season": season,
            "week": week,
            "away_team": g.away_team,
            "home_team": g.home_team,
            "spread": vegas_spread,
            "bet_side": bet,
            "ev": ev,
            "won": won
        })

    return pd.DataFrame(rows)
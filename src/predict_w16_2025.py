import numpy as np
import pandas as pd
import nfl_data_py as nfl
from math import erf, sqrt

from load_data import load_games
from model_margin_decay_weighted import fit_margin_decay_model

# ===== FIXED CALIBRATION PARAMETER =====
TEMPERATURE = 1.75

# =====================================

def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def prob_to_american(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if p > 0.5:
        return int(-100 * p / (1 - p))
    else:
        return int(100 * (1 - p) / p)


def predict_total(spread):
    # simple baseline total model
    return round(44 + 0.4 * abs(spread), 1)


def predict_week(season=2025, week=16, decay_rate=0.02):
    # Load historical data
    df, team_to_idx = load_games()

    # Train on all games before this week
    train = df[
        (df["season"] < season) |
        ((df["season"] == season) & (df["week"] < week))
    ]

    trace = fit_margin_decay_model(train, len(team_to_idx), decay_rate)

    # Load schedule
    sched = nfl.import_schedules([season])
    games = sched[
        (sched["game_type"] == "REG") &
        (sched["week"] == week)
    ][["season", "week", "home_team", "away_team"]].copy()

    games["home_idx"] = games["home_team"].map(team_to_idx)
    games["away_idx"] = games["away_team"].map(team_to_idx)

    # Posterior samples
    post = trace.posterior
    team_s = post["team_strength"].values
    hfa_s = post["hfa"].values
    sigma_s = post["sigma"].values

    rows = []

    for _, g in games.iterrows():
        mu = (
            hfa_s
            + team_s[..., g.home_idx]
            - team_s[..., g.away_idx]
        )

        mu_flat = mu.reshape(-1)
        sigma_flat = sigma_s.reshape(-1)

        # ===== CALIBRATED PROBABILITY =====
        z = mu_flat / sigma_flat
        home_win_prob = np.mean([norm_cdf(v / TEMPERATURE) for v in z])
        away_win_prob = 1 - home_win_prob

        # Spread: Away +/- Home
        spread_away = -np.mean(mu_flat)

        rows.append({
            "Season": season,
            "Week": week,
            "Away Team": g.away_team,
            "Home Team": g.home_team,
            "Spread (Away +/- Home)": round(spread_away, 1),
            "Home Odds": prob_to_american(home_win_prob),
            "Total": predict_total(spread_away),
            "Sigma": round(np.mean(sigma_flat), 2)
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    preds = predict_week()
    out = "../results/week16_2025_lines.csv"
    preds.to_csv(out, index=False)
    print(preds)
    print(f"\nSaved Week 16 predictions to {out}")
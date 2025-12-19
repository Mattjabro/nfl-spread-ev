import numpy as np
import pandas as pd
from pathlib import Path

from load_data import load_games, attach_qbs
from model_margin_decay_weighted import fit_margin_decay_model

RESULTS_DIR = Path("../results")
SEASON = 2025
MAX_WEEK = 15
START_SEASON = SEASON - 3

# -----------------------------
# Load data (last 4 seasons)
# -----------------------------
df, team_to_idx = load_games(
    start_season=START_SEASON,
    end_season=SEASON
)
df = attach_qbs(df)

df = df[
    (df["season"] < SEASON) |
    ((df["season"] == SEASON) & (df["week"] <= MAX_WEEK))
].copy()

# Global week index
df["global_week"] = (df["season"] - df["season"].min()) * 18 + df["week"]

# -----------------------------
# Fit model ONCE
# -----------------------------
trace = fit_margin_decay_model(df, len(team_to_idx))
post = trace.posterior

team_strength = post["team_strength"].mean(axis=(0, 1))

idx_to_team = {v: k for k, v in team_to_idx.items()}

# -----------------------------
# Expand to time-indexed rows
# -----------------------------
rows = []
for _, g in df.iterrows():
    rows.append({
        "team": g["home_team"],
        "global_week": g["global_week"],
        "team_strength": team_strength[team_to_idx[g["home_team"]]]
    })
    rows.append({
        "team": g["away_team"],
        "global_week": g["global_week"],
        "team_strength": team_strength[team_to_idx[g["away_team"]]]
    })

out_df = pd.DataFrame(rows)

out = RESULTS_DIR / "team_power_raw.csv"
out_df.to_csv(out, index=False)

print(f"Saved {out}")
print(out_df.head())
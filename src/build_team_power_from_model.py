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

# Global week index (monotone time)
df["global_week"] = (
    (df["season"] - df["season"].min()) * 18 + df["week"]
)

# -----------------------------
# Fit model ONCE
# -----------------------------
trace = fit_margin_decay_model(df, len(team_to_idx))

post = trace.posterior
team_strength = post["team_strength"].values  # (chains, draws, teams)

# Posterior mean strength per team
mean_strength = team_strength.mean(axis=(0, 1))

idx_to_team = {v: k for k, v in team_to_idx.items()}

# -----------------------------
# Build raw power table
# -----------------------------
rows = []
for team_idx, strength in enumerate(mean_strength):
    rows.append({
        "team": idx_to_team[team_idx],
        "base_power": strength
    })

raw_df = pd.DataFrame(rows)

out = RESULTS_DIR / "team_power_raw.csv"
raw_df.to_csv(out, index=False)

print(f"Saved raw team power to {out}")
print(raw_df.sort_values("base_power", ascending=False).head(10))
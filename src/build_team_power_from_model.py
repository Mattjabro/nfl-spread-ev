import numpy as np
import pandas as pd
from pathlib import Path

from load_data import load_games, attach_qbs
from model_margin_decay_weighted import fit_margin_decay_model

# ============================================================
# CONFIG
# ============================================================
RESULTS_DIR = Path("../results")
SEASON = 2025
MAX_WEEK = 15

# ============================================================
# Load THIS SEASON ONLY
# ============================================================
df, team_to_idx = load_games(
    start_season=SEASON,
    end_season=SEASON
)

df = attach_qbs(df)

df = df[df["week"] <= MAX_WEEK].copy()

# ============================================================
# Fit model
# ============================================================
trace = fit_margin_decay_model(df, len(team_to_idx))
post = trace.posterior

# Latent team strength (numeric)
team_strength = (
    post["team_strength"]
    .mean(axis=(0, 1))
    .values
)

idx_to_team = {v: k for k, v in team_to_idx.items()}

# ============================================================
# QB impact per team (season average)
# ============================================================
qb_cols = ["home_qb_value", "away_qb_value"]

qb_rows = []

for _, g in df.iterrows():
    qb_rows.append({"team": g["home_team"], "qb_value": g["home_qb_value"]})
    qb_rows.append({"team": g["away_team"], "qb_value": g["away_qb_value"]})

qb_df = pd.DataFrame(qb_rows)

qb_team_mean = (
    qb_df.groupby("team")["qb_value"]
    .mean()
    .to_dict()
)

# ============================================================
# Build final team power
# ============================================================
rows = []

for idx, strength in enumerate(team_strength):
    team = idx_to_team[idx]
    qb_adj = qb_team_mean.get(team, 0.0)

    rows.append({
        "team": team,
        "team_strength": float(strength),
        "qb_value": float(qb_adj),
        "team_power": float(strength + qb_adj)
    })

power_df = (
    pd.DataFrame(rows)
    .sort_values("team_power", ascending=False)
    .reset_index(drop=True)
)

out = RESULTS_DIR / "team_power_raw.csv"
power_df.to_csv(out, index=False)

print(f"Saved {out}")
print(power_df.head(10))
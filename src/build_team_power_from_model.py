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
df, team_to_idx = load_games(start_season=START_SEASON, end_season=SEASON)
df = attach_qbs(df)

df = df[
    (df["season"] < SEASON) |
    ((df["season"] == SEASON) & (df["week"] <= MAX_WEEK))
].copy()

# Global week index
df["global_week"] = (
    (df["season"] - df["season"].min()) * 18 + df["week"]
)

max_week = df["global_week"].max()

# -----------------------------
# Fit model
# -----------------------------
trace = fit_margin_decay_model(df, len(team_to_idx))

post = trace.posterior
team_strength = post["team_strength"].values  # (chains, draws, teams)

# -----------------------------
# Recency weighting function
# -----------------------------
def compute_team_power(recency_lambda: float):
    """
    recency_lambda = 0.0 → no recency
    higher = more recent emphasis
    """
    # Effective recency weight
    age = max_week - df["global_week"].values
    weights = np.exp(-recency_lambda * age)

    # Normalize
    weights = weights / weights.mean()

    # Posterior mean team strength
    base_strength = team_strength.mean(axis=(0, 1))

    # Mild recency adjustment (stable, not noisy)
    adj_strength = base_strength * (1 + 0.15 * recency_lambda)

    return adj_strength

# -----------------------------
# Save default rankings (moderate recency)
# -----------------------------
DEFAULT_LAMBDA = 0.05
team_power = compute_team_power(DEFAULT_LAMBDA)

idx_to_team = {v: k for k, v in team_to_idx.items()}

power_df = (
    pd.DataFrame({
        "team": [idx_to_team[i] for i in range(len(team_power))],
        "team_power": team_power
    })
    .sort_values("team_power", ascending=False)
    .reset_index(drop=True)
)

power_df.insert(0, "rank", power_df.index + 1)

out = RESULTS_DIR / "team_power_rankings.csv"
power_df.to_csv(out, index=False)

print(f"Saved team power rankings to {out}")
print(power_df.head(10))
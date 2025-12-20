import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from math import erf, sqrt

# --------------------------------------------------
# Path setup
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from load_data import load_games, attach_qbs
from model_margin_decay_season_boost import fit_margin_decay_model

# --------------------------------------------------
# FINAL TUNED HYPERPARAMETERS
# --------------------------------------------------
DECAY_RATE = 0.015
SEASON_BOOST = 2.5
TEMPERATURE = 1.70

START_SEASON = 2019
END_SEASON = 2025
MAX_WEEK_2025 = 15

OUT_PATH = ROOT / "results" / "final_walkforward_predictions.csv"

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def log(msg):
    print(msg, flush=True)

# --------------------------------------------------
# Load data ONCE
# --------------------------------------------------
log("Loading games + QB data...")
df, team_to_idx = load_games(start_season=START_SEASON, end_season=END_SEASON)
df = attach_qbs(df)

rows = []
run_start = time.time()

# --------------------------------------------------
# Walk-forward
# --------------------------------------------------
for season in range(START_SEASON + 1, END_SEASON + 1):
    max_week = MAX_WEEK_2025 if season == 2025 else 18

    for week in range(1, max_week + 1):
        train = df[
            (df["season"] < season) |
            ((df["season"] == season) & (df["week"] < week))
        ].copy()

        test = df[
            (df["season"] == season) & (df["week"] == week)
        ].copy()

        if len(test) == 0 or len(train) < 200:
            continue

        log(
            f"Training through {season} W{week-1} | "
            f"Predicting {season} W{week}"
        )

        trace = fit_margin_decay_model(
            train,
            len(team_to_idx),
            decay_rate=DECAY_RATE,
            season_boost=SEASON_BOOST,
            prediction_season=season
        )

        post = trace.posterior
        team_s = post["team_strength"].mean(axis=(0, 1))
        hfa = post["hfa"].mean().item()
        sigma = post["sigma"].mean().item()

        for _, g in test.iterrows():
            mu = float(
                hfa
                + team_s[g.home_idx]
                - team_s[g.away_idx]
                + g.home_qb_val
                - g.away_qb_val
            )

            z = float(mu / sigma)
            p_home = float(norm_cdf(z / TEMPERATURE))

            rows.append({
                "season": season,
                "week": week,
                "home_team": g.home_team,
                "away_team": g.away_team,
                "model_spread": mu,
                "sigma": sigma,
                "z": z,
                "p_home_win": p_home,
                "actual_margin": g.margin,
                "home_win": int(g.margin > 0)
            })

# --------------------------------------------------
# Save
# --------------------------------------------------
out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_PATH, index=False)

elapsed = (time.time() - run_start) / 60

log("\n" + "=" * 70)
log("FINAL WALK-FORWARD COMPLETE")
log("=" * 70)
log(f"Saved {len(out_df)} predictions to:")
log(str(OUT_PATH))
log(f"Total runtime: {elapsed:.2f} minutes")
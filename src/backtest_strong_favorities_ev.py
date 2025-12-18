import pandas as pd
import numpy as np
from scipy.stats import norm

# ----------------------------------------
# PARAMETERS (PROFESSIONAL DEFAULTS)
# ----------------------------------------
TEMPERATURE = 1.75        # conservative calibration
STRONG_Z = 1.25           # strong confidence threshold
ODDS = -110               # standard spread pricing

# ----------------------------------------
# LOAD MERGED MODEL + VEGAS DATA
# ----------------------------------------
df = pd.read_csv("../results/predictions_vs_vegas.csv")

# ----------------------------------------
# BASIC SANITY FILTERS
# ----------------------------------------
df = df.dropna(subset=[
    "model_spread",
    "sigma",
    "closing_spread_home",
    "actual_margin"
]).copy()

df = df[df["sigma"] > 0]

# ----------------------------------------
# FAVORITE-ALIGNED ONLY
# (Model + Vegas agree on direction)
# ----------------------------------------
df = df[
    (df["model_spread"] < 0) &
    (df["closing_spread_home"] < 0)
].copy()

# ----------------------------------------
# STANDARDIZED EDGE (CONFIDENCE)
# ----------------------------------------
df["z_edge"] = (
    df["model_spread"] - df["closing_spread_home"]
) / df["sigma"]

df["abs_z"] = df["z_edge"].abs()

# ----------------------------------------
# COVER PROBABILITY (HOME FAVORITE)
# ----------------------------------------
df["p_cover"] = norm.cdf(
    (df["model_spread"] + df["closing_spread_home"]) /
    (df["sigma"] * TEMPERATURE)
)

# ----------------------------------------
# EXPECTED VALUE
# ----------------------------------------
payout = 100 / abs(ODDS)

df["ev"] = (
    df["p_cover"] * payout -
    (1 - df["p_cover"])
)

# ----------------------------------------
# STRONG CONFIDENCE + POSITIVE EV ONLY
# ----------------------------------------
bets = df[
    (df["abs_z"] >= STRONG_Z) &
    (df["ev"] > 0)
].copy()

# ----------------------------------------
# ATS RESULT
# ----------------------------------------
bets["won"] = (
    bets["actual_margin"] + bets["closing_spread_home"]
) > 0

# ----------------------------------------
# OUTPUT COLUMNS
# ----------------------------------------
cols = [
    "season",
    "week",
    "away_team",
    "home_team",
    "closing_spread_home",
    "model_spread",
    "sigma",
    "z_edge",
    "p_cover",
    "ev",
    "won"
]

bets = bets[cols].sort_values(
    ["season", "week"]
)

# ----------------------------------------
# SAVE
# ----------------------------------------
out = "../results/backtest_strong_favorites_ev.csv"
bets.to_csv(out, index=False)

# ----------------------------------------
# SUMMARY
# ----------------------------------------
print("\n================ STRONG FAVORITES EV BACKTEST =================")
print("Total bets:", len(bets))
print("Win rate:", bets["won"].mean())
print("Average EV per $1:", bets["ev"].mean())

print("\nBy season:")
print(
    bets.groupby("season")
        .agg(
            bets=("won", "count"),
            win_rate=("won", "mean"),
            avg_ev=("ev", "mean")
        )
)
print(f"\nSaved to {out}")
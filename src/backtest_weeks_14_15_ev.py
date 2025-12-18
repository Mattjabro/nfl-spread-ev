import pandas as pd
import numpy as np
from scipy.stats import norm

TEMPERATURE = 1.75

df = pd.read_csv("../results/predictions_vs_vegas.csv")

# Filter to 2025 Weeks 14–15
df = df[
    (df["season"] == 2025) &
    (df["week"].isin([14, 15]))
].copy()

if df.empty:
    raise ValueError("No games found for Weeks 14–15.")

# z-score: probability home covers
z = (df["model_spread"] + df["closing_spread_home"]) / df["sigma"]

p_home_cover = norm.cdf(z / TEMPERATURE)

# EV assuming standard -110 pricing
df["home_ev"] = p_home_cover * 0.9091 - (1 - p_home_cover)
df["away_ev"] = (1 - p_home_cover) * 0.9091 - p_home_cover

# Choose side with higher EV
df["bet_side"] = np.where(
    df["home_ev"] > df["away_ev"],
    "home",
    "away"
)

df["bet_ev"] = np.where(
    df["bet_side"] == "home",
    df["home_ev"],
    df["away_ev"]
)

# Actual outcome
df["won"] = np.where(
    df["bet_side"] == "home",
    df["actual_margin"] > 0,
    df["actual_margin"] < 0
)

# Only +EV bets
bets = df[df["bet_ev"] > 0].copy()

print("\n================ WEEKS 14–15 EV BACKTEST =================")
print(
    bets[[
        "week",
        "away_team",
        "home_team",
        "closing_spread_home",
        "model_spread",
        "bet_side",
        "bet_ev",
        "won"
    ]]
)

print("\nSUMMARY")
print("Total bets:", len(bets))
print("Hit rate:", bets["won"].mean())
print("Average EV per $1:", bets["bet_ev"].mean())
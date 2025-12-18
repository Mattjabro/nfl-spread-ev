import pandas as pd
import numpy as np
from math import erf, sqrt

# ----------------------------------------
# PARAMETERS
# ----------------------------------------
SEASON = 2025
WEEKS = {13, 14}
TEMPERATURE = 1.75
ODDS_PAYOUT = 0.9091  # -110

# ----------------------------------------
# HELPERS
# ----------------------------------------
def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def cover_prob_home(mu, sigma, spread_home):
    """
    Probability home team covers the spread.
    mu: model_spread (home - away)
    spread_home: closing_spread_home
    """
    z = (mu + spread_home) / sigma
    return norm_cdf(z / TEMPERATURE)

# ----------------------------------------
# LOAD DATA
# ----------------------------------------
df = pd.read_csv("../results/predictions_vs_vegas.csv")

# Filter to target weeks
df = df[
    (df["season"] == SEASON) &
    (df["week"].isin(WEEKS)) &
    df["closing_spread_home"].notna()
].copy()

if df.empty:
    raise ValueError("No games found for weeks 13–14 in saved data.")

# ----------------------------------------
# COMPUTE EV
# ----------------------------------------
p_home = cover_prob_home(
    df["model_spread"].values,
    df["sigma"].values,
    df["closing_spread_home"].values
)

p_away = 1 - p_home

ev_home = p_home * ODDS_PAYOUT - (1 - p_home)
ev_away = p_away * ODDS_PAYOUT - (1 - p_away)

df["bet_home"] = ev_home >= ev_away
df["bet_ev"] = np.where(df["bet_home"], ev_home, ev_away)

# ----------------------------------------
# SELECT +EV BETS
# ----------------------------------------
bets = df[df["bet_ev"] > 0].copy()

# ----------------------------------------
# DETERMINE BET SIDE STRING
# ----------------------------------------
def bet_side(row):
    s = row["closing_spread_home"]
    if row["bet_home"]:
        return f"{row['home_team']} {s:+}"
    else:
        return f"{row['away_team']} {(-s):+}"

bets["bet_side"] = bets.apply(bet_side, axis=1)

# ----------------------------------------
# ATS RESULT
# ----------------------------------------
bets["home_covers"] = bets["actual_margin"] + bets["closing_spread_home"] > 0
bets["won"] = np.where(
    bets["bet_home"],
    bets["home_covers"],
    ~bets["home_covers"]
)

# ----------------------------------------
# OUTPUT
# ----------------------------------------
cols = [
    "season",
    "week",
    "away_team",
    "home_team",
    "closing_spread_home",
    "model_spread",
    "sigma",
    "bet_side",
    "bet_ev",
    "won"
]

out = bets[cols].sort_values("bet_ev", ascending=False)

print("\n================ WEEKS 13–14 EV BACKTEST =================")
print(out)

print("\nSUMMARY")
print("Total bets:", len(out))
print("Win rate:", out["won"].mean())
print("Average EV per $1:", out["bet_ev"].mean())

out.to_csv("../results/backtest_weeks_13_14_2025_ev.csv", index=False)
print("\nSaved to ../results/backtest_weeks_13_14_2025_ev.csv")
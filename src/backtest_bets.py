import pandas as pd
import numpy as np

# ----------------------------------------
# PARAMETERS (PRO SETTINGS)
# ----------------------------------------
MIN_NORM_EDGE = 0.25          # normalized edge threshold
MAX_SPREAD = 10               # no double-digit spreads
BET_UNDERDOGS_ONLY = True

# ----------------------------------------
# LOAD HISTORICAL MODEL + VEGAS DATA
# ----------------------------------------
df = pd.read_csv("../results/predictions_vs_vegas.csv")

# ----------------------------------------
# COMPUTE RAW EDGE
# ----------------------------------------
df["raw_edge"] = df["model_spread"] - df["closing_spread_home"]

# ----------------------------------------
# NORMALIZE EDGE BY SPREAD MAGNITUDE
# ----------------------------------------
df["norm_edge"] = df["raw_edge"] / df["closing_spread_home"].abs().clip(lower=7)

# ----------------------------------------
# FILTER BAD MARKETS
# ----------------------------------------
df = df[df["closing_spread_home"].abs() <= MAX_SPREAD]
df = df[df["closing_spread_home"].notna()]

if BET_UNDERDOGS_ONLY:
    df = df[df["raw_edge"] < 0]

df = df[df["norm_edge"].abs() >= MIN_NORM_EDGE].copy()

# ----------------------------------------
# DETERMINE BET SIDE
# ----------------------------------------
def pick_side(row):
    spread = row["closing_spread_home"]

    if row["raw_edge"] < 0:
        # Bet away underdog
        if spread < 0:
            return f"{row['away_team']} +{abs(spread)}"
        else:
            return f"{row['away_team']} -{abs(spread)}"
    else:
        # Bet home underdog
        if spread < 0:
            return f"{row['home_team']} {spread}"
        else:
            return f"{row['home_team']} +{abs(spread)}"

df["bet_side"] = df.apply(pick_side, axis=1)

# ----------------------------------------
# ATS RESULT
# ----------------------------------------
df["won"] = np.sign(df["raw_edge"]) == np.sign(df["actual_margin"])

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
    "raw_edge",
    "norm_edge",
    "bet_side",
    "won"
]

df = df[cols].sort_values("norm_edge", ascending=False)

# ----------------------------------------
# SAVE
# ----------------------------------------
out = "../results/backtest_bets_filtered.csv"
df.to_csv(out, index=False)

print("\n================ FILTERED BACKTEST =================")
print(df)
print(f"\nSaved filtered backtest to {out}")

print("\nSUMMARY")
print("Total bets:", len(df))
print("Win rate:", df["won"].mean())

df["favorite"] = df["bet_side"].str.contains("-")
print("\nWin rate: favorites vs underdogs")
print(df.groupby("favorite")["won"].mean())
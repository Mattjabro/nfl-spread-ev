import pandas as pd
import numpy as np
from scipy.stats import norm

# ----------------------------------------
# PARAMETERS (MATCH LIVE SCRIPT)
# ----------------------------------------
TEMPERATURE = 1.75
EV_THRESHOLD = 0.08
ODDS = -110

# ----------------------------------------
# COVER PROBABILITY (HOME TEAM)
# ----------------------------------------
def cover_prob_home(mu, sigma, spread_home):
    z = (mu + spread_home) / sigma
    return norm.cdf(z / TEMPERATURE)

# ----------------------------------------
# EXPECTED VALUE PER $1
# ----------------------------------------
def ev_from_prob(p, odds=-110):
    if odds < 0:
        payout = 100 / abs(odds)
    else:
        payout = odds / 100
    return p * payout - (1 - p)

# ----------------------------------------
# LOAD HISTORICAL MODEL + VEGAS DATA
# ----------------------------------------
df = pd.read_csv("../results/predictions_vs_vegas.csv")

# ----------------------------------------
# SAFETY CHECK
# ----------------------------------------
required = {
    "season",
    "week",
    "home_team",
    "away_team",
    "model_spread",
    "sigma",
    "closing_spread_home",
    "actual_margin"
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ----------------------------------------
# COMPUTE COVER PROBABILITIES
# ----------------------------------------
df["p_home_cover"] = cover_prob_home(
    df["model_spread"].values,
    df["sigma"].values,
    df["closing_spread_home"].values
)

df["p_away_cover"] = 1 - df["p_home_cover"]

# ----------------------------------------
# COMPUTE EV
# ----------------------------------------
df["ev_home"] = ev_from_prob(df["p_home_cover"], ODDS)
df["ev_away"] = ev_from_prob(df["p_away_cover"], ODDS)

# ----------------------------------------
# SELECT BET SIDE
# ----------------------------------------
df["bet_side"] = np.where(
    df["ev_home"] > df["ev_away"],
    "HOME",
    "AWAY"
)

df["bet_ev"] = np.where(
    df["bet_side"] == "HOME",
    df["ev_home"],
    df["ev_away"]
)

# ----------------------------------------
# FILTER: STRONG EV ONLY
# ----------------------------------------
bets = df[df["bet_ev"] >= EV_THRESHOLD].copy()

# ----------------------------------------
# DETERMINE ATS RESULT
# ----------------------------------------
bets["won"] = np.where(
    bets["bet_side"] == "HOME",
    np.sign(bets["actual_margin"] + bets["closing_spread_home"]) > 0,
    np.sign(-bets["actual_margin"] + bets["closing_spread_home"]) > 0
)

# ----------------------------------------
# SUMMARY
# ----------------------------------------
print("\n================ STRONG EV BACKTEST =================")
print("Total bets:", len(bets))
print("Win rate:", bets["won"].mean())
print("Average EV per $1:", bets["bet_ev"].mean())

print("\nBy season:")
print(
    bets.groupby("season")
        .agg(
            bets=("won", "count"),
            win_rate=("won", "mean"),
            avg_ev=("bet_ev", "mean")
        )
)

# ----------------------------------------
# SAVE
# ----------------------------------------
out = "../results/backtest_strong_ev.csv"
bets.to_csv(out, index=False)
print(f"\nSaved to {out}")
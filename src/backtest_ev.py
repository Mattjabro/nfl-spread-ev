import pandas as pd
import numpy as np
from scipy.stats import norm

# ----------------------------------------
# PARAMETERS
# ----------------------------------------
TEMPERATURE = 1.75
ASSUMED_AMERICAN_ODDS = -110
MIN_EV = 0.02
MAX_SPREAD = 10

# ----------------------------------------
# HELPERS
# ----------------------------------------
def american_to_profit_per_1(odds):
    if odds < 0:
        return 100.0 / abs(odds)
    else:
        return odds / 100.0

def breakeven_prob(odds):
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    else:
        return 100.0 / (odds + 100.0)

def cover_prob_home(mu, sigma, spread_home):
    """
    Vectorized:
    P(home covers | model, uncertainty, Vegas spread)
    """
    denom = np.maximum(1e-6, sigma)
    z = (mu + spread_home) / denom
    return norm.cdf(z / TEMPERATURE)

def ev_per_1(p_win, odds):
    win_profit = american_to_profit_per_1(odds)
    return p_win * win_profit - (1 - p_win)

# ----------------------------------------
# LOAD MERGED DATA
# ----------------------------------------
df = pd.read_csv("../results/predictions_vs_vegas.csv")

required = {"model_spread", "closing_spread_home", "actual_margin", "sigma"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"predictions_vs_vegas.csv missing columns: {missing}")

# Optional sanity filter
df = df[df["closing_spread_home"].abs() <= MAX_SPREAD].copy()

# ----------------------------------------
# COMPUTE PROBABILITIES + EV
# ----------------------------------------
odds = ASSUMED_AMERICAN_ODDS
p_be = breakeven_prob(odds)

p_home = cover_prob_home(
    df["model_spread"].values,
    df["sigma"].values,
    df["closing_spread_home"].values
)

p_away = 1.0 - p_home

ev_home = ev_per_1(p_home, odds)
ev_away = ev_per_1(p_away, odds)

choose_home = ev_home >= ev_away

df["chosen_side"] = np.where(choose_home, "HOME", "AWAY")
df["p_win"] = np.where(choose_home, p_home, p_away)
df["ev_per_1"] = np.where(choose_home, ev_home, ev_away)
df["breakeven_p"] = p_be

# ----------------------------------------
# REALIZED ATS OUTCOME
# ----------------------------------------
home_covered = (df["actual_margin"] + df["closing_spread_home"]) > 0
away_covered = ~home_covered

df["won"] = np.where(choose_home, home_covered, away_covered)

# ----------------------------------------
# BET STRING
# ----------------------------------------
def bet_string(row):
    spread_home = row["closing_spread_home"]

    if row["chosen_side"] == "HOME":
        return (
            f"{row['home_team']} {spread_home}"
            if spread_home < 0
            else f"{row['home_team']} +{abs(spread_home)}"
        )
    else:
        away_spread = -spread_home
        return (
            f"{row['away_team']} {away_spread}"
            if away_spread < 0
            else f"{row['away_team']} +{abs(away_spread)}"
        )

df["bet_side"] = df.apply(bet_string, axis=1)

# ----------------------------------------
# FILTER TO BETS WE ACTUALLY PLACE
# ----------------------------------------
bets = df[df["ev_per_1"] >= MIN_EV].copy()
bets = bets.sort_values("ev_per_1", ascending=False)

# ----------------------------------------
# SAVE + REPORT
# ----------------------------------------
out = "../results/backtest_ev.csv"
bets.to_csv(out, index=False)

print("\n================ EV BACKTEST =================")
print("Total bets:", len(bets))
print("Win rate:", bets["won"].mean())
print("Average EV per $1:", bets["ev_per_1"].mean())

print("\nBy season:")
print(
    bets.groupby("season").agg(
        bets=("won", "count"),
        win_rate=("won", "mean"),
        avg_ev=("ev_per_1", "mean")
    )
)

print(f"\nSaved EV backtest to {out}")
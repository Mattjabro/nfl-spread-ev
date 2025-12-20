import pandas as pd
import numpy as np

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
IN_PATH = "../results/final_walkforward_predictions.csv"

# Implied market assumptions
MARKET_ODDS = -110
EDGE_THRESHOLD = 0.02   # minimum edge to bet

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def implied_prob_from_odds(odds):
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)

def ev_from_prob(p, odds=-110):
    if odds < 0:
        win = 100 / abs(odds)
        loss = 1.0
    else:
        win = odds / 100
        loss = 1.0
    return p * win - (1 - p) * loss

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv(IN_PATH)

print(f"\nLoaded {len(df)} walk-forward predictions")

# --------------------------------------------------
# 1. LOG LOSS + BRIER
# --------------------------------------------------
eps = 1e-9
p = df["p_home_win"].clip(eps, 1 - eps)
y = df["home_win"]

log_loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
brier = np.mean((p - y) ** 2)

print("\n=== Overall Accuracy Metrics ===")
print(f"Log Loss : {log_loss:.4f}")
print(f"Brier    : {brier:.4f}")

# --------------------------------------------------
# 2. CALIBRATION TABLE
# --------------------------------------------------
df["prob_bin"] = pd.cut(
    df["p_home_win"],
    bins=np.arange(0.3, 0.76, 0.05),
    include_lowest=True
)

calibration = (
    df.groupby("prob_bin")
      .agg(
          count=("home_win", "size"),
          mean_pred=("p_home_win", "mean"),
          win_rate=("home_win", "mean")
      )
      .reset_index()
)

print("\n=== Calibration Table ===")
print(calibration)

# --------------------------------------------------
# 3. ROI SIMULATION (FAIR MARKET)
# --------------------------------------------------
market_p = implied_prob_from_odds(MARKET_ODDS)

df["edge"] = df["p_home_win"] - market_p
df["bet"] = df["edge"] > EDGE_THRESHOLD

bets = df[df["bet"]].copy()

bets["profit"] = bets.apply(
    lambda r: ev_from_prob(r["p_home_win"], MARKET_ODDS),
    axis=1
)

roi = bets["profit"].mean() if len(bets) else 0.0

print("\n=== Betting Simulation ===")
print(f"Total bets placed : {len(bets)}")
print(f"Average EV / bet  : {roi:.4f}")
print(f"Total EV          : {bets['profit'].sum():.2f}")

# --------------------------------------------------
# 4. EDGE STRATIFICATION (VERY IMPORTANT)
# --------------------------------------------------
df["edge_bin"] = pd.cut(
    df["edge"],
    bins=[-1, -0.05, 0, 0.02, 0.05, 1]
)

edge_perf = (
    df.groupby("edge_bin")
      .agg(
          count=("edge", "size"),
          win_rate=("home_win", "mean"),
          avg_edge=("edge", "mean")
      )
      .reset_index()
)

print("\n=== Edge Stratification ===")
print(edge_perf)
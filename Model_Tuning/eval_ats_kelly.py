import sys
from pathlib import Path
import numpy as np
import pandas as pd
from math import erf, sqrt

# ---------------------------
# Paths
# ---------------------------
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.append(str(ROOT / "src"))

# ---------------------------
# Config
# ---------------------------
TEMPERATURE = 1.70
ODDS_AMERICAN = -110
FRACTIONAL_KELLY = 0.25
KELLY_CAP = 0.05
MIN_EDGE_POINTS = 1.0
START_BANKROLL = 100.0

PRED_PATH = RESULTS / "final_walkforward_predictions.csv"
VEGAS_PATH = RESULTS / "vegas_closing_lines.csv"

# ---------------------------
# Helpers
# ---------------------------
def norm_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))

def american_to_b(odds):
    return 100.0 / abs(odds) if odds < 0 else odds / 100.0

def kelly_fraction(p, odds=-110):
    b = american_to_b(odds)
    q = 1.0 - p
    return (b * p - q) / b

def log_loss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def brier(y, p):
    return np.mean((p - y) ** 2)

# ---------------------------
# Load + merge
# ---------------------------
pred = pd.read_csv(PRED_PATH)
for c in ["model_spread", "sigma", "p_home_win", "actual_margin", "home_win"]:
    pred[c] = pd.to_numeric(pred[c], errors="coerce")

pred = pred.dropna()

vegas = pd.read_csv(VEGAS_PATH).rename(columns={"closing_spread_home": "vegas_spread_home"})

df = pred.merge(
    vegas[["season", "week", "home_team", "away_team", "vegas_spread_home"]],
    on=["season", "week", "home_team", "away_team"],
    how="inner"
)

# ---------------------------
# ATS probabilities
# ---------------------------
df["z_cover"] = (df["model_spread"] + df["vegas_spread_home"]) / df["sigma"]
df["p_home_cover"] = norm_cdf(df["z_cover"] / TEMPERATURE)
df["home_cover"] = ((df["actual_margin"] + df["vegas_spread_home"]) > 0).astype(int)

df["edge_pts"] = df["model_spread"] - df["vegas_spread_home"]
df["abs_edge_pts"] = df["edge_pts"].abs()

# ---------------------------
# Bet selection
# ---------------------------
b = american_to_b(ODDS_AMERICAN)
df["p_best"] = np.where(df["p_home_cover"] >= 0.5, df["p_home_cover"], 1 - df["p_home_cover"])
df["side"] = np.where(df["p_home_cover"] >= 0.5, "HOME", "AWAY")
df["ev_per_unit"] = df["p_best"] * b - (1 - df["p_best"])

bets = df[df["abs_edge_pts"] >= MIN_EDGE_POINTS].copy()

# ---------------------------
# Kelly sizing
# ---------------------------
bets["kelly_raw"] = bets["p_best"].apply(lambda p: kelly_fraction(p, ODDS_AMERICAN))
bets["kelly_capped"] = np.clip(bets["kelly_raw"], 0.0, KELLY_CAP)
bets["stake_frac"] = FRACTIONAL_KELLY * bets["kelly_capped"]

bets["win"] = np.where(bets["side"] == "HOME", bets["home_cover"], 1 - bets["home_cover"])

# ---------------------------
# Bankroll sim
# ---------------------------
bankroll = START_BANKROLL
curve = []

for r in bets.sort_values(["season", "week"]).itertuples(index=False):
    stake = bankroll * r.stake_frac
    if r.win:
        bankroll += stake * b
    else:
        bankroll -= stake
    curve.append(bankroll)

bets["bankroll"] = curve

# ---------------------------
# Metrics (BETTABLE ONLY)
# ---------------------------
ats_ll = np.mean([log_loss(y, p) for y, p in zip(bets["home_cover"], bets["p_home_cover"])])
ats_brier = brier(bets["home_cover"], bets["p_home_cover"])

print("\n=== ATS PROBABILITY QUALITY (BETTABLE GAMES ONLY) ===")
print(f"Log Loss : {ats_ll:.4f}")
print(f"Brier    : {ats_brier:.4f}")

print("\n=== KELLY RESULTS ===")
print(f"Start bankroll : {START_BANKROLL:.2f}")
print(f"End bankroll   : {bets['bankroll'].iloc[-1]:.2f}")
print(f"Win rate       : {bets['win'].mean():.3f}")
print(f"Avg stake %    : {bets['stake_frac'].mean()*100:.2f}%")

out = RESULTS / "ats_kelly_bets.csv"
bets.to_csv(out, index=False)
print(f"\nSaved bets to {out}")
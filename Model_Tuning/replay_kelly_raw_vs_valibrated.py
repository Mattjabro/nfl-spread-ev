import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------
# Paths
# ---------------------------
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

RAW_PATH = RESULTS / "ats_kelly_bets.csv"
CAL_PATH = RESULTS / "ats_kelly_bets_calibrated.csv"

ODDS_AMERICAN = -110
FRACTIONAL_KELLY = 0.25
KELLY_CAP = 0.05
START_BANKROLL = 100.0

# ---------------------------
# Helpers
# ---------------------------
def american_to_b(odds):
    return 100.0 / abs(odds) if odds < 0 else odds / 100.0

def kelly_fraction(p, odds=-110):
    b = american_to_b(odds)
    return max((b * p - (1 - p)) / b, 0.0)

def simulate_bankroll(df, prob_col):
    bankroll = START_BANKROLL
    curve = []

    b = american_to_b(ODDS_AMERICAN)

    for r in df.sort_values(["season", "week"]).itertuples(index=False):
        p = float(getattr(r, prob_col))
        k = min(FRACTIONAL_KELLY * kelly_fraction(p), KELLY_CAP)
        stake = bankroll * k

        if stake <= 0:
            curve.append(bankroll)
            continue

        if int(r.win) == 1:
            bankroll += stake * b
        else:
            bankroll -= stake

        curve.append(bankroll)

    return np.array(curve)

# ---------------------------
# Load data
# ---------------------------
raw = pd.read_csv(RAW_PATH)
cal = pd.read_csv(CAL_PATH)

# ---------------------------
# Construct calibrated bet-side prob
# ---------------------------
cal["p_best_calibrated"] = np.where(
    cal["side"] == "HOME",
    cal["p_home_cover_cal"],
    1.0 - cal["p_home_cover_cal"]
)

raw["p_best_raw"] = raw["p_best"]

# ---------------------------
# Simulate bankrolls
# ---------------------------
raw_curve = simulate_bankroll(raw, "p_best_raw")
cal_curve = simulate_bankroll(cal, "p_best_calibrated")

# ---------------------------
# Summary
# ---------------------------
def summarize(name, curve, df, prob_col):
    returns = np.diff(curve) / curve[:-1]
    max_dd = np.min(curve / np.maximum.accumulate(curve) - 1)

    print(f"\n=== {name} ===")
    print(f"Final bankroll : {curve[-1]:.2f}")
    print(f"Net return     : {(curve[-1] / START_BANKROLL - 1)*100:.2f}%")
    print(f"Win rate       : {df['win'].mean():.3f}")
    print(f"Avg stake %    : {df['stake_frac'].mean()*100:.2f}%")
    print(f"Median stake % : {df['stake_frac'].median()*100:.2f}%")
    print(f"Max drawdown   : {max_dd*100:.2f}%")
    print(f"Return vol     : {np.std(returns):.4f}")

summarize("RAW PROBABILITIES", raw_curve, raw, "p_best_raw")
summarize("CALIBRATED PROBABILITIES", cal_curve, cal, "p_best_calibrated")
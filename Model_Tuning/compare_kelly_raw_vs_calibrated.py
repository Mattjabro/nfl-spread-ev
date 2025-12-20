# compare_kelly_raw_vs_calibrated.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

RAW_PATH = RESULTS / "ats_kelly_bets.csv"
CAL_PATH = RESULTS / "ats_kelly_bets_calibrated.csv"

START_BANKROLL = 100.0

def summarize(bets, label):
    bankroll = bets["bankroll"].values
    returns = np.diff(np.log(bankroll + 1e-12))

    peak = np.maximum.accumulate(bankroll)
    drawdown = (bankroll - peak) / peak

    print(f"\n=== {label} ===")
    print(f"Final bankroll : {bankroll[-1]:.2f}")
    print(f"Net return     : {(bankroll[-1]/START_BANKROLL - 1)*100:.2f}%")
    print(f"Win rate       : {bets['win'].mean():.3f}")
    print(f"Avg stake %    : bets['stake_frac'].mean()*100:.2f")
    print(f"Median stake % : bets['stake_frac'].median()*100:.2f")
    print(f"Max drawdown   : {drawdown.min()*100:.2f}%")
    print(f"Return vol     : {returns.std():.4f}")

# ---------------------------
# Load
# ---------------------------
raw = pd.read_csv(RAW_PATH)
cal = pd.read_csv(CAL_PATH)

# Ensure order
raw = raw.sort_values(["season", "week"]).reset_index(drop=True)
cal = cal.sort_values(["season", "week"]).reset_index(drop=True)

summarize(raw, "RAW PROBABILITIES")
summarize(cal, "CALIBRATED PROBABILITIES")
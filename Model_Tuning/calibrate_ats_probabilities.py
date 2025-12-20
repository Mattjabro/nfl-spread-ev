import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# ---------------------------
# Paths
# ---------------------------
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

IN_PATH = RESULTS / "ats_kelly_bets.csv"
OUT_PATH = RESULTS / "ats_kelly_bets_calibrated.csv"

# ---------------------------
# Load bets
# ---------------------------
df = pd.read_csv(IN_PATH)

# Safety
df = df.dropna(subset=["p_home_cover", "home_cover"])

p = df["p_home_cover"].values
y = df["home_cover"].values

# ---------------------------
# Fit isotonic calibrator
# ---------------------------
iso = IsotonicRegression(out_of_bounds="clip")
p_cal = iso.fit_transform(p, y)

df["p_home_cover_cal"] = p_cal

# ---------------------------
# Compare metrics
# ---------------------------
def log_loss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

ll_raw = np.mean([log_loss(yy, pp) for yy, pp in zip(y, p)])
ll_cal = np.mean([log_loss(yy, pp) for yy, pp in zip(y, p_cal)])

print("\n=== ATS PROBABILITY CALIBRATION ===")
print(f"Raw log loss       : {ll_raw:.4f}")
print(f"Calibrated log loss: {ll_cal:.4f}")

# ---------------------------
# Save
# ---------------------------
df.to_csv(OUT_PATH, index=False)
print(f"\nSaved calibrated bets to {OUT_PATH}")
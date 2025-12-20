import pandas as pd
import numpy as np
from math import sqrt
from scipy.special import erf   # vectorized erf

TEMPERATURE = 1.70
EDGE_MIN = 3.0

def norm_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

df = pd.read_csv("../results/full_report_inputs.csv")

# ================================
# EMBARGO: only seasons NOT used for tuning
# ================================
oos = df[df["season"] >= 2024].copy()

# Safety
for c in ["model_spread", "sigma", "vegas_spread_home", "actual_margin"]:
    oos[c] = pd.to_numeric(oos[c], errors="coerce")
oos = oos.dropna()

# Model probability vs closing line
oos["z"] = (oos["model_spread"] + oos["vegas_spread_home"]) / oos["sigma"]
oos["p_cover"] = norm_cdf(oos["z"] / TEMPERATURE)

# Edge filter (locked production rule)
oos["edge"] = oos["model_spread"] - oos["vegas_spread_home"]
oos = oos[oos["edge"].abs() >= EDGE_MIN]

# True outcome
oos["home_cover"] = ((oos["actual_margin"] + oos["vegas_spread_home"]) > 0).astype(int)

# Locked side decision
oos["side"] = np.where(oos["p_cover"] >= 0.5, 1, 0)
oos["win"] = (oos["side"] == oos["home_cover"]).astype(int)

# ================================
# REPORT
# ================================
print("\n================ EMBARGO TEST =================")
print("Seasons:", sorted(oos["season"].unique()))
print("Bets:", len(oos))
print(f"Win rate: {oos['win'].mean():.3f}")
print(f"Mean edge (pts): {oos['edge'].mean():+.3f}")
print(f"Median edge: {oos['edge'].median():+.3f}")
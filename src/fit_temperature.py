import numpy as np
import pandas as pd
from math import erf, sqrt

def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

# Load historical predictions
df = pd.read_csv("../results/predictions.csv")

# Sanity check
required = {"model_spread", "Sigma", "actual_margin"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# z-score using model uncertainty
df["z"] = df["model_spread"] / df["Sigma"]
df["home_win"] = (df["actual_margin"] > 0).astype(int)

temps = np.linspace(0.5, 3.5, 121)
briers = []

for T in temps:
    probs = df["z"].apply(lambda z: norm_cdf(z / T))
    brier = np.mean((probs - df["home_win"]) ** 2)
    briers.append(brier)

best_idx = np.argmin(briers)
best_T = temps[best_idx]
best_brier = briers[best_idx]

print(f"Best temperature: {best_T:.2f}")
print(f"Brier after calibration: {best_brier:.4f}")
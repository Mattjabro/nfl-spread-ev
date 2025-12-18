import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your historical prediction file
# This should include:
# - model_home_win_prob
# - actual_margin (home - away)
preds = pd.read_csv("../results/predictions.csv")

# Actual outcome: 1 if home wins, 0 otherwise
preds["home_win"] = (preds["actual_margin"] > 0).astype(int)

# ---------- Brier Score ----------
brier = np.mean((preds["model_home_win_prob"] - preds["home_win"]) ** 2)
print(f"Brier score: {brier:.4f}")

# ---------- Calibration bins ----------
bins = np.linspace(0, 1, 11)
preds["bin"] = pd.cut(preds["model_home_win_prob"], bins=bins)

calibration = preds.groupby("bin").agg(
    mean_pred_prob=("model_home_win_prob", "mean"),
    actual_win_rate=("home_win", "mean"),
    count=("home_win", "count")
).reset_index()

print("\nCalibration table:")
print(calibration)

# ---------- Plot ----------
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
plt.scatter(
    calibration["mean_pred_prob"],
    calibration["actual_win_rate"],
    s=calibration["count"],
    alpha=0.7
)
plt.xlabel("Predicted Home Win Probability")
plt.ylabel("Actual Home Win Rate")
plt.title("Calibration Curve")
plt.legend()
plt.grid(True)
plt.show()
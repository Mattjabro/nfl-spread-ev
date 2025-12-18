import pandas as pd
import numpy as np

df = pd.read_csv("../results/predictions_vs_vegas.csv")

# Filter to 2025 Weeks 14–15
df = df[
    (df["season"] == 2025) &
    (df["week"].isin([14, 15]))
].copy()

if df.empty:
    raise ValueError("No games found for Weeks 14–15 of 2025.")

# Model side vs actual
df["model_correct"] = (
    np.sign(df["model_spread"]) ==
    np.sign(df["actual_margin"])
)

print("\n================ WEEKS 14–15 ATS CHECK =================")
print(df[[
    "week",
    "away_team",
    "home_team",
    "closing_spread_home",
    "model_spread",
    "actual_margin",
    "model_correct"
]])

print("\nSUMMARY")
print("Total games:", len(df))
print("ATS win rate:", df["model_correct"].mean())
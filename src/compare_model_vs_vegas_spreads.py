import pandas as pd
import numpy as np

# Load model predictions
model = pd.read_csv("../results/predictions.csv")

# Load Vegas closing spreads
vegas = pd.read_csv("../results/vegas_closing_lines.csv")

# Merge on game identifiers
df = model.merge(
    vegas,
    on=["season", "week", "home_team", "away_team"],
    how="inner"
)

# Compute model edge vs Vegas
# Both spreads are HOME minus AWAY
df["edge"] = df["model_spread"] - df["closing_spread_home"]

# Did model pick the correct side?
df["model_side_correct"] = (
    np.sign(df["model_spread"]) == np.sign(df["actual_margin"])
)

# Did Vegas pick the correct side?
df["vegas_side_correct"] = (
    np.sign(df["closing_spread_home"]) == np.sign(df["actual_margin"])
)

print("\n================ BASIC EDGE STATISTICS ================")
print(df["edge"].describe())

print("\n================ SIDE ACCURACY ========================")
print(f"Model side accuracy: {df['model_side_correct'].mean():.3f}")
print(f"Vegas side accuracy: {df['vegas_side_correct'].mean():.3f}")

# Bucket edges by magnitude
bins = [-15, -7, -5, -3, -2, -1, 1, 2, 3, 5, 7, 15]
df["edge_bucket"] = pd.cut(df["edge"], bins)

bucket_summary = df.groupby("edge_bucket").agg(
    games=("edge", "count"),
    avg_edge=("edge", "mean"),
    model_win_rate=("model_side_correct", "mean"),
    avg_actual_margin=("actual_margin", "mean")
)

print("\n================ EDGE BUCKET PERFORMANCE ===============")
print(bucket_summary)

# Extra diagnostic: do bigger disagreements matter?
df["abs_edge"] = df["edge"].abs()

print("\n================ BY ABSOLUTE EDGE =====================")
print(
    df.groupby(pd.cut(df["abs_edge"], [0, 1, 2, 3, 5, 10]))
      .agg(
          games=("edge", "count"),
          model_win_rate=("model_side_correct", "mean"),
          avg_actual_margin=("actual_margin", "mean")
      )
)

# --------------------------------------------------
# SAVE MERGED DATA FOR BLENDING
# --------------------------------------------------
out = "../results/predictions_vs_vegas.csv"
df.to_csv(out, index=False)
print(f"\nSaved merged model + Vegas data to {out}")
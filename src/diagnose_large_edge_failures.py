import pandas as pd
import numpy as np

# Load merged model + Vegas data
model = pd.read_csv("../results/predictions.csv")
vegas = pd.read_csv("../results/vegas_closing_lines.csv")

df = model.merge(
    vegas,
    on=["season", "week", "home_team", "away_team"],
    how="inner"
)

# Compute edge
df["edge"] = df["model_spread"] - df["closing_spread_home"]
df["abs_edge"] = df["edge"].abs()

# Actual outcome sign
df["actual_sign"] = np.sign(df["actual_margin"])
df["model_sign"] = np.sign(df["model_spread"])
df["vegas_sign"] = np.sign(df["closing_spread_home"])

# Large disagreement games only
large = df[df["abs_edge"] >= 7].copy()

print("\n================ LARGE EDGE SUMMARY ==================")
print(f"Total large-edge games: {len(large)}")
print("Model side accuracy:",
      (large["model_sign"] == large["actual_sign"]).mean())
print("Vegas side accuracy:",
      (large["vegas_sign"] == large["actual_sign"]).mean())

# Which side is model usually wrong on?
large["model_wrong"] = large["model_sign"] != large["actual_sign"]

print("\n================ HOME VS AWAY ========================")
print(
    large.groupby("model_wrong")
         .agg(
             games=("edge", "count"),
             avg_edge=("edge", "mean"),
             avg_actual_margin=("actual_margin", "mean")
         )
)

# Are underdogs the issue?
large["model_favored_home"] = large["model_spread"] < 0
large["vegas_favored_home"] = large["closing_spread_home"] < 0

print("\n================ FAVORITE MISMATCH ==================")
print(
    large.groupby(["model_favored_home", "vegas_favored_home"])
         .agg(
             games=("edge", "count"),
             model_win_rate=("model_sign", lambda x: (x == large.loc[x.index, "actual_sign"]).mean()),
             avg_edge=("edge", "mean")
         )
)

# Teams most often involved in large-edge failures
print("\n================ TEAMS MOST INVOLVED =================")
teams = pd.concat([
    large.loc[large["model_wrong"], "home_team"],
    large.loc[large["model_wrong"], "away_team"]
])

print(teams.value_counts().head(15))

# Seasonality check
print("\n================ BY WEEK =============================")
print(
    large.groupby("week")
         .agg(
             games=("edge", "count"),
             model_win_rate=("model_sign", lambda x: (x == large.loc[x.index, "actual_sign"]).mean())
         )
)

# Show worst misses
print("\n================ WORST MISSES ========================")
worst = large.sort_values("abs_edge", ascending=False).head(15)
print(
    worst[[
        "season",
        "week",
        "away_team",
        "home_team",
        "model_spread",
        "closing_spread_home",
        "actual_margin",
        "edge"
    ]]
)
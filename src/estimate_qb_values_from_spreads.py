import pandas as pd
import nfl_data_py as nfl

START_SEASON = 2018
END_SEASON = 2025
MAX_WEEK_2025 = 15

# --------------------------------------------------
# Load Vegas closing spreads
# --------------------------------------------------
vegas = pd.read_csv("../results/vegas_closing_lines.csv")

# Restrict 2025 to completed weeks only
vegas = vegas[
    (vegas["season"] < 2025) |
    ((vegas["season"] == 2025) & (vegas["week"] <= MAX_WEEK_2025))
]

# --------------------------------------------------
# Load play-by-play data
# --------------------------------------------------
pbp = nfl.import_pbp_data(
    years=list(range(START_SEASON, END_SEASON + 1)),
    downcast=True
)

# Restrict pbp to completed weeks
pbp = pbp[
    (pbp["season"] < 2025) |
    ((pbp["season"] == 2025) & (pbp["week"] <= MAX_WEEK_2025))
]

# Keep only pass plays
passes = pbp[
    (pbp["play_type"] == "pass") &
    (pbp["passer_player_name"].notna())
].copy()

# --------------------------------------------------
# Identify primary QB per team/game
# --------------------------------------------------
qb_games = (
    passes.groupby(
        ["season", "week", "posteam", "passer_player_name"]
    )
    .size()
    .reset_index(name="pass_attempts")
)

# Keep QB with most attempts per team/game
qb_games = (
    qb_games.sort_values("pass_attempts", ascending=False)
    .groupby(["season", "week", "posteam"])
    .head(1)
)

qb_games = qb_games.rename(columns={
    "posteam": "home_team",
    "passer_player_name": "qb_name"
})

# --------------------------------------------------
# Merge with Vegas spreads
# --------------------------------------------------
df = vegas.merge(
    qb_games,
    on=["season", "week", "home_team"],
    how="left"
)

df = df.dropna(subset=["qb_name"])

# --------------------------------------------------
# Estimate QB value (LEAGUE-ANCHORED)
# --------------------------------------------------
# IMPORTANT:
# QB value is measured relative to the league, NOT the team.
# This allows QBs to explain persistent team strength.

league_avg_spread = df["closing_spread_home"].mean()

df["qb_value"] = df["closing_spread_home"] - league_avg_spread

qb_values = (
    df.groupby("qb_name")
      .agg(
          qb_value_points=("qb_value", "mean"),
          games=("qb_value", "count")
      )
      .reset_index()
      .sort_values("qb_value_points", ascending=False)
)

# --------------------------------------------------
# Save output
# --------------------------------------------------
out = "../results/estimated_qb_values.csv"
qb_values.to_csv(out, index=False)

print(f"Saved estimated QB values to {out}")
print(f"Includes data through Week {MAX_WEEK_2025}, 2025")
print("\nTop QBs:")
print(qb_values.head(10))
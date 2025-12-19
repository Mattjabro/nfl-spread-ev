import pandas as pd
import nfl_data_py as nfl

START_SEASON = 2018
END_SEASON = 2025
MAX_WEEK_2025 = 15

# --------------------------------------------------
# Load Vegas closing spreads
# --------------------------------------------------
vegas = pd.read_csv("../results/vegas_closing_lines.csv")

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

pbp = pbp[
    (pbp["season"] < 2025) |
    ((pbp["season"] == 2025) & (pbp["week"] <= MAX_WEEK_2025))
]

# --------------------------------------------------
# Keep pass plays only
# --------------------------------------------------
passes = pbp[
    (pbp["play_type"] == "pass") &
    (pbp["passer_player_name"].notna())
].copy()

# --------------------------------------------------
# Identify primary QB per TEAM per WEEK
# --------------------------------------------------
qb_games = (
    passes.groupby(
        ["season", "week", "posteam", "passer_player_name"]
    )
    .size()
    .reset_index(name="pass_attempts")
)

# QB with most attempts for that team in that week
qb_games = (
    qb_games.sort_values("pass_attempts", ascending=False)
    .groupby(["season", "week", "posteam"])
    .head(1)
)

qb_games = qb_games.rename(columns={
    "posteam": "team",
    "passer_player_name": "qb_name"
})

# --------------------------------------------------
# Attach QBs to Vegas data (SEASON + WEEK + TEAM)
# --------------------------------------------------
df = vegas.merge(
    qb_games.rename(columns={"team": "home_team", "qb_name": "home_qb"}),
    on=["season", "week", "home_team"],
    how="left"
)

df = df.merge(
    qb_games.rename(columns={"team": "away_team", "qb_name": "away_qb"}),
    on=["season", "week", "away_team"],
    how="left"
)

df = df.dropna(subset=["home_qb", "away_qb"])

# --------------------------------------------------
# Estimate QB value (league-anchored)
# --------------------------------------------------
league_avg_spread = df["closing_spread_home"].mean()

home_vals = df.assign(
    qb=df["home_qb"],
    qb_value=df["closing_spread_home"] - league_avg_spread
)[["qb", "qb_value"]]

away_vals = df.assign(
    qb=df["away_qb"],
    qb_value=-(df["closing_spread_home"] - league_avg_spread)
)[["qb", "qb_value"]]

qb_values = (
    pd.concat([home_vals, away_vals])
    .groupby("qb")
    .agg(
        qb_value_points=("qb_value", "mean"),
        games=("qb_value", "count")
    )
    .reset_index()
    .rename(columns={"qb": "qb_name"})
    .sort_values("qb_value_points", ascending=False)
)

# --------------------------------------------------
# Save outputs
# --------------------------------------------------
qb_values.to_csv("../results/estimated_qb_values.csv", index=False)

# ALSO SAVE last-week starter mapping for Streamlit
last_week = df[
    (df["season"] == 2025) & (df["week"] == MAX_WEEK_2025)
]

last_qbs = pd.concat([
    last_week[["home_team", "home_qb"]].rename(
        columns={"home_team": "team", "home_qb": "qb"}
    ),
    last_week[["away_team", "away_qb"]].rename(
        columns={"away_team": "team", "away_qb": "qb"}
    )
])

last_qbs.to_csv("../results/last_week_starting_qbs.csv", index=False)

print("Saved:")
print("- estimated_qb_values.csv")
print("- last_week_starting_qbs.csv")
print(f"Data through Week {MAX_WEEK_2025}, 2025")
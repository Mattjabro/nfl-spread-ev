import pandas as pd
import nfl_data_py as nfl
from pathlib import Path

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
SEASON = 2025
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

OUT_PATH = RESULTS_DIR / "last_week_starting_qbs.csv"

# --------------------------------------------------
# Load schedules to find last completed week
# --------------------------------------------------
sched = nfl.import_schedules([SEASON])

completed = sched[
    (sched["game_type"] == "REG") &
    (sched["home_score"].notna())
]

if completed.empty:
    raise RuntimeError("No completed games found for season")

last_week = int(completed["week"].max())
print(f"Last completed week: {last_week}")

# --------------------------------------------------
# Load play-by-play
# --------------------------------------------------
pbp = nfl.import_pbp_data(
    years=[SEASON],
    downcast=True
)

pbp = pbp[pbp["week"] == last_week]
pbp = pbp[
    (pbp["play_type"] == "pass") &
    (pbp["passer_player_name"].notna())
]

# --------------------------------------------------
# Identify starting QB = most pass attempts
# --------------------------------------------------
qb_games = (
    pbp.groupby(["posteam", "passer_player_name"])
       .size()
       .reset_index(name="pass_attempts")
       .sort_values("pass_attempts", ascending=False)
       .groupby("posteam")
       .head(1)
       .rename(columns={
           "posteam": "team",
           "passer_player_name": "qb"
       })
)

qb_games = qb_games[["team", "qb"]].sort_values("team")

# --------------------------------------------------
# Save
# --------------------------------------------------
qb_games.to_csv(OUT_PATH, index=False)

print(f"Saved {len(qb_games)} teams to {OUT_PATH}")
print(qb_games.head())
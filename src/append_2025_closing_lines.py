import pandas as pd
import nfl_data_py as nfl

# Load existing historical closing spreads
hist = pd.read_csv("../results/vegas_closing_lines.csv")

# Load completed 2025 games
sched = nfl.import_schedules([2025])
sched = sched[
    (sched["game_type"] == "REG") &
    (sched["home_score"].notna())
]

out_2025 = sched[[
    "season",
    "week",
    "home_team",
    "away_team",
    "spread_line"
]].rename(columns={"spread_line": "closing_spread_home"})

# Append and deduplicate
combined = pd.concat([hist, out_2025], ignore_index=True)
combined = combined.drop_duplicates(
    subset=["season", "week", "home_team", "away_team"],
    keep="last"
)

combined.to_csv("../results/vegas_closing_lines.csv", index=False)

print("Updated vegas_closing_lines.csv")
print(combined["season"].value_counts().sort_index())
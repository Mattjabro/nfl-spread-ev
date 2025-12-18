import pandas as pd
import nfl_data_py as nfl

def get_vegas_lines(seasons):
    odds = nfl.import_odds(seasons)

    # Keep only spread markets
    odds = odds[odds["market"] == "spread"]

    # Keep closing line if available, else latest
    odds = odds.sort_values("timestamp").groupby(
        ["season", "week", "home_team", "away_team"]
    ).tail(1)

    vegas = odds[[
        "season",
        "week",
        "home_team",
        "away_team",
        "home_spread"
    ]].rename(columns={
        "home_spread": "closing_spread_home"
    })

    return vegas


if __name__ == "__main__":
    vegas = get_vegas_lines([2025])
    out = "../results/vegas_lines_2025.csv"
    vegas.to_csv(out, index=False)
    print(f"Saved {len(vegas)} Vegas lines to {out}")
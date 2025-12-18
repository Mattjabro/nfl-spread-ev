import pandas as pd
import nfl_data_py as nfl

def get_closing_spreads(start_season=2018, end_season=2024):
    dfs = []

    for season in range(start_season, end_season + 1):
        df = nfl.import_schedules([season])

        # Keep regular season games only
        df = df[df["game_type"] == "REG"]

        # We want closing spread from home perspective
        # nfl_data_py convention:
        # spread_line < 0 means home favorite
        out = df[[
            "season",
            "week",
            "home_team",
            "away_team",
            "spread_line"
        ]].copy()

        out.rename(columns={
            "spread_line": "closing_spread_home"
        }, inplace=True)

        dfs.append(out)

    return pd.concat(dfs, ignore_index=True)


if __name__ == "__main__":
    vegas = get_closing_spreads()
    out = "../results/vegas_closing_lines.csv"
    vegas.to_csv(out, index=False)
    print(f"Saved Vegas closing lines to {out}")
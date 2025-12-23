import nfl_data_py as nfl
import requests
import pandas as pd
from team_map import TEAM_MAP

API_KEY = "b239602837a9227e045963ea28dbb28d"
URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

params = {
    "apiKey": API_KEY,
    "regions": "us",
    "markets": "spreads",
    "oddsFormat": "american",
}

def get_current_vegas_lines(season):
    r = requests.get(URL, params=params)
    r.raise_for_status()
    data = r.json()

    sched = nfl.import_schedules([season])
    sched = sched[sched["game_type"] == "REG"]

    week_lookup = {
        (row.home_team, row.away_team): row.week
        for _, row in sched.iterrows()
    }

    rows = []

    for game in data:
        home = TEAM_MAP.get(game["home_team"])
        away = TEAM_MAP.get(game["away_team"])
        if home is None or away is None:
            continue

        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue

        spreads = bookmakers[0]["markets"][0]["outcomes"]

        home_spread = None
        for o in spreads:
            if TEAM_MAP.get(o["name"]) == home:
                home_spread = o["point"]

        if home_spread is None:
            continue

        wk = week_lookup.get((home, away))
        if wk is None:
            continue

        rows.append({
            "season": season,
            "week": int(wk),
            "home_team": home,
            "away_team": away,
            "closing_spread_home": home_spread
        })

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["season", "week", "home_team", "away_team"],
        keep="last"
    )

    return df

if __name__ == "__main__":
    vegas = get_current_vegas_lines(2025)
    out = "../results/vegas_lines_2025.csv"
    vegas.to_csv(out, index=False)

    print(vegas.sort_values("week"))
    print(f"\nSaved {len(vegas)} Vegas lines to {out}")
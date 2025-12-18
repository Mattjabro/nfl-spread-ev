import requests
import pandas as pd
from team_map import TEAM_MAP

API_KEY = "b239602837a9227e045963ea28dbb28d"  # no quotes around key itself

URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

params = {
    "apiKey": API_KEY,
    "regions": "us",
    "markets": "spreads",
    "oddsFormat": "american",
}

def get_current_vegas_lines(season, week):
    r = requests.get(URL, params=params)
    r.raise_for_status()
    data = r.json()

    rows = []

    for game in data:
        home = TEAM_MAP.get(game["home_team"])
        away = TEAM_MAP.get(game["away_team"])
        if home is None or away is None:
            continue

        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue

        markets = bookmakers[0].get("markets", [])
        spreads = markets[0].get("outcomes", [])

        home_spread = None
        for o in spreads:
            if TEAM_MAP.get(o["name"]) == home:
                home_spread = o["point"]

        if home_spread is None:
            continue

        rows.append({
            "season": season,
            "week": week,
            "home_team": home,
            "away_team": away,
            "closing_spread_home": home_spread
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    vegas = get_current_vegas_lines(2025, 16)

    out = "../results/vegas_lines_2025.csv"
    vegas.to_csv(out, index=False)

    print(vegas)
    print(f"\nSaved {len(vegas)} Vegas lines to {out}")
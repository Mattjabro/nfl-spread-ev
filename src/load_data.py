import nfl_data_py as nfl
import pandas as pd

def load_games(start_season=2015, end_season=2025):
    games = nfl.import_schedules(range(start_season, end_season + 1))

    games = games[
        (games["game_type"] == "REG") &
        (games["home_score"].notna()) &
        (games["away_score"].notna())
    ]

    df = games[
        ["season", "week", "home_team", "away_team",
         "home_score", "away_score"]
    ].copy()

    df["global_week"] = (
        (df["season"] - df["season"].min()) * 18 + df["week"]
    )

    df["margin"] = df["home_score"] - df["away_score"]

    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
    team_to_idx = {team: i for i, team in enumerate(teams)}

    df["home_idx"] = df["home_team"].map(team_to_idx)
    df["away_idx"] = df["away_team"].map(team_to_idx)

    return df.sort_values("global_week").reset_index(drop=True), team_to_idx


def attach_qbs(df, start_season=2015, end_season=2025):
    qb_adj = pd.read_csv("../results/qb_adjustments.csv")

    pbp = nfl.import_pbp_data(
        years=list(range(start_season, end_season + 1)),
        downcast=True
    )

    passes = pbp[pbp["play_type"] == "pass"]

    qb_games = (
        passes.groupby(
            ["season", "week", "posteam", "passer_player_name"]
        )
        .size()
        .reset_index(name="pass_attempts")
        .sort_values("pass_attempts", ascending=False)
        .groupby(["season", "week", "posteam"])
        .head(1)
        .rename(columns={
            "posteam": "team",
            "passer_player_name": "qb_name"
        })
    )

    df = df.merge(
        qb_games.rename(columns={"team": "home_team", "qb_name": "home_qb"}),
        on=["season", "week", "home_team"],
        how="left"
    )

    df = df.merge(
        qb_games.rename(columns={"team": "away_team", "qb_name": "away_qb"}),
        on=["season", "week", "away_team"],
        how="left"
    )

    qb_map = dict(zip(qb_adj["qb_name"], qb_adj["qb_value_shrunk"]))

    df["home_qb_val"] = df["home_qb"].map(qb_map).fillna(0.0)
    df["away_qb_val"] = df["away_qb"].map(qb_map).fillna(0.0)

    return df
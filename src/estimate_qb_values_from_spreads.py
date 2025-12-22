# src/estimate_qb_values_from_spreads.py

import pandas as pd
import nfl_data_py as nfl
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

START_SEASON = 2018

def estimate_qb_values(season, max_week):
    # --------------------------------------------------
    # Load Vegas closing spreads
    # --------------------------------------------------
    vegas = pd.read_csv(RESULTS_DIR / "vegas_closing_lines.csv")

    vegas = vegas[
        (vegas["season"] < season) |
        ((vegas["season"] == season) & (vegas["week"] <= max_week))
    ]

    # --------------------------------------------------
    # Load play-by-play data
    # --------------------------------------------------
    pbp = nfl.import_pbp_data(
        years=list(range(START_SEASON, season + 1)),
        downcast=True
    )

    pbp = pbp[
        (pbp["season"] < season) |
        ((pbp["season"] == season) & (pbp["week"] <= max_week))
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

    qb_games = (
        qb_games.sort_values("pass_attempts", ascending=False)
        .groupby(["season", "week", "posteam"])
        .head(1)
        .rename(columns={
            "posteam": "team",
            "passer_player_name": "qb_name"
        })
    )

    # --------------------------------------------------
    # Attach QBs to Vegas data
    # --------------------------------------------------
    df = vegas.merge(
        qb_games.rename(columns={"team": "home_team", "qb_name": "home_qb"}),
        on=["season", "week", "home_team"],
        how="left"
    ).merge(
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
    )

    return qb_values
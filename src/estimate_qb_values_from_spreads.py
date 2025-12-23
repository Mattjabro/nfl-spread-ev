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

    # ================================
    # TEST 2: VEGAS COVERAGE CHECK
    # ================================
    def debug_vegas_week(season, week):
        v = vegas[(vegas["season"] == season) & (vegas["week"] == week)]
        print(f"\n[VEGAS GAMES — {season} WEEK {week}]")
        print(v[["home_team", "away_team", "closing_spread_home"]])

    vegas = vegas[
        (vegas["season"] < season) |
        ((vegas["season"] == season) & (vegas["week"] <= max_week))
    ]

    debug_vegas_week(season, max_week)

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

    # ================================
    # TEST 1: QB STARTER SANITY CHECK
    # ================================
    def debug_qb_week(season, week, qb_name=None):
        subset = qb_games[
            (qb_games["season"] == season) &
            (qb_games["week"] == week)
        ].sort_values("pass_attempts", ascending=False)

        print("\n[QB STARTERS — PBPs]")
        print(subset.head(10))

        if qb_name is not None:
            found = subset[subset["qb_name"].str.contains(qb_name, case=False, na=False)]
            print(f"\n[LOOKING FOR {qb_name}]")
            print(found if len(found) else "❌ NOT FOUND IN PBPs")

    # Example use
    debug_qb_week(season, max_week, qb_name="Ewers")

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

    # ================================
    # TEST 3: MERGE FAILURE DIAGNOSTIC
    # ================================
    merged = vegas.merge(
        qb_games.rename(columns={"team": "home_team", "qb_name": "home_qb"}),
        on=["season", "week", "home_team"],
        how="left"
    ).merge(
        qb_games.rename(columns={"team": "away_team", "qb_name": "away_qb"}),
        on=["season", "week", "away_team"],
        how="left"
    )

    missing = merged[
        merged["home_qb"].isna() | merged["away_qb"].isna()
    ]

    print("\n[MERGE FAILURES — QB MISSING]")
    print(
        missing[[
            "season", "week",
            "home_team", "away_team",
            "home_qb", "away_qb"
        ]].head(10)
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
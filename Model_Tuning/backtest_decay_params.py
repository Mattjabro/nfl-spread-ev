import sys
from pathlib import Path
import numpy as np
import pandas as pd
from math import erf, sqrt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from load_data import load_games, attach_qbs
from model_margin_decay_season_boost import fit_margin_decay_model


def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def backtest_params(
    decay_rate,
    season_boost,
    start_season=2019,
    end_season=2025,
    max_week_2025=15,
    logger=print
):
    logger(
        f"Starting backtest | λ={decay_rate:.3f}, season_boost={season_boost:.2f}"
    )

    df, team_to_idx = load_games(start_season=start_season, end_season=end_season)
    df = attach_qbs(df)

    results = []

    for season in range(start_season + 1, end_season + 1):
        max_week = max_week_2025 if season == 2025 else 18
        logger(f"Season {season}: weeks 1–{max_week}")

        for week in range(1, max_week + 1):
            logger(
                f"  → Training for prediction: season={season}, week={week} "
                f"(λ={decay_rate:.3f}, c={season_boost:.2f})"
            )

            train = df[
                (df["season"] < season) |
                ((df["season"] == season) & (df["week"] < week))
            ].copy()

            test = df[
                (df["season"] == season) & (df["week"] == week)
            ].copy()

            if len(test) == 0 or len(train) < 200:
                logger("    Skipped (insufficient data)")
                continue

            trace = fit_margin_decay_model(
                train,
                len(team_to_idx),
                decay_rate=decay_rate,
                season_boost=season_boost,
                prediction_season=season
            )

            post = trace.posterior
            team_s = post["team_strength"].mean(axis=(0, 1))
            hfa = post["hfa"].mean().item()
            sigma = post["sigma"].mean().item()

            for _, g in test.iterrows():
                mu = (
                    hfa
                    + team_s[g.home_idx]
                    - team_s[g.away_idx]
                    + g.home_qb_val
                    - g.away_qb_val
                )

                p_home = norm_cdf(mu / sigma)
                y = 1 if g.margin > 0 else 0

                logloss = -(y * np.log(p_home) + (1 - y) * np.log(1 - p_home))
                results.append(logloss)

    avg_loss = float(np.mean(results))
    logger(
        f"Finished backtest | λ={decay_rate:.3f}, c={season_boost:.2f} "
        f"→ mean log loss = {avg_loss:.4f}"
    )

    return avg_loss
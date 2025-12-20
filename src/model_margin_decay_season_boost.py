import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import pymc as pm
import numpy as np


def fit_margin_decay_model(
    train_df,
    n_teams,
    decay_rate=0.02,
    season_boost=1.0,
    prediction_season=None
):
    max_week = train_df["global_week"].max()
    age = max_week - train_df["global_week"].values

    weights = np.exp(-decay_rate * age)

    if prediction_season is not None:
        same_season = (train_df["season"].values == prediction_season)
        weights = weights * np.where(same_season, season_boost, 1.0)

    home_idx = train_df["home_idx"].values
    away_idx = train_df["away_idx"].values
    home_qb_val = train_df["home_qb_val"].values
    away_qb_val = train_df["away_qb_val"].values

    with pm.Model() as model:
        tau = pm.HalfNormal("tau", 6)
        team_strength = pm.Normal("team_strength", 0, tau, shape=n_teams)

        hfa = pm.Normal("hfa", 2.0, 1.0)
        sigma = pm.HalfNormal("sigma", 10)

        mu = (
            hfa
            + team_strength[home_idx]
            - team_strength[away_idx]
            + home_qb_val
            - away_qb_val
        )

        pm.Normal(
            "margin_obs",
            mu=mu,
            sigma=sigma / np.sqrt(weights),
            observed=train_df["margin"].values
        )

        trace = pm.sample(
            1000,
            tune=1000,
            chains=2,
            target_accept=0.9,
            progressbar=False
        )

    return trace
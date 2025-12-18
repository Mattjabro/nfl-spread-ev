import pymc as pm
import numpy as np
import arviz as az

from load_data import load_games, attach_qbs


def fit_margin_decay_model(train_df, n_teams, decay_rate=0.02):
    max_week = train_df["global_week"].max()
    age = max_week - train_df["global_week"].values
    weights = np.exp(-decay_rate * age)

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
            2000,
            tune=2000,
            chains=4,
            target_accept=0.9
        )

    return trace


if __name__ == "__main__":
    df, team_to_idx = load_games()
    df = attach_qbs(df)

    train = df[
        (df["season"] < 2025) |
        ((df["season"] == 2025) & (df["week"] <= 14))
    ]

    trace = fit_margin_decay_model(train, len(team_to_idx))
    print(az.summary(trace, var_names=["hfa", "sigma", "tau"]))
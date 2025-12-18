import numpy as np
import pandas as pd

from load_data import load_games, attach_qbs
from model_margin_decay_weighted import fit_margin_decay_model


def predict_historical(start_season=2018, end_season=2024, decay_rate=0.02):
    # --------------------------------------------------
    # Load data + attach QB values
    # --------------------------------------------------
    df, team_to_idx = load_games(start_season=start_season, end_season=end_season)
    df = attach_qbs(df, start_season=start_season, end_season=end_season)

    # --------------------------------------------------
    # Fit model ONCE on all historical data
    # --------------------------------------------------
    trace = fit_margin_decay_model(df, len(team_to_idx), decay_rate)

    post = trace.posterior
    team_s = post["team_strength"].values      # (chains, draws, teams)
    hfa_s = post["hfa"].values                 # (chains, draws)
    sigma_s = post["sigma"].values             # (chains, draws)

    rows = []

    # --------------------------------------------------
    # Generate predictions game-by-game
    # --------------------------------------------------
    for _, g in df.iterrows():
        mu = (
            hfa_s
            + team_s[..., g.home_idx]
            - team_s[..., g.away_idx]
            + g.home_qb_val
            - g.away_qb_val
        )

        mu_flat = mu.reshape(-1)
        sigma_flat = sigma_s.reshape(-1)

        rows.append({
            "season": g.season,
            "week": g.week,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "model_spread": float(np.mean(mu_flat)),
            "sigma": float(np.mean(sigma_flat)),
            "actual_margin": g.margin
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    preds = predict_historical()
    out = "../results/predictions.csv"
    preds.to_csv(out, index=False)
    print(f"Saved historical predictions to {out}")
import numpy as np
import pandas as pd
import nfl_data_py as nfl

from load_data import load_games, attach_qbs
from model_margin_decay_weighted import fit_margin_decay_model

# --------------------------------------------------
# LOAD BLEND WEIGHTS
# --------------------------------------------------
blend = pd.read_csv("../results/blend_weights_by_edge.csv")
alpha_map = dict(zip(blend["edge_bin"], blend["best_alpha"]))

def get_edge_bin(abs_edge):
    if abs_edge <= 1:
        return "small"
    elif abs_edge <= 3:
        return "medium"
    elif abs_edge <= 5:
        return "large"
    else:
        return "huge"

def predict_week(season, week, decay_rate=0.02):
    # --------------------------------------------------
    # LOAD HISTORICAL DATA (FOR TRAINING)
    # --------------------------------------------------
    hist_df, team_to_idx = load_games()
    hist_df = attach_qbs(hist_df)

    # --------------------------------------------------
    # TRAIN ON PAST GAMES ONLY
    # --------------------------------------------------
    train = hist_df[
        (hist_df["season"] < season) |
        ((hist_df["season"] == season) & (hist_df["week"] < week))
    ]

    trace = fit_margin_decay_model(train, len(team_to_idx), decay_rate)

    post = trace.posterior
    team_s = post["team_strength"].values
    hfa_s = post["hfa"].values
    sigma_s = post["sigma"].values  # (chains, draws)

    # --------------------------------------------------
    # LOAD OFFICIAL WEEK SCHEDULE
    # --------------------------------------------------
    sched = nfl.import_schedules([season])
    games = sched[
        (sched["game_type"] == "REG") &
        (sched["week"] == week)
    ][["season", "week", "home_team", "away_team"]].copy()

    games["home_idx"] = games["home_team"].map(team_to_idx)
    games["away_idx"] = games["away_team"].map(team_to_idx)

    # --------------------------------------------------
    # ATTACH QB VALUES
    # --------------------------------------------------
    games = attach_qbs(games)

    # --------------------------------------------------
    # LOAD CURRENT VEGAS LINES (ODDS API OUTPUT)
    # --------------------------------------------------
    vegas = pd.read_csv("../results/vegas_lines_2025.csv")

    games = games.merge(
        vegas,
        on=["season", "week", "home_team", "away_team"],
        how="left"
    )

    rows = []
    for _, g in games.iterrows():
        mu = (
            hfa_s
            + team_s[..., g.home_idx]
            - team_s[..., g.away_idx]
            + g.home_qb_val
            - g.away_qb_val
        )

        mu_flat = mu.reshape(-1)
        sigma_flat = sigma_s.reshape(-1)

        model_spread = float(np.mean(mu_flat))
        sigma_mean = float(np.mean(sigma_flat))

        has_vegas = not pd.isna(g.closing_spread_home)

        if has_vegas:
            abs_edge = abs(model_spread - g.closing_spread_home)
            edge_bin = get_edge_bin(abs_edge)
            alpha = alpha_map.get(edge_bin, 0.5)

            blended_spread = alpha * g.closing_spread_home + (1 - alpha) * model_spread
        else:
            edge_bin = "no_vegas"
            alpha = None
            blended_spread = None

        rows.append({
            "season": season,
            "week": week,
            "away_team": g.away_team,
            "home_team": g.home_team,
            "model_spread_home": round(model_spread, 2),
            "sigma": round(sigma_mean, 2),
            "vegas_spread_home": g.closing_spread_home,
            "blended_spread_home": round(blended_spread, 2) if has_vegas else None,
            "edge_bin": edge_bin,
            "alpha": alpha
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    preds = predict_week(2025, 16)
    out = "../results/week16_blended_lines.csv"
    preds.to_csv(out, index=False)
    print(preds)
    print(f"\nSaved blended predictions to {out}")
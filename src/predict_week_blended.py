import numpy as np
import pandas as pd
import nfl_data_py as nfl
from pathlib import Path

from load_data import load_games, attach_qbs
from model_margin_decay_season_boost import fit_margin_decay_model

# --------------------------------------------------
# PATH SETUP (ONLY NECESSARY CHANGE)
# --------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

# --------------------------------------------------
# TUNED MODEL HYPERPARAMETERS
# --------------------------------------------------
DECAY_RATE = 0.015
SEASON_BOOST = 2.5

# --------------------------------------------------
# LOAD BLEND WEIGHTS
# --------------------------------------------------
blend = pd.read_csv(RESULTS_DIR / "blend_weights_by_edge.csv")
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
    

def get_last_regular_season_week(season):
    sched = nfl.import_schedules([season])
    return (
        sched[sched["game_type"] == "REG"]["week"]
        .max()
    )

def predict_week(season, week, qb_file=None):
    # --------------------------------------------------
    # LOAD HISTORICAL DATA (FOR TRAINING)
    # --------------------------------------------------
    hist_df, team_to_idx = load_games()
    hist_df = attach_qbs(hist_df, qb_file=qb_file)

    # --------------------------------------------------
    # TRAIN ON PAST GAMES ONLY
    # --------------------------------------------------
    train = hist_df[
        (hist_df["season"] < season) |
        ((hist_df["season"] == season) & (hist_df["week"] < week))
    ]

    trace = fit_margin_decay_model(
        train,
        len(team_to_idx),
        decay_rate=DECAY_RATE,
        season_boost=SEASON_BOOST,
        prediction_season=season
    )

    print(
        f"Model fit: decay_rate={DECAY_RATE}, "
        f"season_boost={SEASON_BOOST}, "
        f"trained through {season} week {week-1}"
    )

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
    games = attach_qbs(games, qb_file=qb_file)

    # --------------------------------------------------
    # LOAD VEGAS LINES
    # --------------------------------------------------
    vegas = pd.read_csv(RESULTS_DIR / "vegas_lines_2025.csv")

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

# --------------------------------------------------
# CLI ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    SEASON = 2025
    WEEK = 17

    preds = predict_week(SEASON, WEEK)

    out = RESULTS_DIR / f"week{WEEK}_blended_lines.csv"
    preds.to_csv(out, index=False)

    print(preds)
    print(f"\nSaved blended predictions to {out}")
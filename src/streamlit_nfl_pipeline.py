import streamlit as st
import pandas as pd
import numpy as np
import math
import nfl_data_py as nfl
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
SEASON = 2025
WEEK = 16
TEMPERATURE = 2.35
ODDS_PRICE = -110
MIN_SIGMA = 1e-6

RESULTS_DIR = Path("results")

# ============================================================
# HELPERS
# ============================================================
def norm_cdf(x):
    x = np.asarray(x)
    return 0.5 * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))

def ev_from_prob(p, odds=-110):
    if odds < 0:
        win = 100 / abs(odds)
        loss = 1.0
    else:
        win = odds / 100
        loss = 1.0
    return p * win - (1 - p) * loss

def format_matchup_spread(away, home, mu_home):
    sign = "+" if mu_home > 0 else ""
    return f"{away} {sign}{mu_home:.2f} {home}"

# ============================================================
# LOAD MODEL OUTPUTS
# ============================================================
@st.cache_data(show_spinner=True)
def load_week_data():
    games = pd.read_csv(RESULTS_DIR / "week16_blended_lines.csv")
    qb_adj = pd.read_csv(RESULTS_DIR / "qb_adjustments.csv")

    qb_map = dict(zip(qb_adj["qb_name"], qb_adj["qb_value_shrunk"]))
    qb_list = sorted(qb_map.keys())

    rookie_baseline = qb_adj["qb_value_shrunk"].quantile(0.35)

    return games, qb_map, qb_list, rookie_baseline

# ============================================================
# DEFAULT QB = MOST PASSES LAST GAME
# ============================================================
@st.cache_data(show_spinner=True, ttl=0)
def load_last_game_qbs(season, week):
    try:
        pbp = nfl.import_pbp_data(years=[season], downcast=True)
    except Exception:
        return {}

    # Only games before the current week
    pbp = pbp[pbp["week"] < week]

    passes = pbp[
        (pbp["play_type"] == "pass") &
        (pbp["passer_player_name"].notna())
    ]

    # Find most recent week played by each team
    last_week_per_team = (
        passes.groupby("posteam")["week"].max().to_dict()
    )

    # Keep only passes from that week
    passes = passes[
        passes["week"] == passes["posteam"].map(last_week_per_team)
    ]

    # QB with most attempts in that game
    qb_counts = (
        passes.groupby(["posteam", "passer_player_name"])
        .size()
        .reset_index(name="attempts")
        .sort_values(["posteam", "attempts"])
    )

    return (
        qb_counts.groupby("posteam")
        .tail(1)
        .set_index("posteam")["passer_player_name"]
        .to_dict()
    )

# ============================================================
# TEAM BASELINES FROM MODEL OUTPUTS
# ============================================================
@st.cache_data(show_spinner=True)
def compute_team_baselines(games):
    home = games[["home_team", "model_spread_home"]].rename(
        columns={"home_team": "team", "model_spread_home": "spread"}
    )
    away = games[["away_team", "model_spread_home"]].rename(
        columns={"away_team": "team", "model_spread_home": "spread"}
    )
    away["spread"] = -away["spread"]

    all_games = pd.concat([home, away], ignore_index=True)
    return all_games.groupby("team")["spread"].mean().to_dict()

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="NFL Spread EV Tool", layout="wide")
st.title("NFL Spread EV Tool — Week 16")

games, QB_MAP, QB_LIST, ROOKIE_BASELINE = load_week_data()
last_qb = load_last_game_qbs(SEASON, WEEK)
TEAM_BASELINE = compute_team_baselines(games)

with st.sidebar:
    st.header("Model Settings")

    temperature = st.number_input("Temperature", value=float(TEMPERATURE), step=0.05)
    st.caption(
        "Controls confidence calibration. Higher values flatten probabilities toward 50/50, "
        "capturing uncertainty not explicitly modeled. Lower values make predictions sharper."
    )

    odds_price = st.number_input("Odds price", value=int(ODDS_PRICE), step=1)

tab1, tab2 = st.tabs(["Week Slate EV", "Matchup Sandbox"])

# ============================================================
# TAB 1: WEEK SLATE EV TOOL
# ============================================================
with tab1:
    rows = []

    for i, g in games.iterrows():
        away = g["away_team"]
        home = g["home_team"]

        default_home_qb = last_qb.get(home)
        default_away_qb = last_qb.get(away)

        st.markdown("---")

        # ---------- TOP ROW: INPUTS ----------
        r1c1, r1c2, r1c3 = st.columns([2.5, 2.5, 2])

        away_qb = r1c1.selectbox(
            f"{away} QB",
            QB_LIST,
            index=QB_LIST.index(default_away_qb) if default_away_qb in QB_LIST else 0,
            key=f"aqb_{i}"
        )

        home_qb = r1c2.selectbox(
            f"{home} QB",
            QB_LIST,
            index=QB_LIST.index(default_home_qb) if default_home_qb in QB_LIST else 0,
            key=f"hqb_{i}"
        )

        spread_away = r1c3.number_input(
            "Market Spread (Away)",
            value=float(-g["vegas_spread_home"]),
            step=0.5,
            key=f"spread_{i}"
        )

        spread_home = -spread_away

        # ---------- MODEL ----------
        qb_delta = (
            (QB_MAP.get(home_qb, ROOKIE_BASELINE)
             - QB_MAP.get(away_qb, ROOKIE_BASELINE))
            - (QB_MAP.get(default_home_qb, ROOKIE_BASELINE)
               - QB_MAP.get(default_away_qb, ROOKIE_BASELINE))
        )

        mu_home = float(g["model_spread_home"]) + qb_delta
        sigma = max(float(g["sigma"]), MIN_SIGMA)

        z_home = (mu_home + spread_home) / sigma
        p_home = float(norm_cdf(z_home / temperature))
        p_away = 1 - p_home

        if p_home >= p_away:
            bet = f"{home} {spread_home:+.1f}"
            prob = p_home
        else:
            bet = f"{away} {-spread_home:+.1f}"
            prob = p_away

        ev = ev_from_prob(prob, odds=odds_price)

        # ---------- OUTPUT ROW ----------
        r2c1, r2c2, r2c3 = st.columns([3, 2, 2])

        r2c1.markdown(
            f"**Predicted Spread:**  {format_matchup_spread(away, home, mu_home)}"
        )

        r2c2.metric("Cover Prob", f"{prob:.3f}")
        r2c3.metric("Best Bet EV", f"{ev:.3f}")

        rows.append({
            "matchup": f"{away} @ {home}",
            "bet": bet,
            "model_mu": format_matchup_spread(away, home, mu_home),
            "cover_prob": prob,
            "bet_ev": ev
        })

    out_df = pd.DataFrame(rows).sort_values("bet_ev", ascending=False)

    st.subheader("EV Ranking (Highest → Lowest)")
    st.dataframe(
        out_df.style.format({
            "cover_prob": "{:.3f}",
            "bet_ev": "{:.3f}"
        }),
        use_container_width=True
    )

# ============================================================
# TAB 2: MATCHUP SANDBOX
# ============================================================
with tab2:
    st.caption(
        "The Matchup Sandbox provides a simplified, hypothetical view of a game."
        "It estimates the spread using each team’s average model-implied strength and your selected quarterbacks."
        "It does not include week-specific context, opponent-specific interactions, or game-level adjustments."
        "Because of this, sandbox spreads may differ from the Week Slate EV view, which reflects the full model for the scheduled matchup."
        "Use the sandbox to explore “what-if” scenarios rather than exact weekly predictions."
    )
    teams = sorted(TEAM_BASELINE.keys())

    away = st.selectbox("Away team", teams)
    home = st.selectbox("Home team", teams, index=1)

    default_away_qb = last_qb.get(away)
    default_home_qb = last_qb.get(home)

    away_qb = st.selectbox(
        "Away QB",
        QB_LIST,
        index=QB_LIST.index(default_away_qb) if default_away_qb in QB_LIST else 0
    )

    home_qb = st.selectbox(
        "Home QB",
        QB_LIST,
        index=QB_LIST.index(default_home_qb) if default_home_qb in QB_LIST else 0
    )

    base_mu_home = TEAM_BASELINE.get(home, 0.0) - TEAM_BASELINE.get(away, 0.0)
    qb_delta = QB_MAP.get(home_qb, ROOKIE_BASELINE) - QB_MAP.get(away_qb, ROOKIE_BASELINE)
    mu_home = base_mu_home + qb_delta

    st.metric(
        "Predicted Spread",
        format_matchup_spread(away, home, mu_home),
        help="Displayed as: Away ± X Home. Positive = home favored."
    )

    market_spread_away = st.slider(
        "Market Spread (Away)",
        min_value=-21.0,
        max_value=21.0,
        value=0.0,
        step=0.5
    )

    market_spread = -market_spread_away

    sigma = max(games["sigma"].mean(), MIN_SIGMA)

    z_home = (mu_home + market_spread) / sigma
    p_home = float(norm_cdf(z_home / temperature))
    p_away = 1 - p_home

    ev_home = ev_from_prob(p_home, odds=odds_price)
    ev_away = ev_from_prob(p_away, odds=odds_price)

    st.metric("Home Cover Prob", f"{p_home:.3f}")
    st.metric("Away Cover Prob", f"{p_away:.3f}")

    if ev_home >= ev_away:
        st.metric("Best Bet", f"{home} {market_spread:+.1f}")
        st.metric("EV", f"{ev_home:.3f}")
    else:
        st.metric("Best Bet", f"{away} {-market_spread:+.1f}")
        st.metric("EV", f"{ev_away:.3f}")
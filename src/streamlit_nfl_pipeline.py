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
# DEFAULT QB = STARTER LAST WEEK
# ============================================================
@st.cache_data(show_spinner=True)
def load_last_week_qbs():
    df = pd.read_csv(RESULTS_DIR / "last_week_starting_qbs.csv")
    return dict(zip(df["team"], df["qb"]))

# ============================================================
# RECENCY-AWARE TEAM POWER (NEW)
# ============================================================
@st.cache_data(show_spinner=True)
def compute_team_power_with_recency(recency_lambda: float):
    """
    Recency-weighted team power using time-indexed latent strengths.
    """
    df = pd.read_csv(RESULTS_DIR / "team_power_raw.csv")

    df["team_strength"] = pd.to_numeric(df["team_strength"], errors="coerce")
    df["global_week"] = pd.to_numeric(df["global_week"], errors="coerce")

    df = df.dropna(subset=["team_strength", "global_week"])

    required = {"team", "global_week", "team_strength"}
    if not required.issubset(df.columns):
        raise ValueError(f"team_power_raw.csv must contain {required}")

    max_week = df["global_week"].max()

    age = max_week - df["global_week"]

    # IMPORTANT FIX: make weights an indexed Series
    weights = pd.Series(
        np.exp(-recency_lambda * age.values),
        index=df.index
    )

    df["weighted_strength"] = df["team_strength"] * weights

    team_power = (
        df.groupby("team")
          .apply(lambda x: x["weighted_strength"].sum() / weights.loc[x.index].sum())
          .to_dict()
    )

    return team_power

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="NFL Spread EV Tool", layout="wide")
st.title("NFL Spread EV Tool — Week 16")

games, QB_MAP, QB_LIST, ROOKIE_BASELINE = load_week_data()
last_qb = load_last_week_qbs()

with st.sidebar:
    st.header("Model Settings")

    temperature = st.number_input("Temperature", value=float(TEMPERATURE), step=0.05)

    odds_price = st.number_input("Odds price", value=int(ODDS_PRICE), step=1)

tab1, tab2, tab3 = st.tabs(
    ["Week Slate EV", "Matchup Sandbox", "Power Rankings"]
)

# Default team power (used by Tab 1 & 2)
DEFAULT_RECENCY = 0.05
TEAM_BASELINE = compute_team_power_with_recency(DEFAULT_RECENCY)

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

        r2c1, r2c2, r2c3 = st.columns([3, 2, 2])

        r2c1.markdown(
            f"""
            **Predicted Spread:** {format_matchup_spread(away, home, mu_home)}  
            **Best Bet:** {bet}
            """
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
    teams = sorted(TEAM_BASELINE.keys())

    away = st.selectbox("Away team", teams)
    home = st.selectbox("Home team", teams, index=1)

    away_qb = st.selectbox("Away QB", QB_LIST)
    home_qb = st.selectbox("Home QB", QB_LIST)

    base_mu_home = TEAM_BASELINE.get(home, 0.0) - TEAM_BASELINE.get(away, 0.0)
    qb_delta = QB_MAP.get(home_qb, ROOKIE_BASELINE) - QB_MAP.get(away_qb, ROOKIE_BASELINE)
    mu_home = base_mu_home + qb_delta

    st.metric("Predicted Spread", format_matchup_spread(away, home, mu_home))

# ============================================================
# TAB 3: POWER RANKINGS
# ============================================================
with tab3:
    st.subheader("Quarterback Power Rankings")

    qb_rankings = (
        pd.DataFrame({
            "QB": list(QB_MAP.keys()),
            "QB Value (pts)": list(QB_MAP.values())
        })
        .sort_values("QB Value (pts)", ascending=False)
        .reset_index(drop=True)
    )

    qb_rankings.insert(0, "Rank", qb_rankings.index + 1)

    st.dataframe(
        qb_rankings.style.format({"QB Value (pts)": "{:+.2f}"}),
        use_container_width=True,
        hide_index=True
    )
    st.subheader("Power Rankings")

    recency_lambda = st.slider(
        "Team Power Recency Weight",
        min_value=0.0,
        max_value=0.15,
        value=0.05,
        step=0.01,
        help="Controls how much recent seasons matter vs long-run strength"
    )

    TEAM_BASELINE = compute_team_power_with_recency(recency_lambda)
    
    team_rankings = (
        pd.DataFrame({
            "Team": list(TEAM_BASELINE.keys()),
            "Power": list(TEAM_BASELINE.values())
        })
        .sort_values("Power", ascending=False)
        .reset_index(drop=True)
    )

    team_rankings.insert(0, "Rank", team_rankings.index + 1)

    st.dataframe(
        team_rankings.style.format({"Power": "{:+.2f}"}),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Team power is derived from Bayesian latent strength (last ~4 seasons) "
        "with user-controlled recency weighting. Sandbox uses team power + QB deltas. "
        "Week Slate EV uses full matchup-specific modeling."
    )
import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================
SEASON = 2025
WEEK = 16
TEMPERATURE = 1.7
ODDS_PRICE = -110
MIN_SIGMA = 1e-6

RESULTS_DIR = Path("results")

# ============================================================
# HELPERS
# ============================================================
def sigma_adjusted(base_sigma, spread_home):
    """
    Inflate variance slightly for large spreads to reflect
    higher blowout uncertainty.
    """
    return base_sigma * (1 + 0.08 * abs(spread_home))


def norm_cdf(x):
    x = np.asarray(x)
    return 0.5 * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))


def student_t_cdf(x, df):
    x = np.asarray(x, dtype=float)
    adj = x * np.sqrt((df - 2) / df)
    return norm_cdf(adj)


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


def american_to_b(odds):
    if odds < 0:
        return 100 / abs(odds)
    return odds / 100


def kelly_fraction(p, odds=-110):
    b = american_to_b(odds)
    q = 1 - p
    f = (b * p - q) / b
    return max(f, 0.0)


def color_results(val):
    if "Win" in val:
        return "background-color: #d4f8d4; color: black"
    if "Loss" in val:
        return "background-color: #f8d4d4; color: black"
    if "Push" in val:
        return "background-color: #e6e6e6; color: black"
    return ""


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


HIST_DIR = Path("historical/outputs")


@st.cache_data(show_spinner=True)
def load_historical_week(season: int, week: int):
    path = HIST_DIR / f"season_{season}_week_{week}_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=True)
def load_actual_results(season: int):
    scores = pd.read_csv(RESULTS_DIR / "final_walkforward_predictions.csv")[
        ["season", "week", "home_team", "away_team", "actual_margin"]
    ]
    return scores


# ============================================================
# DEFAULT QB = STARTER LAST WEEK
# ============================================================
@st.cache_data(show_spinner=True)
def load_last_week_qbs():
    df = pd.read_csv(RESULTS_DIR / "last_week_starting_qbs.csv")
    return dict(zip(df["team"], df["qb"]))


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="NFL Spread EV Tool", layout="wide")
st.title("NFL Spread EV Tool — Week 16")

games, QB_MAP, QB_LIST, ROOKIE_BASELINE = load_week_data()
last_qb = load_last_week_qbs()

with st.sidebar:
    st.header("Model Settings")

    temperature = st.number_input(
        "Temperature",
        value=float(TEMPERATURE),
        step=0.05,
    )

    odds_price = st.number_input(
        "Odds price",
        value=int(ODDS_PRICE),
        step=1,
    )

    st.divider()
    st.header("Bet Sizing")

    bankroll = st.number_input(
        "Bankroll ($)",
        min_value=0.0,
        value=100.0,
        step=10.0,
    )

    fractional_kelly = st.slider(
        "Fractional Kelly",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.05,
    )

    kelly_cap = st.slider(
        "Max % of bankroll per bet",
        min_value=0.5,
        max_value=100.0,
        value=2.0,
        step=0.5,
    ) / 100.0


tab1, tab2, tab3 = st.tabs(
    ["Week Slate EV", "QB Power Rankings", "Historical Predictions"]
)

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
            key=f"aqb_{i}",
        )

        home_qb = r1c2.selectbox(
            f"{home} QB",
            QB_LIST,
            index=QB_LIST.index(default_home_qb) if default_home_qb in QB_LIST else 0,
            key=f"hqb_{i}",
        )

        spread_away = r1c3.number_input(
            "Market Spread (Away)",
            value=float(-g["vegas_spread_home"]),
            step=0.5,
            key=f"spread_{i}",
        )

        spread_home = -spread_away

        qb_delta = (
            (QB_MAP.get(home_qb, ROOKIE_BASELINE)
             - QB_MAP.get(away_qb, ROOKIE_BASELINE))
            - (QB_MAP.get(default_home_qb, ROOKIE_BASELINE)
               - QB_MAP.get(default_away_qb, ROOKIE_BASELINE))
        )

        mu_home = float(g["model_spread_home"]) + qb_delta

        base_sigma = max(float(g["sigma"]), MIN_SIGMA)
        sigma = sigma_adjusted(base_sigma, spread_home)

        skew_adj = 0.15 * np.sign(mu_home)
        z_home = (mu_home + spread_home + skew_adj) / sigma

        p_home = float(student_t_cdf(z_home / temperature, df=6))
        p_away = 1 - p_home

        if p_home >= p_away:
            bet = f"{home} {spread_home:+.1f}"
            prob = p_home
        else:
            bet = f"{away} {-spread_home:+.1f}"
            prob = p_away

        ev = ev_from_prob(prob, odds=odds_price)

        kelly_raw = kelly_fraction(prob, odds_price)
        kelly_frac = min(fractional_kelly * kelly_raw, kelly_cap)
        stake_dollars = bankroll * kelly_frac

        r2c1, r2c2, r2c3 = st.columns([3, 2, 2])

        r2c1.markdown(
            f"""
            **Predicted Spread:** {format_matchup_spread(away, home, mu_home)}  
            **Best Bet:** {bet}
            """
        )

        r2c2.metric("Cover Prob", f"{prob:.3f}")
        r2c3.metric("EV", f"{ev:.3f}")

        if kelly_frac > 0 and ev > 0:
            st.caption(
                f"**Kelly stake:** {kelly_frac*100:.2f}% "
                f"(${stake_dollars:,.2f})"
            )
        else:
            st.caption("**Kelly stake:** $0.00 (no positive edge)")

        rows.append({
            "matchup": f"{away} @ {home}",
            "bet": bet,
            "model_mu": format_matchup_spread(away, home, mu_home),
            "cover_prob": prob,
            "bet_ev": ev,
            "kelly_pct": kelly_frac * 100,
            "stake_$": stake_dollars,
        })

    out_df = pd.DataFrame(rows).sort_values("bet_ev", ascending=False)

    st.subheader("EV Ranking (Highest → Lowest)")
    st.dataframe(
        out_df.style.format({
            "cover_prob": "{:.3f}",
            "bet_ev": "{:.3f}",
            "kelly_pct": "{:.2f}%",
            "stake_$": "${:,.2f}",
        }),
        use_container_width=True,
    )


# ============================================================
# TAB 2: QB POWER RANKINGS
# ============================================================
with tab2:
    st.subheader("Quarterback Power Rankings")

    qb_rankings = (
        pd.DataFrame({
            "QB": list(QB_MAP.keys()),
            "QB Value (pts)": list(QB_MAP.values()),
        })
        .sort_values("QB Value (pts)", ascending=False)
        .reset_index(drop=True)
    )

    qb_rankings.insert(0, "Rank", qb_rankings.index + 1)

    st.dataframe(
        qb_rankings.style.format({"QB Value (pts)": "{:+.2f}"}),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TAB 3: HISTORICAL PREDICTIONS
# ============================================================
with tab3:
    st.subheader("Historical Predictions (Model Bets)")

    week = st.selectbox(
        "Select Week",
        options=list(range(1, 16)),
        index=0,
    )

    hist = load_historical_week(SEASON, week)
    actuals = load_actual_results(SEASON)

    if hist is None:
        st.warning("No data found for this week.")
        st.stop()

    hist = hist.merge(
        actuals,
        on=["season", "week", "home_team", "away_team"],
        how="left"
    )

    rows = []

    for _, g in hist.iterrows():
        away = g["away_team"]
        home = g["home_team"]

        mu_home = float(g["model_spread_home"])
        spread_home = float(g["vegas_spread_home"])  # trust file as-is
        sigma = max(float(g["sigma"]), MIN_SIGMA)

        margin_away = g["actual_margin"]  # away - home
        if pd.isna(margin_away):
            continue

        margin_away = float(margin_away)
        margin_home = -margin_away

        # ---- probabilities (identical to Tab 1 math) ----
        skew_adj = 0.15 * np.sign(mu_home)
        z_home = (mu_home + spread_home + skew_adj) / sigma
        prob_home = float(student_t_cdf(z_home / temperature, df=6))
        prob_away = 1 - prob_home

        # ---- EXACT SAME BET CONSTRUCTION AS TAB 1 ----
        if prob_home >= prob_away:
            bet_team = home
            bet_line = spread_home
            prob = prob_home
            margin_for_bet = margin_home
        else:
            bet_team = away
            bet_line = -spread_home
            prob = prob_away
            margin_for_bet = margin_away

        bet = f"{bet_team} {bet_line:+.1f}"
        ev = ev_from_prob(prob, odds_price)

        # ---- single grading rule ----
        cover_val = margin_for_bet + bet_line

        if abs(cover_val) < 1e-9:
            result = "➖ Push"
        elif cover_val > 0:
            result = "✅ Win"
        else:
            result = "❌ Loss"

        # ---- display actual margin ----
        if abs(margin_away - round(margin_away)) < 1e-9:
            actual_margin_display = f"{away} {int(round(margin_away)):+d}"
        else:
            actual_margin_display = f"{away} {margin_away:+.1f}"

        rows.append({
            "matchup": f"{away} @ {home}",
            "bet": bet,
            "model_mu": format_matchup_spread(away, home, mu_home),
            "cover_prob": prob,
            "bet_ev": ev,
            "actual_margin": actual_margin_display,
            "result": result,
        })

    table = pd.DataFrame(rows).sort_values("bet_ev", ascending=False)

    st.dataframe(
        table.style
        .format({
            "cover_prob": "{:.3f}",
            "bet_ev": "{:.3f}",
        })
        .applymap(color_results, subset=["result"]),
        use_container_width=True,
    )
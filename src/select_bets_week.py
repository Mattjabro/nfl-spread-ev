import pandas as pd
import numpy as np
from math import erf, sqrt

# ----------------------------------------
# PARAMETERS
# ----------------------------------------
TEMPERATURE = 1.75          # keep your calibration idea
ASSUMED_AMERICAN_ODDS = -110
MIN_EV = 0.02               # minimum expected value per $1 staked (2 cents)
MAX_BETS = 5                # cap volume like a pro

# ----------------------------------------
# HELPERS
# ----------------------------------------
def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def american_to_profit_per_1(odds):
    # Profit (not return) for $1 stake if the bet wins
    # -110 => profit = 100/110 ≈ 0.9091
    if odds < 0:
        return 100.0 / abs(odds)
    else:
        return odds / 100.0

def breakeven_prob(odds):
    # probability needed for EV = 0
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    else:
        return 100.0 / (odds + 100.0)

def cover_prob_home(mu, sigma, spread_home):
    # Home covers if margin + spread_home > 0
    # p = Φ((mu + spread_home) / (sigma * T))
    z = (mu + spread_home) / max(1e-6, sigma)
    return norm_cdf(z / TEMPERATURE)

def ev_per_1(p_win, odds):
    win_profit = american_to_profit_per_1(odds)
    return p_win * win_profit - (1 - p_win) * 1.0

# ----------------------------------------
# LOAD WEEKLY DATA
# ----------------------------------------
df = pd.read_csv("../results/week16_blended_lines.csv")

# Required columns produced by your pipeline
required = {
    "away_team",
    "home_team",
    "vegas_spread_home",
    "blended_spread_home"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"week16_blended_lines.csv missing columns: {missing}")

# IMPORTANT:
# Your week file currently does NOT include sigma.
# So we use a conservative fixed sigma. This is still better than raw edge.
# If you want the pro version, see section 3 where we add sigma to predict_week_blended.py.
DEFAULT_SIGMA = 13.5

df["sigma"] = DEFAULT_SIGMA

# ----------------------------------------
# COMPUTE COVER PROBS + EV FOR BOTH SIDES
# ----------------------------------------
odds = ASSUMED_AMERICAN_ODDS
p_be = breakeven_prob(odds)

rows = []
for _, r in df.iterrows():
    mu = float(r["blended_spread_home"])
    sigma = float(r["sigma"])
    spread_home = float(r["vegas_spread_home"])

    p_home = cover_prob_home(mu, sigma, spread_home)
    p_away = 1 - p_home

    ev_home = ev_per_1(p_home, odds)
    ev_away = ev_per_1(p_away, odds)

    if ev_home >= ev_away:
        side = "HOME"
        bet_team = r["home_team"]
        p_win = p_home
        ev = ev_home
        bet_side = f"{bet_team} {spread_home}" if spread_home < 0 else f"{bet_team} +{abs(spread_home)}"
    else:
        side = "AWAY"
        bet_team = r["away_team"]
        p_win = p_away
        ev = ev_away
        # Away spread is the negative of home spread
        away_spread = -spread_home
        bet_side = f"{bet_team} {away_spread}" if away_spread < 0 else f"{bet_team} +{abs(away_spread)}"

    rows.append({
        "away_team": r["away_team"],
        "home_team": r["home_team"],
        "vegas_spread_home": spread_home,
        "model_spread_home": mu,
        "sigma": sigma,
        "p_home_cover": p_home,
        "p_away_cover": p_away,
        "chosen_side": side,
        "p_win": p_win,
        "breakeven_p": p_be,
        "ev_per_1": ev,
        "bet_side": bet_side
    })

bets = pd.DataFrame(rows)

# ----------------------------------------
# FILTER: ONLY POSITIVE-EV BETS, CAP VOLUME
# ----------------------------------------
bets = bets[bets["ev_per_1"] >= MIN_EV].copy()
bets = bets.sort_values("ev_per_1", ascending=False).head(MAX_BETS)

# ----------------------------------------
# SAVE
# ----------------------------------------
out = "../results/week16_bets.csv"
bets.to_csv(out, index=False)

print("\n================ WEEK 16 EV BETS =================")
print(bets[[
    "away_team","home_team","vegas_spread_home",
    "model_spread_home","sigma","p_win","breakeven_p",
    "ev_per_1","bet_side"
]])
print(f"\nSaved bets to {out}")
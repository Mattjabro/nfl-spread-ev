import pandas as pd
import numpy as np
import math

# ----------------------------------------
# PARAMETERS
# ----------------------------------------
TEMPERATURE = 1.75
ODDS_PRICE = -110
MIN_SIGMA = 1e-6

# ----------------------------------------
# HELPERS (ROBUST)
# ----------------------------------------
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

# ----------------------------------------
# LOAD WEEKLY DATA
# ----------------------------------------
df = pd.read_csv("../results/week16_blended_lines.csv")

required = {
    "home_team",
    "away_team",
    "model_spread_home",
    "sigma",
    "vegas_spread_home"
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ----------------------------------------
# COMPUTE COVER PROBABILITIES
# ----------------------------------------
mu = df["model_spread_home"].values
sigma = np.maximum(df["sigma"].values, MIN_SIGMA)
spread_home = df["vegas_spread_home"].values

z_home = (mu + spread_home) / sigma
p_home_cover = norm_cdf(z_home / TEMPERATURE)
p_away_cover = 1 - p_home_cover

# ----------------------------------------
# DETERMINE BEST SIDE + EV (SIGN-CORRECT)
# ----------------------------------------
rows = []

for i, row in df.iterrows():
    spread = row["vegas_spread_home"]

    if p_home_cover[i] >= p_away_cover[i]:
        # Bet HOME team → spread used as-is
        team = row["home_team"]
        bet_spread = spread
        prob = p_home_cover[i]
    else:
        # Bet AWAY team → flip sign
        team = row["away_team"]
        bet_spread = -spread
        prob = p_away_cover[i]

    # Proper sportsbook formatting
    if bet_spread > 0:
        bet_side = f"{team} +{abs(bet_spread):.1f}"
    else:
        bet_side = f"{team} {bet_spread:.1f}"

    ev = ev_from_prob(prob, ODDS_PRICE)

    rows.append({
        "away_team": row["away_team"],
        "home_team": row["home_team"],
        "vegas_spread_home": spread,
        "model_spread_home": row["model_spread_home"],
        "sigma": row["sigma"],
        "bet_side": bet_side,
        "cover_prob": round(prob, 3),
        "bet_ev": round(ev, 3)
    })

out_df = pd.DataFrame(rows)

# ----------------------------------------
# SORT BY EV
# ----------------------------------------
out_df = out_df.sort_values("bet_ev", ascending=False).reset_index(drop=True)

# ----------------------------------------
# SAVE
# ----------------------------------------
out = "../results/week16_all_ev_ranked.csv"
out_df.to_csv(out, index=False)

print("\n================ WEEK 16 EV RANKING =================")
print(out_df)
print(f"\nSaved EV-ranked games to {out}")
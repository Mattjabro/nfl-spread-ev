import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--stress", action="store_true")
parser.add_argument("--edge", type=float, default=1.0)
parser.add_argument("--shift", type=float, default=0.0)
args = parser.parse_args()

MIN_EDGE_POINTS = args.edge
EXECUTION_SHIFT = args.shift
import pandas as pd
from pathlib import Path
from math import erf, sqrt

STRESS_SCENARIOS = [
    ("baseline", 0.00),
    ("minus_0.25", -0.25),
    ("minus_0.50", -0.50),
    ("minus_1.00", -1.00),
]

# ============================================================
# CONFIG
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Inputs
FULL_PATH = RESULTS / "full_report_inputs.csv"      # produced by your full_report_walkforward.py
BETLINE_PATH = RESULTS / "vegas_bet_time_lines.csv" # you will create/fill this (open or time-of-bet)
OUT_MERGED = RESULTS / "execution_eval_merged.csv"
OUT_BETS = RESULTS / "execution_bets.csv"

# Model calibration (use what you tuned)
TEMPERATURE = 1.70

# Execution assumptions
ODDS_AMERICAN = -110
FRACTIONAL_KELLY = 0.25     # 25% Kelly
KELLY_CAP = 0.02            # 2% bankroll max per bet (keeps it sane)   
MIN_EV_PER_UNIT = 0.00      # require positive EV vs price (can raise later)
MAX_BETS_PER_WEEK = 6       # realism constraint
START_BANKROLL = 100.0

# ============================================================
# Helpers
# ============================================================
def norm_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))

def american_to_b(odds):
    # profit per 1 staked if win
    if odds < 0:
        return 100.0 / abs(odds)
    return odds / 100.0

def kelly_fraction(p, odds=-110):
    b = american_to_b(odds)
    q = 1.0 - p
    return (b * p - q) / b

def log_loss(y, p, eps=1e-12):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

def brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))

def ece(y, p, bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges, right=True)
    out = 0.0
    n = len(y)
    for b in range(1, bins + 1):
        m = (idx == b)
        if not np.any(m):
            continue
        out += (np.sum(m) / n) * abs(np.mean(p[m]) - np.mean(y[m]))
    return float(out)

def max_drawdown(curve):
    curve = np.asarray(curve, dtype=float)
    peak = np.maximum.accumulate(curve)
    dd = (curve / peak) - 1.0
    return float(dd.min())

# ============================================================
# Load full dataset
# ============================================================
if not FULL_PATH.exists():
    raise FileNotFoundError(f"Missing {FULL_PATH}. Run: python full_report_walkforward.py")

df = pd.read_csv(FULL_PATH)

# Required columns from your pipeline
need = ["season", "week", "home_team", "away_team", "model_spread", "sigma", "vegas_spread_home", "actual_margin"]
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"{FULL_PATH} is missing columns: {missing}")

# Numeric cleanup
for c in ["model_spread", "sigma", "vegas_spread_home", "actual_margin"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=need).copy()

# ATS outcome vs the market spread used for evaluation
# home covers if (margin + spread_home) > 0
df["home_cover_close"] = ((df["actual_margin"] + df["vegas_spread_home"]) > 0).astype(int)

# ============================================================
# Ensure bet-time line file exists (open/time-of-bet)
# ============================================================
if not BETLINE_PATH.exists():
    template = df[["season", "week", "home_team", "away_team", "vegas_spread_home"]].copy()
    template = template.rename(columns={"vegas_spread_home": "closing_spread_home"})
    template["bet_spread_home"] = np.nan  # fill this with open line or time-of-bet line
    template["bet_time_note"] = ""        # optional: "open", "Wed", "Sat", etc.
    template.to_csv(BETLINE_PATH, index=False)
    print("\nCreated template:", BETLINE_PATH)
    print("Fill bet_spread_home with the line you could actually bet (open or time-of-bet), then rerun.")
    raise SystemExit(0)

betlines = pd.read_csv(BETLINE_PATH)
need_bet = ["season", "week", "home_team", "away_team", "bet_spread_home"]
missing_bet = [c for c in need_bet if c not in betlines.columns]
if missing_bet:
    raise ValueError(f"{BETLINE_PATH} missing columns: {missing_bet}")

betlines["bet_spread_home"] = pd.to_numeric(betlines["bet_spread_home"], errors="coerce")

# Merge
m = df.merge(
    betlines[["season", "week", "home_team", "away_team", "bet_spread_home"]],
    on=["season", "week", "home_team", "away_team"],
    how="left"
)

# If bet lines are missing, you can still compute CLV quality later, but execution won’t run.
usable = m.dropna(subset=["bet_spread_home"]).copy()
usable["bet_spread_home"] = usable["bet_spread_home"] + EXECUTION_SHIFT
if len(usable) == 0:
    print("\nNo bet_spread_home values found in", BETLINE_PATH)
    print("Fill bet_spread_home (open/time-of-bet lines) and rerun.")
    raise SystemExit(0)

# ============================================================
# Compute executable ATS probabilities vs BET-TIME line
# ============================================================
# P(home covers bet line) = P(margin + bet_spread_home > 0) = P(margin > -bet_spread_home)
# margin ~ Normal(mu=model_spread, sigma)
# z = (mu + bet_spread_home) / sigma
usable["z_bet_cover"] = (usable["model_spread"] + usable["bet_spread_home"]) / usable["sigma"]
usable["p_home_cover_bet"] = norm_cdf(usable["z_bet_cover"] / TEMPERATURE)

# True label for bet-time ATS
usable["home_cover_bet"] = ((usable["actual_margin"] + usable["bet_spread_home"]) > 0).astype(int)

# Edge in points against bet-time line (this is what you actually had)
usable["edge_pts_bet"] = usable["model_spread"] - usable["bet_spread_home"]
usable["abs_edge_pts_bet"] = usable["edge_pts_bet"].abs()

# CLV: closing - bet line (positive CLV means you beat the market)
usable["clv_pts"] = usable["vegas_spread_home"] - usable["bet_spread_home"]

# Lock side: choose HOME if p>=0.5 else AWAY (no switching later)
usable["side"] = np.where(usable["p_home_cover_bet"] >= 0.5, "HOME", "AWAY")
usable["p_best"] = np.where(usable["side"] == "HOME", usable["p_home_cover_bet"], 1.0 - usable["p_home_cover_bet"])

# Price EV at -110
b = american_to_b(ODDS_AMERICAN)
usable["ev_per_unit"] = usable["p_best"] * b - (1.0 - usable["p_best"]) * 1.0

# Filter bettable
bets = usable[
    (usable["abs_edge_pts_bet"] >= MIN_EDGE_POINTS) &
    (usable["ev_per_unit"] > MIN_EV_PER_UNIT)
].copy()

# Limit action per week (realism)
bets = bets.sort_values(["season", "week", "ev_per_unit"], ascending=[True, True, False])
bets = bets.groupby(["season", "week"]).head(MAX_BETS_PER_WEEK).reset_index(drop=True)

# Outcomes for chosen side
bets["win"] = np.where(bets["side"] == "HOME", bets["home_cover_bet"], 1 - bets["home_cover_bet"]).astype(int)

# Kelly sizing (fractional + cap)
bets["kelly_raw"] = bets["p_best"].apply(lambda p: kelly_fraction(p, ODDS_AMERICAN))
bets["stake_frac"] = np.clip(
    FRACTIONAL_KELLY * np.tanh(2.0 * bets["kelly_raw"]),
    0.0,
    KELLY_CAP
)

# Bankroll sim
bankroll = START_BANKROLL
curve = []
for r in bets.itertuples(index=False):
    stake = bankroll * float(r.stake_frac)
    if stake <= 0:
        curve.append(bankroll)
        continue
    if int(r.win) == 1:
        bankroll += stake * b
    else:
        bankroll -= stake
    curve.append(bankroll)

bets["bankroll"] = curve

# ============================================================
# Reporting
# ============================================================
print("\n" + "=" * 70)
print("EXECUTION REPORT (BET-TIME LINES + CLV)")
print("=" * 70)
print(f"Games with bet-time lines: {len(usable)} / {len(df)}")
print(f"Bets placed: {len(bets)} | edge>= {MIN_EDGE_POINTS:.1f} | max/week={MAX_BETS_PER_WEEK}")
print(f"Kelly: frac={FRACTIONAL_KELLY:.2f} cap={KELLY_CAP:.3f} | odds={ODDS_AMERICAN}")
print(f"Temperature: {TEMPERATURE:.3f}")

# Probability quality (bet-time ATS)
ll = log_loss(usable["home_cover_bet"], usable["p_home_cover_bet"])
br = brier(usable["home_cover_bet"], usable["p_home_cover_bet"])
ec = ece(usable["home_cover_bet"], usable["p_home_cover_bet"], bins=10)
print("\n=== Bet-time ATS probability quality (ALL games with bet lines) ===")
print(f"LogLoss: {ll:.4f} | Brier: {br:.4f} | ECE: {ec:.4f}")

# CLV
print("\n=== CLV (closing - bet line) ===")
print(f"Mean CLV (pts): {bets['clv_pts'].mean():+.3f}  | Median: {bets['clv_pts'].median():+.3f}")
print(f"% Positive CLV: {(bets['clv_pts'] > 0).mean()*100:.1f}%")

# Betting results
if len(bets) > 0:
    end_bank = float(bets["bankroll"].iloc[-1])
    dd = max_drawdown(bets["bankroll"].values)
    print("\n=== Execution results (Kelly) ===")
    print(f"Start bankroll: {START_BANKROLL:.2f}")
    print(f"End bankroll  : {end_bank:.2f}")
    print(f"Net return    : {(end_bank / START_BANKROLL - 1)*100:.2f}%")
    print(f"Win rate      : {bets['win'].mean():.3f}")
    print(f"Avg stake %   : {bets['stake_frac'].mean()*100:.2f}%")
    print(f"Median stake %: {bets['stake_frac'].median()*100:.2f}%")
    print(f"Max drawdown  : {dd*100:.2f}%")

# Save outputs
m.to_csv(OUT_MERGED, index=False)
bets.to_csv(OUT_BETS, index=False)

print("\nSaved merged dataset:", OUT_MERGED)
print("Saved bets:", OUT_BETS)
print("Bet line source:", BETLINE_PATH)
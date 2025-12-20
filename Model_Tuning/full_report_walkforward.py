import sys
from pathlib import Path
import numpy as np
import pandas as pd
from math import erf, sqrt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.append(str(ROOT / "src"))

PRED_PATH = RESULTS / "final_walkforward_predictions.csv"
VEGAS_PATH = RESULTS / "vegas_closing_lines.csv"

TEMPERATURE = 1.70
ODDS_AMERICAN = -110

EDGE_THRESHOLDS = [0.0, 1.0, 2.0, 3.0, 5.0]

def norm_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))

def american_to_b(odds):
    return 100.0 / abs(odds) if odds < 0 else odds / 100.0

def logloss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(np.mean(-(y*np.log(p) + (1-y)*np.log(1-p))))

def brier(y, p):
    return float(np.mean((p - y) ** 2))

def calibration_table(y, p, bins=10):
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    out = (df.groupby("bin")
             .agg(count=("y", "size"), mean_pred=("p", "mean"), win_rate=("y", "mean"))
             .reset_index()
             .rename(columns={"bin": "prob_bin"}))
    return out

def ece(y, p, bins=10):
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    g = df.groupby("bin").agg(n=("y","size"), mp=("p","mean"), wy=("y","mean")).dropna()
    w = g["n"] / g["n"].sum()
    return float(np.sum(w * np.abs(g["mp"] - g["wy"])))

def max_drawdown(curve):
    peak = curve[0]
    mdd = 0.0
    for x in curve:
        peak = max(peak, x)
        mdd = min(mdd, (x/peak) - 1.0)
    return float(mdd)

# ---------------------------
# Load + merge
# ---------------------------
pred = pd.read_csv(PRED_PATH)
for col in ["model_spread", "sigma", "actual_margin", "home_win"]:
    pred[col] = pd.to_numeric(pred[col], errors="coerce")
pred = pred.dropna(subset=["season","week","home_team","away_team","model_spread","sigma","actual_margin"])

vegas = pd.read_csv(VEGAS_PATH).rename(columns={"closing_spread_home":"vegas_spread_home"})
df = pred.merge(
    vegas[["season","week","home_team","away_team","vegas_spread_home"]],
    on=["season","week","home_team","away_team"],
    how="inner"
).dropna(subset=["vegas_spread_home"])

# ---------------------------
# Core quantities
# ---------------------------
df["spread_err"] = df["model_spread"] - df["actual_margin"]

# ATS: home covers if actual_margin + vegas_spread_home > 0
df["home_cover"] = ((df["actual_margin"] + df["vegas_spread_home"]) > 0).astype(int)

# Model cover prob at closing (Normal margin w/ temperature)
df["z_cover"] = (df["model_spread"] + df["vegas_spread_home"]) / df["sigma"]
df["p_home_cover"] = norm_cdf(df["z_cover"] / TEMPERATURE)

# Edge in points vs market (your “CLV proxy”)
df["edge_pts"] = df["model_spread"] - df["vegas_spread_home"]
df["abs_edge_pts"] = df["edge_pts"].abs()

# Flat-stake betting rule: take the side with higher cover prob if edge >= threshold
b = american_to_b(ODDS_AMERICAN)
df["side"] = np.where(df["p_home_cover"] >= 0.5, "HOME", "AWAY")
df["p_best"] = np.where(df["side"] == "HOME", df["p_home_cover"], 1.0 - df["p_home_cover"])
df["win"] = np.where(df["side"] == "HOME", df["home_cover"], 1 - df["home_cover"]).astype(int)

def season_report(d, title):
    y = d["home_cover"].values.astype(int)
    p = d["p_home_cover"].values.astype(float)

    ll = logloss(y, p)
    br = brier(y, p)
    _ece = ece(y, p)

    mae = float(np.mean(np.abs(d["spread_err"].values)))
    rmse = float(np.sqrt(np.mean(d["spread_err"].values**2)))

    print("\n" + "="*70)
    print(title)
    print("="*70)
    print(f"Games: {len(d)}")
    print(f"ATS LogLoss: {ll:.4f} | Brier: {br:.4f} | ECE: {_ece:.4f}")
    print(f"Spread MAE: {mae:.3f} | RMSE: {rmse:.3f}")

    print("\nCalibration (home cover):")
    print(calibration_table(y, p, bins=10))

    # Edge thresholds with flat 1u staking
    print("\nFlat-stake ATS by edge threshold (units assume -110):")
    for thr in EDGE_THRESHOLDS:
        bb = d[d["abs_edge_pts"] >= thr].copy()
        if len(bb) == 0:
            continue
        # 1 unit stake per bet
        profit = np.where(bb["win"].values==1, b, -1.0).sum()
        roi = profit / len(bb)
        winr = bb["win"].mean()
        avg_edge = bb["edge_pts"].mean()
        print(f"  abs_edge >= {thr:>3.1f} | bets={len(bb):4d} | win%={winr:.3f} | ROI/bet={roi:.4f} | avg_edge={avg_edge:+.3f}")

    # Simple bankroll curve for flat stakes to compute drawdown
    bb = d[d["abs_edge_pts"] >= 1.0].sort_values(["season","week"]).copy()
    if len(bb) > 0:
        bank = 100.0
        curve = []
        rets = []
        for w in bb["win"].values.astype(int):
            r = b if w==1 else -1.0
            bank += r
            curve.append(bank)
            rets.append(r)
        print(f"\nFlat-stake (abs_edge>=1) end bank: {curve[-1]:.2f} | max DD: {max_drawdown(curve)*100:.2f}% | return vol: {np.std(rets):.4f}")

# Overall + by season
season_report(df, "OVERALL REPORT (closing ATS + spread error)")
for s in sorted(df["season"].unique()):
    season_report(df[df["season"]==s], f"SEASON {int(s)}")

out_path = RESULTS / "full_report_inputs.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved merged evaluation dataset to: {out_path}")
import pandas as pd
from pathlib import Path

# ============================================================
# PATHS (CORRECTED)
# ============================================================
ROOT = Path(__file__).resolve().parent.parent
HIST_DIR = ROOT / "historical" / "outputs"
RESULTS_DIR = ROOT / "results"

print("ROOT:", ROOT)
print("HIST_DIR exists:", HIST_DIR.exists())
print("RESULTS_DIR exists:", RESULTS_DIR.exists())

# ============================================================
# LIST HISTORICAL FILES
# ============================================================
print("\n==============================")
print("HISTORICAL FILES")
print("==============================")

files = sorted(HIST_DIR.glob("*.csv"))
for f in files:
    print(f.name)

assert files, "❌ No historical prediction files found"

# ============================================================
# LOAD ONE FILE TO INSPECT SCHEMA
# ============================================================
hist_path = files[0]
hist = pd.read_csv(hist_path)

print("\n==============================")
print("USING FILE:", hist_path.name)
print("==============================")
print(hist.columns.tolist())

print("\n==============================")
print("SAMPLE ROW")
print("==============================")
print(hist.head(1).T)

# ============================================================
# LOAD ACTUAL RESULTS
# ============================================================
actuals = pd.read_csv(
    RESULTS_DIR / "final_walkforward_predictions.csv"
)[["season", "week", "home_team", "away_team", "actual_margin"]]

print("\n==============================")
print("ACTUAL RESULTS SAMPLE")
print("==============================")
print(actuals.head(1).T)
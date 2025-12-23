import pandas as pd
from pathlib import Path

from estimate_qb_values_from_spreads import estimate_qb_values

# ============================================================
# CONFIG — CHANGE THIS ONLY
# ============================================================
SEASON = 2025
WEEK = 17              # QB values trained through WEEK-1
K = 12                 # shrinkage strength

# ============================================================
# PATHS
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

EST_PATH = RESULTS_DIR / "estimated_qb_values.csv"
OUT_PATH = RESULTS_DIR / "qb_adjustments.csv"

# ============================================================
# 1. ESTIMATE RAW QB VALUES (through WEEK-1)
# ============================================================
qb = estimate_qb_values(
    season=SEASON,
    max_week=WEEK - 1
)

qb = qb.sort_values("qb_value_points", ascending=False).reset_index(drop=True)

qb.to_csv(EST_PATH, index=False)

print("=" * 60)
print(f"Estimated QB values through {SEASON} Week {WEEK - 1}")
print(f"Saved raw estimates → {EST_PATH}")
print("=" * 60)

# ============================================================
# 2. APPLY SHRINKAGE
# ============================================================
qb["qb_value_shrunk"] = (
    qb["games"] / (qb["games"] + K)
) * qb["qb_value_points"]

qb_adj = qb[["qb_name", "qb_value_shrunk"]]

# ================================
# TEST 4: QB ADJUSTMENT PRESENCE
# ================================
def assert_qb_present(qb_name):
    matches = qb[qb["qb_name"].str.contains(qb_name, case=False, na=False)]
    print(f"\n[QB VALUE CHECK: {qb_name}]")
    print(matches if len(matches) else "❌ QB NOT PRESENT")

assert_qb_present("Ewers")

qb_adj.to_csv(OUT_PATH, index=False)

print(f"\nSaved QB adjustments → {OUT_PATH}")

print("\nTop 10 QBs (shrunk):")
print(qb_adj.sort_values("qb_value_shrunk", ascending=False).head(10))

print("\nBottom 10 QBs (shrunk):")
print(qb_adj.sort_values("qb_value_shrunk").head(10))
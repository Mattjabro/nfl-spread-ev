from pathlib import Path
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from estimate_qb_values_from_spreads import estimate_qb_values

SEASON = 2025
START_WEEK = 1
END_WEEK = 15
K = 12  # shrinkage strength

OUT_DIR = REPO_ROOT / "historical" / "qb"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for week in range(START_WEEK, END_WEEK + 1):
    print(f"Building QB adjustments through Week {week-1}")

    qb = estimate_qb_values(SEASON, max_week=week - 1)

    qb["qb_value_shrunk"] = (
        qb["games"] / (qb["games"] + K)
    ) * qb["qb_value_points"]

    qb_adj = qb[["qb_name", "qb_value_shrunk"]]

    out = OUT_DIR / f"qb_adjustments_week_{week}.csv"
    qb_adj.to_csv(out, index=False)

    print(f"Saved → {out}")
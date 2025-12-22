from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from predict_week_blended import predict_week

SEASON = 2025
START_WEEK = 1
END_WEEK = 15

OUT_DIR = REPO_ROOT / "historical" / "outputs"
QB_DIR = REPO_ROOT / "historical" / "qb"

OUT_DIR.mkdir(parents=True, exist_ok=True)

for week in range(START_WEEK, END_WEEK + 1):
    print(f"=== Predicting {SEASON} Week {week} ===")

    qb_file = QB_DIR / f"qb_adjustments_week_{week}.csv"

    df = predict_week(
        SEASON,
        week,
        qb_file=qb_file   # one extra argument
    )

    out = OUT_DIR / f"season_{SEASON}_week_{week}_predictions.csv"
    df.to_csv(out, index=False)

    print(f"Saved → {out}")
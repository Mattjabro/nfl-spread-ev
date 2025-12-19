from pathlib import Path
from predict_week_blended import predict_week

SEASON = 2025
START_WEEK = 1
END_WEEK = 15   # up to prior week only

OUT_DIR = Path("../results")
OUT_DIR.mkdir(exist_ok=True)

for week in range(START_WEEK, END_WEEK + 1):
    print(f"Generating blended lines for Week {week}...")

    df = predict_week(SEASON, week)

    out = OUT_DIR / f"week{week}_blended_lines.csv"
    df.to_csv(out, index=False)

    print(f"Saved {out}")
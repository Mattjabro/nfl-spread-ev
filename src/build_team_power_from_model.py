import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("../results")
SEASON = 2025
MAX_WEEK = 15   # power rankings must be through prior week only

rows = []

for week in range(1, MAX_WEEK + 1):
    path = RESULTS_DIR / f"week{week}_blended_lines.csv"

    if not path.exists():
        print(f"Skipping Week {week} (file not found)")
        continue

    df = pd.read_csv(path)

    if "model_spread_home" not in df.columns:
        print(f"Skipping Week {week} (missing model_spread_home)")
        continue

    for _, g in df.iterrows():
        rows.append({
            "team": g["home_team"],
            "power": g["model_spread_home"]
        })
        rows.append({
            "team": g["away_team"],
            "power": -g["model_spread_home"]
        })

if not rows:
    raise RuntimeError("No blended week files found. Cannot build team power.")

power_df = (
    pd.DataFrame(rows)
      .groupby("team", as_index=False)["power"]
      .mean()
      .sort_values("power", ascending=False)
      .reset_index(drop=True)
)

power_df.insert(0, "rank", power_df.index + 1)

out = RESULTS_DIR / "team_power_rankings.csv"
power_df.to_csv(out, index=False)

print(f"Saved team power rankings to {out}")
print(power_df.head(10))
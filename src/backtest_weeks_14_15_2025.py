import pandas as pd
from backtest_week_strong_ev import backtest_week

results = []

for wk in [14, 15]:
    wk_df = backtest_week(2025, wk)
    results.append(wk_df)

df = pd.concat(results, ignore_index=True)

out = "../results/backtest_weeks_14_15_2025.csv"
df.to_csv(out, index=False)

print("\n================ WEEKS 14–15 BACKTEST =================")
print(df)

print("\nSUMMARY")
print("Total bets:", len(df))
print("Win rate:", df["won"].mean())
print("Average EV:", df["ev"].mean())

print("\nBy week:")
print(
    df.groupby("week")
      .agg(
          bets=("won", "count"),
          win_rate=("won", "mean"),
          avg_ev=("ev", "mean")
      )
)

print(f"\nSaved to {out}")
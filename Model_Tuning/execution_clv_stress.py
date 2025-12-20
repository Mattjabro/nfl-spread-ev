import pandas as pd
import numpy as np

PATH = "../results/vegas_bet_time_lines.csv"

df = pd.read_csv(PATH)

shifts = [-1.5, -1.0, -0.5, 0.0, +0.5]

rows = []

for s in shifts:
    tmp = df.copy()
    tmp["bet_spread_home"] = tmp["closing_spread_home"] - np.sign(tmp["closing_spread_home"]) * s
    tmp["scenario"] = f"shift_{s:+.1f}"

    rows.append(tmp)

out = pd.concat(rows, ignore_index=True)
out.to_csv("../results/vegas_bet_time_lines_stress.csv", index=False)

print("Created stress-test bet lines")
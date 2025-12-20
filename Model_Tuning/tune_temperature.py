import numpy as np
import pandas as pd
from scipy.stats import norm

# -----------------------------
# Load predictions
# -----------------------------
df = pd.read_csv("../results/predictions.csv")

required = {"model_spread", "sigma", "actual_margin"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["home_win"] = (df["actual_margin"] > 0).astype(int)

# -----------------------------
# Temperature grid
# -----------------------------
temps = np.linspace(0.5, 3.5, 121)

rows = []

for T in temps:
    z = df["model_spread"] / (df["sigma"] * T)
    probs = norm.cdf(z)

    eps = 1e-9
    probs = np.clip(probs, eps, 1 - eps)

    logloss = -(
        df["home_win"] * np.log(probs)
        + (1 - df["home_win"]) * np.log(1 - probs)
    ).mean()

    rows.append({
        "temperature": T,
        "log_loss": logloss
    })

out = pd.DataFrame(rows).sort_values("log_loss")
out.to_csv("../results/temperature_tuning.csv", index=False)

print("\nBest temperatures:")
print(out.head(10))
import sys
import time
import itertools
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# Path setup
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from backtest_decay_params import backtest_params

# --------------------------------------------------
# Hyperparameter grid
# --------------------------------------------------
lambdas = [0.015, 0.02, 0.03, 0.04, 0.05]
season_boosts = [1.0, 1.5, 2.0, 2.5]

rows = []

total_runs = len(lambdas) * len(season_boosts)
run_id = 1

# --------------------------------------------------
# Simple structured logger
# --------------------------------------------------
def log(msg):
    print(msg, flush=True)

# --------------------------------------------------
# Grid search
# --------------------------------------------------
for lam, c in itertools.product(lambdas, season_boosts):
    print("\n" + "=" * 70)
    print(f"[RUN {run_id}/{total_runs}]  λ={lam:.3f}  |  season_boost={c:.2f}")
    print("=" * 70)

    t0 = time.time()

    loss = backtest_params(
        decay_rate=lam,
        season_boost=c,
        logger=log   # <-- THIS is how we see season/week progress
    )

    elapsed = time.time() - t0

    print(
        f"[RUN {run_id}/{total_runs}] "
        f"λ={lam:.3f}, c={c:.2f} completed in {elapsed/60:.2f} min "
        f"(log loss = {loss:.4f})"
    )

    rows.append({
        "lambda": lam,
        "season_boost": c,
        "log_loss": loss
    })

    run_id += 1

# --------------------------------------------------
# Save + report
# --------------------------------------------------
out = (
    pd.DataFrame(rows)
      .sort_values("log_loss")
      .reset_index(drop=True)
)

out_path = ROOT / "results" / "decay_hyperparam_results.csv"
out.to_csv(out_path, index=False)

print("\n" + "=" * 70)
print("TOP 10 HYPERPARAMETER SETTINGS")
print("=" * 70)
print(out.head(10))
print(f"\nSaved results to {out_path}")
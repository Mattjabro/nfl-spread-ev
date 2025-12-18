import pandas as pd

# Shrinkage strength (games until we "trust" QB)
K = 12

qb = pd.read_csv("../results/estimated_qb_values.csv")

qb["qb_value_shrunk"] = (qb["games"] / (qb["games"] + K)) * qb["qb_value_points"]

qb_adj = qb[["qb_name", "qb_value_shrunk"]]

out = "../results/qb_adjustments.csv"
qb_adj.to_csv(out, index=False)

print(f"Saved QB adjustments to {out}")
print(qb_adj.sort_values("qb_value_shrunk", ascending=False).head(10))
print(qb_adj.sort_values("qb_value_shrunk").head(10))
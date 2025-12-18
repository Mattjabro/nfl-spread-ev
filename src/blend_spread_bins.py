import pandas as pd
import numpy as np

df = pd.read_csv("../results/predictions_vs_vegas.csv")

# Define edge size bins
bins = [0, 1, 3, 5, 100]
labels = ["small", "medium", "large", "huge"]
df["edge_size"] = pd.cut(df["edge"].abs(), bins=bins, labels=labels)

results = []

print("\n================ BLEND RESULTS BY EDGE SIZE =================")

for label in labels:
    subset = df[df["edge_size"] == label].copy()
    if len(subset) < 50:
        continue

    best_alpha = None
    best_acc = 0

    for alpha in np.linspace(0, 1, 21):
        blended = alpha * subset["closing_spread_home"] + (1 - alpha) * subset["model_spread"]
        correct = np.sign(blended) == np.sign(subset["actual_margin"])
        acc = correct.mean()

        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    results.append({
        "edge_bin": label,
        "games": len(subset),
        "best_alpha": best_alpha,
        "blended_accuracy": best_acc,
        "model_accuracy": (
            np.sign(subset["model_spread"]) == np.sign(subset["actual_margin"])
        ).mean(),
        "vegas_accuracy": (
            np.sign(subset["closing_spread_home"]) == np.sign(subset["actual_margin"])
        ).mean()
    })

out = pd.DataFrame(results)
print(out)

out.to_csv("../results/blend_weights_by_edge.csv", index=False)
print("\nSaved blend weights to ../results/blend_weights_by_edge.csv")
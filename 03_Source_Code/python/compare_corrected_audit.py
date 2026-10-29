
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(PROJECT_ROOT / "06_Results_and_Demo_Evidence" / "audit_results" / "outputs_corrected" / "post_GA90_predictions.csv")
df = df[df["split"] == "test"].copy()

df["error"] = (
    pd.to_numeric(df["actual"]) -
    pd.to_numeric(df["prediction"])
).abs()

pivot = df.pivot(
    index="transfer_row_id",
    columns="model",
    values="error"
)

print("Unique test transfer IDs:", len(pivot))
print("\nTest MAE by model:")
print(pivot.mean().sort_values().to_string())

print("\nPaired comparisons against core:")
for model in pivot.columns:
    if model == "core":
        continue

    pair = pivot[["core", model]].dropna()
    difference = pair["core"] - pair[model]
    _, p = wilcoxon(pair["core"], pair[model])

    reduction_pct = (
        100 * difference.mean() / pair["core"].mean()
    )

    print(
        f"{model}: n={len(pair)}, "
        f"MAE={pair[model].mean():.6f}, "
        f"MAE reduction={difference.mean():.6f}, "
        f"reduction={reduction_pct:.3f}%, "
        f"Wilcoxon p={p:.6g}"
    )

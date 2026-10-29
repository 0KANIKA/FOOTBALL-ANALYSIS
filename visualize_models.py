
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Find the actual model comparison file from the current folder
matches = list(Path.cwd().rglob("ABC_model_comparison.csv"))

if not matches:
    raise FileNotFoundError(
        "ABC_model_comparison.csv not found. "
        "Run your model training script first, then try again."
    )

csv_file = matches[0]
df = pd.read_csv(csv_file)

required = ["model", "test_MAE", "test_RMSE", "test_R2"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in results CSV: {missing}")

df["model"] = df["model"].astype(str).str.strip()
df = df.set_index("model").reindex(["A", "B", "C"]).reset_index()

if df[required].isna().any().any():
    raise ValueError("Could not find valid metric values for Models A, B and C.")

out = Path.cwd() / "visual_results"
out.mkdir(exist_ok=True)

# FIGURE 1: Actual test performance of Models A, B and C
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = [
    ("test_MAE", "Test MAE (lower is better)"),
    ("test_RMSE", "Test RMSE (lower is better)"),
    ("test_R2", "Test R² (higher is better)")
]

for ax, (column, title) in zip(axes, metrics):
    bars = ax.bar(["Model A", "Model B", "Model C"], df[column])
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.tick_params(axis="x", rotation=15)

fig.suptitle("Football Analytics: Models A vs B vs C", fontsize=15)
fig.tight_layout()
fig.savefig(out / "Figure1_Model_ABC_Comparison.png", dpi=250)
plt.close(fig)

# FIGURE 2: The three components inside Model B
fig, ax = plt.subplots(figsize=(13, 6))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.axis("off")

def box(x, y, width, height, title, body, color):
    ax.text(
        x + width/2, y + height/2,
        title + "\n\n" + body,
        ha="center", va="center", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.8",
                  facecolor=color, edgecolor="black", linewidth=1.4)
    )

box(0.5, 3.6, 3.5, 2.6,
    "MODE 1 — TARGET",
    "post_GA90\nWhat the model predicts",
    "#dcecff")

box(4.8, 3.6, 3.5, 2.6,
    "MODE 2 — METHOD",
    "RandomForestRegressor\nHow predictions are made",
    "#dff3e4")

box(9.0, 3.6, 3.5, 2.6,
    "MODE 3 — INPUTS",
    "Player features + 6 team features\nInformation used for prediction",
    "#fff0d8")

ax.annotate("", xy=(4.6, 4.9), xytext=(4.1, 4.9),
            arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(8.8, 4.9), xytext=(8.4, 4.9),
            arrowprops=dict(arrowstyle="->", lw=1.5))

ax.text(6.5, 2.5,
        "Player data: age, position, previous goals/assists, minutes, values",
        ha="center", fontsize=10)

ax.text(6.5, 1.9,
        "Team data: goals for/against, goal difference, win rate, points per game, match count",
        ha="center", fontsize=10)

ax.text(6.5, 0.8,
        "One model: one target + one algorithm + a set of input features",
        ha="center", fontsize=12, fontweight="bold")

ax.set_title("Inside Model B: Modes 1, 2 and 3", fontsize=15, pad=15)
fig.tight_layout()
fig.savefig(out / "Figure2_ModelB_Modes.png", dpi=250)
plt.close(fig)

print("\nVisualisations created successfully.")
print(f"Results read from: {csv_file}")
print(f"Images saved in: {out.resolve()}")
print("1. Figure1_Model_ABC_Comparison.png")
print("2. Figure2_ModelB_Modes.png")

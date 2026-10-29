# ================================================================
# 04_visualizations.py
# Research Visualizations for Models A, B and C
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ================================================================
# 1. PATHS
# ================================================================

BASE = Path(__file__).resolve().parents[2]

DATA_FILE = BASE / "04_Datasets" / "derived" / "FINAL_MODEL_C_CLEAN.csv"

RESULTS_DIR = BASE / "06_Results_and_Demo_Evidence" / "research_results" / "model_abc_results"

ANALYSIS_DIR = BASE / "06_Results_and_Demo_Evidence" / "research_results" / "model_analysis_results"

OUTPUT_DIR = BASE / "06_Results_and_Demo_Evidence" / "model_visualizations" / "model_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)


# ================================================================
# 2. LOAD DATA
# ================================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_csv(DATA_FILE)

df["transfer_date"] = pd.to_datetime(
    df["transfer_date"]
)

print("Dataset rows:", len(df))


# ================================================================
# 3. LOAD MODEL COMPARISON
# ================================================================

comparison = pd.read_csv(
    RESULTS_DIR / "ABC_model_comparison.csv"
)

print("\nModel comparison loaded.")


# ================================================================
# 4. LOAD ERROR ANALYSIS
# ================================================================

error_data = pd.read_csv(
    ANALYSIS_DIR / "ABC_test_error_analysis.csv"
)

print(
    "Test error-analysis records:",
    len(error_data)
)


# ================================================================
# 5. LOAD FEATURE IMPORTANCE
# ================================================================

importance_C = pd.read_csv(
    RESULTS_DIR / "model_C_feature_importance.csv"
)

importance_A = pd.read_csv(
    RESULTS_DIR / "model_A_feature_importance.csv"
)

importance_B = pd.read_csv(
    RESULTS_DIR / "model_B_feature_importance.csv"
)


# ================================================================
# HELPER
# ================================================================

def save_show(fig, filename):

    filepath = OUTPUT_DIR / filename

    fig.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight"
    )

    print("Saved:", filename)

    plt.show()

    plt.close(fig)


# ================================================================
# FIGURE 1
# MODEL PERFORMANCE COMPARISON — R²
# ================================================================

print("\n" + "=" * 80)
print("FIGURE 1 — MODEL R² COMPARISON")
print("=" * 80)

models = comparison["model"].tolist()

test_r2 = comparison["test_R2"].tolist()

fig = plt.figure(figsize=(8, 5))

plt.bar(
    models,
    test_r2
)

plt.xlabel("Model")

plt.ylabel("Test R²")

plt.title(
    "Test R² Comparison of Models A, B and C"
)

plt.ylim(
    max(0, min(test_r2) - 0.05),
    max(test_r2) + 0.05
)

for i, value in enumerate(test_r2):

    plt.text(
        i,
        value + 0.003,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

save_show(
    fig,
    "figure_1_test_r2_comparison.png"
)


# ================================================================
# FIGURE 2
# MAE COMPARISON
# ================================================================

print("\n" + "=" * 80)
print("FIGURE 2 — MODEL MAE COMPARISON")
print("=" * 80)

test_mae = comparison["test_MAE"].tolist()

fig = plt.figure(figsize=(8, 5))

plt.bar(
    models,
    test_mae
)

plt.xlabel("Model")

plt.ylabel("Test MAE")

plt.title(
    "Test MAE Comparison of Models A, B and C"
)

plt.ylim(
    0,
    max(test_mae) + 0.02
)

for i, value in enumerate(test_mae):

    plt.text(
        i,
        value + 0.002,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

save_show(
    fig,
    "figure_2_test_mae_comparison.png"
)


# ================================================================
# FIGURE 3
# ACTUAL VS PREDICTED — MODEL C
# ================================================================

print("\n" + "=" * 80)
print("FIGURE 3 — ACTUAL VS PREDICTED")
print("=" * 80)

actual = error_data["post_GA90"]

predicted_C = error_data["pred_C"]

fig = plt.figure(figsize=(7, 7))

plt.scatter(
    actual,
    predicted_C,
    alpha=0.45,
    s=25
)

min_value = min(
    actual.min(),
    predicted_C.min()
)

max_value = max(
    actual.max(),
    predicted_C.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel(
    "Actual Post-Transfer G+A/90"
)

plt.ylabel(
    "Predicted Post-Transfer G+A/90"
)

plt.title(
    "Model C: Actual vs Predicted Post-Transfer G+A/90"
)

plt.tight_layout()

save_show(
    fig,
    "figure_3_model_C_actual_vs_predicted.png"
)


# ================================================================
# FIGURE 4
# ERROR DISTRIBUTION
# ================================================================

print("\n" + "=" * 80)
print("FIGURE 4 — ERROR DISTRIBUTION")
print("=" * 80)

fig = plt.figure(figsize=(9, 5))

plt.hist(
    error_data["error_A"],
    bins=40,
    alpha=0.45,
    label="Model A"
)

plt.hist(
    error_data["error_B"],
    bins=40,
    alpha=0.45,
    label="Model B"
)

plt.hist(
    error_data["error_C"],
    bins=40,
    alpha=0.45,
    label="Model C"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Prediction Error (Predicted − Actual)"
)

plt.ylabel(
    "Number of Test Transfers"
)

plt.title(
    "Prediction Error Distribution"
)

plt.legend()

plt.tight_layout()

save_show(
    fig,
    "figure_4_error_distribution.png"
)


# ================================================================
# FIGURE 5
# MAE BY POSITION
# ================================================================

print("\n" + "=" * 80)
print("FIGURE 5 — MAE BY POSITION")
print("=" * 80)

position_data = pd.read_csv(
    ANALYSIS_DIR / "error_by_position.csv"
)

positions = position_data["position"]

x = np.arange(
    len(positions)
)

width = 0.25

fig = plt.figure(figsize=(10, 6))

plt.bar(
    x - width,
    position_data["A_MAE"],
    width,
    label="Model A"
)

plt.bar(
    x,
    position_data["B_MAE"],
    width,
    label="Model B"
)

plt.bar(
    x + width,
    position_data["C_MAE"],
    width,
    label="Model C"
)

plt.xticks(
    x,
    positions
)

plt.xlabel(
    "Player Position"
)

plt.ylabel(
    "MAE"
)

plt.title(
    "Prediction Error by Player Position"
)

plt.legend()

plt.tight_layout()

save_show(
    fig,
    "figure_5_mae_by_position.png"
)


# ================================================================
# FIGURE 6
# MAE BY PERFORMANCE BAND
# ================================================================

print("\n" + "=" * 80)
print("FIGURE 6 — MAE BY PERFORMANCE BAND")
print("=" * 80)

performance_data = pd.read_csv(
    ANALYSIS_DIR /
    "error_by_performance_band.csv"
)

bands = performance_data[
    "performance_band"
]

x = np.arange(
    len(bands)
)

fig = plt.figure(figsize=(10, 6))

plt.bar(
    x - width,
    performance_data["A_MAE"],
    width,
    label="Model A"
)

plt.bar(
    x,
    performance_data["B_MAE"],
    width,
    label="Model B"
)

plt.bar(
    x + width,
    performance_data["C_MAE"],
    width,
    label="Model C"
)

plt.xticks(
    x,
    bands,
    rotation=15
)

plt.xlabel(
    "Actual Post-Transfer Performance Band"
)

plt.ylabel(
    "MAE"
)

plt.title(
    "Prediction Error Across Performance Bands"
)

plt.legend()

plt.tight_layout()

save_show(
    fig,
    "figure_6_mae_by_performance_band.png"
)


# ================================================================
# FIGURE 7
# MODEL C FEATURE IMPORTANCE
# ================================================================

print("\n" + "=" * 80)
print("FIGURE 7 — MODEL C FEATURE IMPORTANCE")
print("=" * 80)

top_features = (
    importance_C
    .sort_values(
        "importance",
        ascending=False
    )
    .head(15)
    .sort_values(
        "importance",
        ascending=True
    )
)

fig = plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.xlabel(
    "Random Forest Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 15 Features in Model C"
)

plt.tight_layout()

save_show(
    fig,
    "figure_7_model_C_feature_importance.png"
)


# ================================================================
# FIGURE 8
# INCREMENTAL R²
# ================================================================

print("\n" + "=" * 80)
print("FIGURE 8 — INCREMENTAL R²")
print("=" * 80)

r2_values = [
    comparison.loc[
        comparison["model"] == "A",
        "test_R2"
    ].iloc[0],

    comparison.loc[
        comparison["model"] == "B",
        "test_R2"
    ].iloc[0],

    comparison.loc[
        comparison["model"] == "C",
        "test_R2"
    ].iloc[0]
]

fig = plt.figure(figsize=(8, 5))

plt.plot(
    ["A\nPlayer",
     "B\nPlayer + Team",
     "C\nRich Context"],
    r2_values,
    marker="o"
)

plt.xlabel(
    "Model Specification"
)

plt.ylabel(
    "Test R²"
)

plt.title(
    "Incremental Predictive Performance"
)

plt.ylim(
    min(r2_values) - 0.02,
    max(r2_values) + 0.02
)

for i, value in enumerate(r2_values):

    plt.text(
        i,
        value + 0.002,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

save_show(
    fig,
    "figure_8_incremental_R2.png"
)


# ================================================================
# FIGURE 9
# TRAIN vs VALIDATION vs TEST R²
# ================================================================

print("\n" + "=" * 80)
print("FIGURE 9 — R² ACROSS DATA SPLITS")
print("=" * 80)

x = np.arange(
    len(models)
)

width = 0.25

fig = plt.figure(figsize=(9, 6))

plt.bar(
    x - width,
    comparison["train_R2"],
    width,
    label="Train"
)

plt.bar(
    x,
    comparison["validation_R2"],
    width,
    label="Validation"
)

plt.bar(
    x + width,
    comparison["test_R2"],
    width,
    label="Test"
)

plt.xticks(
    x,
    models
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "R²"
)

plt.title(
    "Model R² Across Training, Validation and Test Sets"
)

plt.legend()

plt.tight_layout()

save_show(
    fig,
    "figure_9_r2_train_validation_test.png"
)


# ================================================================
# 6. SAVE NUMERICAL VISUALIZATION DATA
# ================================================================

comparison.to_csv(
    OUTPUT_DIR / "visualization_model_metrics.csv",
    index=False
)

position_data.to_csv(
    OUTPUT_DIR / "visualization_position_metrics.csv",
    index=False
)

performance_data.to_csv(
    OUTPUT_DIR / "visualization_performance_metrics.csv",
    index=False
)

importance_C.head(30).to_csv(
    OUTPUT_DIR / "visualization_top_C_features.csv",
    index=False
)
# ================================================================
# FIGURE 3
# RMSE COMPARISON
# ================================================================

print("\n" + "=" * 80)
print("FIGURE 3 — MODEL RMSE COMPARISON")
print("=" * 80)

test_rmse = comparison["test_RMSE"].tolist()

fig = plt.figure(figsize=(8, 5))

plt.bar(
    models,
    test_rmse
)

plt.xlabel("Model")

plt.ylabel("Test RMSE")

plt.title(
    "Test RMSE Comparison of Models A, B and C"
)

plt.ylim(
    0,
    max(test_rmse) + 0.02
)

for i, value in enumerate(test_rmse):

    plt.text(
        i,
        value + 0.002,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

save_show(
    fig,
    "figure_3_test_rmse_comparison.png"
)

# ================================================================
# FINAL
# ================================================================

print("\n")
print("=" * 80)
print("VISUALIZATION GENERATION COMPLETE")
print("=" * 80)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated figures:")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(" -", file.name)

print("\nDone.")

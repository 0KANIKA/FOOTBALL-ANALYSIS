import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_FILE = os.path.join(PROJECT_ROOT, "04_Datasets", "derived", "post_transfer_performance.csv")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "06_Results_and_Demo_Evidence", "research_results", "model_a_results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("MODEL A - DIAGNOSTICS")
print("=" * 60)

print("\nLoading datasets...")

df = pd.read_csv(DATA_FILE)

pred_file = os.path.join(
    RESULTS_DIR,
    "test_predictions.csv"
)

predictions = pd.read_csv(pred_file)

df["transfer_date"] = pd.to_datetime(
    df["transfer_date"],
    errors="coerce"
)

predictions["transfer_date"] = pd.to_datetime(
    predictions["transfer_date"],
    errors="coerce"
)

print("Full dataset:", len(df))
print("Prediction records:", len(predictions))


# ============================================================
# FILTER RANDOM FOREST PREDICTIONS
# ============================================================

rf_pred = predictions[
    predictions["Model"] == "Random Forest"
].copy()

print(
    "Random Forest test predictions:",
    len(rf_pred)
)


# ============================================================
# 1. ACTUAL VS PREDICTED
# ============================================================

print("\nCreating actual vs predicted plot...")

plt.figure(figsize=(8, 6))

plt.scatter(
    rf_pred["post_GA90"],
    rf_pred["predicted_GA90"],
    alpha=0.45
)

min_value = min(
    rf_pred["post_GA90"].min(),
    rf_pred["predicted_GA90"].min()
)

max_value = max(
    rf_pred["post_GA90"].max(),
    rf_pred["predicted_GA90"].max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Post-transfer G+A/90")
plt.ylabel("Predicted Post-transfer G+A/90")
plt.title("Random Forest: Actual vs Predicted")

plt.tight_layout()

actual_pred_file = os.path.join(
    RESULTS_DIR,
    "actual_vs_predicted.png"
)

plt.savefig(
    actual_pred_file,
    dpi=300
)

plt.close()


# ============================================================
# 2. RESIDUAL DISTRIBUTION
# ============================================================

print("Creating residual distribution...")

plt.figure(figsize=(8, 6))

plt.hist(
    rf_pred["error"],
    bins=40
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Random Forest Residual Distribution")

plt.tight_layout()

residual_file = os.path.join(
    RESULTS_DIR,
    "residual_distribution.png"
)

plt.savefig(
    residual_file,
    dpi=300
)

plt.close()


# ============================================================
# 3. ABSOLUTE ERROR DISTRIBUTION
# ============================================================

print("Creating absolute error distribution...")

plt.figure(figsize=(8, 6))

plt.hist(
    rf_pred["absolute_error"],
    bins=40
)

plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.title("Random Forest Absolute Prediction Error")

plt.tight_layout()

absolute_error_file = os.path.join(
    RESULTS_DIR,
    "absolute_error_distribution.png"
)

plt.savefig(
    absolute_error_file,
    dpi=300
)

plt.close()


# ============================================================
# 4. ERROR VS PREDICTION
# ============================================================

print("Creating residual scatter plot...")

plt.figure(figsize=(8, 6))

plt.scatter(
    rf_pred["predicted_GA90"],
    rf_pred["error"],
    alpha=0.45
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted G+A/90")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted G+A/90")

plt.tight_layout()

residual_scatter_file = os.path.join(
    RESULTS_DIR,
    "residuals_vs_predicted.png"
)

plt.savefig(
    residual_scatter_file,
    dpi=300
)

plt.close()


# ============================================================
# 5. LARGEST PREDICTION ERRORS
# ============================================================

print("\nFinding largest prediction errors...")

largest_errors = rf_pred.sort_values(
    "absolute_error",
    ascending=False
).head(25).copy()

largest_errors_file = os.path.join(
    RESULTS_DIR,
    "largest_prediction_errors.csv"
)

largest_errors.to_csv(
    largest_errors_file,
    index=False
)


# ============================================================
# 6. PERFORMANCE BY POSITION
# ============================================================

print("Calculating performance by position...")

position_data = df[
    [
        "player_id",
        "transfer_date",
        "position",
        "post_GA90"
    ]
].copy()

position_data = position_data.merge(
    rf_pred[
        [
            "player_id",
            "transfer_date",
            "predicted_GA90",
            "absolute_error"
        ]
    ],
    on=[
        "player_id",
        "transfer_date"
    ],
    how="inner"
)

position_summary = (
    position_data
    .groupby("position")
    .agg(
        Players=("player_id", "count"),
        Mean_Actual_GA90=("post_GA90", "mean"),
        Mean_Predicted_GA90=("predicted_GA90", "mean"),
        MAE=("absolute_error", "mean")
    )
    .reset_index()
)

position_summary = position_summary.sort_values(
    "Players",
    ascending=False
)

position_file = os.path.join(
    RESULTS_DIR,
    "performance_by_position.csv"
)

position_summary.to_csv(
    position_file,
    index=False
)


# ============================================================
# POSITION MAE PLOT
# ============================================================

print("Creating position error plot...")

position_plot_data = position_summary[
    position_summary["Players"] >= 20
].copy()

plt.figure(figsize=(10, 6))

plt.bar(
    position_plot_data["position"].astype(str),
    position_plot_data["MAE"]
)

plt.xlabel("Position")
plt.ylabel("Mean Absolute Error")
plt.title("Random Forest MAE by Player Position")

plt.xticks(rotation=30)

plt.tight_layout()

position_plot_file = os.path.join(
    RESULTS_DIR,
    "mae_by_position.png"
)

plt.savefig(
    position_plot_file,
    dpi=300
)

plt.close()


# ============================================================
# 7. PERFORMANCE BY TRANSFER YEAR
# ============================================================

print("Calculating performance by transfer year...")

rf_pred["transfer_year"] = (
    rf_pred["transfer_date"]
    .dt.year
)

year_summary = (
    rf_pred
    .groupby("transfer_year")
    .agg(
        Transfers=("player_id", "count"),
        MAE=("absolute_error", "mean"),
        Mean_Actual_GA90=("post_GA90", "mean"),
        Mean_Predicted_GA90=("predicted_GA90", "mean")
    )
    .reset_index()
)

year_file = os.path.join(
    RESULTS_DIR,
    "performance_by_transfer_year.csv"
)

year_summary.to_csv(
    year_file,
    index=False
)


# ============================================================
# 8. TARGET DISTRIBUTION
# ============================================================

print("Creating target distribution...")

eligible = df[
    (df["complete_365_day_window"] == True) &
    (df["eligible_450min"] == True)
].copy()

plt.figure(figsize=(8, 6))

plt.hist(
    eligible["post_GA90"],
    bins=50
)

plt.xlabel("Post-transfer G+A/90")
plt.ylabel("Number of Transfers")
plt.title("Distribution of Post-transfer G+A/90")

plt.tight_layout()

target_file = os.path.join(
    RESULTS_DIR,
    "target_distribution.png"
)

plt.savefig(
    target_file,
    dpi=300
)

plt.close()


# ============================================================
# 9. SUMMARY STATISTICS
# ============================================================

summary = {
    "Test records": len(rf_pred),
    "Mean actual G+A/90": rf_pred["post_GA90"].mean(),
    "Median actual G+A/90": rf_pred["post_GA90"].median(),
    "Mean predicted G+A/90": rf_pred["predicted_GA90"].mean(),
    "MAE": rf_pred["absolute_error"].mean(),
    "Mean residual": rf_pred["error"].mean(),
    "Maximum absolute error": rf_pred["absolute_error"].max()
}

summary_df = pd.DataFrame(
    [summary]
)

summary_file = os.path.join(
    RESULTS_DIR,
    "diagnostic_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 60)
print("MODEL A DIAGNOSTICS COMPLETE")
print("=" * 60)

print("\nSummary:")

for key, value in summary.items():

    if isinstance(value, (float, np.floating)):
        print(
            key + ":",
            round(float(value), 4)
        )
    else:
        print(
            key + ":",
            value
        )

print("\nFiles created in:")

print(RESULTS_DIR)

print("\nImportant:")
print("These are still preliminary research results.")

print("\nDone.")

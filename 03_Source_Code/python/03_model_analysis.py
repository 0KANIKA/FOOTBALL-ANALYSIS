# ================================================================
# 03_model_analysis.py
# Error Analysis for Models A, B and C
# ================================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ================================================================
# 1. PATHS
# ================================================================

BASE = Path(__file__).resolve().parents[2]

DATA_FILE = BASE / "04_Datasets" / "derived" / "FINAL_MODEL_C_CLEAN.csv"
RESULTS_DIR = BASE / "06_Results_and_Demo_Evidence" / "research_results" / "model_abc_results"

ANALYSIS_DIR = BASE / "06_Results_and_Demo_Evidence" / "research_results" / "model_analysis_results"
ANALYSIS_DIR.mkdir(exist_ok=True)


# ================================================================
# 2. LOAD DATA
# ================================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_csv(DATA_FILE)

df["transfer_date"] = pd.to_datetime(df["transfer_date"])

print("Final dataset:", len(df))


# ================================================================
# 3. LOAD MODEL PREDICTIONS
# ================================================================

print("\n" + "=" * 80)
print("LOADING MODEL PREDICTIONS")
print("=" * 80)

pred_A = pd.read_csv(
    RESULTS_DIR / "model_A_test_predictions.csv"
)

pred_B = pd.read_csv(
    RESULTS_DIR / "model_B_test_predictions.csv"
)

pred_C = pd.read_csv(
    RESULTS_DIR / "model_C_test_predictions.csv"
)

print("Model A predictions:", len(pred_A))
print("Model B predictions:", len(pred_B))
print("Model C predictions:", len(pred_C))


# ================================================================
# 4. STANDARDIZE PREDICTION COLUMNS
# ================================================================

pred_A = pred_A.rename(
    columns={
        "predicted_GA90": "pred_A"
    }
)

pred_B = pred_B.rename(
    columns={
        "predicted_GA90": "pred_B"
    }
)

pred_C = pred_C.rename(
    columns={
        "predicted_GA90": "pred_C"
    }
)


# ================================================================
# 5. MERGE A/B/C
# ================================================================

merge_cols = [
    "transfer_row_id"
]

comparison = pred_A[
    merge_cols
    + [
        "post_GA90",
        "pred_A"
    ]
].merge(
    pred_B[
        merge_cols
        + ["pred_B"]
    ],
    on="transfer_row_id",
    how="inner"
).merge(
    pred_C[
        merge_cols
        + ["pred_C"]
    ],
    on="transfer_row_id",
    how="inner"
)


print("\nCommon test records:", len(comparison))


if len(comparison) != len(pred_A):
    raise ValueError(
        "A/B/C do not contain the same test records."
    )


# ================================================================
# 6. ADD PLAYER INFORMATION
# ================================================================

info_cols = [
    "transfer_row_id",
    "player_id",
    "transfer_date",
    "from_club_name",
    "to_club_name",
    "age_at_transfer",
    "position",
    "sub_position",
    "target_position_group",
    "minutes_pre",
    "goals_pre",
    "assists_pre",
    "goals_per90_pre",
    "assists_per90_pre",
    "market_value_pre"
]

info = df[info_cols].copy()

comparison = comparison.merge(
    info,
    on="transfer_row_id",
    how="left"
)


# ================================================================
# 7. ERROR CALCULATIONS
# ================================================================

for model in ["A", "B", "C"]:

    comparison[f"error_{model}"] = (
        comparison[f"pred_{model}"]
        - comparison["post_GA90"]
    )

    comparison[f"absolute_error_{model}"] = (
        comparison[f"error_{model}"]
        .abs()
    )


# ================================================================
# 8. ERROR DIRECTION
# ================================================================

comparison["best_model"] = comparison[
    [
        "absolute_error_A",
        "absolute_error_B",
        "absolute_error_C"
    ]
].idxmin(axis=1)

comparison["best_model"] = (
    comparison["best_model"]
    .str.replace(
        "absolute_error_",
        "",
        regex=False
    )
)


# ================================================================
# 9. OVERALL ERROR SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("OVERALL TEST ERROR")
print("=" * 80)

summary_rows = []

for model in ["A", "B", "C"]:

    errors = comparison[
        f"absolute_error_{model}"
    ]

    signed = comparison[
        f"error_{model}"
    ]

    summary_rows.append({
        "model": model,
        "MAE": errors.mean(),
        "RMSE": np.sqrt(
            np.mean(
                signed ** 2
            )
        ),
        "mean_signed_error": signed.mean(),
        "median_absolute_error": errors.median(),
        "90th_percentile_absolute_error":
            errors.quantile(0.90),
        "max_absolute_error": errors.max()
    })


overall_error = pd.DataFrame(
    summary_rows
)

print(
    overall_error.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

overall_error.to_csv(
    ANALYSIS_DIR / "overall_error_summary.csv",
    index=False
)


# ================================================================
# 10. POSITION ERROR ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("ERROR BY POSITION")
print("=" * 80)

position_rows = []

for position, group in comparison.groupby(
    "position",
    dropna=False
):

    row = {
        "position": position,
        "n": len(group)
    }

    for model in ["A", "B", "C"]:

        row[f"{model}_MAE"] = group[
            f"absolute_error_{model}"
        ].mean()

        row[f"{model}_RMSE"] = np.sqrt(
            np.mean(
                group[f"error_{model}"] ** 2
            )
        )

    position_rows.append(row)


error_by_position = pd.DataFrame(
    position_rows
).sort_values(
    "n",
    ascending=False
)

print(
    error_by_position.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

error_by_position.to_csv(
    ANALYSIS_DIR / "error_by_position.csv",
    index=False
)


# ================================================================
# 11. TARGET POSITION GROUP ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("ERROR BY DESTINATION POSITION GROUP")
print("=" * 80)

target_position_rows = []

for group_name, group in comparison.groupby(
    "target_position_group",
    dropna=False
):

    row = {
        "target_position_group": group_name,
        "n": len(group)
    }

    for model in ["A", "B", "C"]:

        row[f"{model}_MAE"] = group[
            f"absolute_error_{model}"
        ].mean()

        row[f"{model}_RMSE"] = np.sqrt(
            np.mean(
                group[f"error_{model}"] ** 2
            )
        )

    target_position_rows.append(row)


error_by_target_position = pd.DataFrame(
    target_position_rows
).sort_values(
    "n",
    ascending=False
)

print(
    error_by_target_position.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

error_by_target_position.to_csv(
    ANALYSIS_DIR / "error_by_target_position_group.csv",
    index=False
)


# ================================================================
# 12. AGE GROUP ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("ERROR BY AGE GROUP")
print("=" * 80)


def age_group(age):

    if pd.isna(age):
        return "Unknown"

    if age < 21:
        return "Under 21"

    if age < 25:
        return "21-24"

    if age < 29:
        return "25-28"

    if age < 33:
        return "29-32"

    return "33+"


comparison["age_group"] = (
    comparison["age_at_transfer"]
    .apply(age_group)
)


age_rows = []

for group_name, group in comparison.groupby(
    "age_group",
    dropna=False
):

    row = {
        "age_group": group_name,
        "n": len(group)
    }

    for model in ["A", "B", "C"]:

        row[f"{model}_MAE"] = group[
            f"absolute_error_{model}"
        ].mean()

        row[f"{model}_RMSE"] = np.sqrt(
            np.mean(
                group[f"error_{model}"] ** 2
            )
        )

    age_rows.append(row)


error_by_age = pd.DataFrame(
    age_rows
)

print(
    error_by_age.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

error_by_age.to_csv(
    ANALYSIS_DIR / "error_by_age_group.csv",
    index=False
)


# ================================================================
# 13. PERFORMANCE BANDS
# ================================================================

print("\n" + "=" * 80)
print("ERROR BY ACTUAL POST-TRANSFER PERFORMANCE")
print("=" * 80)


def performance_band(value):

    if value < 0.10:
        return "Low (<0.10)"

    if value < 0.25:
        return "Medium (0.10-0.25)"

    if value < 0.50:
        return "High (0.25-0.50)"

    return "Very High (>=0.50)"


comparison["performance_band"] = (
    comparison["post_GA90"]
    .apply(performance_band)
)


performance_rows = []

for band, group in comparison.groupby(
    "performance_band",
    dropna=False
):

    row = {
        "performance_band": band,
        "n": len(group),
        "actual_mean_GA90":
            group["post_GA90"].mean()
    }

    for model in ["A", "B", "C"]:

        row[f"{model}_MAE"] = group[
            f"absolute_error_{model}"
        ].mean()

        row[f"{model}_RMSE"] = np.sqrt(
            np.mean(
                group[f"error_{model}"] ** 2
            )
        )

        row[f"{model}_mean_error"] = group[
            f"error_{model}"
        ].mean()

    performance_rows.append(row)


error_by_performance = pd.DataFrame(
    performance_rows
)

print(
    error_by_performance.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

error_by_performance.to_csv(
    ANALYSIS_DIR / "error_by_performance_band.csv",
    index=False
)


# ================================================================
# 14. UNDER / OVER-PREDICTION
# ================================================================

print("\n" + "=" * 80)
print("UNDER / OVER PREDICTION")
print("=" * 80)

direction_rows = []

for model in ["A", "B", "C"]:

    error = comparison[
        f"error_{model}"
    ]

    direction_rows.append({
        "model": model,
        "underpredictions":
            (error < 0).sum(),
        "overpredictions":
            (error > 0).sum(),
        "exact_predictions":
            (error == 0).sum(),
        "mean_error":
            error.mean()
    })


direction_df = pd.DataFrame(
    direction_rows
)

print(
    direction_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

direction_df.to_csv(
    ANALYSIS_DIR / "under_over_prediction.csv",
    index=False
)


# ================================================================
# 15. MODEL WIN COUNTS
# ================================================================

print("\n" + "=" * 80)
print("WHICH MODEL HAS THE LOWEST ERROR?")
print("=" * 80)

win_counts = (
    comparison["best_model"]
    .value_counts()
    .rename_axis("model")
    .reset_index(name="cases")
)

win_counts["percentage"] = (
    win_counts["cases"]
    / len(comparison)
    * 100
)

print(
    win_counts.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

win_counts.to_csv(
    ANALYSIS_DIR / "model_win_counts.csv",
    index=False
)


# ================================================================
# 16. A -> B IMPROVEMENT BY PLAYER
# ================================================================

comparison["A_to_B_error_change"] = (
    comparison["absolute_error_A"]
    - comparison["absolute_error_B"]
)

comparison["A_to_C_error_change"] = (
    comparison["absolute_error_A"]
    - comparison["absolute_error_C"]
)

comparison["B_to_C_error_change"] = (
    comparison["absolute_error_B"]
    - comparison["absolute_error_C"]
)


# ================================================================
# 17. CONTEXT IMPROVEMENT SUMMARY
# ================================================================

improvement_summary = pd.DataFrame({

    "comparison": [
        "A -> B",
        "B -> C",
        "A -> C"
    ],

    "mean_absolute_error_change": [
        comparison["A_to_B_error_change"].mean(),
        comparison["B_to_C_error_change"].mean(),
        comparison["A_to_C_error_change"].mean()
    ],

    "median_absolute_error_change": [
        comparison["A_to_B_error_change"].median(),
        comparison["B_to_C_error_change"].median(),
        comparison["A_to_C_error_change"].median()
    ],

    "cases_improved": [
        (
            comparison["A_to_B_error_change"] > 0
        ).sum(),

        (
            comparison["B_to_C_error_change"] > 0
        ).sum(),

        (
            comparison["A_to_C_error_change"] > 0
        ).sum()
    ],

    "cases_worsened": [
        (
            comparison["A_to_B_error_change"] < 0
        ).sum(),

        (
            comparison["B_to_C_error_change"] < 0
        ).sum(),

        (
            comparison["A_to_C_error_change"] < 0
        ).sum()
    ]
})


improvement_summary.to_csv(
    ANALYSIS_DIR / "context_improvement_summary.csv",
    index=False
)


# ================================================================
# 18. LARGEST ERRORS
# ================================================================

print("\n" + "=" * 80)
print("LARGEST MODEL C ERRORS")
print("=" * 80)

largest_errors = (
    comparison
    .sort_values(
        "absolute_error_C",
        ascending=False
    )
    .head(25)
)

largest_columns = [
    "transfer_row_id",
    "player_id",
    "transfer_date",
    "from_club_name",
    "to_club_name",
    "position",
    "age_at_transfer",
    "post_GA90",
    "pred_A",
    "pred_B",
    "pred_C",
    "absolute_error_A",
    "absolute_error_B",
    "absolute_error_C"
]

print(
    largest_errors[largest_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

largest_errors[largest_columns].to_csv(
    ANALYSIS_DIR / "largest_model_C_errors.csv",
    index=False
)


# ================================================================
# 19. COMPLETE COMPARISON DATASET
# ================================================================

comparison.to_csv(
    ANALYSIS_DIR / "ABC_test_error_analysis.csv",
    index=False
)


# ================================================================
# 20. FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 80)
print("ERROR ANALYSIS COMPLETE")
print("=" * 80)

print("\nOutput directory:")
print(ANALYSIS_DIR)

print("\nGenerated files:")

for file in sorted(ANALYSIS_DIR.iterdir()):
    print(" -", file.name)

print("\nDone.")

# ================================================================
# 07_final_results_summary.py
# Final Research Results Consolidation
# ================================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ================================================================
# 1. PATHS
# ================================================================

BASE = Path(__file__).resolve().parents[2]

DATA_FILE = BASE / "04_Datasets" / "derived" / "FINAL_MODEL_C_CLEAN.csv"

MODEL_DIR = BASE / "06_Results_and_Demo_Evidence" / "research_results" / "model_abc_results"

ANALYSIS_DIR = BASE / "06_Results_and_Demo_Evidence" / "research_results" / "model_analysis_results"

ROBUSTNESS_DIR = BASE / "06_Results_and_Demo_Evidence" / "audit_results" / "statistical_robustness_results"

OUTPUT_DIR = BASE / "06_Results_and_Demo_Evidence" / "research_results" / "FINAL_RESEARCH_RESULTS"

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ================================================================
# 2. LOAD FINAL DATASET
# ================================================================

print("=" * 90)
print("FINAL RESEARCH RESULTS SUMMARY")
print("=" * 90)

df = pd.read_csv(
    DATA_FILE
)

print("\nFinal dataset:")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ================================================================
# 3. DATASET SUMMARY
# ================================================================

dataset_summary = pd.DataFrame({

    "Metric": [
        "Final cohort size",
        "Total dataset columns",
        "Minimum transfer date",
        "Maximum transfer date",
        "Unique players",
        "Unique destination clubs",
        "Duplicate transfer IDs",
        "Missing target values",
        "Target mean",
        "Target median",
        "Target standard deviation",
        "Target minimum",
        "Target maximum"
    ],

    "Value": [
        len(df),
        len(df.columns),
        df["transfer_date"].min(),
        df["transfer_date"].max(),
        df["player_id"].nunique(),
        df["to_club_id"].nunique(),
        df["transfer_row_id"].duplicated().sum(),
        df["post_GA90"].isna().sum(),
        df["post_GA90"].mean(),
        df["post_GA90"].median(),
        df["post_GA90"].std(),
        df["post_GA90"].min(),
        df["post_GA90"].max()
    ]

})

dataset_summary.to_csv(
    OUTPUT_DIR / "01_dataset_summary.csv",
    index=False
)


# ================================================================
# 4. MODEL PERFORMANCE
# ================================================================

comparison_file = (
    MODEL_DIR /
    "ABC_model_comparison.csv"
)

comparison = pd.read_csv(
    comparison_file
)

# Keep the important columns
model_performance = comparison[
    [
        "model",
        "train_MAE",
        "train_RMSE",
        "train_R2",
        "validation_MAE",
        "validation_RMSE",
        "validation_R2",
        "test_MAE",
        "test_RMSE",
        "test_R2"
    ]
].copy()


model_performance.to_csv(
    OUTPUT_DIR /
    "02_model_performance.csv",
    index=False
)


# ================================================================
# 5. INCREMENTAL IMPROVEMENT
# ================================================================

incremental_file = (
    MODEL_DIR /
    "incremental_improvement.csv"
)

incremental = pd.read_csv(
    incremental_file
)

incremental.to_csv(
    OUTPUT_DIR /
    "03_incremental_improvement.csv",
    index=False
)


# ================================================================
# 6. STATISTICAL ROBUSTNESS
# ================================================================

basic_metrics = pd.read_csv(
    ROBUSTNESS_DIR /
    "basic_metrics.csv"
)

bootstrap = pd.read_csv(
    ROBUSTNESS_DIR /
    "bootstrap_mae_confidence_intervals.csv"
)

paired = pd.read_csv(
    ROBUSTNESS_DIR /
    "paired_mae_differences.csv"
)

wilcoxon = pd.read_csv(
    ROBUSTNESS_DIR /
    "wilcoxon_signed_rank_tests.csv"
)

direction = pd.read_csv(
    ROBUSTNESS_DIR /
    "paired_error_direction.csv"
)

generalization = pd.read_csv(
    ROBUSTNESS_DIR /
    "generalization_gaps.csv"
)


basic_metrics.to_csv(
    OUTPUT_DIR /
    "04_basic_test_metrics.csv",
    index=False
)

bootstrap.to_csv(
    OUTPUT_DIR /
    "05_bootstrap_confidence_intervals.csv",
    index=False
)

paired.to_csv(
    OUTPUT_DIR /
    "06_paired_mae_comparisons.csv",
    index=False
)

wilcoxon.to_csv(
    OUTPUT_DIR /
    "07_wilcoxon_tests.csv",
    index=False
)

direction.to_csv(
    OUTPUT_DIR /
    "08_error_direction.csv",
    index=False
)

generalization.to_csv(
    OUTPUT_DIR /
    "09_generalization_gaps.csv",
    index=False
)


# ================================================================
# 7. FEATURE IMPORTANCE
# ================================================================

feature_importance_C = pd.read_csv(
    MODEL_DIR /
    "model_C_feature_importance.csv"
)

feature_importance_C = (
    feature_importance_C
    .sort_values(
        "importance",
        ascending=False
    )
)

feature_importance_C.to_csv(
    OUTPUT_DIR /
    "10_model_C_feature_importance.csv",
    index=False
)


# ================================================================
# 8. ERROR ANALYSIS
# ================================================================

overall_error = pd.read_csv(
    ANALYSIS_DIR /
    "overall_error_summary.csv"
)

position_error = pd.read_csv(
    ANALYSIS_DIR /
    "error_by_position.csv"
)

performance_error = pd.read_csv(
    ANALYSIS_DIR /
    "error_by_performance_band.csv"
)

age_error = pd.read_csv(
    ANALYSIS_DIR /
    "error_by_age_group.csv"
)

under_over = pd.read_csv(
    ANALYSIS_DIR /
    "under_over_prediction.csv"
)

largest_errors = pd.read_csv(
    ANALYSIS_DIR /
    "largest_model_C_errors.csv"
)

win_counts = pd.read_csv(
    ANALYSIS_DIR /
    "model_win_counts.csv"
)


overall_error.to_csv(
    OUTPUT_DIR /
    "11_overall_error_analysis.csv",
    index=False
)

position_error.to_csv(
    OUTPUT_DIR /
    "12_error_by_position.csv",
    index=False
)

performance_error.to_csv(
    OUTPUT_DIR /
    "13_error_by_performance_band.csv",
    index=False
)

age_error.to_csv(
    OUTPUT_DIR /
    "14_error_by_age_group.csv",
    index=False
)

under_over.to_csv(
    OUTPUT_DIR /
    "15_under_over_prediction.csv",
    index=False
)

largest_errors.to_csv(
    OUTPUT_DIR /
    "16_largest_model_C_errors.csv",
    index=False
)

win_counts.to_csv(
    OUTPUT_DIR /
    "17_model_win_counts.csv",
    index=False
)


# ================================================================
# 9. FORMATTED MAIN RESULTS TABLE
# ================================================================

main_results = model_performance[
    [
        "model",
        "test_MAE",
        "test_RMSE",
        "test_R2"
    ]
].copy()


main_results.columns = [
    "Model",
    "Test MAE",
    "Test RMSE",
    "Test R²"
]


main_results[
    "Test MAE"
] = main_results[
    "Test MAE"
].map(
    lambda x: f"{x:.4f}"
)

main_results[
    "Test RMSE"
] = main_results[
    "Test RMSE"
].map(
    lambda x: f"{x:.4f}"
)

main_results[
    "Test R²"
] = main_results[
    "Test R²"
].map(
    lambda x: f"{x:.4f}"
)


main_results.to_csv(
    OUTPUT_DIR /
    "18_main_results_table.csv",
    index=False
)


# ================================================================
# 10. FORMATTED STATISTICAL COMPARISON
# ================================================================

statistical_table = paired.copy()


statistical_table[
    "Comparison"
] = (
    statistical_table[
        "comparison"
    ]
    .str.replace(
        "_vs_",
        " vs "
    )
)


statistical_table[
    "MAE (Model 1)"
] = (
    statistical_table[
        "MAE_model_1"
    ]
    .map(
        lambda x: f"{x:.4f}"
    )
)


statistical_table[
    "MAE (Model 2)"
] = (
    statistical_table[
        "MAE_model_2"
    ]
    .map(
        lambda x: f"{x:.4f}"
    )
)


statistical_table[
    "Mean MAE Difference"
] = (
    statistical_table[
        "MAE_difference_model1_minus_model2"
    ]
    .map(
        lambda x: f"{x:+.4f}"
    )
)


statistical_table[
    "95% Bootstrap CI"
] = statistical_table.apply(
    lambda row:
    f"[{row['Bootstrap_95CI_Lower']:.4f}, "
    f"{row['Bootstrap_95CI_Upper']:.4f}]",
    axis=1
)


statistical_table[
    "MAE Change"
] = (
    statistical_table[
        "MAE_percent_change"
    ]
    .map(
        lambda x: f"{x:+.2f}%"
    )
)


statistical_table[
    "CI Excludes Zero"
] = (
    statistical_table[
        "CI_excludes_zero"
    ]
    .map(
        {
            True: "Yes",
            False: "No"
        }
    )
)


statistical_table = statistical_table[
    [
        "Comparison",
        "model_1",
        "model_2",
        "MAE (Model 1)",
        "MAE (Model 2)",
        "Mean MAE Difference",
        "95% Bootstrap CI",
        "MAE Change",
        "CI Excludes Zero"
    ]
]


statistical_table.columns = [
    "Comparison",
    "Model 1",
    "Model 2",
    "MAE (Model 1)",
    "MAE (Model 2)",
    "Mean MAE Difference",
    "95% Bootstrap CI",
    "MAE Change",
    "CI Excludes Zero"
]


statistical_table.to_csv(
    OUTPUT_DIR /
    "19_statistical_comparison_table.csv",
    index=False
)


# ================================================================
# 11. FINAL TEXT SUMMARY
# ================================================================

A = model_performance[
    model_performance["model"] == "A"
].iloc[0]

B = model_performance[
    model_performance["model"] == "B"
].iloc[0]

C = model_performance[
    model_performance["model"] == "C"
].iloc[0]


AB = paired[
    paired["comparison"] == "A_vs_B"
].iloc[0]

BC = paired[
    paired["comparison"] == "B_vs_C"
].iloc[0]

AC = paired[
    paired["comparison"] == "A_vs_C"
].iloc[0]


AB_W = wilcoxon[
    wilcoxon["comparison"] == "A_vs_B"
].iloc[0]

BC_W = wilcoxon[
    wilcoxon["comparison"] == "B_vs_C"
].iloc[0]

AC_W = wilcoxon[
    wilcoxon["comparison"] == "A_vs_C"
].iloc[0]


summary_text = f"""
============================================================
FINAL RESEARCH RESULTS SUMMARY
============================================================

DATASET
------------------------------------------------------------
Final cohort: {len(df):,} transfers
Unique players: {df["player_id"].nunique():,}
Unique destination clubs: {df["to_club_id"].nunique():,}

Transfer date range:
{df["transfer_date"].min()} to {df["transfer_date"].max()}

Target:
Post-transfer G+A/90

Target mean: {df["post_GA90"].mean():.6f}
Target median: {df["post_GA90"].median():.6f}
Target standard deviation: {df["post_GA90"].std():.6f}
Target minimum: {df["post_GA90"].min():.6f}
Target maximum: {df["post_GA90"].max():.6f}


============================================================
TEST SET
============================================================

Common test records: 1,194


============================================================
MODEL PERFORMANCE
============================================================

MODEL A — PLAYER CHARACTERISTICS

Test MAE:  {A["test_MAE"]:.6f}
Test RMSE: {A["test_RMSE"]:.6f}
Test R²:   {A["test_R2"]:.6f}


MODEL B — PLAYER + BASIC TEAM CONTEXT

Test MAE:  {B["test_MAE"]:.6f}
Test RMSE: {B["test_RMSE"]:.6f}
Test R²:   {B["test_R2"]:.6f}


MODEL C — PLAYER + RICH CONTEXT

Test MAE:  {C["test_MAE"]:.6f}
Test RMSE: {C["test_RMSE"]:.6f}
Test R²:   {C["test_R2"]:.6f}


============================================================
INCREMENTAL PERFORMANCE
============================================================

A -> B

MAE change:
{AB["MAE_percent_change"]:.4f}%

Mean paired MAE difference:
{AB["MAE_difference_model1_minus_model2"]:.6f}

Bootstrap 95% CI:
[{AB["Bootstrap_95CI_Lower"]:.6f},
 {AB["Bootstrap_95CI_Upper"]:.6f}]

Wilcoxon p-value:
{AB_W["p_value"]:.6f}


B -> C

MAE change:
{BC["MAE_percent_change"]:.4f}%

Mean paired MAE difference:
{BC["MAE_difference_model1_minus_model2"]:.6f}

Bootstrap 95% CI:
[{BC["Bootstrap_95CI_Lower"]:.6f},
 {BC["Bootstrap_95CI_Upper"]:.6f}]

Wilcoxon p-value:
{BC_W["p_value"]:.6f}


A -> C

MAE change:
{AC["MAE_percent_change"]:.4f}%

Mean paired MAE difference:
{AC["MAE_difference_model1_minus_model2"]:.6f}

Bootstrap 95% CI:
[{AC["Bootstrap_95CI_Lower"]:.6f},
 {AC["Bootstrap_95CI_Upper"]:.6f}]

Wilcoxon p-value:
{AC_W["p_value"]:.6f}


============================================================
GENERALIZATION
============================================================

Model A train-test R² gap:
{A["train_R2"] - A["test_R2"]:.6f}

Model B train-test R² gap:
{B["train_R2"] - B["test_R2"]:.6f}

Model C train-test R² gap:
{C["train_R2"] - C["test_R2"]:.6f}


============================================================
PRIMARY FINDING
============================================================

Adding basic destination-team context from Model A to Model B
reduced test MAE by approximately
{abs(AB["MAE_percent_change"]):.2f}% and increased test R² from
{A["test_R2"]:.4f} to {B["test_R2"]:.4f}.

The paired bootstrap confidence interval for the A -> B
MAE difference did not include zero, and the Wilcoxon
signed-rank test produced p = {AB_W["p_value"]:.6f}.

Adding the richer contextual features in Model C did not
produce a statistically detectable reduction in MAE compared
with Model B.

The B -> C bootstrap confidence interval included zero and
the Wilcoxon test produced p = {BC_W["p_value"]:.6f}.

Model C achieved higher training R² than Model B, but its
test R² was slightly lower than Model B. This indicates that
the additional contextual representation increased training
fit without producing an additional out-of-sample performance
gain in the present experiment.


============================================================
IMPORTANT INTERPRETATION
============================================================

The results support an incremental-value interpretation:

Player characteristics provide substantial predictive
information.

Adding basic destination-team context provides a measurable
improvement in out-of-sample prediction.

Adding the richer contextual feature set does not provide
additional aggregate predictive improvement beyond the basic
team-context model in this dataset and experimental setup.

These results should not be interpreted as evidence that
richer contextual factors are universally unimportant.
Their usefulness may depend on feature representation,
coverage, data quality, and model specification.


============================================================
END OF SUMMARY
============================================================
"""


summary_file = (
    OUTPUT_DIR /
    "20_FINAL_RESULTS_SUMMARY.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        summary_text
    )


# ================================================================
# 12. CREATE MASTER EXCEL-LIKE CSV INDEX
# ================================================================

file_index = []

for file in sorted(
    OUTPUT_DIR.iterdir()
):

    if file.is_file():

        file_index.append({

            "file": file.name,

            "type": file.suffix,

            "description":
                "Final consolidated research result"

        })


pd.DataFrame(
    file_index
).to_csv(
    OUTPUT_DIR /
    "00_results_file_index.csv",
    index=False
)


# ================================================================
# 13. PRINT FINAL SUMMARY
# ================================================================

print("\n")
print("=" * 90)
print("MAIN TEST RESULTS")
print("=" * 90)

print(
    main_results.to_string(
        index=False
    )
)


print("\n")
print("=" * 90)
print("STATISTICAL COMPARISONS")
print("=" * 90)

print(
    statistical_table.to_string(
        index=False
    )
)


print("\n")
print("=" * 90)
print("FINAL RESEARCH CONCLUSION")
print("=" * 90)

print(
    f"""
A -> B:
MAE improvement = {abs(AB["MAE_percent_change"]):.2f}%
Bootstrap CI =
[{AB["Bootstrap_95CI_Lower"]:.4f},
 {AB["Bootstrap_95CI_Upper"]:.4f}]
Wilcoxon p = {AB_W["p_value"]:.6f}

B -> C:
MAE change = {BC["MAE_percent_change"]:+.2f}%
Bootstrap CI =
[{BC["Bootstrap_95CI_Lower"]:.4f},
 {BC["Bootstrap_95CI_Upper"]:.4f}]
Wilcoxon p = {BC_W["p_value"]:.6f}
"""
)


print("\n")
print("=" * 90)
print("OUTPUT DIRECTORY")
print("=" * 90)

print(
    OUTPUT_DIR
)


print("\nFiles created:")

for file in sorted(
    OUTPUT_DIR.iterdir()
):

    if file.is_file():

        print(
            " -",
            file.name
        )


print("\n")
print("=" * 90)
print("FINAL RESULTS CONSOLIDATION COMPLETE")
print("=" * 90)

# ================================================================
# 06_statistical_robustness.py
# Statistical Robustness Analysis for Models A, B and C
# ================================================================

import pandas as pd
import numpy as np
from pathlib import Path

# Optional statistical test
try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ================================================================
# 1. PATHS
# ================================================================

BASE = Path(r"C:\Users\rsuba\Downloads\R football analytics")

ANALYSIS_DIR = BASE / "model_analysis_results"
OUTPUT_DIR = BASE / "statistical_robustness_results"

OUTPUT_DIR.mkdir(exist_ok=True)


# ================================================================
# 2. SETTINGS
# ================================================================

RANDOM_SEED = 42

N_BOOTSTRAP = 5000

np.random.seed(RANDOM_SEED)


# ================================================================
# 3. LOAD TEST ERROR DATA
# ================================================================

print("=" * 80)
print("STATISTICAL ROBUSTNESS ANALYSIS")
print("=" * 80)

error_file = ANALYSIS_DIR / "ABC_test_error_analysis.csv"

df = pd.read_csv(error_file)

print("\nLoaded:", error_file)
print("Common test records:", len(df))


# ================================================================
# 4. VERIFY REQUIRED COLUMNS
# ================================================================

required_columns = [
    "post_GA90",
    "pred_A",
    "pred_B",
    "pred_C"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:

    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ================================================================
# 5. CALCULATE ERRORS
# ================================================================

df["error_A"] = df["pred_A"] - df["post_GA90"]

df["error_B"] = df["pred_B"] - df["post_GA90"]

df["error_C"] = df["pred_C"] - df["post_GA90"]


df["abs_error_A"] = np.abs(
    df["error_A"]
)

df["abs_error_B"] = np.abs(
    df["error_B"]
)

df["abs_error_C"] = np.abs(
    df["error_C"]
)


df["sq_error_A"] = (
    df["error_A"] ** 2
)

df["sq_error_B"] = (
    df["error_B"] ** 2
)

df["sq_error_C"] = (
    df["error_C"] ** 2
)


# ================================================================
# 6. BASIC METRICS
# ================================================================

def calculate_metrics(errors):

    errors = np.asarray(errors)

    return {
        "MAE": np.mean(np.abs(errors)),
        "RMSE": np.sqrt(
            np.mean(errors ** 2)
        ),
        "Median_Absolute_Error": np.median(
            np.abs(errors)
        ),
        "Mean_Signed_Error": np.mean(
            errors
        )
    }


metrics = []

for model in ["A", "B", "C"]:

    values = calculate_metrics(
        df[f"error_{model}"]
    )

    values["model"] = model

    metrics.append(values)


metrics_df = pd.DataFrame(metrics)

metrics_df = metrics_df[
    [
        "model",
        "MAE",
        "RMSE",
        "Median_Absolute_Error",
        "Mean_Signed_Error"
    ]
]


# ================================================================
# 7. BOOTSTRAP CONFIDENCE INTERVAL
# ================================================================

def bootstrap_mae_ci(
    errors,
    n_bootstrap=5000,
    confidence=0.95
):

    errors = np.asarray(errors)

    n = len(errors)

    bootstrap_maes = np.empty(
        n_bootstrap
    )

    for i in range(n_bootstrap):

        sample = np.random.choice(
            errors,
            size=n,
            replace=True
        )

        bootstrap_maes[i] = np.mean(
            np.abs(sample)
        )

    alpha = 1 - confidence

    lower = np.percentile(
        bootstrap_maes,
        100 * alpha / 2
    )

    upper = np.percentile(
        bootstrap_maes,
        100 * (1 - alpha / 2)
    )

    return (
        np.mean(np.abs(errors)),
        lower,
        upper
    )


# ================================================================
# 8. BOOTSTRAP RESULTS
# ================================================================

bootstrap_results = []

for model in ["A", "B", "C"]:

    mean_mae, lower, upper = (
        bootstrap_mae_ci(
            df[f"error_{model}"],
            N_BOOTSTRAP
        )
    )

    bootstrap_results.append({

        "model": model,

        "MAE": mean_mae,

        "Bootstrap_95CI_Lower": lower,

        "Bootstrap_95CI_Upper": upper,

        "Bootstrap_Width":
            upper - lower

    })


bootstrap_df = pd.DataFrame(
    bootstrap_results
)


# ================================================================
# 9. PAIRED MAE DIFFERENCES
# ================================================================

# Positive value = second model has lower error
#
# A -> B:
# abs_error_A - abs_error_B
#
# B -> C:
# abs_error_B - abs_error_C
#
# A -> C:
# abs_error_A - abs_error_C


df["A_minus_B_abs_error"] = (
    df["abs_error_A"]
    -
    df["abs_error_B"]
)

df["B_minus_C_abs_error"] = (
    df["abs_error_B"]
    -
    df["abs_error_C"]
)

df["A_minus_C_abs_error"] = (
    df["abs_error_A"]
    -
    df["abs_error_C"]
)


# ================================================================
# 10. PAIRED DIFFERENCE BOOTSTRAP
# ================================================================

def bootstrap_difference_ci(
    differences,
    n_bootstrap=5000,
    confidence=0.95
):

    differences = np.asarray(
        differences
    )

    n = len(differences)

    bootstrap_means = np.empty(
        n_bootstrap
    )

    for i in range(n_bootstrap):

        sample = np.random.choice(
            differences,
            size=n,
            replace=True
        )

        bootstrap_means[i] = np.mean(
            sample
        )

    alpha = 1 - confidence

    lower = np.percentile(
        bootstrap_means,
        100 * alpha / 2
    )

    upper = np.percentile(
        bootstrap_means,
        100 * (1 - alpha / 2)
    )

    return (
        np.mean(differences),
        lower,
        upper
    )


comparisons = [
    (
        "A_vs_B",
        "A",
        "B",
        "A_minus_B_abs_error"
    ),

    (
        "B_vs_C",
        "B",
        "C",
        "B_minus_C_abs_error"
    ),

    (
        "A_vs_C",
        "A",
        "C",
        "A_minus_C_abs_error"
    )
]


difference_results = []


for name, model1, model2, column in comparisons:

    difference = df[column]

    mean_difference, lower, upper = (
        bootstrap_difference_ci(
            difference,
            N_BOOTSTRAP
        )
    )

    mae1 = df[
        f"abs_error_{model1}"
    ].mean()

    mae2 = df[
        f"abs_error_{model2}"
    ].mean()

    percent_change = (
        (mae2 - mae1)
        /
        mae1
        *
        100
    )

    difference_results.append({

        "comparison": name,

        "model_1": model1,

        "model_2": model2,

        "MAE_model_1": mae1,

        "MAE_model_2": mae2,

        "MAE_difference_model1_minus_model2":
            mean_difference,

        "Bootstrap_95CI_Lower":
            lower,

        "Bootstrap_95CI_Upper":
            upper,

        "MAE_percent_change":
            percent_change,

        "CI_excludes_zero":
            not (
                lower <= 0 <= upper
            )

    })


difference_df = pd.DataFrame(
    difference_results
)


# ================================================================
# 11. WILCOXON SIGNED-RANK TEST
# ================================================================

wilcoxon_results = []


def run_wilcoxon(
    values1,
    values2
):

    if not SCIPY_AVAILABLE:

        return np.nan, np.nan

    statistic, p_value = wilcoxon(
        values1,
        values2,
        zero_method="wilcox",
        alternative="two-sided"
    )

    return statistic, p_value


for name, model1, model2, _ in comparisons:

    values1 = df[
        f"abs_error_{model1}"
    ].values

    values2 = df[
        f"abs_error_{model2}"
    ].values

    statistic, p_value = (
        run_wilcoxon(
            values1,
            values2
        )
    )

    wilcoxon_results.append({

        "comparison": name,

        "model_1": model1,

        "model_2": model2,

        "Wilcoxon_statistic":
            statistic,

        "p_value":
            p_value,

        "significant_at_0.05":
            (
                p_value < 0.05
                if not np.isnan(p_value)
                else np.nan
            )

    })


wilcoxon_df = pd.DataFrame(
    wilcoxon_results
)


# ================================================================
# 12. EFFECT DIRECTION
# ================================================================

direction_results = []


for name, model1, model2, column in comparisons:

    differences = df[column]

    improved = (
        differences > 0
    ).sum()

    worsened = (
        differences < 0
    ).sum()

    tied = (
        differences == 0
    ).sum()

    total = len(differences)

    direction_results.append({

        "comparison": name,

        "model_1_better_cases":
            improved,

        "model_2_better_cases":
            worsened,

        "ties":
            tied,

        "model_1_better_percent":
            improved / total * 100,

        "model_2_better_percent":
            worsened / total * 100

    })


direction_df = pd.DataFrame(
    direction_results
)


# ================================================================
# 13. GENERALIZATION GAP
# ================================================================

# These are taken from the known final model results.

generalization_data = pd.DataFrame({

    "model": [
        "A",
        "B",
        "C"
    ],

    "train_R2": [
        0.667883,
        0.688827,
        0.702842
    ],

    "validation_R2": [
        0.567797,
        0.586790,
        0.590468
    ],

    "test_R2": [
        0.551043,
        0.568156,
        0.564382
    ]

})


generalization_data[
    "train_test_R2_gap"
] = (
    generalization_data["train_R2"]
    -
    generalization_data["test_R2"]
)


generalization_data[
    "validation_test_R2_gap"
] = (
    generalization_data["validation_R2"]
    -
    generalization_data["test_R2"]
)


# ================================================================
# 14. SAVE RESULTS
# ================================================================

metrics_df.to_csv(
    OUTPUT_DIR /
    "basic_metrics.csv",
    index=False
)


bootstrap_df.to_csv(
    OUTPUT_DIR /
    "bootstrap_mae_confidence_intervals.csv",
    index=False
)


difference_df.to_csv(
    OUTPUT_DIR /
    "paired_mae_differences.csv",
    index=False
)


wilcoxon_df.to_csv(
    OUTPUT_DIR /
    "wilcoxon_signed_rank_tests.csv",
    index=False
)


direction_df.to_csv(
    OUTPUT_DIR /
    "paired_error_direction.csv",
    index=False
)


generalization_data.to_csv(
    OUTPUT_DIR /
    "generalization_gaps.csv",
    index=False
)


# ================================================================
# 15. PRINT RESULTS
# ================================================================

print("\n")
print("=" * 80)
print("BASIC TEST METRICS")
print("=" * 80)

print(
    metrics_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 80)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS FOR MAE")
print("=" * 80)

print(
    bootstrap_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 80)
print("PAIRED MAE DIFFERENCES")
print("=" * 80)

print(
    difference_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 80)
print("WILCOXON SIGNED-RANK TEST")
print("=" * 80)

if not SCIPY_AVAILABLE:

    print(
        "SciPy is not installed."
    )

    print(
        "Install using: pip install scipy"
    )

else:

    print(
        wilcoxon_df.to_string(
            index=False
        )
    )


print("\n")
print("=" * 80)
print("ERROR DIRECTION")
print("=" * 80)

print(
    direction_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 80)
print("GENERALIZATION GAPS")
print("=" * 80)

print(
    generalization_data.to_string(
        index=False
    )
)


# ================================================================
# 16. INTERPRETATION
# ================================================================

print("\n")
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)


ab_row = difference_df[
    difference_df["comparison"] == "A_vs_B"
].iloc[0]

bc_row = difference_df[
    difference_df["comparison"] == "B_vs_C"
].iloc[0]


print("\nA -> B:")

print(
    f"MAE change: "
    f"{ab_row['MAE_percent_change']:.2f}%"
)

print(
    f"Mean paired MAE difference: "
    f"{ab_row['MAE_difference_model1_minus_model2']:.6f}"
)

print(
    f"95% CI: "
    f"[{ab_row['Bootstrap_95CI_Lower']:.6f}, "
    f"{ab_row['Bootstrap_95CI_Upper']:.6f}]"
)


print("\nB -> C:")

print(
    f"MAE change: "
    f"{bc_row['MAE_percent_change']:.2f}%"
)

print(
    f"Mean paired MAE difference: "
    f"{bc_row['MAE_difference_model1_minus_model2']:.6f}"
)

print(
    f"95% CI: "
    f"[{bc_row['Bootstrap_95CI_Lower']:.6f}, "
    f"{bc_row['Bootstrap_95CI_Upper']:.6f}]"
)


print("\n")
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

for file in sorted(
    OUTPUT_DIR.glob("*.csv")
):

    print(
        " -",
        file.name
    )

# ================================================================
# 10A. FORMATTED PAIRED MAE DIFFERENCES
# ================================================================

print("\n")
print("=" * 100)
print("PAIRED MAE DIFFERENCES")
print("=" * 100)

formatted_difference = difference_df.copy()

# Round numerical values for clean display
formatted_difference["MAE_model_1"] = (
    formatted_difference["MAE_model_1"].round(4)
)

formatted_difference["MAE_model_2"] = (
    formatted_difference["MAE_model_2"].round(4)
)

formatted_difference[
    "MAE_difference_model1_minus_model2"
] = (
    formatted_difference[
        "MAE_difference_model1_minus_model2"
    ].round(4)
)

# Format confidence interval as one readable column
formatted_difference["95% Bootstrap CI"] = (
    formatted_difference.apply(
        lambda row:
        f"[{row['Bootstrap_95CI_Lower']:.4f}, "
        f"{row['Bootstrap_95CI_Upper']:.4f}]",
        axis=1
    )
)

# Format percentage change
formatted_difference["MAE Change"] = (
    formatted_difference["MAE_percent_change"]
    .map(lambda x: f"{x:+.2f}%")
)

# Format comparison names
formatted_difference["Comparison"] = (
    formatted_difference["comparison"]
    .str.replace("_vs_", " vs ")
)

# Create clean display table
display_difference = formatted_difference[
    [
        "Comparison",
        "model_1",
        "model_2",
        "MAE_model_1",
        "MAE_model_2",
        "MAE_difference_model1_minus_model2",
        "95% Bootstrap CI",
        "MAE Change"
    ]
].copy()

# Rename columns
display_difference.columns = [
    "Comparison",
    "Model 1",
    "Model 2",
    "MAE (Model 1)",
    "MAE (Model 2)",
    "Mean MAE Difference",
    "95% Bootstrap CI",
    "MAE Change"
]

# Print clean table
print(
    display_difference.to_string(
        index=False
    )
)


# ================================================================
# 10B. CI SIGNIFICANCE SUMMARY
# ================================================================

print("\n")
print("=" * 100)
print("BOOTSTRAP CI SUMMARY")
print("=" * 100)

for _, row in formatted_difference.iterrows():

    status = (
        "Yes"
        if row["CI_excludes_zero"]
        else "No"
    )

    print(
        f"{row['Comparison']}: "
        f"CI excludes zero = {status}"
    )
print("\n")
print("=" * 80)
print("STATISTICAL ROBUSTNESS ANALYSIS COMPLETE")
print("=" * 80)
# ================================================================
# 05_sensitivity_analysis.py
# Post-Transfer Minutes Sensitivity Analysis
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ================================================================
# 1. PATHS
# ================================================================

BASE = Path(__file__).resolve().parents[2]

POST_FILE = BASE / "04_Datasets" / "derived" / "post_transfer_performance.csv"

FINAL_FILE = BASE / "04_Datasets" / "derived" / "FINAL_MODEL_C_CLEAN.csv"

OUTPUT_DIR = BASE / "06_Results_and_Demo_Evidence" / "audit_results" / "sensitivity_analysis_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ================================================================
# 2. SETTINGS
# ================================================================

MINUTE_THRESHOLDS = [
    450,
    600,
    900,
    1200,
    1500
]


# ================================================================
# 3. LOAD DATA
# ================================================================

print("=" * 80)
print("POST-TRANSFER PERFORMANCE SENSITIVITY ANALYSIS")
print("=" * 80)

post = pd.read_csv(POST_FILE)

final = pd.read_csv(FINAL_FILE)

print("\nPost-transfer dataset:")
print("Rows:", len(post))

print("\nFinal Model C dataset:")
print("Rows:", len(final))


# ================================================================
# 4. BASIC COLUMN CHECK
# ================================================================

required_columns = [
    "transfer_row_id",
    "post_minutes",
    "post_GA90"
]

missing = [
    col
    for col in required_columns
    if col not in post.columns
]

if missing:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing)
    )


# ================================================================
# 5. REMOVE INVALID TARGETS
# ================================================================

post = post.copy()

post["post_minutes"] = pd.to_numeric(
    post["post_minutes"],
    errors="coerce"
)

post["post_GA90"] = pd.to_numeric(
    post["post_GA90"],
    errors="coerce"
)

valid = post[
    post["post_minutes"].notna()
    &
    post["post_GA90"].notna()
].copy()

print("\nValid target records:", len(valid))


# ================================================================
# 6. SENSITIVITY TABLE
# ================================================================

results = []

for threshold in MINUTE_THRESHOLDS:

    subset = valid[
        valid["post_minutes"] >= threshold
    ].copy()

    target = subset["post_GA90"]

    if len(subset) == 0:
        continue

    results.append({

        "minimum_post_minutes": threshold,

        "n_players": len(subset),

        "retention_percent":
            len(subset) / len(valid) * 100,

        "mean_GA90":
            target.mean(),

        "median_GA90":
            target.median(),

        "std_GA90":
            target.std(),

        "min_GA90":
            target.min(),

        "q25_GA90":
            target.quantile(0.25),

        "q75_GA90":
            target.quantile(0.75),

        "max_GA90":
            target.max()
    })


sensitivity = pd.DataFrame(results)


# ================================================================
# 7. DISPLAY RESULTS
# ================================================================

print("\n")
print("=" * 80)
print("SENSITIVITY RESULTS")
print("=" * 80)

print(
    sensitivity.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ================================================================
# 8. COMPARE AGAINST PRIMARY COHORT
# ================================================================

primary = sensitivity[
    sensitivity["minimum_post_minutes"] == 450
]

if len(primary) > 0:

    primary_n = primary["n_players"].iloc[0]
    primary_median = primary["median_GA90"].iloc[0]

    sensitivity["n_change_vs_450"] = (
        sensitivity["n_players"]
        - primary_n
    )

    sensitivity["median_change_vs_450"] = (
        sensitivity["median_GA90"]
        - primary_median
    )

else:

    sensitivity["n_change_vs_450"] = np.nan

    sensitivity["median_change_vs_450"] = np.nan


# ================================================================
# 9. SAVE TABLE
# ================================================================

table_file = (
    OUTPUT_DIR /
    "post_minutes_sensitivity.csv"
)

sensitivity.to_csv(
    table_file,
    index=False
)

print("\nSaved:")
print(table_file)


# ================================================================
# FIGURE 1
# COHORT SIZE VS MINIMUM MINUTES
# ================================================================

fig = plt.figure(figsize=(8, 5))

plt.plot(
    sensitivity["minimum_post_minutes"],
    sensitivity["n_players"],
    marker="o"
)

plt.xlabel(
    "Minimum post-transfer minutes"
)

plt.ylabel(
    "Number of eligible players"
)

plt.title(
    "Sensitivity of Cohort Size to Post-Transfer Minutes"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR /
    "figure_1_cohort_size_sensitivity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close(fig)


# ================================================================
# FIGURE 2
# MEDIAN TARGET VS MINIMUM MINUTES
# ================================================================

fig = plt.figure(figsize=(8, 5))

plt.plot(
    sensitivity["minimum_post_minutes"],
    sensitivity["median_GA90"],
    marker="o"
)

plt.xlabel(
    "Minimum post-transfer minutes"
)

plt.ylabel(
    "Median post-transfer G+A/90"
)

plt.title(
    "Sensitivity of Target Distribution to Minutes Threshold"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR /
    "figure_2_median_target_sensitivity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close(fig)


# ================================================================
# FIGURE 3
# MEAN TARGET VS MINIMUM MINUTES
# ================================================================

fig = plt.figure(figsize=(8, 5))

plt.plot(
    sensitivity["minimum_post_minutes"],
    sensitivity["mean_GA90"],
    marker="o"
)

plt.xlabel(
    "Minimum post-transfer minutes"
)

plt.ylabel(
    "Mean post-transfer G+A/90"
)

plt.title(
    "Mean Post-Transfer Performance Across Cohorts"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR /
    "figure_3_mean_target_sensitivity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close(fig)


# ================================================================
# FIGURE 4
# TARGET DISPERSION
# ================================================================

fig = plt.figure(figsize=(8, 5))

plt.plot(
    sensitivity["minimum_post_minutes"],
    sensitivity["std_GA90"],
    marker="o"
)

plt.xlabel(
    "Minimum post-transfer minutes"
)

plt.ylabel(
    "Standard deviation of G+A/90"
)

plt.title(
    "Target Variability Across Minutes Thresholds"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR /
    "figure_4_target_variability_sensitivity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close(fig)


# ================================================================
# 10. PRIMARY COHORT CHECK
# ================================================================

print("\n")
print("=" * 80)
print("PRIMARY COHORT CHECK")
print("=" * 80)

primary_final = final.copy()

print(
    "\nFinal Model C cohort:",
    len(primary_final)
)

print(
    "Minimum post-transfer minutes in final cohort:",
    primary_final["post_minutes"].min()
)

print(
    "Median post-transfer minutes:",
    primary_final["post_minutes"].median()
)

print(
    "Maximum post-transfer minutes:",
    primary_final["post_minutes"].max()
)


# ================================================================
# 11. SUMMARY TEXT
# ================================================================

summary_lines = []

summary_lines.append(
    "Post-Transfer Minutes Sensitivity Analysis"
)

summary_lines.append(
    "============================================================"
)

summary_lines.append(
    "Primary threshold: 450 post-transfer minutes"
)

summary_lines.append(
    f"Primary cohort size: {int(primary_n) if len(primary) else 'N/A'}"
)

summary_lines.append("")

for _, row in sensitivity.iterrows():

    summary_lines.append(
        f"Threshold {int(row['minimum_post_minutes'])} min: "
        f"n={int(row['n_players'])}, "
        f"median G+A/90={row['median_GA90']:.4f}, "
        f"mean={row['mean_GA90']:.4f}, "
        f"std={row['std_GA90']:.4f}"
    )

summary_lines.append("")

summary_lines.append(
    "Interpretation:"
)

summary_lines.append(
    "Increasing the minimum post-transfer minutes threshold "
    "reduces the cohort size while selecting players with "
    "greater observed playing exposure."
)

summary_lines.append(
    "The 450-minute threshold remains the primary cohort "
    "because it provides the largest eligible sample among "
    "the tested thresholds while reducing extremely low-"
    "exposure observations."
)

summary_lines.append(
    "Higher thresholds are treated as sensitivity cohorts "
    "rather than replacements for the primary cohort."
)

summary_file = (
    OUTPUT_DIR /
    "sensitivity_analysis_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(summary_lines)
    )


# ================================================================
# 12. FINAL OUTPUT
# ================================================================

print("\n")
print("=" * 80)
print("SENSITIVITY ANALYSIS COMPLETE")
print("=" * 80)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nFiles generated:")

for file in sorted(OUTPUT_DIR.iterdir()):

    print(
        " -",
        file.name
    )

print("\nDone.")

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "04_Datasets" / "derived"

INPUT = DATA_DIR / "FINAL_MODEL_C_6108_RECORD_AUDIT.csv"

CLEAN_OUTPUT = DATA_DIR / "FINAL_MODEL_C_CLEAN.csv"
REMOVED_OUTPUT = DATA_DIR / "FINAL_MODEL_C_REMOVED.csv"
SUMMARY_OUTPUT = DATA_DIR / "FINAL_MODEL_C_CLEANING_SUMMARY.csv"


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("FINAL MODEL C DATA CLEANING")
print("=" * 70)

df = pd.read_csv(INPUT)

original_count = len(df)

print(f"\nOriginal records: {original_count:,}")
print(f"Original columns: {len(df.columns):,}")


# ============================================================
# PREPARE TYPES
# ============================================================

df["transfer_date"] = pd.to_datetime(
    df["transfer_date"],
    errors="coerce"
)

team_matches = pd.to_numeric(
    df["rich_team_matches_last10"],
    errors="coerce"
)

post_minutes = pd.to_numeric(
    df["post_minutes"],
    errors="coerce"
)

post_ga90 = pd.to_numeric(
    df["post_GA90"],
    errors="coerce"
)


# ============================================================
# ELIGIBILITY CONDITIONS
# ============================================================

conditions = {}

# 1. Valid transfer ID
conditions["valid_transfer_id"] = (
    df["transfer_row_id"].notna()
)

# 2. No duplicate transfer ID
conditions["unique_transfer_id"] = (
    ~df["transfer_row_id"].duplicated(keep=False)
)

# 3. Valid transfer date
conditions["valid_transfer_date"] = (
    df["transfer_date"].notna()
)

# 4. Valid destination club
conditions["valid_destination_club"] = (
    df["to_club_id"].notna()
    &
    df["to_club_name"].notna()
)

# 5. Complete 365-day post-transfer window
conditions["complete_365_day_window"] = (
    df["complete_365_day_window"]
    .astype(str)
    .str.upper()
    .eq("TRUE")
)

# 6. >=450 post-transfer minutes
conditions["eligible_450min"] = (
    df["eligible_450min"]
    .astype(str)
    .str.upper()
    .eq("TRUE")
)

# 7. At least 10 previous destination-team matches
conditions["at_least_10_team_matches"] = (
    team_matches >= 10
)

# 8. Valid target
conditions["valid_target"] = (
    post_ga90.notna()
)


# ============================================================
# PRINT CONDITION COUNTS
# ============================================================

print("\nEligibility audit")
print("-" * 50)

for name, mask in conditions.items():
    print(f"{name:35s}: {mask.sum():,}")


# ============================================================
# COMBINE CONDITIONS
# ============================================================

clean_mask = pd.Series(
    True,
    index=df.index
)

for mask in conditions.values():
    clean_mask &= mask


clean = df[clean_mask].copy()
removed = df[~clean_mask].copy()


# ============================================================
# RECORD EXCLUSION REASONS
# ============================================================

def exclusion_reason(row):
    reasons = []

    if pd.isna(row["transfer_row_id"]):
        reasons.append("missing_transfer_id")

    if pd.isna(row["transfer_date"]):
        reasons.append("invalid_transfer_date")

    if pd.isna(row["to_club_id"]) or pd.isna(row["to_club_name"]):
        reasons.append("invalid_destination_club")

    if str(row["complete_365_day_window"]).upper() != "TRUE":
        reasons.append("incomplete_365_day_window")

    if str(row["eligible_450min"]).upper() != "TRUE":
        reasons.append("less_than_450_post_minutes")

    matches = pd.to_numeric(
        row["rich_team_matches_last10"],
        errors="coerce"
    )

    if pd.isna(matches):
        reasons.append("missing_team_match_history")
    elif matches < 10:
        reasons.append("fewer_than_10_team_matches")

    if pd.isna(row["post_GA90"]):
        reasons.append("missing_target")

    return "; ".join(reasons)


if len(removed) > 0:
    removed["exclusion_reason"] = removed.apply(
        exclusion_reason,
        axis=1
    )


# ============================================================
# IMPORTANT:
# Do NOT fill coach NaNs with zero
# ============================================================

print("\nCoach missing values")
print("-" * 50)

coach_cols = [
    "coach_position_group_minutes_share",
    "coach_position_group_start_rate",
    "coach_exact_position_minutes_share",
    "coach_exact_position_start_rate",
    "coach_position_competitor_count",
    "coach_position_competitor_minutes",
    "coach_position_rotation_hhi",
    "coach_formation_stability",
    "coach_observed_tenure_days"
]

for col in coach_cols:
    if col in clean.columns:
        print(
            f"{col:45s}: "
            f"{clean[col].isna().sum():,} missing"
        )


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

clean = clean.sort_values(
    ["transfer_date", "transfer_row_id"]
).reset_index(drop=True)


# ============================================================
# CREATE CHRONOLOGICAL 60 / 20 / 20 SPLIT
# ============================================================

n = len(clean)

train_end = int(n * 0.60)
validation_end = train_end + int(n * 0.20)

clean["dataset_split"] = "test"

clean.loc[
    :train_end - 1,
    "dataset_split"
] = "train"

clean.loc[
    train_end:validation_end - 1,
    "dataset_split"
] = "validation"


# ============================================================
# AUDIT FLAGS
# ============================================================

clean["clean_final_cohort"] = 1
clean["team_history_requirement"] = ">=10_previous_matches"


# ============================================================
# SAVE
# ============================================================

clean.to_csv(
    CLEAN_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

removed.to_csv(
    REMOVED_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# CLEANING SUMMARY
# ============================================================

summary = pd.DataFrame({
    "metric": [
        "original_records",
        "clean_records",
        "removed_records",
        "retention_percent",
        "duplicate_transfer_ids_clean",
        "missing_target_clean",
        "minimum_team_matches",
        "train_records",
        "validation_records",
        "test_records"
    ],
    "value": [
        original_count,
        len(clean),
        len(removed),
        round(len(clean) / original_count * 100, 2),
        clean["transfer_row_id"].duplicated().sum(),
        clean["post_GA90"].isna().sum(),
        team_matches.loc[clean.index].min()
        if len(clean) > 0 else None,
        (clean["dataset_split"] == "train").sum(),
        (clean["dataset_split"] == "validation").sum(),
        (clean["dataset_split"] == "test").sum()
    ]
})

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print(f"\nOriginal records:       {original_count:,}")
print(f"Clean records:          {len(clean):,}")
print(f"Removed records:        {len(removed):,}")
print(
    f"Retention:              "
    f"{len(clean) / original_count * 100:.2f}%"
)

print("\nTeam match history:")
print(
    clean["rich_team_matches_last10"]
    .value_counts()
    .sort_index()
)

print("\nDataset split:")
print(
    clean["dataset_split"]
    .value_counts()
)

print("\nDate range:")
print(f"First: {clean['transfer_date'].min().date()}")
print(f"Last:  {clean['transfer_date'].max().date()}")

print("\nDuplicate transfer IDs:")
print(
    clean["transfer_row_id"].duplicated().sum()
)

print("\nMissing target:")
print(
    clean["post_GA90"].isna().sum()
)

print("\nOutput files:")
print(CLEAN_OUTPUT)
print(REMOVED_OUTPUT)
print(SUMMARY_OUTPUT)

print("=" * 70)

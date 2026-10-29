import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "04_Datasets" / "derived"

TRANSFER_FILE = DATA_DIR / "professional_transfer_dataset.csv"
TARGET_FILE = DATA_DIR / "post_transfer_performance.csv"
TEAM_FILE = DATA_DIR / "rich_team_context_dataset.csv"
COACH_FILE = DATA_DIR / "coach_tactical_context_v21.csv"

OUTPUT_FILE = DATA_DIR / "FINAL_MODEL_C_6108_RECORD_AUDIT.csv"


def load_csv(path):
    print(f"Loading: {path.name}")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df):,}")
    return df


# ============================================================
# 1. LOAD DATA
# ============================================================

transfers = load_csv(TRANSFER_FILE)
target = load_csv(TARGET_FILE)
team = load_csv(TEAM_FILE)
coach = load_csv(COACH_FILE)


# ============================================================
# 2. ENSURE TRANSFER ROW ID
# ============================================================

if "transfer_row_id" not in transfers.columns:
    transfers = transfers.reset_index(drop=True)
    transfers["transfer_row_id"] = transfers.index + 1

if "transfer_row_id" not in target.columns:
    raise ValueError("post_transfer_performance.csv has no transfer_row_id")

if "transfer_row_id" not in team.columns:
    raise ValueError("rich_team_context_dataset.csv has no transfer_row_id")

if "transfer_row_id" not in coach.columns:
    raise ValueError("coach_tactical_context_v21.csv has no transfer_row_id")


# ============================================================
# 3. DEFINE FINAL MODEL C ELIGIBILITY
# ============================================================

eligible = target[
    (target["complete_365_day_window"] == True) &
    (target["eligible_450min"] == True)
].copy()

print("\nEligibility")
print("-----------")
print(f"Eligible records: {len(eligible):,}")


# ============================================================
# 4. FIND COMMON COHORT
# ============================================================

common_ids = set(eligible["transfer_row_id"])

common_ids &= set(transfers["transfer_row_id"])
common_ids &= set(team["transfer_row_id"])
common_ids &= set(coach["transfer_row_id"])

print(f"Common cohort: {len(common_ids):,}")


# ============================================================
# 5. START WITH TRANSFER DATA
# ============================================================

audit = transfers[
    transfers["transfer_row_id"].isin(common_ids)
].copy()


# ============================================================
# 6. MERGE POST-TRANSFER TARGET
# ============================================================

target_cols = [
    "transfer_row_id",
    "post_appearances",
    "post_minutes",
    "post_goals",
    "post_assists",
    "post_GA90",
    "complete_365_day_window",
    "eligible_450min",
]

target_keep = target[
    [c for c in target_cols if c in target.columns]
].copy()

audit = audit.merge(
    target_keep,
    on="transfer_row_id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 7. MERGE RICH TEAM CONTEXT
# ============================================================

team_keep = team[
    team["transfer_row_id"].isin(common_ids)
].copy()

# Avoid duplicate base columns if present
team_drop = [
    c for c in team_keep.columns
    if c in audit.columns and c != "transfer_row_id"
]

team_keep = team_keep.drop(columns=team_drop)

audit = audit.merge(
    team_keep,
    on="transfer_row_id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 8. MERGE COACH / TACTICAL CONTEXT
# ============================================================

coach_keep = coach[
    coach["transfer_row_id"].isin(common_ids)
].copy()

coach_drop = [
    c for c in coach_keep.columns
    if c in audit.columns and c != "transfer_row_id"
]

coach_keep = coach_keep.drop(columns=coach_drop)

audit = audit.merge(
    coach_keep,
    on="transfer_row_id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 9. FINAL AUDIT FLAGS
# ============================================================

audit["included_in_final_model_c"] = 1

audit["final_model_c_cohort"] = "complete_365d_and_450min"

# Chronological split exactly matching final Model C
audit = audit.sort_values("transfer_date").reset_index(drop=True)

n = len(audit)

train_end = int(n * 0.60)
val_end = train_end + int(n * 0.20)

audit["dataset_split"] = "test"
audit.loc[:train_end - 1, "dataset_split"] = "train"
audit.loc[train_end:val_end - 1, "dataset_split"] = "validation"


# ============================================================
# 10. BASIC DATA QUALITY FLAGS
# ============================================================

audit["has_target"] = audit["post_GA90"].notna().astype(int)
audit["has_destination_club"] = audit["to_club_id"].notna().astype(int)
audit["has_transfer_date"] = audit["transfer_date"].notna().astype(int)

if "to_club_name" in audit.columns:
    audit["destination_club_available"] = (
        audit["to_club_name"].notna()
        & (audit["to_club_name"].astype(str).str.strip() != "")
    ).astype(int)


# ============================================================
# 11. SAVE
# ============================================================

audit.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 12. FINAL VALIDATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL C RECORD AUDIT")
print("=" * 60)

print(f"Final records:       {len(audit):,}")
print(f"Unique transfer IDs: {audit['transfer_row_id'].nunique():,}")
print(f"Duplicate IDs:       {audit['transfer_row_id'].duplicated().sum():,}")

print("\nSplit:")
print(audit["dataset_split"].value_counts().to_string())

print("\nTarget:")
print(f"Target records:      {audit['has_target'].sum():,}")

print("\nDate range:")
print(f"First transfer:      {audit['transfer_date'].min()}")
print(f"Last transfer:       {audit['transfer_date'].max()}")

print("\nDestination clubs:")
print(f"Unique destination clubs: {audit['to_club_id'].nunique():,}")

print("\nColumns:")
print(f"Total columns:       {len(audit.columns):,}")

print("\nSaved to:")
print(OUTPUT_FILE)

print("=" * 60)

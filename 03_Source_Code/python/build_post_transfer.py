import pandas as pd
import os

# ============================================================
# FILE LOCATIONS
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "04_Datasets", "derived")
RAW_DIR = os.path.join(BASE_DIR, "04_Datasets", "raw_transfer_data", "archive")

TRANSFER_FILE = os.path.join(
    DATA_DIR,
    "professional_transfer_dataset.csv"
)

APPEARANCES_FILE = os.path.join(
    RAW_DIR,
    "appearances.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "post_transfer_performance.csv"
)

WINDOW_DAYS = 365


# ============================================================
# STEP 1: LOAD TRANSFER DATA
# ============================================================

print("=" * 60)
print("POST-TRANSFER PERFORMANCE DATASET BUILDER")
print("=" * 60)

print("\nLoading transfer dataset...")

transfers = pd.read_csv(
    TRANSFER_FILE,
    low_memory=False
)

print("Transfer records:", len(transfers))


# ============================================================
# STEP 2: CLEAN TRANSFER DATA
# ============================================================

print("\nPreparing transfer records...")

transfers["transfer_date"] = pd.to_datetime(
    transfers["transfer_date"],
    errors="coerce"
)

transfers["player_id"] = pd.to_numeric(
    transfers["player_id"],
    errors="coerce"
)

transfers["to_club_id"] = pd.to_numeric(
    transfers["to_club_id"],
    errors="coerce"
)

# Remove records without essential information
transfers = transfers.dropna(
    subset=[
        "player_id",
        "transfer_date",
        "to_club_id"
    ]
).copy()

# Keep IDs as integers
transfers["player_id"] = transfers["player_id"].astype(int)
transfers["to_club_id"] = transfers["to_club_id"].astype(int)


# ============================================================
# STEP 3: CREATE UNIQUE TRANSFER ROW ID
# ============================================================

# IMPORTANT:
# Use the actual row identity rather than only player/date/club.
# This prevents repeated-transfer records from being accidentally
# merged together.

transfers = transfers.reset_index(drop=True)

transfers["transfer_row_id"] = (
    transfers.index + 1
)

transfers["window_start"] = (
    transfers["transfer_date"]
    + pd.Timedelta(days=1)
)

transfers["window_end"] = (
    transfers["transfer_date"]
    + pd.Timedelta(days=WINDOW_DAYS)
)


# ============================================================
# STEP 4: LOAD APPEARANCES
# ============================================================

print("\nLoading appearances dataset...")
print("This may take a while...")

appearances = pd.read_csv(
    APPEARANCES_FILE,
    low_memory=False
)

print("Appearance records:", len(appearances))


# ============================================================
# STEP 5: PREPARE APPEARANCES
# ============================================================

print("\nPreparing appearances...")

appearances["date"] = pd.to_datetime(
    appearances["date"],
    errors="coerce"
)

appearances["player_id"] = pd.to_numeric(
    appearances["player_id"],
    errors="coerce"
)

appearances["player_club_id"] = pd.to_numeric(
    appearances["player_club_id"],
    errors="coerce"
)

for col in [
    "minutes_played",
    "goals",
    "assists"
]:
    appearances[col] = pd.to_numeric(
        appearances[col],
        errors="coerce"
    ).fillna(0)

appearances = appearances.dropna(
    subset=[
        "player_id",
        "player_club_id",
        "date"
    ]
).copy()

appearances["player_id"] = appearances["player_id"].astype(int)
appearances["player_club_id"] = appearances["player_club_id"].astype(int)


# ============================================================
# STEP 6: REDUCE APPEARANCES TO RELEVANT PLAYERS
# ============================================================

print("\nFiltering to players in transfer dataset...")

player_ids = set(
    transfers["player_id"].unique()
)

appearances = appearances[
    appearances["player_id"].isin(player_ids)
].copy()

print(
    "Relevant appearance records:",
    len(appearances)
)


# ============================================================
# STEP 7: MATCH APPEARANCES TO TRANSFERS
# ============================================================

print("\nMatching appearances to individual transfers...")

# We first match on player_id.
# The exact transfer is then determined using date + destination club.

merged = appearances.merge(
    transfers[
        [
            "transfer_row_id",
            "player_id",
            "transfer_date",
            "window_start",
            "window_end",
            "to_club_id"
        ]
    ],
    on="player_id",
    how="inner"
)

print(
    "Potential player-transfer matches:",
    len(merged)
)


# ============================================================
# STEP 8: KEEP ONLY VALID POST-TRANSFER APPEARANCES
# ============================================================

print("\nFiltering post-transfer appearances...")

merged = merged[
    (merged["player_club_id"] == merged["to_club_id"])
    &
    (merged["date"] >= merged["window_start"])
    &
    (merged["date"] <= merged["window_end"])
].copy()

print(
    "Valid post-transfer appearance records:",
    len(merged)
)


# ============================================================
# STEP 9: AGGREGATE PERFORMANCE
# ============================================================

print("\nAggregating player performance...")

post_stats = (
    merged
    .groupby("transfer_row_id")
    .agg(
        post_appearances=(
            "date",
            "nunique"
        ),
        post_minutes=(
            "minutes_played",
            "sum"
        ),
        post_goals=(
            "goals",
            "sum"
        ),
        post_assists=(
            "assists",
            "sum"
        )
    )
    .reset_index()
)


# ============================================================
# STEP 10: CALCULATE G+A/90
# ============================================================

post_stats["post_GA90"] = float("nan")

valid_minutes = (
    post_stats["post_minutes"] > 0
)

post_stats.loc[
    valid_minutes,
    "post_GA90"
] = (
    (
        post_stats.loc[
            valid_minutes,
            "post_goals"
        ]
        +
        post_stats.loc[
            valid_minutes,
            "post_assists"
        ]
    )
    /
    post_stats.loc[
        valid_minutes,
        "post_minutes"
    ]
    * 90
)


# ============================================================
# STEP 11: MERGE BACK TO ALL TRANSFERS
# ============================================================

print("\nCreating final dataset...")

result = transfers.merge(
    post_stats,
    on="transfer_row_id",
    how="left"
)


# Transfers with no recorded appearances
performance_columns = [
    "post_appearances",
    "post_minutes",
    "post_goals",
    "post_assists",
    "post_GA90"
]

for col in performance_columns:
    if col == "post_GA90":
        result.loc[result["post_minutes"] <= 0, col] = float("nan")
    else:
        result[col] = result[col].fillna(0)

# ============================================================
# STEP 12: DETERMINE COMPLETE 365-DAY WINDOW
# ============================================================

latest_appearance_date = appearances["date"].max()

result["complete_365_day_window"] = (
    result["window_end"]
    <= latest_appearance_date
)


# ============================================================
# STEP 13: 450-MINUTE ELIGIBILITY
# ============================================================

result["eligible_450min"] = (
    result["complete_365_day_window"]
    &
    (result["post_minutes"] >= 450)
)


# ============================================================
# STEP 14: TARGET QUALITY CHECKS
# ============================================================

print("\n" + "=" * 60)
print("QUALITY CHECKS")
print("=" * 60)

print(
    "\nDuplicate transfer_row_id:",
    result["transfer_row_id"].duplicated().sum()
)

print(
    "Missing transfer dates:",
    result["transfer_date"].isna().sum()
)

print(
    "Missing destination club IDs:",
    result["to_club_id"].isna().sum()
)

print(
    "Complete 365-day windows:",
    int(result["complete_365_day_window"].sum())
)

print(
    "Eligible >=450 post-transfer minutes:",
    int(result["eligible_450min"].sum())
)

eligible = result[
    result["eligible_450min"]
].copy()

print("\nEligible target statistics:")

if len(eligible) > 0:

    print(
        "Count:",
        len(eligible)
    )

    print(
        "Mean G+A/90:",
        round(
            eligible["post_GA90"].mean(),
            4
        )
    )

    print(
        "Median G+A/90:",
        round(
            eligible["post_GA90"].median(),
            4
        )
    )

    print(
        "Std G+A/90:",
        round(
            eligible["post_GA90"].std(),
            4
        )
    )

    print(
        "Min G+A/90:",
        round(
            eligible["post_GA90"].min(),
            4
        )
    )

    print(
        "Max G+A/90:",
        round(
            eligible["post_GA90"].max(),
            4
        )
    )


# ============================================================
# STEP 15: SAVE
# ============================================================

print("\nSaving dataset...")

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# STEP 16: FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nTotal transfer records:", len(result))

print(
    "Complete 365-day windows:",
    int(result["complete_365_day_window"].sum())
)

print(
    ">=450 post-transfer minutes:",
    int(result["eligible_450min"].sum())
)

print(
    "Latest appearance date:",
    latest_appearance_date
)

print("\nSample:")

print(
    result[
        [
            "transfer_row_id",
            "player_id",
            "transfer_date",
            "from_club_name",
            "to_club_name",
            "post_minutes",
            "post_goals",
            "post_assists",
            "post_GA90",
            "eligible_450min"
        ]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 60)
print("Post-transfer target dataset created successfully.")
print("=" * 60)

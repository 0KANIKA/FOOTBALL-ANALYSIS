import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "04_Datasets" / "derived"
RAW_DIR = BASE / "04_Datasets" / "raw_transfer_data" / "archive"

TRANSFER_FILE = DATA_DIR / "post_transfer_performance.csv"
TEAM_FILE = DATA_DIR / "team_context_dataset.csv"
GAMES_FILE = RAW_DIR / "games.csv"

OUTPUT_FILE = DATA_DIR / "team_coach_context_dataset.csv"

print("=" * 65)
print("BUILDING DESTINATION TEAM + COACH CONTEXT")
print("=" * 65)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("\nLoading transfer dataset...")
transfers = pd.read_csv(TRANSFER_FILE)
transfers["transfer_date"] = pd.to_datetime(
    transfers["transfer_date"], errors="coerce"
)

print(f"Transfer records: {len(transfers)}")

print("\nLoading team context...")
team = pd.read_csv(TEAM_FILE)

print("\nLoading games...")
games = pd.read_csv(GAMES_FILE)

games["date"] = pd.to_datetime(
    games["date"], errors="coerce"
)

print(f"Games: {len(games)}")


# ---------------------------------------------------------
# PREPARE GAME DATA
# ---------------------------------------------------------

required = [
    "date",
    "home_club_id",
    "away_club_id",
    "home_club_manager_name",
    "away_club_manager_name"
]

games = games.dropna(
    subset=["date", "home_club_id", "away_club_id"]
).copy()

print(f"Valid games: {len(games)}")


# ---------------------------------------------------------
# CREATE CLUB-LEVEL MANAGER RECORDS
# ---------------------------------------------------------

home = games[
    [
        "date",
        "home_club_id",
        "home_club_manager_name"
    ]
].copy()

home.columns = [
    "date",
    "club_id",
    "manager_name"
]

away = games[
    [
        "date",
        "away_club_id",
        "away_club_manager_name"
    ]
].copy()

away.columns = [
    "date",
    "club_id",
    "manager_name"
]

club_games = pd.concat(
    [home, away],
    ignore_index=True
)

club_games["club_id"] = pd.to_numeric(
    club_games["club_id"],
    errors="coerce"
)

club_games = club_games.dropna(
    subset=["club_id"]
)

club_games = club_games.sort_values(
    ["club_id", "date"]
)

print(
    f"Club-manager records: {len(club_games)}"
)


# ---------------------------------------------------------
# MANAGER MATCHING
# ---------------------------------------------------------

print("\nMatching destination-club coaches...")

results = []

for i, row in transfers.iterrows():

    transfer_date = row["transfer_date"]
    destination = row["to_club_id"]

    if pd.isna(transfer_date) or pd.isna(destination):
        results.append({
            "coach_name": np.nan,
            "coach_match_date": pd.NaT,
            "coach_days_from_transfer": np.nan,
            "coach_match_type": "unmatched"
        })
        continue

    club_matches = club_games[
        club_games["club_id"] == destination
    ]

    if club_matches.empty:
        results.append({
            "coach_name": np.nan,
            "coach_match_date": pd.NaT,
            "coach_days_from_transfer": np.nan,
            "coach_match_type": "unmatched"
        })
        continue

    # -----------------------------------------------------
    # PRIORITY 1:
    # First destination-club match AFTER transfer
    # within 60 days
    # -----------------------------------------------------

    after = club_matches[
        club_matches["date"] >= transfer_date
    ].copy()

    after["days"] = (
        after["date"] - transfer_date
    ).dt.days

    after = after[
        after["days"] <= 60
    ]

    after = after[
        after["manager_name"].notna()
    ]

    if not after.empty:

        selected = after.iloc[0]

        results.append({
            "coach_name": selected["manager_name"],
            "coach_match_date": selected["date"],
            "coach_days_from_transfer": selected["days"],
            "coach_match_type": "post_transfer"
        })

        continue

    # -----------------------------------------------------
    # PRIORITY 2:
    # Most recent destination-club match BEFORE transfer
    # within 60 days
    # -----------------------------------------------------

    before = club_matches[
        club_matches["date"] < transfer_date
    ].copy()

    before["days"] = (
        transfer_date - before["date"]
    ).dt.days

    before = before[
        before["days"] <= 60
    ]

    before = before[
        before["manager_name"].notna()
    ]

    if not before.empty:

        selected = before.iloc[-1]

        results.append({
            "coach_name": selected["manager_name"],
            "coach_match_date": selected["date"],
            "coach_days_from_transfer": -selected["days"],
            "coach_match_type": "pre_transfer"
        })

        continue

    # -----------------------------------------------------
    # NO MATCH
    # -----------------------------------------------------

    results.append({
        "coach_name": np.nan,
        "coach_match_date": pd.NaT,
        "coach_days_from_transfer": np.nan,
        "coach_match_type": "unmatched"
    })

    if (i + 1) % 1000 == 0:
        print(
            f"Processed {i + 1} / {len(transfers)} transfers..."
        )


# ---------------------------------------------------------
# MERGE
# ---------------------------------------------------------

coach_df = pd.DataFrame(results)

final = pd.concat(
    [
        transfers.reset_index(drop=True),
        coach_df.reset_index(drop=True)
    ],
    axis=1
)


# ---------------------------------------------------------
# COACH COVERAGE
# ---------------------------------------------------------

matched = final["coach_name"].notna()

print("\n" + "=" * 65)
print("COACH CONTEXT COVERAGE")
print("=" * 65)

print(f"Total transfers: {len(final)}")
print(f"Coach matched: {matched.sum()}")

coverage = matched.mean() * 100

print(f"Coverage: {coverage:.2f}%")

print("\nMatch type:")

print(
    final["coach_match_type"]
    .value_counts(dropna=False)
)

print("\nDays from transfer:")

print(
    final.loc[
        matched,
        "coach_days_from_transfer"
    ].describe()
)


# ---------------------------------------------------------
# COACH FREQUENCY
# ---------------------------------------------------------

print("\nUnique coaches:")

print(
    final["coach_name"]
    .nunique()
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

final.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 65)
print("COMPLETE")
print("=" * 65)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nNew coach features:")
print("  - coach_name")
print("  - coach_match_date")
print("  - coach_days_from_transfer")
print("  - coach_match_type")

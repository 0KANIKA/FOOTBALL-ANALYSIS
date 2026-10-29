import os
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

ROOT = os.environ.get(
    "FOOTBALL_PROJECT_ROOT",
    str(Path(__file__).resolve().parents[2]),
)
DATA_DIR = os.path.join(ROOT, "04_Datasets", "derived")
ARCHIVE = os.path.join(ROOT, "04_Datasets", "raw_transfer_data", "archive")

TRANSFER_FILE = os.path.join(
    DATA_DIR, "professional_transfer_dataset.csv"
)

GAMES_FILE = os.path.join(
    ARCHIVE, "games.csv"
)

LINEUPS_FILE = os.path.join(
    ARCHIVE, "game_lineups.csv"
)

APPEARANCES_FILE = os.path.join(
    ARCHIVE, "appearances.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "coach_tactical_context_v21.csv"
)

WINDOW_DAYS = 365


# ============================================================
# HELPERS
# ============================================================

def clean_position(x):
    if pd.isna(x):
        return "Unknown"

    x = str(x).strip()

    mapping = {
        "midfield": "Midfield",
        "Midfield": "Midfield",
        "Defender": "Defender",
        "Attack": "Attack",
        "Sweeper": "Defender",
    }

    return mapping.get(x, x)


def position_group(position):
    if pd.isna(position):
        return "Unknown"

    p = str(position).lower()

    if "goalkeeper" in p:
        return "Goalkeeper"

    if any(x in p for x in [
        "back",
        "centre-back",
        "sweeper",
        "defender"
    ]):
        return "Defender"

    if any(x in p for x in [
        "midfield",
        "winger",
        "second striker"
    ]):
        return "Midfield"

    if any(x in p for x in [
        "forward",
        "attack"
    ]):
        return "Attack"

    return "Unknown"


# ============================================================
# 1. TRANSFERS
# ============================================================

print("=" * 70)
print("LOADING TRANSFERS")
print("=" * 70)

transfers = pd.read_csv(TRANSFER_FILE)

transfers["transfer_date"] = pd.to_datetime(
    transfers["transfer_date"],
    errors="coerce"
)

if "transfer_row_id" not in transfers.columns:
    transfers["transfer_row_id"] = (
        np.arange(len(transfers)) + 1
    )

required_transfer_cols = [
    "transfer_row_id",
    "transfer_date",
    "to_club_id",
    "position",
    "sub_position",
]

transfers = transfers[
    required_transfer_cols
].copy()

transfers["to_club_id"] = pd.to_numeric(
    transfers["to_club_id"],
    errors="coerce"
)

transfers = transfers.dropna(
    subset=[
        "transfer_date",
        "to_club_id"
    ]
)

transfers["target_position"] = (
    transfers["position"]
    .map(clean_position)
)

transfers["target_position_group"] = (
    transfers["target_position"]
    .map(position_group)
)

print(
    f"Transfers: {len(transfers):,}"
)


# ============================================================
# 2. GAMES
# ============================================================

print("\n" + "=" * 70)
print("LOADING GAMES")
print("=" * 70)

games = pd.read_csv(
    GAMES_FILE,
    usecols=[
        "game_id",
        "date",
        "home_club_id",
        "away_club_id",
        "home_club_manager_name",
        "away_club_manager_name",
        "home_club_formation",
        "away_club_formation",
    ]
)

games["date"] = pd.to_datetime(
    games["date"],
    errors="coerce"
)

games["home_club_id"] = pd.to_numeric(
    games["home_club_id"],
    errors="coerce"
)

games["away_club_id"] = pd.to_numeric(
    games["away_club_id"],
    errors="coerce"
)

print(
    f"Games: {len(games):,}"
)


# ============================================================
# 3. TEAM-GAME TABLE
# ============================================================

print("\nBuilding team-game table...")

home = games[
    [
        "game_id",
        "date",
        "home_club_id",
        "home_club_manager_name",
        "home_club_formation",
    ]
].copy()

home.columns = [
    "game_id",
    "date",
    "club_id",
    "coach",
    "formation",
]

away = games[
    [
        "game_id",
        "date",
        "away_club_id",
        "away_club_manager_name",
        "away_club_formation",
    ]
].copy()

away.columns = [
    "game_id",
    "date",
    "club_id",
    "coach",
    "formation",
]

team_games = pd.concat(
    [home, away],
    ignore_index=True
)

team_games["club_id"] = pd.to_numeric(
    team_games["club_id"],
    errors="coerce"
)

team_games["coach"] = (
    team_games["coach"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

team_games["formation"] = (
    team_games["formation"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

team_games = team_games.drop_duplicates(
    subset=[
        "game_id",
        "club_id"
    ]
)

print(
    f"Team-games: {len(team_games):,}"
)

print(
    "Known coaches:",
    f"{(team_games['coach'] != 'Unknown').mean() * 100:.2f}%"
)


# ============================================================
# 4. LINEUPS
# ============================================================

print("\n" + "=" * 70)
print("LOADING LINEUPS")
print("=" * 70)

lineups = pd.read_csv(
    LINEUPS_FILE,
    usecols=[
        "game_id",
        "date",
        "player_id",
        "club_id",
        "type",
        "position",
    ]
)

lineups["date"] = pd.to_datetime(
    lineups["date"],
    errors="coerce"
)

lineups["player_id"] = pd.to_numeric(
    lineups["player_id"],
    errors="coerce"
)

lineups["club_id"] = pd.to_numeric(
    lineups["club_id"],
    errors="coerce"
)

lineups["position"] = (
    lineups["position"]
    .map(clean_position)
)

lineups["position_group"] = (
    lineups["position"]
    .map(position_group)
)

lineups["is_starter"] = (
    lineups["type"]
    .astype(str)
    .str.lower()
    .eq("starting_lineup")
    .astype(int)
)

lineups = lineups.drop_duplicates(
    subset=[
        "game_id",
        "player_id",
        "club_id"
    ]
)

print(
    f"Unique lineup records: {len(lineups):,}"
)


# ============================================================
# 5. ATTACH COACH / FORMATION
# ============================================================

print("\nAttaching coach and formation...")

lineups = lineups.merge(
    team_games[
        [
            "game_id",
            "club_id",
            "coach",
            "formation",
        ]
    ],
    on=[
        "game_id",
        "club_id"
    ],
    how="left"
)

lineups["coach"] = (
    lineups["coach"]
    .fillna("Unknown")
)

lineups["formation"] = (
    lineups["formation"]
    .fillna("Unknown")
)


# ============================================================
# 6. APPEARANCES
# ============================================================

print("\n" + "=" * 70)
print("LOADING APPEARANCES")
print("=" * 70)

appearances = pd.read_csv(
    APPEARANCES_FILE,
    usecols=[
        "game_id",
        "player_id",
        "player_club_id",
        "date",
        "minutes_played",
    ]
)

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

appearances["minutes_played"] = pd.to_numeric(
    appearances["minutes_played"],
    errors="coerce"
).fillna(0)

appearances = appearances.rename(
    columns={
        "player_club_id": "club_id"
    }
)

appearances = appearances.drop_duplicates(
    subset=[
        "game_id",
        "player_id",
        "club_id"
    ]
)

print(
    f"Appearance records: {len(appearances):,}"
)


# ============================================================
# 7. PLAYER MATCH HISTORY
# ============================================================

print("\n" + "=" * 70)
print("BUILDING PLAYER-COACH HISTORY")
print("=" * 70)

history = appearances.merge(
    lineups[
        [
            "game_id",
            "player_id",
            "club_id",
            "position",
            "position_group",
            "is_starter",
            "coach",
            "formation",
        ]
    ],
    on=[
        "game_id",
        "player_id",
        "club_id"
    ],
    how="inner"
)

history["coach"] = (
    history["coach"]
    .fillna("Unknown")
)

history["formation"] = (
    history["formation"]
    .fillna("Unknown")
)

history = history[
    history["minutes_played"] > 0
].copy()

history = history[
    history["date"].notna()
].copy()

print(
    f"Historical player appearances: {len(history):,}"
)

print(
    "Known coach:",
    f"{(history['coach'] != 'Unknown').mean() * 100:.2f}%"
)


# ============================================================
# 8. TRANSFER-LEVEL CALCULATION
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING LEAKAGE-SAFE COACH/Tactical FEATURES")
print("=" * 70)

results = []

for idx, transfer in transfers.iterrows():

    if idx % 1000 == 0:
        print(
            f"Processing transfer "
            f"{idx + 1:,}/{len(transfers):,}"
        )

    transfer_id = transfer[
        "transfer_row_id"
    ]

    transfer_date = transfer[
        "transfer_date"
    ]

    destination = transfer[
        "to_club_id"
    ]

    target_position = transfer[
        "target_position"
    ]

    target_group = transfer[
        "target_position_group"
    ]

    start_date = (
        transfer_date
        - pd.Timedelta(days=WINDOW_DAYS)
    )

    # --------------------------------------------------------
    # Destination club history before transfer
    # --------------------------------------------------------

    club_history = history[
        (history["club_id"] == destination)
        & (history["date"] < transfer_date)
        & (history["date"] >= start_date)
    ].copy()

    if len(club_history) == 0:

        results.append({
            "transfer_row_id": transfer_id,
            "coach_known_pretransfer": 0,
        })

        continue

    # --------------------------------------------------------
    # Last known coach BEFORE transfer
    # --------------------------------------------------------

    known = club_history[
        club_history["coach"] != "Unknown"
    ].sort_values("date")

    if len(known) == 0:

        results.append({
            "transfer_row_id": transfer_id,
            "coach_known_pretransfer": 0,
        })

        continue

    coach = known.iloc[-1]["coach"]

    coach_history = club_history[
        club_history["coach"] == coach
    ].copy()

    if len(coach_history) == 0:

        results.append({
            "transfer_row_id": transfer_id,
            "coach_known_pretransfer": 0,
        })

        continue

    # --------------------------------------------------------
    # GROUP-LEVEL POSITION USAGE
    # --------------------------------------------------------

    group_history = coach_history[
        coach_history["position_group"]
        == target_group
    ]

    total_minutes = (
        coach_history["minutes_played"]
        .sum()
    )

    group_minutes = (
        group_history["minutes_played"]
        .sum()
    )

    group_starts = (
        group_history["is_starter"]
        .sum()
    )

    group_appearances = len(
        group_history
    )

    group_players = (
        group_history["player_id"]
        .nunique()
    )

    group_minutes_share = (
        group_minutes / total_minutes
        if total_minutes > 0
        else 0
    )

    group_start_rate = (
        group_starts / group_appearances
        if group_appearances > 0
        else 0
    )

    # --------------------------------------------------------
    # OTHER-PLAYER COMPETITION
    # --------------------------------------------------------

    competitor_history = group_history[
        group_history["player_id"]
        != transfer.get("player_id", -999999)
    ]

    competitor_minutes = (
        competitor_history[
            "minutes_played"
        ].sum()
    )

    competitor_count = (
        competitor_history[
            "player_id"
        ].nunique()
    )

    # --------------------------------------------------------
    # POSITION ROTATION
    # --------------------------------------------------------

    player_minutes = (
        group_history
        .groupby("player_id")[
            "minutes_played"
        ]
        .sum()
    )

    if player_minutes.sum() > 0:

        shares = (
            player_minutes
            / player_minutes.sum()
        )

        rotation_hhi = float(
            (shares ** 2).sum()
        )

    else:

        rotation_hhi = 0.0

    # --------------------------------------------------------
    # EXACT POSITION FIT
    # --------------------------------------------------------

    exact_history = coach_history[
        coach_history["position"]
        == target_position
    ]

    exact_minutes = (
        exact_history[
            "minutes_played"
        ].sum()
    )

    exact_minutes_share = (
        exact_minutes / total_minutes
        if total_minutes > 0
        else 0
    )

    exact_start_rate = (
        exact_history["is_starter"].sum()
        / len(exact_history)
        if len(exact_history) > 0
        else 0
    )

    # --------------------------------------------------------
    # FORMATION STABILITY
    # --------------------------------------------------------

    formations = coach_history[
        coach_history["formation"] != "Unknown"
    ]

    if len(formations) > 0:

        formation_counts = (
            formations[
                "formation"
            ]
            .value_counts()
        )

        formation_stability = (
            formation_counts.iloc[0]
            / formation_counts.sum()
        )

    else:

        formation_stability = 0.0

    # --------------------------------------------------------
    # COACH OBSERVED TENURE
    # --------------------------------------------------------

    first_coach_date = known[
        known["coach"] == coach
    ]["date"].min()

    if pd.notna(first_coach_date):

        observed_tenure_days = max(
            0,
            (
                transfer_date
                - first_coach_date
            ).days
        )

    else:

        observed_tenure_days = 0

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    results.append({

        "transfer_row_id": transfer_id,

        "coach_known_pretransfer": 1,

        # Broad positional opportunity
        "coach_position_group_minutes_share":
            group_minutes_share,

        "coach_position_group_start_rate":
            group_start_rate,

        # Exact positional fit
        "coach_exact_position_minutes_share":
            exact_minutes_share,

        "coach_exact_position_start_rate":
            exact_start_rate,

        # Competition
        "coach_position_competitor_count":
            competitor_count,

        "coach_position_competitor_minutes":
            competitor_minutes,

        # Rotation
        "coach_position_rotation_hhi":
            rotation_hhi,

        # Tactical stability
        "coach_formation_stability":
            formation_stability,

        # Coach continuity
        "coach_observed_tenure_days":
            observed_tenure_days,

    })


# ============================================================
# 9. SAVE
# ============================================================

result_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("FINAL C v2.1 CONTEXT DATASET")
print("=" * 70)

print(
    f"Rows: {len(result_df):,}"
)

print(
    f"Columns: {len(result_df.columns)}"
)

print("\nMissingness:")

print(
    result_df.isna()
    .mean()
    .sort_values(
        ascending=False
    )
    .to_string()
)

print("\nFeature statistics:")

numeric = result_df.select_dtypes(
    include=np.number
).columns

print(
    result_df[numeric]
    .describe()
    .T
    .to_string()
)

print("\nCoach coverage:")

print(
    result_df[
        "coach_known_pretransfer"
    ].value_counts(
        dropna=False
    )
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nDONE.")

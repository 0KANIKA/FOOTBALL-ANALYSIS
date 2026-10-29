import os
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

ROOT = str(Path(__file__).resolve().parents[2])
DATA_DIR = os.path.join(ROOT, "04_Datasets", "derived")
ARCHIVE = os.path.join(ROOT, "04_Datasets", "raw_transfer_data", "archive")

TRANSFER_FILE = os.path.join(DATA_DIR, "professional_transfer_dataset.csv")
GAMES_FILE = os.path.join(ARCHIVE, "games.csv")
LINEUPS_FILE = os.path.join(ARCHIVE, "game_lineups.csv")
APPEARANCES_FILE = os.path.join(ARCHIVE, "appearances.csv")

OUTPUT_FILE = os.path.join(DATA_DIR, "coach_tactical_context_v2.csv")

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
        "winger",
        "midfield",
        "attacking midfield",
        "defensive midfield",
        "second striker"
    ]):
        return "Midfield"

    if any(x in p for x in [
        "forward",
        "centre-forward",
        "attack"
    ]):
        return "Attack"

    return "Unknown"


def safe_div(a, b):
    if b == 0 or pd.isna(b):
        return 0.0
    return float(a) / float(b)


# ============================================================
# 1. LOAD TRANSFERS
# ============================================================

print("=" * 70)
print("LOADING TRANSFER DATA")
print("=" * 70)

transfers = pd.read_csv(TRANSFER_FILE)

transfers["transfer_date"] = pd.to_datetime(
    transfers["transfer_date"],
    errors="coerce"
)

if "transfer_row_id" not in transfers.columns:
    transfers["transfer_row_id"] = np.arange(
        1, len(transfers) + 1
    )

transfers = transfers[
    [
        "transfer_row_id",
        "transfer_date",
        "to_club_id",
        "position",
        "sub_position",
    ]
].copy()

transfers = transfers.dropna(
    subset=["transfer_date", "to_club_id"]
)

transfers["to_club_id"] = pd.to_numeric(
    transfers["to_club_id"],
    errors="coerce"
)

transfers["target_position"] = transfers["position"].map(
    clean_position
)

transfers["target_position_group"] = transfers[
    "target_position"
].map(position_group)

print(f"Transfers: {len(transfers):,}")


# ============================================================
# 2. LOAD GAMES
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

print(f"Games: {len(games):,}")


# ============================================================
# 3. CONVERT GAMES INTO TEAM-GAME TABLE
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
    subset=["game_id", "club_id"]
)

print(f"Team-games: {len(team_games):,}")
print(
    f"Known coaches: "
    f"{(team_games['coach'] != 'Unknown').mean() * 100:.2f}%"
)


# ============================================================
# 4. LOAD LINEUPS
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

lineups["club_id"] = pd.to_numeric(
    lineups["club_id"],
    errors="coerce"
)

lineups["player_id"] = pd.to_numeric(
    lineups["player_id"],
    errors="coerce"
)

lineups["position"] = lineups["position"].map(
    clean_position
)

lineups["position_group"] = lineups["position"].map(
    position_group
)

lineups["is_starter"] = (
    lineups["type"].astype(str).str.lower()
    == "starting_lineup"
).astype(int)

print(f"Lineup records: {len(lineups):,}")
print(
    "Starting lineup records:",
    int(lineups["is_starter"].sum())
)

# Remove accidental duplicate player-game-club records
lineups = lineups.drop_duplicates(
    subset=["game_id", "player_id", "club_id"]
)

print(f"Unique lineup records: {len(lineups):,}")


# ============================================================
# 5. ATTACH COACH TO EVERY LINEUP
# ============================================================

print("\nAttaching historical coach information...")

lineups = lineups.merge(
    team_games[
        [
            "game_id",
            "club_id",
            "coach",
            "formation",
        ]
    ],
    on=["game_id", "club_id"],
    how="left",
)

lineups["coach"] = (
    lineups["coach"]
    .fillna("Unknown")
    .astype(str)
)

lineups["formation"] = (
    lineups["formation"]
    .fillna("Unknown")
    .astype(str)
)

print(
    "Lineups with known coach:",
    f"{(lineups['coach'] != 'Unknown').mean() * 100:.2f}%"
)


# ============================================================
# 6. LOAD APPEARANCES
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
    columns={"player_club_id": "club_id"}
)

appearances = appearances.drop_duplicates(
    subset=["game_id", "player_id", "club_id"]
)

print(f"Appearance records: {len(appearances):,}")


# ============================================================
# 7. MERGE APPEARANCES WITH LINEUPS
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
            "type",
            "is_starter",
            "coach",
            "formation",
        ]
    ],
    on=["game_id", "player_id", "club_id"],
    how="inner",
)

history = history.merge(
    team_games[
        [
            "game_id",
            "club_id",
            "date",
        ]
    ],
    on=["game_id", "club_id"],
    how="left",
    suffixes=("", "_team"),
)

history["date"] = history["date_team"].fillna(
    history["date"]
)

history = history.drop(
    columns=["date_team"],
    errors="ignore"
)

history["coach"] = (
    history["coach"]
    .fillna("Unknown")
    .astype(str)
)

history["formation"] = (
    history["formation"]
    .fillna("Unknown")
    .astype(str)
)

history["minutes_played"] = pd.to_numeric(
    history["minutes_played"],
    errors="coerce"
).fillna(0)

history["is_starter"] = history[
    "is_starter"
].fillna(0).astype(int)

history = history[
    history["minutes_played"] > 0
].copy()

history = history[
    history["date"].notna()
].copy()

print(f"Historical player appearances: {len(history):,}")
print(
    "Known coach:",
    f"{(history['coach'] != 'Unknown').mean() * 100:.2f}%"
)


# ============================================================
# 8. NORMALIZE POSITION
# ============================================================

history["position"] = history["position"].map(
    clean_position
)

history["position_group"] = history[
    "position"
].map(position_group)


# ============================================================
# 9. CREATE COACH/CLUB POSITION STATISTICS
# ============================================================

print("\nCreating coach positional statistics...")

# We will use the player's exact Transfermarkt position where
# possible. Position group is retained as a more stable backup.

position_stats = (
    history[
        history["coach"] != "Unknown"
    ]
    .groupby(
        [
            "club_id",
            "coach",
            "position",
        ],
        dropna=False
    )
    .agg(
        position_minutes=(
            "minutes_played",
            "sum"
        ),
        position_starts=(
            "is_starter",
            "sum"
        ),
        position_players=(
            "player_id",
            "nunique"
        ),
    )
    .reset_index()
)

coach_totals = (
    history[
        history["coach"] != "Unknown"
    ]
    .groupby(
        ["club_id", "coach"],
        dropna=False
    )
    .agg(
        coach_total_minutes=(
            "minutes_played",
            "sum"
        ),
        coach_total_players=(
            "player_id",
            "nunique"
        ),
        coach_total_starts=(
            "is_starter",
            "sum"
        ),
    )
    .reset_index()
)

position_stats = position_stats.merge(
    coach_totals,
    on=["club_id", "coach"],
    how="left"
)

position_stats["position_minutes_share"] = (
    position_stats["position_minutes"]
    / position_stats["coach_total_minutes"]
).fillna(0)


# ============================================================
# 10. POSITION ROTATION
# ============================================================

print("Calculating positional rotation...")

# Player minutes within each coach/club/position
player_position = (
    history[
        history["coach"] != "Unknown"
    ]
    .groupby(
        [
            "club_id",
            "coach",
            "position",
            "player_id",
        ],
        dropna=False
    )
    .agg(
        player_minutes=(
            "minutes_played",
            "sum"
        )
    )
    .reset_index()
)

player_position = player_position.merge(
    position_stats[
        [
            "club_id",
            "coach",
            "position",
            "position_minutes",
        ]
    ],
    on=["club_id", "coach", "position"],
    how="left"
)

player_position["minute_share"] = (
    player_position["player_minutes"]
    / player_position["position_minutes"]
).fillna(0)

rotation_stats = (
    player_position
    .groupby(
        ["club_id", "coach", "position"],
        dropna=False
    )
    .agg(
        position_rotation_hhi=(
            "minute_share",
            lambda x: float((x ** 2).sum())
        )
    )
    .reset_index()
)

position_stats = position_stats.merge(
    rotation_stats,
    on=["club_id", "coach", "position"],
    how="left"
)


# ============================================================
# 11. FORMATION STABILITY
# ============================================================

print("Calculating formation stability...")

formation_counts = (
    history[
        history["coach"] != "Unknown"
    ]
    .groupby(
        [
            "club_id",
            "coach",
            "formation",
        ],
        dropna=False
    )
    .size()
    .reset_index(name="formation_matches")
)

formation_totals = (
    formation_counts
    .groupby(
        ["club_id", "coach"],
        dropna=False
    )["formation_matches"]
    .sum()
    .reset_index(
        name="total_formation_matches"
    )
)

formation_counts = formation_counts.merge(
    formation_totals,
    on=["club_id", "coach"],
    how="left"
)

formation_counts["formation_frequency"] = (
    formation_counts["formation_matches"]
    / formation_counts["total_formation_matches"]
)

formation_stability = (
    formation_counts
    .groupby(
        ["club_id", "coach"],
        dropna=False
    )["formation_frequency"]
    .max()
    .reset_index(
        name="coach_formation_stability"
    )
)


# ============================================================
# 12. COACH OBSERVED TENURE
# ============================================================

print("Calculating observed coach tenure...")

coach_dates = (
    history[
        history["coach"] != "Unknown"
    ]
    .groupby(
        ["club_id", "coach"],
        dropna=False
    )["date"]
    .agg(
        coach_first_observed_date="min",
        coach_last_observed_date="max",
    )
    .reset_index()
)


# ============================================================
# 13. BUILD TRANSFER-LEVEL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("BUILDING TRANSFER-LEVEL CONTEXT")
print("=" * 70)

results = []

history_dates = history[
    [
        "club_id",
        "coach",
        "date",
        "formation",
    ]
].drop_duplicates()

# Sort for efficient lookup
history_dates = history_dates.sort_values(
    ["club_id", "date"]
)

for i, row in transfers.iterrows():

    if i % 1000 == 0:
        print(
            f"Processing transfer "
            f"{i + 1:,}/{len(transfers):,}"
        )

    transfer_id = row["transfer_row_id"]
    transfer_date = row["transfer_date"]
    club_id = row["to_club_id"]

    if pd.isna(club_id) or pd.isna(transfer_date):
        continue

    start_date = (
        transfer_date
        - pd.Timedelta(days=WINDOW_DAYS)
    )

    # --------------------------------------------------------
    # Historical destination-club matches
    # --------------------------------------------------------

    club_matches = history_dates[
        (history_dates["club_id"] == club_id)
        & (history_dates["date"] < transfer_date)
        & (history_dates["date"] >= start_date)
    ].copy()

    if len(club_matches) == 0:
        results.append({
            "transfer_row_id": transfer_id
        })
        continue

    # --------------------------------------------------------
    # Most recently observed coach BEFORE transfer
    # --------------------------------------------------------

    known_coach_rows = club_matches[
        club_matches["coach"] != "Unknown"
    ].sort_values("date")

    if len(known_coach_rows) == 0:
        coach = "Unknown"
        coach_known = 0
    else:
        coach = known_coach_rows.iloc[-1]["coach"]
        coach_known = 1

    # --------------------------------------------------------
    # Coach history within the 365-day window
    # --------------------------------------------------------

    coach_history = history[
        (history["club_id"] == club_id)
        & (history["coach"] == coach)
        & (history["date"] < transfer_date)
        & (history["date"] >= start_date)
    ].copy()

    # Position
    target_position = row["position"]
    target_position = clean_position(target_position)

    target_group = row["target_position_group"]

    # Exact-position history
    pos_history = coach_history[
        coach_history["position"] == target_position
    ]

    # Group-level fallback
    group_history = coach_history[
        coach_history["position_group"] == target_group
    ]

    # --------------------------------------------------------
    # Feature defaults
    # --------------------------------------------------------

    position_minutes = 0.0
    position_starts = 0
    position_players = 0
    total_coach_minutes = 0.0
    total_coach_players = 0

    competitor_minutes = 0.0
    competitor_count = 0

    rotation_hhi = 0.0

    formation_stability = 0.0
    formation_count = 0
    dominant_formation = "Unknown"

    coach_tenure_days = 0.0

    # --------------------------------------------------------
    # Position statistics
    # --------------------------------------------------------

    if len(pos_history) > 0:

        position_minutes = (
            pos_history["minutes_played"].sum()
        )

        position_starts = (
            pos_history["is_starter"].sum()
        )

        position_players = (
            pos_history["player_id"].nunique()
        )

    elif len(group_history) > 0:

        # Fallback when exact position is too sparse
        position_minutes = (
            group_history["minutes_played"].sum()
        )

        position_starts = (
            group_history["is_starter"].sum()
        )

        position_players = (
            group_history["player_id"].nunique()
        )

    if len(coach_history) > 0:

        total_coach_minutes = (
            coach_history["minutes_played"].sum()
        )

        total_coach_players = (
            coach_history["player_id"].nunique()
        )

    # --------------------------------------------------------
    # Competitor metrics
    # --------------------------------------------------------

    if len(pos_history) > 0:

        competitor_minutes = (
            pos_history
            .groupby("player_id")[
                "minutes_played"
            ]
            .sum()
            .sum()
        )

        competitor_count = (
            pos_history["player_id"]
            .nunique()
        )

        # Position-level player minute concentration
        player_minutes = (
            pos_history
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

    # --------------------------------------------------------
    # Formation usage
    # --------------------------------------------------------

    formation_history = coach_history[
        coach_history["formation"] != "Unknown"
    ]

    if len(formation_history) > 0:

        formation_counts_local = (
            formation_history[
                "formation"
            ]
            .value_counts()
        )

        formation_count = (
            len(formation_history)
        )

        dominant_formation = (
            formation_counts_local.index[0]
        )

        formation_stability = (
            formation_counts_local.iloc[0]
            / formation_count
        )

    # --------------------------------------------------------
    # Coach observed tenure
    # --------------------------------------------------------

    if coach_known:

        first_coach_date = (
            known_coach_rows[
                known_coach_rows["coach"] == coach
            ]["date"]
            .min()
        )

        if pd.notna(first_coach_date):
            coach_tenure_days = max(
                0,
                (
                    transfer_date
                    - first_coach_date
                ).days
            )

    # --------------------------------------------------------
    # Position start rate
    # --------------------------------------------------------

    if len(pos_history) > 0:

        appearances_in_position = (
            pos_history["player_id"]
            .count()
        )

        position_start_rate = safe_div(
            position_starts,
            appearances_in_position
        )

    else:
        position_start_rate = 0.0

    # --------------------------------------------------------
    # Position minutes share
    # --------------------------------------------------------

    position_minutes_share = safe_div(
        position_minutes,
        total_coach_minutes
    )

    # --------------------------------------------------------
    # Append
    # --------------------------------------------------------

    results.append({

        "transfer_row_id": transfer_id,

        # Coach
        "coach_known_pretransfer": coach_known,

        # Position opportunity
        "coach_position_minutes_share":
            position_minutes_share,

        "coach_position_start_rate":
            position_start_rate,

        "coach_position_competitor_count":
            competitor_count,

        "coach_same_position_competitor_minutes":
            competitor_minutes,

        # Rotation
        "coach_position_rotation_hhi":
            rotation_hhi,

        # Tactical
        "coach_formation_stability":
            formation_stability,

        "coach_observed_tenure_days":
            coach_tenure_days,

        # Diagnostics
        "coach_history_matches":
            len(coach_history),

        "coach_position_matches":
            len(pos_history),

        "coach_position_players":
            position_players,

        "coach_total_players":
            total_coach_players,

        "dominant_formation":
            dominant_formation,
    })


# ============================================================
# 14. SAVE
# ============================================================

result_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print("Rows:", len(result_df))
print("Columns:", len(result_df.columns))

print("\nMissingness:")
print(
    result_df.isna().mean()
    .sort_values(ascending=False)
    .head(20)
)

print("\nCoach known:")
if "coach_known_pretransfer" in result_df:
    print(
        result_df[
            "coach_known_pretransfer"
        ].value_counts(
            dropna=False
        )
    )

print("\nFeature summary:")
numeric_cols = result_df.select_dtypes(
    include=np.number
).columns

print(
    result_df[numeric_cols]
    .describe()
    .T
    .to_string()
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nDONE.")

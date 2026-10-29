import os
from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE = str(Path(__file__).resolve().parents[2])
DATA_DIR = os.path.join(BASE, "04_Datasets", "derived")
ARCHIVE = os.path.join(BASE, "04_Datasets", "raw_transfer_data", "archive")

TRANSFER_FILE = os.path.join(
    DATA_DIR, "professional_transfer_dataset.csv"
)

GAMES_FILE = os.path.join(
    ARCHIVE, "games.csv"
)

APPEARANCES_FILE = os.path.join(
    ARCHIVE, "appearances.csv"
)

LINEUPS_FILE = os.path.join(
    ARCHIVE, "game_lineups.csv"
)

VALUATIONS_FILE = os.path.join(
    ARCHIVE, "player_valuations.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR, "rich_team_context_dataset.csv"
)


# ============================================================
# Helpers
# ============================================================

def safe_div(a, b):
    if b is None or b == 0:
        return 0.0

    return float(a) / float(b)


def normalize_position_group(x):

    if pd.isna(x):
        return "Unknown"

    x = str(x).strip().lower()

    # Goalkeeper
    if x in [
        "goalkeeper",
        "keeper"
    ]:
        return "Goalkeeper"

    # Defender
    if x in [
        "centre-back",
        "center-back",
        "right-back",
        "left-back",
        "sweeper",
        "defender",
        "defence"
    ]:
        return "Defender"

    # Midfield
    if x in [
        "defensive midfield",
        "central midfield",
        "attacking midfield",
        "left midfield",
        "right midfield",
        "midfield",
        "midfielder"
    ]:
        return "Midfield"

    # Attack
    if x in [
        "centre-forward",
        "center-forward",
        "left winger",
        "right winger",
        "second striker",
        "attack",
        "forward"
    ]:
        return "Attack"

    return str(x).title()


def clean_sub_position(x):

    if pd.isna(x):
        return "Unknown"

    x = str(x).strip()

    if x == "" or x.lower() in [
        "nan",
        "none",
        "null"
    ]:
        return "Unknown"

    return x


def normalize_formation(x):

    if pd.isna(x):
        return "Unknown"

    x = str(x).strip()

    if x == "" or x.lower() in [
        "nan",
        "none",
        "null"
    ]:
        return "Unknown"

    return x


# ============================================================
# Load datasets
# ============================================================

print("Loading datasets...")

transfers = pd.read_csv(TRANSFER_FILE)
games = pd.read_csv(GAMES_FILE)
appearances = pd.read_csv(APPEARANCES_FILE)
lineups = pd.read_csv(LINEUPS_FILE)
valuations = pd.read_csv(VALUATIONS_FILE)

print(f"Transfers:       {len(transfers):,}")
print(f"Games:           {len(games):,}")
print(f"Appearances:     {len(appearances):,}")
print(f"Lineups:         {len(lineups):,}")
print(f"Valuations:      {len(valuations):,}")


# ============================================================
# Date preparation
# ============================================================

transfers["transfer_date"] = pd.to_datetime(
    transfers["transfer_date"],
    errors="coerce"
)

games["date"] = pd.to_datetime(
    games["date"],
    errors="coerce"
)

appearances["date"] = pd.to_datetime(
    appearances["date"],
    errors="coerce"
)

lineups["date"] = pd.to_datetime(
    lineups["date"],
    errors="coerce"
)

valuations["date"] = pd.to_datetime(
    valuations["date"],
    errors="coerce"
)


# ============================================================
# Numeric IDs
# ============================================================

for df, cols in [

    (
        transfers,
        [
            "player_id",
            "from_club_id",
            "to_club_id"
        ]
    ),

    (
        games,
        [
            "game_id",
            "home_club_id",
            "away_club_id"
        ]
    ),

    (
        appearances,
        [
            "game_id",
            "player_id",
            "player_club_id"
        ]
    ),

    (
        lineups,
        [
            "game_id",
            "player_id",
            "club_id"
        ]
    ),

    (
        valuations,
        [
            "player_id",
            "current_club_id"
        ]
    )

]:

    for col in cols:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


# ============================================================
# Historical lineup position information
#
# IMPORTANT:
# We use game_lineups.csv rather than players.csv.
# Therefore the player's role comes from historical match data.
# ============================================================

print("\nPreparing historical lineup positions...")

lineup_position = lineups[
    [
        "game_id",
        "player_id",
        "club_id",
        "position"
    ]
].copy()

lineup_position["position_group"] = (
    lineup_position["position"]
    .apply(normalize_position_group)
)

lineup_position["sub_position"] = (
    lineup_position["position"]
    .apply(clean_sub_position)
)

# Keep one position record per player/game/club
lineup_position = (
    lineup_position
    .drop_duplicates(
        subset=[
            "game_id",
            "player_id",
            "club_id"
        ]
    )
)

print(
    f"Historical lineup position records: "
    f"{len(lineup_position):,}"
)


# ============================================================
# Historical market valuation table
#
# We NEVER use the current player value from players.csv.
# We use the latest valuation ON OR BEFORE the transfer date.
# ============================================================

print("\nPreparing historical player valuations...")

valuations = valuations[
    [
        "player_id",
        "date",
        "market_value_in_eur"
    ]
].copy()

valuations["market_value_in_eur"] = pd.to_numeric(
    valuations["market_value_in_eur"],
    errors="coerce"
)

valuations = valuations.dropna(
    subset=[
        "player_id",
        "date"
    ]
)

valuations = valuations.sort_values(
    [
        "player_id",
        "date"
    ]
)

print(
    f"Historical valuation records: "
    f"{len(valuations):,}"
)


# ============================================================
# Team-game records
#
# Each match becomes two records:
# one for home team
# one for away team
# ============================================================

print("\nBuilding team-game records...")

home = games[
    [
        "game_id",
        "date",
        "home_club_id",
        "away_club_id",
        "home_club_goals",
        "away_club_goals",
        "home_club_formation",
        "home_club_manager_name"
    ]
].copy()

home = home.rename(
    columns={
        "home_club_id": "club_id",
        "away_club_id": "opponent_id",
        "home_club_goals": "own_goals",
        "away_club_goals": "opponent_goals",
        "home_club_formation": "formation",
        "home_club_manager_name": "manager_name"
    }
)

home["hosting"] = "Home"


away = games[
    [
        "game_id",
        "date",
        "away_club_id",
        "home_club_id",
        "away_club_goals",
        "home_club_goals",
        "away_club_formation",
        "away_club_manager_name"
    ]
].copy()

away = away.rename(
    columns={
        "away_club_id": "club_id",
        "home_club_id": "opponent_id",
        "away_club_goals": "own_goals",
        "home_club_goals": "opponent_goals",
        "away_club_formation": "formation",
        "away_club_manager_name": "manager_name"
    }
)

away["hosting"] = "Away"


team_games = pd.concat(
    [
        home,
        away
    ],
    ignore_index=True
)

team_games = team_games.dropna(
    subset=[
        "club_id",
        "date"
    ]
)

team_games["own_goals"] = pd.to_numeric(
    team_games["own_goals"],
    errors="coerce"
).fillna(0)

team_games["opponent_goals"] = pd.to_numeric(
    team_games["opponent_goals"],
    errors="coerce"
).fillna(0)


# ============================================================
# Team performance features
# ============================================================

team_games["goal_difference"] = (
    team_games["own_goals"]
    -
    team_games["opponent_goals"]
)

team_games["win"] = (
    team_games["own_goals"]
    >
    team_games["opponent_goals"]
).astype(int)

team_games["draw"] = (
    team_games["own_goals"]
    ==
    team_games["opponent_goals"]
).astype(int)

team_games["points"] = np.select(
    [
        team_games["own_goals"]
        >
        team_games["opponent_goals"],

        team_games["own_goals"]
        ==
        team_games["opponent_goals"]
    ],
    [
        3,
        1
    ],
    default=0
)

team_games["formation"] = (
    team_games["formation"]
    .apply(normalize_formation)
)

team_games["manager_name"] = (
    team_games["manager_name"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

team_games = team_games.sort_values(
    [
        "club_id",
        "date",
        "game_id"
    ]
).reset_index(
    drop=True
)

print(
    f"Team-game records: {len(team_games):,}"
)


# ============================================================
# Transfer preparation
#
# IMPORTANT:
# 1-based ID to match post_transfer_performance.csv
# ============================================================

transfers = transfers.reset_index(
    drop=True
)

transfers["transfer_row_id"] = (
    np.arange(len(transfers))
    +
    1
)

valid_transfer_mask = (
    transfers["transfer_date"].notna()
    &
    transfers["to_club_id"].notna()
    &
    transfers["player_id"].notna()
)

print(
    "Transfers with valid "
    "date + destination + player:",
    valid_transfer_mask.sum()
)


# ============================================================
# Build rich context
# ============================================================

records = []

print("\nBuilding leakage-safe destination-team context...")


for idx, transfer in transfers.iterrows():

    if not valid_transfer_mask.loc[idx]:
        continue

    transfer_date = transfer[
        "transfer_date"
    ]

    club_id = transfer[
        "to_club_id"
    ]

    player_id = transfer[
        "player_id"
    ]


    # ========================================================
    # 1. Destination team's previous matches
    #
    # STRICTLY BEFORE transfer date
    # ========================================================

    previous_games = team_games[
        (
            team_games["club_id"]
            ==
            club_id
        )
        &
        (
            team_games["date"]
            <
            transfer_date
        )
    ].sort_values(
        [
            "date",
            "game_id"
        ],
        ascending=[
            False,
            False
        ]
    )

    recent = previous_games.head(
        10
    ).copy()


    # ========================================================
    # 2. Basic team performance
    # ========================================================

    if len(recent) > 0:

        team_matches = len(
            recent
        )

        goals_for_avg = (
            recent["own_goals"]
            .mean()
        )

        goals_against_avg = (
            recent["opponent_goals"]
            .mean()
        )

        goal_diff_avg = (
            recent["goal_difference"]
            .mean()
        )

        win_rate = (
            recent["win"]
            .mean()
        )

        points_per_game = (
            recent["points"]
            .mean()
        )

    else:

        team_matches = 0

        goals_for_avg = np.nan
        goals_against_avg = np.nan
        goal_diff_avg = np.nan
        win_rate = np.nan
        points_per_game = np.nan


    # ========================================================
    # 3. Formation context
    # ========================================================

    if len(recent) > 0:

        formation_counts = (
            recent["formation"]
            .value_counts()
        )

        if len(formation_counts) > 0:

            dominant_formation = (
                formation_counts.index[0]
            )

            dominant_formation_frequency = (
                formation_counts.iloc[0]
                /
                len(recent)
            )

        else:

            dominant_formation = "Unknown"
            dominant_formation_frequency = np.nan

        formation_diversity = (
            recent["formation"]
            .nunique()
        )

    else:

        dominant_formation = "Unknown"
        dominant_formation_frequency = np.nan
        formation_diversity = np.nan


    # ========================================================
    # 4. Manager stability
    # ========================================================

    if len(recent) > 0:

        manager_counts = (
            recent["manager_name"]
            .replace(
                "Unknown",
                np.nan
            )
            .dropna()
            .value_counts()
        )

        if len(manager_counts) > 0:

            dominant_manager = (
                manager_counts.index[0]
            )

            manager_matches_last10 = (
                manager_counts.iloc[0]
            )

        else:

            dominant_manager = "Unknown"
            manager_matches_last10 = 0

        unique_managers_last10 = (
            recent["manager_name"]
            .replace(
                "Unknown",
                np.nan
            )
            .dropna()
            .nunique()
        )

    else:

        dominant_manager = "Unknown"
        manager_matches_last10 = 0
        unique_managers_last10 = 0


    # ========================================================
    # 5. Historical squad appearances
    #
    # Previous 365 days ONLY.
    # ========================================================

    squad_start = (
        transfer_date
        -
        pd.Timedelta(
            days=365
        )
    )

    squad_apps = appearances[
        (
            appearances["player_club_id"]
            ==
            club_id
        )
        &
        (
            appearances["date"]
            >=
            squad_start
        )
        &
        (
            appearances["date"]
            <
            transfer_date
        )
    ].copy()


    if len(squad_apps) > 0:

        squad_apps["minutes_played"] = (
            pd.to_numeric(
                squad_apps[
                    "minutes_played"
                ],
                errors="coerce"
            )
            .fillna(0)
        )

        # ----------------------------------------------------
        # Attach HISTORICAL lineup position
        # ----------------------------------------------------

        squad_apps = squad_apps.merge(
            lineup_position[
                [
                    "game_id",
                    "player_id",
                    "club_id",
                    "position_group",
                    "sub_position"
                ]
            ],
            left_on=[
                "game_id",
                "player_id",
                "player_club_id"
            ],
            right_on=[
                "game_id",
                "player_id",
                "club_id"
            ],
            how="left"
        )

        squad_apps["position_group"] = (
            squad_apps[
                "position_group"
            ]
            .fillna("Unknown")
        )

        squad_apps["sub_position"] = (
            squad_apps[
                "sub_position"
            ]
            .apply(clean_sub_position)
        )


        # ----------------------------------------------------
        # Aggregate squad usage
        # ----------------------------------------------------

        squad_summary = (
            squad_apps
            .groupby(
                [
                    "player_id",
                    "position_group",
                    "sub_position"
                ],
                dropna=False
            )
            .agg(
                squad_minutes=(
                    "minutes_played",
                    "sum"
                ),

                squad_appearances=(
                    "game_id",
                    "nunique"
                )
            )
            .reset_index()
        )

    else:

        squad_summary = pd.DataFrame(
            columns=[
                "player_id",
                "position_group",
                "sub_position",
                "squad_minutes",
                "squad_appearances"
            ]
        )


    # ========================================================
    # 6. Target player's position
    #
    # Use professional_transfer_dataset.csv,
    # NOT current players.csv.
    # ========================================================

    target_position = normalize_position_group(
        transfer.get(
            "position",
            "Unknown"
        )
    )

    target_sub_position = clean_sub_position(
        transfer.get(
            "sub_position",
            "Unknown"
        )
    )


    # ========================================================
    # 7. Same-position competition
    # ========================================================

    if len(squad_summary) > 0:

        same_position = (
            squad_summary[
                squad_summary[
                    "position_group"
                ]
                ==
                target_position
            ]
            .copy()
        )

        # Exclude target if already present
        same_position = (
            same_position[
                same_position[
                    "player_id"
                ]
                !=
                player_id
            ]
        )

    else:

        same_position = (
            squad_summary.copy()
        )


    if len(same_position) > 0:

        same_position_competitor_count = (
            same_position[
                "player_id"
            ]
            .nunique()
        )

        same_position_competitor_minutes = (
            same_position[
                "squad_minutes"
            ]
            .sum()
        )

        same_position_top_competitor_minutes = (
            same_position[
                "squad_minutes"
            ]
            .max()
        )

    else:

        same_position_competitor_count = 0
        same_position_competitor_minutes = 0
        same_position_top_competitor_minutes = 0


    # ========================================================
    # 8. Same-sub-position competition
    # ========================================================

    if len(squad_summary) > 0:

        same_sub_position = (
            squad_summary[
                squad_summary[
                    "sub_position"
                ]
                ==
                target_sub_position
            ]
            .copy()
        )

        same_sub_position = (
            same_sub_position[
                same_sub_position[
                    "player_id"
                ]
                !=
                player_id
            ]
        )

    else:

        same_sub_position = (
            squad_summary.copy()
        )


    if len(same_sub_position) > 0:

        same_sub_position_competitor_count = (
            same_sub_position[
                "player_id"
            ]
            .nunique()
        )

        same_sub_position_competitor_minutes = (
            same_sub_position[
                "squad_minutes"
            ]
            .sum()
        )

        same_sub_position_top_minutes = (
            same_sub_position[
                "squad_minutes"
            ]
            .max()
        )

    else:

        same_sub_position_competitor_count = 0
        same_sub_position_competitor_minutes = 0
        same_sub_position_top_minutes = 0


    # ========================================================
    # 9. Positional usage
    #
    # Share of total squad minutes represented by
    # the target player's position.
    # ========================================================

    total_squad_minutes = (
        squad_summary[
            "squad_minutes"
        ]
        .sum()
        if len(squad_summary) > 0
        else 0
    )

    position_minutes = (
        same_position[
            "squad_minutes"
        ]
        .sum()
        if len(same_position) > 0
        else 0
    )

    position_minutes_share = safe_div(
        position_minutes,
        total_squad_minutes
    )


    # ========================================================
    # 10. Historical competitor market value
    #
    # IMPORTANT:
    # Latest valuation ON OR BEFORE transfer date.
    #
    # This avoids using a player's future/current value.
    # ========================================================

    competitor_ids = set(
        same_position[
            "player_id"
        ].dropna().astype(int)
    )

    competitor_ids.update(
        same_sub_position[
            "player_id"
        ].dropna().astype(int)
    )


    competitor_values = []


    if len(competitor_ids) > 0:

        relevant_values = valuations[
            valuations[
                "player_id"
            ].isin(
                competitor_ids
            )
            &
            (
                valuations[
                    "date"
                ]
                <=
                transfer_date
            )
        ].copy()

        if len(relevant_values) > 0:

            relevant_values = (
                relevant_values
                .sort_values(
                    [
                        "player_id",
                        "date"
                    ]
                )
                .drop_duplicates(
                    "player_id",
                    keep="last"
                )
            )

            competitor_values = (
                relevant_values[
                    "market_value_in_eur"
                ]
                .dropna()
                .tolist()
            )


    # Only same-position competitor values are used
    same_position_ids = set(
        same_position[
            "player_id"
        ].dropna().astype(int)
    )

    same_position_values = []

    if len(same_position_ids) > 0:

        position_values = valuations[
            valuations[
                "player_id"
            ].isin(
                same_position_ids
            )
            &
            (
                valuations[
                    "date"
                ]
                <=
                transfer_date
            )
        ].copy()

        if len(position_values) > 0:

            position_values = (
                position_values
                .sort_values(
                    [
                        "player_id",
                        "date"
                    ]
                )
                .drop_duplicates(
                    "player_id",
                    keep="last"
                )
            )

            same_position_values = (
                position_values[
                    "market_value_in_eur"
                ]
                .dropna()
                .tolist()
            )


    if len(same_position_values) > 0:

        same_position_avg_market_value = (
            np.mean(
                same_position_values
            )
        )

    else:

        same_position_avg_market_value = 0


    # ========================================================
    # 11. Previous destination familiarity
    #
    # Kept for a future Model D.
    # ========================================================

    player_dest_apps = appearances[
        (
            appearances["player_id"]
            ==
            player_id
        )
        &
        (
            appearances["player_club_id"]
            ==
            club_id
        )
        &
        (
            appearances["date"]
            <
            transfer_date
        )
    ].copy()


    if len(player_dest_apps) > 0:

        previous_destination_appearances = (
            player_dest_apps[
                "game_id"
            ]
            .nunique()
        )

        previous_destination_minutes = (
            pd.to_numeric(
                player_dest_apps[
                    "minutes_played"
                ],
                errors="coerce"
            )
            .fillna(0)
            .sum()
        )

    else:

        previous_destination_appearances = 0
        previous_destination_minutes = 0


    # ========================================================
    # 12. Create record
    # ========================================================

    record = {

        # ----------------------------------------------------
        # Identity
        # ----------------------------------------------------

        "transfer_row_id":
            transfer[
                "transfer_row_id"
            ],

        "player_id":
            player_id,

        "transfer_date":
            transfer_date,

        "to_club_id":
            club_id,


        # ----------------------------------------------------
        # Basic team context
        # ----------------------------------------------------

        "rich_team_matches_last10":
            team_matches,

        "rich_team_goals_for_avg10":
            goals_for_avg,

        "rich_team_goals_against_avg10":
            goals_against_avg,

        "rich_team_goal_difference_avg10":
            goal_diff_avg,

        "rich_team_win_rate10":
            win_rate,

        "rich_team_points_per_game10":
            points_per_game,


        # ----------------------------------------------------
        # Formation
        # ----------------------------------------------------

        "dominant_formation10":
            dominant_formation,

        "dominant_formation_frequency10":
            dominant_formation_frequency,

        "formation_diversity10":
            formation_diversity,


        # ----------------------------------------------------
        # Manager
        # ----------------------------------------------------

        "dominant_manager10":
            dominant_manager,

        "manager_matches_last10":
            manager_matches_last10,

        "unique_managers_last10":
            unique_managers_last10,


        # ----------------------------------------------------
        # Target role
        # ----------------------------------------------------

        "target_position_group":
            target_position,

        "target_sub_position":
            target_sub_position,


        # ----------------------------------------------------
        # Same-position competition
        # ----------------------------------------------------

        "same_position_competitor_count":
            same_position_competitor_count,

        "same_position_competitor_minutes":
            same_position_competitor_minutes,

        "same_position_avg_market_value":
            same_position_avg_market_value,

        "same_position_top_competitor_minutes":
            same_position_top_competitor_minutes,


        # ----------------------------------------------------
        # Same-sub-position competition
        # ----------------------------------------------------

        "same_sub_position_competitor_count":
            same_sub_position_competitor_count,

        "same_sub_position_competitor_minutes":
            same_sub_position_competitor_minutes,

        "same_sub_position_top_minutes":
            same_sub_position_top_minutes,


        # ----------------------------------------------------
        # Positional usage
        # ----------------------------------------------------

        "position_minutes_share":
            position_minutes_share,


        # ----------------------------------------------------
        # Previous familiarity
        # ----------------------------------------------------

        "previous_destination_appearances":
            previous_destination_appearances,

        "previous_destination_minutes":
            previous_destination_minutes
    }


    records.append(
        record
    )


    # Progress
    if (idx + 1) % 500 == 0:

        print(
            f"Processed "
            f"{idx + 1:,}/"
            f"{len(transfers):,}"
        )


# ============================================================
# Create context dataframe
# ============================================================

rich_context = pd.DataFrame(
    records
)

print(
    "\nContext dataset created."
)

print(
    f"Rows: "
    f"{len(rich_context):,}"
)

print(
    f"Columns: "
    f"{len(rich_context.columns):,}"
)


# ============================================================
# Merge back to original transfers
# ============================================================

output = transfers.merge(
    rich_context,
    on=[
        "transfer_row_id",
        "player_id",
        "transfer_date",
        "to_club_id"
    ],
    how="left",
    suffixes=(
        "",
        "_context"
    )
)


# ============================================================
# Save
# ============================================================

output.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# Validation
# ============================================================

print(
    "\n========== VALIDATION =========="
)

print(
    f"Output rows: "
    f"{len(output):,}"
)

print(
    f"Output columns: "
    f"{len(output.columns):,}"
)

print(
    f"Unique transfer IDs: "
    f"{output['transfer_row_id'].nunique():,}"
)

print(
    f"Duplicate transfer IDs: "
    f"{output['transfer_row_id'].duplicated().sum():,}"
)


print(
    "\nCoverage:"
)


context_cols = [

    "rich_team_matches_last10",

    "rich_team_goals_for_avg10",

    "rich_team_goals_against_avg10",

    "rich_team_goal_difference_avg10",

    "rich_team_win_rate10",

    "rich_team_points_per_game10",

    "dominant_formation10",

    "dominant_formation_frequency10",

    "formation_diversity10",

    "manager_matches_last10",

    "unique_managers_last10",

    "same_position_competitor_count",

    "same_position_competitor_minutes",

    "same_position_avg_market_value",

    "same_position_top_competitor_minutes",

    "same_sub_position_competitor_count",

    "same_sub_position_competitor_minutes",

    "same_sub_position_top_minutes",

    "position_minutes_share",

    "previous_destination_appearances",

    "previous_destination_minutes"
]


for col in context_cols:

    if col in output.columns:

        coverage = (
            output[col]
            .notna()
            .mean()
            *
            100
        )

        print(
            f"{col}: "
            f"{coverage:.2f}%"
        )


# ============================================================
# Sanity checks
# ============================================================

print(
    "\nSanity checks:"
)

print(
    "Transfer IDs start:",
    output[
        "transfer_row_id"
    ].min()
)

print(
    "Transfer IDs end:",
    output[
        "transfer_row_id"
    ].max()
)

print(
    "Formation examples:"
)

print(
    output[
        "dominant_formation10"
    ]
    .value_counts()
    .head(10)
)

print(
    "\nPosition group examples:"
)

print(
    output[
        "target_position_group"
    ]
    .value_counts()
)

print(
    "\nDone."
)

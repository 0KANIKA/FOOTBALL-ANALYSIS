import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("BUILDING DESTINATION TEAM CONTEXT")
print("=" * 60)

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "04_Datasets" / "derived"
ARCHIVE = BASE / "04_Datasets" / "raw_transfer_data" / "archive"

TRANSFER_FILE = DATA_DIR / "post_transfer_performance.csv"
GAMES_FILE = ARCHIVE / "games.csv"

OUTPUT_FILE = DATA_DIR / "team_context_dataset.csv"

print("\nLoading transfer dataset...")
transfers = pd.read_csv(TRANSFER_FILE)

print(f"Transfer records: {len(transfers)}")

print("\nLoading games...")
games = pd.read_csv(
    GAMES_FILE,
    usecols=[
        "game_id",
        "date",
        "home_club_id",
        "away_club_id",
        "home_club_goals",
        "away_club_goals"
    ]
)

games["date"] = pd.to_datetime(games["date"], errors="coerce")
transfers["transfer_date"] = pd.to_datetime(
    transfers["transfer_date"],
    errors="coerce"
)

games = games.dropna(
    subset=[
        "date",
        "home_club_id",
        "away_club_id",
        "home_club_goals",
        "away_club_goals"
    ]
)

print(f"Valid games: {len(games)}")

print("\nCreating club match records...")

home = pd.DataFrame({
    "club_id": games["home_club_id"],
    "date": games["date"],
    "goals_for": games["home_club_goals"],
    "goals_against": games["away_club_goals"]
})

away = pd.DataFrame({
    "club_id": games["away_club_id"],
    "date": games["date"],
    "goals_for": games["away_club_goals"],
    "goals_against": games["home_club_goals"]
})

club_games = pd.concat([home, away], ignore_index=True)

club_games["win"] = (
    club_games["goals_for"] > club_games["goals_against"]
).astype(int)

club_games["points"] = np.where(
    club_games["goals_for"] > club_games["goals_against"],
    3,
    np.where(
        club_games["goals_for"] == club_games["goals_against"],
        1,
        0
    )
)

club_games = club_games.sort_values(["club_id", "date"])

print("\nCalculating destination-team context...")

results = []

for i, row in transfers.iterrows():

    if pd.isna(row["transfer_date"]) or pd.isna(row["to_club_id"]):
        continue

    club_id = row["to_club_id"]
    transfer_date = row["transfer_date"]

    previous_games = club_games[
        (club_games["club_id"] == club_id) &
        (club_games["date"] < transfer_date)
    ]

    recent = previous_games.tail(10)

    if len(recent) == 0:
        results.append({
            "player_id": row["player_id"],
            "transfer_date": row["transfer_date"],
            "to_club_id": row["to_club_id"],
            "team_matches_last10": 0,
            "team_goals_for_avg10": np.nan,
            "team_goals_against_avg10": np.nan,
            "team_win_rate10": np.nan,
            "team_points_per_game10": np.nan,
            "team_goal_difference_avg10": np.nan
        })
        continue

    results.append({
        "player_id": row["player_id"],
        "transfer_date": row["transfer_date"],
        "to_club_id": row["to_club_id"],
        "team_matches_last10": len(recent),
        "team_goals_for_avg10": recent["goals_for"].mean(),
        "team_goals_against_avg10": recent["goals_against"].mean(),
        "team_win_rate10": recent["win"].mean(),
        "team_points_per_game10": recent["points"].mean(),
        "team_goal_difference_avg10": (
            recent["goals_for"] -
            recent["goals_against"]
        ).mean()
    })

    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1} / {len(transfers)} transfers...")

context = pd.DataFrame(results)

print(f"\nContext records created: {len(context)}")

print("\nMerging team context...")

dataset = transfers.merge(
    context,
    on=["player_id", "transfer_date", "to_club_id"],
    how="left"
)

context_columns = [
    "team_matches_last10",
    "team_goals_for_avg10",
    "team_goals_against_avg10",
    "team_win_rate10",
    "team_points_per_game10",
    "team_goal_difference_avg10"
]

has_context = dataset["team_matches_last10"] > 0

print("\n" + "=" * 60)
print("TEAM CONTEXT COVERAGE")
print("=" * 60)

print(f"Total transfers: {len(dataset)}")
print(f"With team context: {has_context.sum()}")
print(f"Coverage: {has_context.mean() * 100:.2f}%")

print("\nContext summary:")
print(dataset[context_columns].describe())

dataset.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nNew Model B features:")
for col in context_columns:
    print(f"  - {col}")

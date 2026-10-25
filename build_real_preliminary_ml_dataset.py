import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\rsuba\Downloads\R football analytics")
TM = BASE / "archive"
OUT = BASE / "preliminary_real_ml_dataset.csv"

print("=" * 80)
print("BUILDING REAL PRELIMINARY TRANSFER-LEVEL DATASET")
print("=" * 80)

# ------------------------------------------------------------------
# Load only the columns we actually need.
# ------------------------------------------------------------------
players = pd.read_csv(
    TM / "players.csv",
    usecols=[
        "player_id", "date_of_birth", "sub_position", "position",
        "foot", "height_in_cm"
    ],
    low_memory=False
)

transfers = pd.read_csv(
    TM / "transfers.csv",
    usecols=[
        "player_id", "transfer_date", "transfer_season",
        "from_club_id", "to_club_id", "from_club_name",
        "to_club_name", "transfer_fee", "market_value_in_eur"
    ],
    low_memory=False
)

appearances = pd.read_csv(
    TM / "appearances.csv",
    usecols=[
        "game_id", "player_id", "player_club_id", "date",
        "competition_id", "yellow_cards", "red_cards",
        "goals", "assists", "minutes_played"
    ],
    low_memory=False
)

valuations = pd.read_csv(
    TM / "player_valuations.csv",
    usecols=[
        "player_id", "date", "market_value_in_eur"
    ],
    low_memory=False
)

print(f"Loaded players:       {len(players):,}")
print(f"Loaded transfers:     {len(transfers):,}")
print(f"Loaded appearances:   {len(appearances):,}")
print(f"Loaded valuations:    {len(valuations):,}")

# ------------------------------------------------------------------
# Clean basic types.
# ------------------------------------------------------------------
players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")

for col in ["goals", "assists", "minutes_played", "yellow_cards", "red_cards"]:
    appearances[col] = pd.to_numeric(appearances[col], errors="coerce").fillna(0)

transfers["transfer_fee"] = pd.to_numeric(transfers["transfer_fee"], errors="coerce")
transfers["market_value_in_eur"] = pd.to_numeric(
    transfers["market_value_in_eur"], errors="coerce"
)

# ------------------------------------------------------------------
# Remove unusable/future transfer records.
# Current date: 2026-08-24.
# ------------------------------------------------------------------
cutoff = pd.Timestamp("2026-08-24")
transfers = transfers[
    transfers["transfer_date"].notna() &
    (transfers["transfer_date"] <= cutoff)
].copy()

# Keep actual club-to-club transfers.
transfers = transfers[
    transfers["from_club_id"].notna() &
    transfers["to_club_id"].notna() &
    (transfers["from_club_id"] != transfers["to_club_id"])
].copy()

# Remove exact duplicate transfer records.
transfers = transfers.drop_duplicates(
    subset=["player_id", "transfer_date", "from_club_id", "to_club_id"]
)

print(f"Valid historical club transfers: {len(transfers):,}")

# ------------------------------------------------------------------
# Merge player characteristics.
# ------------------------------------------------------------------
df = transfers.merge(players, on="player_id", how="left")

# ------------------------------------------------------------------
# Age at transfer.
# ------------------------------------------------------------------
df["age_at_transfer"] = (
    (df["transfer_date"] - df["date_of_birth"]).dt.days / 365.2425
).round(2)

# ------------------------------------------------------------------
# For each transfer, calculate REAL pre-transfer performance.
#
# Window: 365 days immediately before transfer.
# This is deliberately simple for the preliminary dataset.
# ------------------------------------------------------------------
app = appearances.copy()

# Merge appearances with transfers by player first.
# Then retain only appearances occurring before each transfer.
pre = app.merge(
    transfers[["player_id", "transfer_date", "from_club_id", "to_club_id"]],
    on="player_id",
    how="inner"
)

pre = pre[
    (pre["date"] < pre["transfer_date"]) &
    (pre["date"] >= pre["transfer_date"] - pd.Timedelta(days=365)) &
    (pre["player_club_id"] == pre["from_club_id"])
].copy()

pre["minutes_played"] = pre["minutes_played"].fillna(0)

pre_group = (
    pre.groupby(
        ["player_id", "transfer_date", "from_club_id", "to_club_id"],
        as_index=False
    )
    .agg(
        appearances_pre=("game_id", "nunique"),
        minutes_pre=("minutes_played", "sum"),
        goals_pre=("goals", "sum"),
        assists_pre=("assists", "sum"),
        yellow_cards_pre=("yellow_cards", "sum"),
        red_cards_pre=("red_cards", "sum")
    )
)

pre_group["goals_per90_pre"] = np.where(
    pre_group["minutes_pre"] > 0,
    pre_group["goals_pre"] / pre_group["minutes_pre"] * 90,
    np.nan
)

pre_group["assists_per90_pre"] = np.where(
    pre_group["minutes_pre"] > 0,
    pre_group["assists_pre"] / pre_group["minutes_pre"] * 90,
    np.nan
)

# ------------------------------------------------------------------
# Attach the real pre-transfer performance.
# ------------------------------------------------------------------
df = df.merge(
    pre_group,
    on=["player_id", "transfer_date", "from_club_id", "to_club_id"],
    how="left"
)

# ------------------------------------------------------------------
# Get the most recent valuation on or before the transfer.
# This is a real value from player_valuations.csv.
# ------------------------------------------------------------------
val = valuations.rename(
    columns={"date": "valuation_date", "market_value_in_eur": "valuation_market_value"}
)

val = val.sort_values(["player_id", "valuation_date"])
df = df.sort_values(["player_id", "transfer_date"])

# Make sure both sides are sorted exactly as required.
df = df.sort_values(["transfer_date", "player_id"]).reset_index(drop=True)
val = val.sort_values(["valuation_date", "player_id"]).reset_index(drop=True)

df = pd.merge_asof(
    df,
    val,
    left_on="transfer_date",
    right_on="valuation_date",
    by="player_id",
    direction="backward"
)
# ------------------------------------------------------------------
# Clean up / select the project-level columns.
# ------------------------------------------------------------------
keep = [
    "player_id",
    "transfer_date",
    "transfer_season",
    "from_club_id",
    "to_club_id",
    "from_club_name",
    "to_club_name",

    "age_at_transfer",
    "position",
    "sub_position",
    "height_in_cm",
    "foot",

    "appearances_pre",
    "minutes_pre",
    "goals_pre",
    "assists_pre",
    "goals_per90_pre",
    "assists_per90_pre",
    "yellow_cards_pre",
    "red_cards_pre",

    "market_value_in_eur",
    "valuation_market_value",

    "transfer_fee"
]

df = df[keep]

# Rename for clarity.
df = df.rename(columns={
    "market_value_in_eur": "transfer_market_value",
    "valuation_market_value": "market_value_pre"
})

# Remove records where we have no player identity or transfer date.
df = df[
    df["player_id"].notna() &
    df["transfer_date"].notna()
].copy()

# Sort chronologically.
df = df.sort_values("transfer_date").reset_index(drop=True)

# ------------------------------------------------------------------
# Save.
# ------------------------------------------------------------------
df.to_csv(OUT, index=False)

print("\n" + "=" * 80)
print("DATASET CREATED")
print("=" * 80)
print(f"File: {OUT}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Date range: {df['transfer_date'].min().date()} to {df['transfer_date'].max().date()}")

print("\nColumns:")
for i, c in enumerate(df.columns, 1):
    print(f"{i:02d}. {c}")

print("\nMissing values:")
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing):
    for c, n in missing.items():
        print(f"{c}: {n:,} ({n/len(df)*100:.1f}%)")
else:
    print("None")

print("\nFirst 10 REAL records:")
print(df.head(10).to_string(index=False))

print("\nNOTE:")
print("This is a REAL preliminary dataset generated from your Transfermarkt files.")
print("It is NOT yet the final integrated ML dataset.")
print("StatsBomb tactical features and the final post-transfer target still need to be added.")

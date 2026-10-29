import os
import pandas as pd

# ============================================================
# COACH / TACTICAL DATA AUDIT
# ============================================================

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "04_Datasets", "raw_transfer_data", "archive"))

FILES = [
    "games.csv",
    "game_lineups.csv",
    "club_games.csv",
    "appearances.csv"
]

print("=" * 70)
print("COACH / TACTICAL DATA AUDIT")
print("=" * 70)
print("Archive:", BASE)

# ------------------------------------------------------------
# 1. Basic schema inspection
# ------------------------------------------------------------

for filename in FILES:

    path = os.path.join(BASE, filename)

    if not os.path.exists(path):
        print(f"\n[MISSING] {filename}")
        continue

    print(f"\n{'-' * 70}")
    print(f"FILE: {filename}")
    print(f"{'-' * 70}")

    df = pd.read_csv(path, nrows=5)

    print("Columns:")
    for col in df.columns:
        print(f"  - {col}")

    relevant = [
        col for col in df.columns
        if any(keyword in col.lower() for keyword in [
            "coach",
            "manager",
            "formation",
            "position",
            "player",
            "club",
            "date",
            "minute",
            "substitute",
            "lineup",
            "team"
        ])
    ]

    print("\nPotentially relevant columns:")
    for col in relevant:
        print(f"  - {col}")


# ------------------------------------------------------------
# 2. Load Games + Lineups
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DEEP INSPECTION")
print("=" * 70)

games_path = os.path.join(BASE, "games.csv")
lineups_path = os.path.join(BASE, "game_lineups.csv")

games = pd.read_csv(games_path)
lineups = pd.read_csv(lineups_path)

print("\nGAMES")
print("Shape:", games.shape)
print("Columns:", games.columns.tolist())

print("\nGAME LINEUPS")
print("Shape:", lineups.shape)
print("Columns:", lineups.columns.tolist())


# ------------------------------------------------------------
# 3. Search Games for coach / manager / formation information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GAMES: COACH / MANAGER / FORMATION")
print("=" * 70)

game_special_cols = [
    col for col in games.columns
    if any(keyword in col.lower() for keyword in [
        "coach",
        "manager",
        "formation"
    ])
]

print("Detected columns:", game_special_cols)

for col in game_special_cols:

    print(f"\n--- {col} ---")

    print(
        games[col]
        .dropna()
        .value_counts()
        .head(20)
        .to_string()
    )


# ------------------------------------------------------------
# 4. Search Lineups for coach / manager / formation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LINEUPS: COACH / MANAGER / FORMATION")
print("=" * 70)

lineup_special_cols = [
    col for col in lineups.columns
    if any(keyword in col.lower() for keyword in [
        "coach",
        "manager",
        "formation"
    ])
]

print("Detected columns:", lineup_special_cols)

for col in lineup_special_cols:

    print(f"\n--- {col} ---")

    print(
        lineups[col]
        .dropna()
        .value_counts()
        .head(20)
        .to_string()
    )


# ------------------------------------------------------------
# 5. Position information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LINEUP POSITION INFORMATION")
print("=" * 70)

position_cols = [
    col for col in lineups.columns
    if "position" in col.lower()
]

print("Position columns:", position_cols)

for col in position_cols:

    print(f"\n--- {col} ---")

    print(
        lineups[col]
        .dropna()
        .value_counts()
        .head(30)
        .to_string()
    )


# ------------------------------------------------------------
# 6. Starting XI / substitute information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STARTING / SUBSTITUTE INFORMATION")
print("=" * 70)

sub_cols = [
    col for col in lineups.columns
    if any(keyword in col.lower() for keyword in [
        "substitute",
        "starter",
        "lineup"
    ])
]

print("Detected columns:", sub_cols)

for col in sub_cols:

    print(f"\n--- {col} ---")

    print(
        lineups[col]
        .dropna()
        .value_counts()
        .head(20)
        .to_string()
    )


# ------------------------------------------------------------
# 7. Common columns between games and lineups
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("GAMES <-> LINEUPS JOIN")
print("=" * 70)

common_columns = sorted(
    set(games.columns) & set(lineups.columns)
)

print("Common columns:")

for col in common_columns:
    print(f"  - {col}")


# ------------------------------------------------------------
# 8. Date columns
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATE INFORMATION")
print("=" * 70)

game_date_cols = [
    col for col in games.columns
    if "date" in col.lower()
]

lineup_date_cols = [
    col for col in lineups.columns
    if "date" in col.lower()
]

print("Games date columns:", game_date_cols)
print("Lineups date columns:", lineup_date_cols)


# ------------------------------------------------------------
# 9. Potential team identifiers
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TEAM / CLUB IDENTIFIERS")
print("=" * 70)

for df_name, df in [
    ("games", games),
    ("lineups", lineups)
]:

    cols = [
        col for col in df.columns
        if any(keyword in col.lower() for keyword in [
            "club_id",
            "team_id",
            "home_club",
            "away_club"
        ])
    ]

    print(f"\n{df_name}:")
    print(cols)


# ------------------------------------------------------------
# 10. Save audit summary
# ------------------------------------------------------------

report = {
    "games_rows": len(games),
    "games_columns": len(games.columns),

    "lineups_rows": len(lineups),
    "lineups_columns": len(lineups.columns),

    "games_special_columns":
        ", ".join(game_special_cols),

    "lineups_special_columns":
        ", ".join(lineup_special_cols),

    "lineup_position_columns":
        ", ".join(position_cols),

    "lineup_substitution_columns":
        ", ".join(sub_cols),

    "common_join_columns":
        ", ".join(common_columns),

    "games_date_columns":
        ", ".join(game_date_cols),

    "lineups_date_columns":
        ", ".join(lineup_date_cols)
}

output_path = os.path.join(
    BASE,
    "coach_tactical_audit.csv"
)

pd.DataFrame([report]).to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print("\nSaved:")
print(output_path)

print("\nNEXT STEP:")
print("""
If the archive contains:

[1] Coach/manager information
[2] Formation information
[3] Player positions
[4] Starting/substitute information
[5] Match dates
[6] Team identifiers

then we can build a leakage-safe Coach + Tactical Context layer.

IMPORTANT:
Every feature will use only matches occurring BEFORE
the player's transfer date.

We will NOT use information occurring after the transfer.
""")

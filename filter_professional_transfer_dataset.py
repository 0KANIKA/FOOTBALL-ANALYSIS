import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\rsuba\Downloads\R football analytics")
INPUT = BASE / "preliminary_real_ml_dataset.csv"
OUTPUT = BASE / "professional_transfer_dataset.csv"

print("=" * 80)
print("FILTERING PRELIMINARY DATASET TO PROFESSIONAL PLAYER CASES")
print("=" * 80)

df = pd.read_csv(INPUT, low_memory=False)

print(f"Input rows: {len(df):,}")

# ------------------------------------------------------------
# 1. Professional age
# ------------------------------------------------------------
df = df[df["age_at_transfer"].notna() & (df["age_at_transfer"] >= 16)].copy()
print(f"After age >= 16: {len(df):,}")

# ------------------------------------------------------------
# 2. Require meaningful pre-transfer playing data.
# 450 minutes ~= five full matches.
# ------------------------------------------------------------
df = df[
    df["minutes_pre"].notna() &
    (df["minutes_pre"] >= 450)
].copy()

print(f"After >= 450 pre-transfer minutes: {len(df):,}")

# ------------------------------------------------------------
# 3. Remove implausible performance rates caused by bad/zero
#    denominators or corrupted records.
# ------------------------------------------------------------
df = df[
    df["goals_per90_pre"].notna() &
    df["assists_per90_pre"].notna()
].copy()

print(f"After valid per-90 metrics: {len(df):,}")

# ------------------------------------------------------------
# 4. Remove duplicate player-transfer records.
# ------------------------------------------------------------
df = df.drop_duplicates(
    subset=["player_id", "transfer_date", "from_club_id", "to_club_id"]
)

print(f"After duplicate removal: {len(df):,}")

# ------------------------------------------------------------
# 5. Sort chronologically.
# ------------------------------------------------------------
df = df.sort_values(
    ["transfer_date", "player_id"]
).reset_index(drop=True)

# ------------------------------------------------------------
# 6. Report missingness.
# ------------------------------------------------------------
print("\n" + "=" * 80)
print("FILTERED DATASET")
print("=" * 80)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(
    f"Date range: "
    f"{pd.to_datetime(df['transfer_date']).min().date()} "
    f"to "
    f"{pd.to_datetime(df['transfer_date']).max().date()}"
)

print("\nMissing values:")
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing):
    for col, count in missing.items():
        print(f"{col}: {count:,} ({count / len(df) * 100:.1f}%)")
else:
    print("None")

# ------------------------------------------------------------
# 7. Save.
# ------------------------------------------------------------
df.to_csv(OUTPUT, index=False)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)
print(OUTPUT)

print("\nFirst 10 records:")
print(df.head(10).to_string(index=False))

print("\nNOTE:")
print("This is a real Transfermarkt-derived professional-player subset.")
print("It is still NOT the final ML dataset.")
print("Post-transfer performance and StatsBomb tactical features have not yet been integrated.")

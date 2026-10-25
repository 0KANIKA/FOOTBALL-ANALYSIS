import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\rsuba\Downloads\R football analytics")

FILE = BASE / "FINAL_MODEL_C_CLEAN.csv"

df = pd.read_csv(FILE)

print("=" * 70)
print("FINAL DATASET CHECK")
print("=" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2}. {col}")

print("\n" + "=" * 70)
print("TARGET CHECK")
print("=" * 70)

print(df["post_GA90"].describe())

print("\nMissing target:", df["post_GA90"].isna().sum())

print("\n" + "=" * 70)
print("SPLIT CHECK")
print("=" * 70)

if "split" in df.columns:
    print(df["split"].value_counts())
else:
    print("No split column found.")

print("\n" + "=" * 70)
print("DATE RANGE")
print("=" * 70)

df["transfer_date"] = pd.to_datetime(df["transfer_date"])

print("Earliest:", df["transfer_date"].min())
print("Latest:", df["transfer_date"].max())

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

print("Duplicate transfer IDs:",
      df["transfer_row_id"].duplicated().sum())

print("\n" + "=" * 70)
print("COACH MISSINGNESS")
print("=" * 70)

coach_cols = [
    c for c in df.columns
    if c.startswith("coach_")
]

for col in coach_cols:
    print(f"{col}: {df[col].isna().sum()} missing")

print("\nDataset check complete.")
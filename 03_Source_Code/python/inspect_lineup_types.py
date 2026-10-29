import pandas as pd
from pathlib import Path

path = str(Path(__file__).resolve().parents[2] / "04_Datasets" / "raw_transfer_data" / "archive" / "game_lineups.csv")

df = pd.read_csv(
    path,
    usecols=["game_id", "date", "player_id", "club_id", "type", "position"]
)

print("\nTYPE VALUES")
print(df["type"].value_counts(dropna=False).head(20))

print("\nPOSITION VALUES")
print(df["position"].value_counts(dropna=False).head(30))

print("\nDATE RANGE")
print(df["date"].min(), "to", df["date"].max())

print("\nSAMPLE")
print(df.head(20).to_string(index=False))

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "04_Datasets", "derived")

TRANSFER_FILE = os.path.join(DATA_DIR, "professional_transfer_dataset.csv")
TARGET_FILE = os.path.join(DATA_DIR, "post_transfer_performance.csv")
RICH_FILE = os.path.join(DATA_DIR, "rich_team_context_dataset.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "06_Results_and_Demo_Evidence", "research_results", "model_c_rich_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# LOAD
# ============================================================

print("Loading datasets...")

transfers = pd.read_csv(TRANSFER_FILE)
target = pd.read_csv(TARGET_FILE)
rich = pd.read_csv(RICH_FILE)

print(f"Transfers: {len(transfers):,}")
print(f"Target:    {len(target):,}")
print(f"Rich:      {len(rich):,}")


# ============================================================
# ENSURE TRANSFER IDs MATCH
# ============================================================

# professional_transfer_dataset.csv follows the same row order
# used when transfer_row_id was created.

if "transfer_row_id" not in transfers.columns:
    transfers["transfer_row_id"] = np.arange(1, len(transfers) + 1)

if "transfer_row_id" not in target.columns:
    raise ValueError("Target dataset has no transfer_row_id")

if "transfer_row_id" not in rich.columns:
    raise ValueError("Rich context dataset has no transfer_row_id")


# ============================================================
# MERGE
# ============================================================

df = transfers.merge(
    target,
    on="transfer_row_id",
    how="inner",
    suffixes=("", "_target")
)

df = df.merge(
    rich,
    on="transfer_row_id",
    how="inner",
    suffixes=("", "_rich")
)

print(f"\nMerged rows: {len(df):,}")


# ============================================================
# IDENTIFY TARGET
# ============================================================

# The rebuilt post_transfer_performance dataset contains
# these target-related columns. Adjust automatically if needed.

possible_targets = [
    "post_GA90",
    "post_goals_assists_per90",
    "goals_assists_per90",
    "post_ga_per90",
    "target"
]

target_col = None

for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    print("\nAvailable target-like columns:")
    print([c for c in df.columns if "90" in c.lower() or "target" in c.lower()])
    raise ValueError("Could not automatically identify G+A/90 target column.")

print(f"Target column: {target_col}")


# ============================================================
# TARGET ELIGIBILITY
# ============================================================

# Primary cohort:
# - complete 365-day post-transfer window
# - at least 450 post-transfer minutes

conditions = pd.Series(True, index=df.index)

if "complete_365d_window" in df.columns:
    conditions &= df["complete_365d_window"].astype(bool)

if "eligible_450min" in df.columns:
    conditions &= df["eligible_450min"].astype(bool)

elif "post_minutes" in df.columns:
    conditions &= df["post_minutes"] >= 450

elif "minutes_post" in df.columns:
    conditions &= df["minutes_post"] >= 450

else:
    raise ValueError("Cannot identify post-transfer minutes eligibility.")


df = df[conditions].copy()

df = df[
    df[target_col].notna() &
    np.isfinite(df[target_col])
].copy()

print(f"Eligible cohort: {len(df):,}")


# ============================================================
# TRANSFER DATE
# ============================================================

if "transfer_date" not in df.columns:
    raise ValueError("transfer_date not found.")

df["transfer_date"] = pd.to_datetime(
    df["transfer_date"],
    errors="coerce"
)

df = df[df["transfer_date"].notna()].copy()

df = df.sort_values("transfer_date").reset_index(drop=True)


# ============================================================
# FEATURE SELECTION
# ============================================================

# ------------------------------------------------------------
# PLAYER / TRANSFER FEATURES
# ------------------------------------------------------------

player_numeric = [
    "age_at_transfer",
    "height_in_cm",
    "appearances_pre",
    "minutes_pre",
    "goals_pre",
    "assists_pre",
    "goals_per90_pre",
    "assists_per90_pre",
    "yellow_cards_pre",
    "red_cards_pre",
    "transfer_market_value",
    "market_value_pre",
    "transfer_fee"
]

player_categorical = [
    "position",
    "sub_position",
    "foot"
]


# ------------------------------------------------------------
# RICH DESTINATION CONTEXT
# ------------------------------------------------------------

rich_numeric = [
    # Team performance
    "rich_team_matches_last10",
    "rich_team_goals_for_avg10",
    "rich_team_goals_against_avg10",
    "rich_team_goal_difference_avg10",
    "rich_team_win_rate10",
    "rich_team_points_per_game10",

    # Tactical context
    "dominant_formation_frequency10",
    "formation_diversity10",

    # Manager context
    "manager_matches_last10",
    "unique_managers_last10",

    # Squad competition
    "same_position_competitor_count",
    "same_position_competitor_minutes",
    "same_position_avg_market_value",
    "same_position_top_competitor_minutes",

    "same_sub_position_competitor_count",
    "same_sub_position_competitor_minutes",
    "same_sub_position_top_minutes",

    # Position opportunity
    "position_minutes_share",

    # Previous destination familiarity
    "previous_destination_appearances",
    "previous_destination_minutes"
]

rich_categorical = [
    "dominant_formation10",
    "target_position_group"
]


# ============================================================
# ONLY USE FEATURES THAT ACTUALLY EXIST
# ============================================================

numeric_features = [
    c for c in player_numeric + rich_numeric
    if c in df.columns
]

categorical_features = [
    c for c in player_categorical + rich_categorical
    if c in df.columns
]

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

print(f"\nTotal features before encoding: "
      f"{len(numeric_features) + len(categorical_features)}")


# ============================================================
# X / y
# ============================================================

X = df[numeric_features + categorical_features].copy()
y = df[target_col].astype(float)


# ============================================================
# TEMPORAL 60 / 20 / 20 SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

X_train = X.iloc[:train_end].copy()
X_val = X.iloc[train_end:val_end].copy()
X_test = X.iloc[val_end:].copy()

y_train = y.iloc[:train_end].copy()
y_val = y.iloc[train_end:val_end].copy()
y_test = y.iloc[val_end:].copy()

dates_train = df["transfer_date"].iloc[:train_end]
dates_val = df["transfer_date"].iloc[train_end:val_end]
dates_test = df["transfer_date"].iloc[val_end:]

print("\n========== SPLIT ==========")

print(
    f"Train: {len(X_train):,} | "
    f"{dates_train.min().date()} → {dates_train.max().date()}"
)

print(
    f"Validation: {len(X_val):,} | "
    f"{dates_val.min().date()} → {dates_val.max().date()}"
)

print(
    f"Test: {len(X_test):,} | "
    f"{dates_test.min().date()} → {dates_test.max().date()}"
)


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ],
    remainder="drop"
)


# ============================================================
# MODEL C
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    max_features="sqrt"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# ============================================================
# TRAIN
# ============================================================

print("\nTraining rich Model C...")

pipeline.fit(X_train, y_train)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, X_split, y_split):

    pred = pipeline.predict(X_split)

    mae = mean_absolute_error(y_split, pred)
    rmse = np.sqrt(mean_squared_error(y_split, pred))
    r2 = r2_score(y_split, pred)

    print(
        f"{name:<12} "
        f"MAE={mae:.6f}  "
        f"RMSE={rmse:.6f}  "
        f"R²={r2:.6f}"
    )

    return pred, mae, rmse, r2


print("\n========== MODEL C RESULTS ==========")

train_pred, train_mae, train_rmse, train_r2 = evaluate(
    "Train", X_train, y_train
)

val_pred, val_mae, val_rmse, val_r2 = evaluate(
    "Validation", X_val, y_val
)

test_pred, test_mae, test_rmse, test_r2 = evaluate(
    "Test", X_test, y_test
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

test_results = df.iloc[val_end:].copy()

test_results["actual"] = y_test.values
test_results["predicted"] = test_pred
test_results["residual"] = (
    test_results["actual"] -
    test_results["predicted"]
)

test_results["absolute_error"] = (
    test_results["actual"] -
    test_results["predicted"]
).abs()

test_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_c_rich_test_predictions.csv"
    ),
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "split": [
        "train",
        "validation",
        "test"
    ],
    "n": [
        len(y_train),
        len(y_val),
        len(y_test)
    ],
    "mae": [
        train_mae,
        val_mae,
        test_mae
    ],
    "rmse": [
        train_rmse,
        val_rmse,
        test_rmse
    ],
    "r2": [
        train_r2,
        val_r2,
        test_r2
    ]
})

metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_c_rich_metrics.csv"
    ),
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========== SUMMARY ==========")

print(f"Cohort size: {len(df):,}")
print(f"Train:       {len(X_train):,}")
print(f"Validation:  {len(X_val):,}")
print(f"Test:        {len(X_test):,}")

print("\nTest performance:")
print(f"MAE  : {test_mae:.6f}")
print(f"RMSE : {test_rmse:.6f}")
print(f"R²   : {test_r2:.6f}")

print("\nSaved:")
print(
    os.path.join(
        OUTPUT_DIR,
        "model_c_rich_metrics.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "model_c_rich_test_predictions.csv"
    )
)

print("\nDone.")

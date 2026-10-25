import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE = r"C:\Users\rsuba\Downloads\R football analytics"

TRANSFER_FILE = BASE + r"\professional_transfer_dataset.csv"
TARGET_FILE = BASE + r"\post_transfer_performance.csv"
TEAM_FILE = BASE + r"\rich_team_context_dataset.csv"


print("=" * 100)
print("MODEL C — REGULARIZATION SEARCH")
print("=" * 100)


# ============================================================
# LOAD DATA
# ============================================================

transfers = pd.read_csv(TRANSFER_FILE)
target = pd.read_csv(TARGET_FILE)
team = pd.read_csv(TEAM_FILE)

print(f"Transfers: {len(transfers)}")
print(f"Target:    {len(target)}")
print(f"Team:      {len(team)}")


# ============================================================
# COMMON ELIGIBLE COHORT
# ============================================================

eligible_target = target[
    (target["complete_365_day_window"] == True) &
    (target["eligible_450min"] == True)
].copy()

# professional_transfer_dataset.csv does not contain transfer_row_id.
# Recreate the same transfer identity used by post_transfer_performance.csv
# and rich_team_context_dataset.csv.
#
# The original transfer dataset was indexed from 0, while transfer_row_id
# was defined as original row index + 1.

transfers = transfers.reset_index(drop=True)
transfers["transfer_row_id"] = transfers.index + 1

eligible_ids = set(eligible_target["transfer_row_id"])

transfers = transfers[
    transfers["transfer_row_id"].isin(eligible_ids)
].copy()

team = team[
    team["transfer_row_id"].isin(eligible_ids)
].copy()
# ============================================================
# MERGE
# ============================================================

# Keep the transfer dataset as the base table.
# This preserves all player-level pre-transfer variables.

transfer_base = transfers.copy()

# Keep only the target columns needed for the experiment.
target_base = eligible_target[
    [
        "transfer_row_id",
        "post_GA90",
        "complete_365_day_window",
        "eligible_450min"
    ]
].copy()

# Keep the rich team context.
team_base = team.copy()

# Avoid accidental duplicate columns from the context datasets.
# transfer_row_id is the only intended merge key.

target_extra = [
    c for c in target_base.columns
    if c != "transfer_row_id"
]

team_extra = [
    c for c in team_base.columns
    if c != "transfer_row_id"
]

# Merge player/transfer data + target
df = transfer_base.merge(
    target_base,
    on="transfer_row_id",
    how="inner",
    suffixes=("", "_target")
)

# Merge rich team context
df = df.merge(
    team_base,
    on="transfer_row_id",
    how="inner",
    suffixes=("", "_team")
)

print(f"Common cohort after merge: {len(df)}")

# Verify the required player features survived the merge.
required_player_columns = [
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
    "transfer_fee",
    "position",
    "sub_position",
    "foot"
]

missing_player = [
    c for c in required_player_columns
    if c not in df.columns
]

if missing_player:
    print("\nERROR — required player columns missing:")
    print(missing_player)
    print("\nAvailable columns:")
    print(df.columns.tolist())
    raise SystemExit(1)

print("Player feature check: PASSED")

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

team_numeric = [
    "rich_team_matches_last10",
    "rich_team_goals_for_avg10",
    "rich_team_goals_against_avg10",
    "rich_team_goal_difference_avg10",
    "rich_team_win_rate10",
    "rich_team_points_per_game10",
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

team_categorical = [
    "dominant_formation10",
    "target_position_group"
]

numeric_features = player_numeric + team_numeric
categorical_features = player_categorical + team_categorical

features = numeric_features + categorical_features

X = df[features].copy()
y = df["post_GA90"].copy()


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

df = df.sort_values("transfer_date").reset_index(drop=True)

X = df[features].copy()
y = df["post_GA90"].copy()

n = len(df)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:val_end]

y_train = y.iloc[:train_end]
y_val = y.iloc[train_end:val_end]

print("\nSplit:")
print(f"Train      : {len(X_train)}")
print(f"Validation : {len(X_val)}")
print(f"Test       : {n - val_end}")

print(
    f"Train dates: "
    f"{df.iloc[0]['transfer_date']} to "
    f"{df.iloc[train_end - 1]['transfer_date']}"
)

print(
    f"Validation dates: "
    f"{df.iloc[train_end]['transfer_date']} to "
    f"{df.iloc[val_end - 1]['transfer_date']}"
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
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)


# ============================================================
# REGULARIZATION CONFIGURATIONS
# ============================================================

configs = [
    {
        "name": "Depth8_Leaf10",
        "max_depth": 8,
        "min_samples_leaf": 10,
        "min_samples_split": 2,
        "max_features": "sqrt"
    },
    {
        "name": "Depth10_Leaf10",
        "max_depth": 10,
        "min_samples_leaf": 10,
        "min_samples_split": 2,
        "max_features": "sqrt"
    },
    {
        "name": "Depth12_Leaf10",
        "max_depth": 12,
        "min_samples_leaf": 10,
        "min_samples_split": 2,
        "max_features": "sqrt"
    },
    {
        "name": "Depth15_Leaf10",
        "max_depth": 15,
        "min_samples_leaf": 10,
        "min_samples_split": 2,
        "max_features": "sqrt"
    },
    {
        "name": "Depth10_Leaf15",
        "max_depth": 10,
        "min_samples_leaf": 15,
        "min_samples_split": 2,
        "max_features": "sqrt"
    },
    {
        "name": "Depth12_Leaf15",
        "max_depth": 12,
        "min_samples_leaf": 15,
        "min_samples_split": 2,
        "max_features": "sqrt"
    },
    {
        "name": "Depth15_Leaf15",
        "max_depth": 15,
        "min_samples_leaf": 15,
        "min_samples_split": 2,
        "max_features": "sqrt"
    },
    {
        "name": "Depth12_Leaf20",
        "max_depth": 12,
        "min_samples_leaf": 20,
        "min_samples_split": 2,
        "max_features": "sqrt"
    }
]


# ============================================================
# RUN SEARCH
# ============================================================

results = []

for cfg in configs:

    print("\n" + "-" * 100)
    print(cfg["name"])

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=cfg["max_depth"],
        min_samples_split=cfg["min_samples_split"],
        min_samples_leaf=cfg["min_samples_leaf"],
        max_features=cfg["max_features"],
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_processed, y_train)

    train_pred = model.predict(X_train_processed)
    val_pred = model.predict(X_val_processed)

    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)

    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_r2 = r2_score(y_val, val_pred)

    gap = train_r2 - val_r2

    print(f"Train R²       : {train_r2:.6f}")
    print(f"Validation R²  : {val_r2:.6f}")
    print(f"Validation MAE : {val_mae:.6f}")
    print(f"Validation RMSE: {val_rmse:.6f}")
    print(f"R² Gap         : {gap:.6f}")

    results.append({
        "configuration": cfg["name"],
        "max_depth": cfg["max_depth"],
        "min_samples_leaf": cfg["min_samples_leaf"],
        "min_samples_split": cfg["min_samples_split"],
        "max_features": cfg["max_features"],
        "train_r2": train_r2,
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "validation_r2": val_r2,
        "validation_mae": val_mae,
        "validation_rmse": val_rmse,
        "r2_gap": gap
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "validation_r2",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 100)
print("REGULARIZATION SEARCH RESULTS")
print("=" * 100)

print(
    results_df[
        [
            "configuration",
            "train_r2",
            "validation_r2",
            "validation_mae",
            "validation_rmse",
            "r2_gap"
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR = BASE + r"\model_c_regularization_results"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_file = OUTPUT_DIR + r"\model_c_regularization_search.csv"

results_df.to_csv(
    output_file,
    index=False
)

print("\nSaved:")
print(output_file)

best = results_df.iloc[0]

print("\n" + "=" * 100)
print("BEST VALIDATION CONFIGURATION")
print("=" * 100)

print(f"Configuration   : {best['configuration']}")
print(f"Max depth       : {best['max_depth']}")
print(f"Min leaf        : {best['min_samples_leaf']}")
print(f"Validation R²   : {best['validation_r2']:.6f}")
print(f"Validation MAE  : {best['validation_mae']:.6f}")
print(f"Validation RMSE : {best['validation_rmse']:.6f}")
print(f"Train R²        : {best['train_r2']:.6f}")
print(f"R² gap          : {best['r2_gap']:.6f}")

print("\nIMPORTANT:")
print("The test set was NOT used in this search.")
print("=" * 100)
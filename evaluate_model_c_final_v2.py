import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\rsuba\Downloads\R football analytics"

TRANSFER_FILE = os.path.join(
    BASE_DIR, "professional_transfer_dataset.csv"
)

TARGET_FILE = os.path.join(
    BASE_DIR, "post_transfer_performance.csv"
)

TEAM_FILE = os.path.join(
    BASE_DIR, "rich_team_context_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "model_c_final_v2_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

print("=" * 100)
print("MODEL C — FINAL UNTOUCHED TEST EVALUATION")
print("=" * 100)

transfers = pd.read_csv(TRANSFER_FILE)
target = pd.read_csv(TARGET_FILE)
team = pd.read_csv(TEAM_FILE)

print(f"Transfers: {len(transfers)}")
print(f"Target:    {len(target)}")
print(f"Team:      {len(team)}")


# ============================================================
# TRANSFER ID
# ============================================================

if "transfer_row_id" not in transfers.columns:
    transfers = transfers.reset_index(drop=True)
    transfers["transfer_row_id"] = transfers.index + 1

if "transfer_row_id" not in target.columns:
    raise ValueError("Target file has no transfer_row_id.")

if "transfer_row_id" not in team.columns:
    raise ValueError("Team file has no transfer_row_id.")


# ============================================================
# TARGET
# ============================================================

target["transfer_date"] = pd.to_datetime(
    target["transfer_date"],
    errors="coerce"
)

target["post_GA90"] = pd.to_numeric(
    target["post_GA90"],
    errors="coerce"
)


# ============================================================
# EXACT PRIMARY COHORT
# ============================================================
#
# post_transfer_performance.csv already contains the validated
# eligibility flags. Use them directly rather than reconstructing
# the cohort.
#
# Primary cohort:
#   1. Complete 365-day post-transfer window
#   2. At least 450 post-transfer minutes
#
# ============================================================

target["transfer_date"] = pd.to_datetime(
    target["transfer_date"],
    errors="coerce"
)

target["post_GA90"] = pd.to_numeric(
    target["post_GA90"],
    errors="coerce"
)

target["complete_365_day_window"] = target[
    "complete_365_day_window"
].astype(bool)

target["eligible_450min"] = target[
    "eligible_450min"
].astype(bool)


eligible_target = target[
    target["complete_365_day_window"]
    & target["eligible_450min"]
    & target["post_GA90"].notna()
].copy()


print(
    f"\nEligible target records: "
    f"{len(eligible_target)}"
)


# ============================================================
# EXACT COMMON COHORT
# ============================================================

common_ids = set(
    eligible_target["transfer_row_id"]
)

common_ids &= set(
    transfers["transfer_row_id"]
)

common_ids &= set(
    team["transfer_row_id"]
)


print(
    f"Common cohort after merge: "
    f"{len(common_ids)}"
)


# SAFETY CHECK:
# The validated primary cohort is exactly 6,108 records.
# Stop rather than accidentally evaluating a different cohort.

if len(common_ids) != 6108:
    raise ValueError(
        f"Expected exactly 6108 records, "
        f"but got {len(common_ids)}. "
        f"Stop before evaluating the test set."
    )
# ============================================================
# MERGE
# ============================================================

df = transfers[
    transfers["transfer_row_id"].isin(common_ids)
].copy()

target_small = eligible_target[
    [
        "transfer_row_id",
        "post_GA90"
    ]
].copy()

team_small = team[
    team["transfer_row_id"].isin(common_ids)
].copy()


df = df.merge(
    target_small,
    on="transfer_row_id",
    how="inner"
)

df = df.merge(
    team_small,
    on="transfer_row_id",
    how="inner",
    suffixes=("", "_team")
)

df = df.drop_duplicates(
    subset=["transfer_row_id"]
).copy()

print(
    f"Merged final cohort: {len(df)}"
)


# ============================================================
# DATE
# ============================================================

df["transfer_date"] = pd.to_datetime(
    df["transfer_date"],
    errors="coerce"
)

df = df.dropna(
    subset=["transfer_date", "post_GA90"]
).copy()


# ============================================================
# MODEL C FEATURES
# ============================================================

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

numeric_features = (
    player_numeric
    + team_numeric
)

categorical_features = (
    player_categorical
    + team_categorical
)

all_features = (
    numeric_features
    + categorical_features
)


# ============================================================
# FEATURE CHECK
# ============================================================

missing_features = [
    f for f in all_features
    if f not in df.columns
]

if missing_features:
    print("\nMISSING FEATURES:")
    for f in missing_features:
        print(" -", f)

    raise ValueError(
        "Required Model C features are missing."
    )

print("\nFeature check: PASSED")
print(
    f"Numeric features     : "
    f"{len(numeric_features)}"
)

print(
    f"Categorical features : "
    f"{len(categorical_features)}"
)

print(
    f"Total raw features   : "
    f"{len(all_features)}"
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

df = df.sort_values(
    "transfer_date"
).reset_index(drop=True)

n = len(df)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()


print("\n" + "=" * 100)
print("CHRONOLOGICAL SPLIT")
print("=" * 100)

print(f"Train      : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")

print(
    f"Train dates: "
    f"{train_df['transfer_date'].min().date()} "
    f"to "
    f"{train_df['transfer_date'].max().date()}"
)

print(
    f"Validation dates: "
    f"{val_df['transfer_date'].min().date()} "
    f"to "
    f"{val_df['transfer_date'].max().date()}"
)

print(
    f"Test dates: "
    f"{test_df['transfer_date'].min().date()} "
    f"to "
    f"{test_df['transfer_date'].max().date()}"
)


# ============================================================
# X / Y
# ============================================================

X_train = train_df[all_features].copy()
y_train = train_df["post_GA90"].copy()

X_val = val_df[all_features].copy()
y_val = val_df["post_GA90"].copy()

X_test = test_df[all_features].copy()
y_test = test_df["post_GA90"].copy()


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
# FIT PREPROCESSOR ONLY ON TRAINING
# ============================================================

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_val_processed = preprocessor.transform(
    X_val
)

X_test_processed = preprocessor.transform(
    X_test
)

print("\nProcessed feature dimensions:")
print(
    f"Train      : "
    f"{X_train_processed.shape}"
)

print(
    f"Validation : "
    f"{X_val_processed.shape}"
)

print(
    f"Test       : "
    f"{X_test_processed.shape}"
)


# ============================================================
# FINAL VALIDATION-SELECTED MODEL
# ============================================================

print("\n" + "=" * 100)
print("FINAL MODEL C CONFIGURATION")
print("=" * 100)

print("n_estimators      : 300")
print("max_depth         : 12")
print("min_samples_split : 2")
print("min_samples_leaf  : 10")
print("max_features      : sqrt")
print("random_state      : 42")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_split=2,
    min_samples_leaf=10,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining...")

model.fit(
    X_train_processed,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

train_pred = model.predict(
    X_train_processed
)

val_pred = model.predict(
    X_val_processed
)

test_pred = model.predict(
    X_test_processed
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return mae, rmse, r2


train_mae, train_rmse, train_r2 = calculate_metrics(
    y_train,
    train_pred
)

val_mae, val_rmse, val_r2 = calculate_metrics(
    y_val,
    val_pred
)

test_mae, test_rmse, test_r2 = calculate_metrics(
    y_test,
    test_pred
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 100)
print("FINAL MODEL C RESULTS")
print("=" * 100)

print("\nTRAIN")
print(f"MAE  : {train_mae:.6f}")
print(f"RMSE : {train_rmse:.6f}")
print(f"R²   : {train_r2:.6f}")

print("\nVALIDATION")
print(f"MAE  : {val_mae:.6f}")
print(f"RMSE : {val_rmse:.6f}")
print(f"R²   : {val_r2:.6f}")

print("\nTEST — FINAL UNTOUCHED EVALUATION")
print(f"MAE  : {test_mae:.6f}")
print(f"RMSE : {test_rmse:.6f}")
print(f"R²   : {test_r2:.6f}")

print(
    f"\nTrain–Validation R² Gap : "
    f"{train_r2 - val_r2:.6f}"
)

print(
    f"Validation–Test R² Difference : "
    f"{val_r2 - test_r2:.6f}"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame([
    {
        "split": "train",
        "n": len(train_df),
        "mae": train_mae,
        "rmse": train_rmse,
        "r2": train_r2
    },
    {
        "split": "validation",
        "n": len(val_df),
        "mae": val_mae,
        "rmse": val_rmse,
        "r2": val_r2
    },
    {
        "split": "test",
        "n": len(test_df),
        "mae": test_mae,
        "rmse": test_rmse,
        "r2": test_r2
    }
])

metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_c_final_v2_metrics.csv"
    ),
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_predictions = test_df[
    [
        "transfer_row_id",
        "transfer_date"
    ]
].copy()

test_predictions[
    "actual_post_GA90"
] = y_test.values

test_predictions[
    "predicted_post_GA90"
] = test_pred

test_predictions[
    "absolute_error"
] = np.abs(
    test_predictions["actual_post_GA90"]
    -
    test_predictions["predicted_post_GA90"]
)

test_predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_c_final_v2_test_predictions.csv"
    ),
    index=False
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_names = (
    preprocessor.get_feature_names_out()
)

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)

importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_c_final_v2_feature_importance.csv"
    ),
    index=False
)


# ============================================================
# TOP FEATURES
# ============================================================

print("\n" + "=" * 100)
print("TOP 20 FEATURE IMPORTANCES")
print("=" * 100)

print(
    importance.head(20).to_string(
        index=False
    )
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 100)
print("FINAL EVALUATION COMPLETE")
print("=" * 100)

print("\nSaved:")

print(
    os.path.join(
        OUTPUT_DIR,
        "model_c_final_v2_metrics.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "model_c_final_v2_test_predictions.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "model_c_final_v2_feature_importance.csv"
    )
)

print("\nIMPORTANT:")
print(
    "The test set was evaluated only after "
    "model selection."
)

print(
    "Do NOT tune the model based on these "
    "test results."
)

print("=" * 100)
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

ROOT = r"C:\Users\rsuba\Downloads\R football analytics"

TRANSFER_FILE = os.path.join(
    ROOT, "professional_transfer_dataset.csv"
)

TARGET_FILE = os.path.join(
    ROOT, "post_transfer_performance.csv"
)

TEAM_FILE = os.path.join(
    ROOT, "rich_team_context_dataset.csv"
)

COACH_FILE = os.path.join(
    ROOT, "coach_tactical_context_v21.csv"
)

OUT_DIR = os.path.join(
    ROOT, "model_c_v21_results"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

transfers = pd.read_csv(TRANSFER_FILE)
target = pd.read_csv(TARGET_FILE)
team = pd.read_csv(TEAM_FILE)
coach = pd.read_csv(COACH_FILE)

# Recreate the exact transfer identity used when
# post_transfer_performance.csv and context datasets were built.
transfers["transfer_row_id"] = np.arange(
    1,
    len(transfers) + 1
)

print(
    f"Created transfer_row_id: "
    f"{transfers['transfer_row_id'].min()} "
    f"to {transfers['transfer_row_id'].max()}"
)

print(f"Transfers: {len(transfers):,}")
print(f"Target rows: {len(target):,}")
print(f"Team context rows: {len(team):,}")
print(f"Coach context rows: {len(coach):,}")


# ============================================================
# BUILD COMMON COHORT
# ============================================================

print()
print("=" * 70)
print("BUILDING COMMON COHORT")
print("=" * 70)

df = transfers.merge(
    target,
    on="transfer_row_id",
    how="inner",
    suffixes=("", "_target")
)

df = df.merge(
    team,
    on="transfer_row_id",
    how="left",
    suffixes=("", "_team")
)

df = df.merge(
    coach,
    on="transfer_row_id",
    how="left",
    suffixes=("", "_coach")
)

# Primary eligibility threshold
df = df[
    df["eligible_450min"] == 1
].copy()

# Guarantee one row per transfer
df = df.drop_duplicates(
    subset=["transfer_row_id"]
).copy()

print(
    f"Exact C v2.1 cohort: {len(df):,}"
)


# ============================================================
# TARGET
# ============================================================

TARGET = "post_GA90"

if TARGET not in df.columns:
    print()
    print("Available columns containing 'GA' or '90':")

    candidates = [
        c for c in df.columns
        if "ga" in c.lower()
        or "90" in c.lower()
        or "goal" in c.lower()
    ]

    print(candidates)

    raise ValueError(
        "Target column post_GA90 was not found."
    )

print(f"Target: {TARGET}")


# ============================================================
# PLAYER FEATURES
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

# ============================================================
# DESTINATION TEAM FEATURES
# Same six features used in B-NEW
# ============================================================

team_features = [
    "rich_team_matches_last10",
    "rich_team_goals_for_avg10",
    "rich_team_goals_against_avg10",
    "rich_team_goal_difference_avg10",
    "rich_team_win_rate10",
    "rich_team_points_per_game10"
]


# ============================================================
# COACH / TACTICAL FEATURES
# C v2.1
# ============================================================

coach_features = [
    "coach_known_pretransfer",
    "coach_position_group_minutes_share",
    "coach_position_group_start_rate",
    "coach_exact_position_minutes_share",
    "coach_exact_position_start_rate",
    "coach_position_competitor_count",
    "coach_position_competitor_minutes",
    "coach_position_rotation_hhi",
    "coach_formation_stability",
    "coach_observed_tenure_days"
]


# ============================================================
# FEATURE VALIDATION
# ============================================================

all_features = (
    player_numeric
    + player_categorical
    + team_features
    + coach_features
)

missing_features = [
    feature
    for feature in all_features
    if feature not in df.columns
]

if missing_features:

    print()
    print("=" * 70)
    print("MISSING FEATURES")
    print("=" * 70)

    for feature in missing_features:
        print(" -", feature)

    raise ValueError(
        "One or more required features are missing."
    )

print()
print("=" * 70)
print("FEATURE SUMMARY")
print("=" * 70)

print(
    f"Player features: "
    f"{len(player_numeric) + len(player_categorical)}"
)

print(
    f"Team features: "
    f"{len(team_features)}"
)

print(
    f"Coach/tactical features: "
    f"{len(coach_features)}"
)

print(
    f"Total before encoding: "
    f"{len(all_features)}"
)


# ============================================================
# CHECK COACH COVERAGE
# ============================================================

print()
print("=" * 70)
print("COACH COVERAGE IN C v2.1 COHORT")
print("=" * 70)

coach_known_count = (
    df["coach_known_pretransfer"] == 1
).sum()

coach_unknown_count = (
    df["coach_known_pretransfer"] == 0
).sum()

print(
    f"Coach known:   {coach_known_count:,} "
    f"({coach_known_count / len(df) * 100:.2f}%)"
)

print(
    f"Coach unknown: {coach_unknown_count:,} "
    f"({coach_unknown_count / len(df) * 100:.2f}%)"
)


# ============================================================
# CHRONOLOGICAL ORDER
# ============================================================

df["transfer_date"] = pd.to_datetime(
    df["transfer_date"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "transfer_date",
        TARGET
    ]
).copy()

df = df.sort_values(
    "transfer_date"
).reset_index(drop=True)


# ============================================================
# CHRONOLOGICAL 60 / 20 / 20 SPLIT
# ============================================================

print()
print("=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

n = len(df)

train_end = int(
    n * 0.60
)

val_end = int(
    n * 0.80
)

train_df = df.iloc[
    :train_end
].copy()

val_df = df.iloc[
    train_end:val_end
].copy()

test_df = df.iloc[
    val_end:
].copy()

print(
    f"Train:      {len(train_df):,}"
)

print(
    f"Validation:  {len(val_df):,}"
)

print(
    f"Test:       {len(test_df):,}"
)

print()

print(
    "Train dates: "
    f"{train_df['transfer_date'].min().date()} "
    "-> "
    f"{train_df['transfer_date'].max().date()}"
)

print(
    "Validation dates: "
    f"{val_df['transfer_date'].min().date()} "
    "-> "
    f"{val_df['transfer_date'].max().date()}"
)

print(
    "Test dates: "
    f"{test_df['transfer_date'].min().date()} "
    "-> "
    f"{test_df['transfer_date'].max().date()}"
)


# ============================================================
# PREPROCESSING
# ============================================================

numeric_features = (
    player_numeric
    + team_features
    + coach_features
)

categorical_features = (
    player_categorical
)

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    max_features=1.0
)


# ============================================================
# COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# TRAIN / VALIDATION / TEST DATA
# ============================================================

X_train = train_df[
    all_features
]

y_train = train_df[
    TARGET
]

X_val = val_df[
    all_features
]

y_val = val_df[
    TARGET
]

X_test = test_df[
    all_features
]

y_test = test_df[
    TARGET
]


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("=" * 70)
print("TRAINING C v2.1 RANDOM FOREST")
print("=" * 70)

pipeline.fit(
    X_train,
    y_train
)

print(
    "Training complete."
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    split_name,
    X,
    y
):

    predictions = pipeline.predict(
        X
    )

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions
        )
    )

    r2 = r2_score(
        y,
        predictions
    )

    print()
    print(split_name)
    print("-" * 40)

    print(
        f"MAE :  {mae:.6f}"
    )

    print(
        f"RMSE:  {rmse:.6f}"
    )

    print(
        f"R²  :  {r2:.6f}"
    )

    return {
        "split": split_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }, predictions


# ============================================================
# EVALUATE
# ============================================================

train_metrics, train_predictions = evaluate_model(
    "TRAIN",
    X_train,
    y_train
)

val_metrics, val_predictions = evaluate_model(
    "VALIDATION",
    X_val,
    y_val
)

test_metrics, test_predictions = evaluate_model(
    "TEST",
    X_test,
    y_test
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame(
    [
        train_metrics,
        val_metrics,
        test_metrics
    ]
)

metrics_path = os.path.join(
    OUT_DIR,
    "model_c_v21_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_columns = [
    "transfer_row_id",
    "player_id",
    "transfer_date",
    "from_club_name",
    "to_club_name",
    TARGET
]

# Only use columns that actually exist
prediction_columns = [
    c for c in prediction_columns
    if c in test_df.columns
]

predictions_df = test_df[
    prediction_columns
].copy()

predictions_df[
    "predicted"
] = test_predictions

predictions_df[
    "residual"
] = (
    predictions_df[TARGET]
    - predictions_df["predicted"]
)

predictions_df[
    "absolute_error"
] = np.abs(
    predictions_df["residual"]
)

predictions_path = os.path.join(
    OUT_DIR,
    "model_c_v21_test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 70)
print("EXTRACTING FEATURE IMPORTANCE")
print("=" * 70)

fitted_preprocessor = (
    pipeline.named_steps[
        "preprocessor"
    ]
)

rf = (
    pipeline.named_steps[
        "model"
    ]
)

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": rf.feature_importances_
    }
)

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)

importance_path = os.path.join(
    OUT_DIR,
    "model_c_v21_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)


# ============================================================
# TOP FEATURES
# ============================================================

print()
print("=" * 70)
print("TOP 30 FEATURES")
print("=" * 70)

print(
    importance_df
    .head(30)
    .to_string(index=False)
)


# ============================================================
# COACH / TACTICAL FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 70)
print("COACH / TACTICAL FEATURE IMPORTANCE")
print("=" * 70)

for feature in coach_features:

    matching = importance_df[
        importance_df["feature"].str.contains(
            feature,
            regex=False
        )
    ]

    if len(matching) > 0:

        total_importance = (
            matching["importance"].sum()
        )

        print(
            f"{feature:50s}"
            f"{total_importance:.6f}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("C v2.1 COMPLETE")
print("=" * 70)

print(
    f"Cohort: {len(df):,}"
)

print(
    f"Train: {len(train_df):,}"
)

print(
    f"Validation: {len(val_df):,}"
)

print(
    f"Test: {len(test_df):,}"
)

print()

print(
    "TEST RESULTS"
)

print(
    f"MAE :  {test_metrics['MAE']:.6f}"
)

print(
    f"RMSE:  {test_metrics['RMSE']:.6f}"
)

print(
    f"R²  :  {test_metrics['R2']:.6f}"
)

print()
print(
    "Files saved:"
)

print(
    metrics_path
)

print(
    predictions_path
)

print(
    importance_path
)

print()
print(
    "DONE."
)
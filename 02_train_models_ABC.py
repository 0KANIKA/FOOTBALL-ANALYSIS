# ================================================================
# 02_train_models_ABC.py
# Context-Aware Prediction of Post-Transfer Football Performance
#
# Model A = Player characteristics
# Model B = Player + destination-team performance
# Model C = Player + richer destination-team context
# ================================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ================================================================
# 1. PATHS
# ================================================================

BASE = Path(r"C:\Users\rsuba\Downloads\R football analytics")

DATA_FILE = BASE / "FINAL_MODEL_C_CLEAN.csv"

OUTPUT_DIR = BASE / "model_ABC_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ================================================================
# 2. LOAD DATA
# ================================================================

print("=" * 80)
print("LOADING FINAL DATASET")
print("=" * 80)

df = pd.read_csv(DATA_FILE)

df["transfer_date"] = pd.to_datetime(df["transfer_date"])

print(f"Dataset rows: {len(df)}")


# ================================================================
# 3. BASIC VALIDATION
# ================================================================

required_columns = [
    "transfer_row_id",
    "transfer_date",
    "post_GA90"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")

if df["transfer_row_id"].duplicated().any():
    raise ValueError("Duplicate transfer_row_id detected.")

if df["post_GA90"].isna().any():
    raise ValueError("Missing target values detected.")


# ================================================================
# 4. SORT CHRONOLOGICALLY
# ================================================================

df = df.sort_values("transfer_date").reset_index(drop=True)


# ================================================================
# 5. CHRONOLOGICAL 60/20/20 SPLIT
# ================================================================

n = len(df)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

train_df["dataset_split"] = "train"
val_df["dataset_split"] = "validation"
test_df["dataset_split"] = "test"

print("\n" + "=" * 80)
print("CHRONOLOGICAL SPLIT")
print("=" * 80)

print(f"Train      : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")

print("\nDate ranges:")

print(
    "Train:",
    train_df["transfer_date"].min().date(),
    "->",
    train_df["transfer_date"].max().date()
)

print(
    "Validation:",
    val_df["transfer_date"].min().date(),
    "->",
    val_df["transfer_date"].max().date()
)

print(
    "Test:",
    test_df["transfer_date"].min().date(),
    "->",
    test_df["transfer_date"].max().date()
)


# ================================================================
# 6. TARGET
# ================================================================

TARGET = "post_GA90"


# ================================================================
# 7. MODEL A FEATURES
# ================================================================
#
# Player characteristics available BEFORE the transfer.
#
# We intentionally exclude:
# - post-transfer information
# - destination-team context
# - transfer_row_id
# - cleaning/audit columns
# - player_id (identifier)
# - club names/IDs
#
# ================================================================

PLAYER_NUMERIC = [
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

PLAYER_CATEGORICAL = [
    "position",
    "sub_position",
    "foot"
]


# ================================================================
# 8. MODEL B FEATURES
# ================================================================
#
# Player characteristics
# +
# destination team's recent performance
#
# The six core team-performance variables are:
# - goals for
# - goals against
# - goal difference
# - win rate
# - points per game
# - recent match count
#
# ================================================================

TEAM_PERFORMANCE_NUMERIC = [
    "rich_team_matches_last10",
    "rich_team_goals_for_avg10",
    "rich_team_goals_against_avg10",
    "rich_team_goal_difference_avg10",
    "rich_team_win_rate10",
    "rich_team_points_per_game10"
]


# ================================================================
# 9. MODEL C FEATURES
# ================================================================
#
# Player
# +
# team performance
# +
# tactical
# +
# manager
# +
# squad opportunity
# +
# player-team context
#
# ================================================================

TEAM_RICH_NUMERIC = [
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

TEAM_RICH_CATEGORICAL = [
    "dominant_formation10",
    "target_position_group"
]


# ================================================================
# 10. COACH / TACTICAL FEATURES
# ================================================================

COACH_NUMERIC = [
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


# ================================================================
# 11. MODEL FEATURE SETS
# ================================================================

MODEL_A_NUMERIC = PLAYER_NUMERIC

MODEL_A_CATEGORICAL = PLAYER_CATEGORICAL


MODEL_B_NUMERIC = (
    PLAYER_NUMERIC
    + TEAM_PERFORMANCE_NUMERIC
)

MODEL_B_CATEGORICAL = PLAYER_CATEGORICAL


MODEL_C_NUMERIC = (
    PLAYER_NUMERIC
    + TEAM_PERFORMANCE_NUMERIC
    + TEAM_RICH_NUMERIC
    + COACH_NUMERIC
)

MODEL_C_CATEGORICAL = (
    PLAYER_CATEGORICAL
    + TEAM_RICH_CATEGORICAL
)


# ================================================================
# 12. CHECK FEATURES EXIST
# ================================================================

all_features = (
    MODEL_C_NUMERIC
    + MODEL_C_CATEGORICAL
)

missing_features = [
    col for col in all_features
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        "Missing model features:\n"
        + "\n".join(missing_features)
    )

print("\n" + "=" * 80)
print("FEATURE COUNTS")
print("=" * 80)

print(
    "Model A:",
    len(MODEL_A_NUMERIC),
    "numeric +",
    len(MODEL_A_CATEGORICAL),
    "categorical"
)

print(
    "Model B:",
    len(MODEL_B_NUMERIC),
    "numeric +",
    len(MODEL_B_CATEGORICAL),
    "categorical"
)

print(
    "Model C:",
    len(MODEL_C_NUMERIC),
    "numeric +",
    len(MODEL_C_CATEGORICAL),
    "categorical"
)


# ================================================================
# 13. PREPROCESSOR
# ================================================================

def create_preprocessor(numeric_features, categorical_features):

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
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
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop"
    )

    return preprocessor


# ================================================================
# 14. MODEL CREATION
# ================================================================

def create_model(numeric_features, categorical_features):

    preprocessor = create_preprocessor(
        numeric_features,
        categorical_features
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


# ================================================================
# 15. METRIC FUNCTION
# ================================================================

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

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ================================================================
# 16. TRAIN + EVALUATE MODEL
# ================================================================

def run_model(
    model_name,
    numeric_features,
    categorical_features
):

    print("\n")
    print("=" * 80)
    print(f"TRAINING MODEL {model_name}")
    print("=" * 80)

    feature_columns = (
        numeric_features
        + categorical_features
    )

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET]

    X_val = val_df[feature_columns]
    y_val = val_df[TARGET]

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET]

    print(f"Features used: {len(feature_columns)}")

    print("Training...")

    pipeline = create_model(
        numeric_features,
        categorical_features
    )

    pipeline.fit(
        X_train,
        y_train
    )

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------

    train_pred = pipeline.predict(X_train)
    val_pred = pipeline.predict(X_val)
    test_pred = pipeline.predict(X_test)

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    train_metrics = calculate_metrics(
        y_train,
        train_pred
    )

    val_metrics = calculate_metrics(
        y_val,
        val_pred
    )

    test_metrics = calculate_metrics(
        y_test,
        test_pred
    )

    print("\nResults:")
    print("-" * 80)

    print(
        f"Train      | "
        f"MAE={train_metrics['MAE']:.6f} | "
        f"RMSE={train_metrics['RMSE']:.6f} | "
        f"R²={train_metrics['R2']:.6f}"
    )

    print(
        f"Validation | "
        f"MAE={val_metrics['MAE']:.6f} | "
        f"RMSE={val_metrics['RMSE']:.6f} | "
        f"R²={val_metrics['R2']:.6f}"
    )

    print(
        f"Test       | "
        f"MAE={test_metrics['MAE']:.6f} | "
        f"RMSE={test_metrics['RMSE']:.6f} | "
        f"R²={test_metrics['R2']:.6f}"
    )

    # ------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------

    prediction_df = test_df[
        [
            "transfer_row_id",
            "player_id",
            "transfer_date",
            "from_club_name",
            "to_club_name",
            "position",
            "post_GA90"
        ]
    ].copy()

    prediction_df["predicted_GA90"] = test_pred

    prediction_df["absolute_error"] = (
        prediction_df["post_GA90"]
        - prediction_df["predicted_GA90"]
    ).abs()

    prediction_df["signed_error"] = (
        prediction_df["predicted_GA90"]
        - prediction_df["post_GA90"]
    )

    prediction_file = (
        OUTPUT_DIR
        / f"model_{model_name}_test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_file,
        index=False
    )

    # ------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------

    metrics_rows = []

    for split_name, metrics in [
        ("train", train_metrics),
        ("validation", val_metrics),
        ("test", test_metrics)
    ]:

        metrics_rows.append({
            "model": model_name,
            "split": split_name,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"]
        })

    metrics_df = pd.DataFrame(metrics_rows)

    metrics_file = (
        OUTPUT_DIR
        / f"model_{model_name}_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_file,
        index=False
    )

    # ------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------

    preprocessor = pipeline.named_steps["preprocessor"]
    rf_model = pipeline.named_steps["model"]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance_file = (
        OUTPUT_DIR
        / f"model_{model_name}_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_file,
        index=False
    )

    print(
        f"\nSaved predictions: {prediction_file.name}"
    )

    print(
        f"Saved metrics: {metrics_file.name}"
    )

    print(
        f"Saved feature importance: "
        f"{importance_file.name}"
    )

    print("\nTop 15 features:")

    print(
        importance_df.head(15).to_string(
            index=False
        )
    )

    return {
        "model": model_name,
        "pipeline": pipeline,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "importance": importance_df
    }


# ================================================================
# 17. RUN MODEL A
# ================================================================

result_A = run_model(
    "A",
    MODEL_A_NUMERIC,
    MODEL_A_CATEGORICAL
)


# ================================================================
# 18. RUN MODEL B
# ================================================================

result_B = run_model(
    "B",
    MODEL_B_NUMERIC,
    MODEL_B_CATEGORICAL
)


# ================================================================
# 19. RUN MODEL C
# ================================================================

result_C = run_model(
    "C",
    MODEL_C_NUMERIC,
    MODEL_C_CATEGORICAL
)


# ================================================================
# 20. FINAL COMPARISON
# ================================================================

comparison_rows = []

for result in [
    result_A,
    result_B,
    result_C
]:

    comparison_rows.append({
        "model": result["model"],

        "train_MAE":
            result["train_metrics"]["MAE"],

        "train_RMSE":
            result["train_metrics"]["RMSE"],

        "train_R2":
            result["train_metrics"]["R2"],

        "validation_MAE":
            result["val_metrics"]["MAE"],

        "validation_RMSE":
            result["val_metrics"]["RMSE"],

        "validation_R2":
            result["val_metrics"]["R2"],

        "test_MAE":
            result["test_metrics"]["MAE"],

        "test_RMSE":
            result["test_metrics"]["RMSE"],

        "test_R2":
            result["test_metrics"]["R2"]
    })


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_file = (
    OUTPUT_DIR
    / "ABC_model_comparison.csv"
)

comparison_df.to_csv(
    comparison_file,
    index=False
)


# ================================================================
# 21. PRINT FINAL COMPARISON
# ================================================================

print("\n")
print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ================================================================
# 22. CALCULATE INCREMENTAL IMPROVEMENT
# ================================================================

A_test = result_A["test_metrics"]
B_test = result_B["test_metrics"]
C_test = result_C["test_metrics"]


def pct_change(old, new):

    if old == 0:
        return np.nan

    return ((new - old) / old) * 100


improvement_rows = [

    {
        "comparison": "A -> B",
        "MAE_change_percent":
            pct_change(A_test["MAE"], B_test["MAE"]),
        "RMSE_change_percent":
            pct_change(A_test["RMSE"], B_test["RMSE"]),
        "R2_change":
            B_test["R2"] - A_test["R2"]
    },

    {
        "comparison": "B -> C",
        "MAE_change_percent":
            pct_change(B_test["MAE"], C_test["MAE"]),
        "RMSE_change_percent":
            pct_change(B_test["RMSE"], C_test["RMSE"]),
        "R2_change":
            C_test["R2"] - B_test["R2"]
    },

    {
        "comparison": "A -> C",
        "MAE_change_percent":
            pct_change(A_test["MAE"], C_test["MAE"]),
        "RMSE_change_percent":
            pct_change(A_test["RMSE"], C_test["RMSE"]),
        "R2_change":
            C_test["R2"] - A_test["R2"]
    }
]


improvement_df = pd.DataFrame(
    improvement_rows
)

improvement_file = (
    OUTPUT_DIR
    / "incremental_improvement.csv"
)

improvement_df.to_csv(
    improvement_file,
    index=False
)


print("\n")
print("=" * 80)
print("INCREMENTAL IMPROVEMENT")
print("=" * 80)

print(
    improvement_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ================================================================
# 23. SAVE SPLIT INFORMATION
# ================================================================

split_df = pd.concat(
    [
        train_df[
            [
                "transfer_row_id",
                "transfer_date"
            ]
        ].assign(split="train"),

        val_df[
            [
                "transfer_row_id",
                "transfer_date"
            ]
        ].assign(split="validation"),

        test_df[
            [
                "transfer_row_id",
                "transfer_date"
            ]
        ].assign(split="test")
    ],
    ignore_index=True
)

split_file = (
    OUTPUT_DIR
    / "chronological_split.csv"
)

split_df.to_csv(
    split_file,
    index=False
)


# ================================================================
# 24. FINISHED
# ================================================================

print("\n")
print("=" * 80)
print("MODEL IMPLEMENTATION COMPLETE")
print("=" * 80)

print(f"\nResults folder:")
print(OUTPUT_DIR)

print("\nFiles generated:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print(" -", file.name)

print("\nDone.")
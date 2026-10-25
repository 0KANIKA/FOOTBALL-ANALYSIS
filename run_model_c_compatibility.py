import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

BASE = r"C:\Users\rsuba\Downloads\R football analytics"

TRANSFER_FILE = os.path.join(
    BASE, "professional_transfer_dataset.csv"
)

TARGET_FILE = os.path.join(
    BASE, "post_transfer_performance.csv"
)

TEAM_FILE = os.path.join(
    BASE, "rich_team_context_dataset.csv"
)

COACH_FILE = os.path.join(
    BASE, "coach_tactical_context_v21.csv"
)

OUTPUT_DIR = os.path.join(
    BASE, "model_c_compatibility_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "post_GA90"

RANDOM_STATE = 42

RF_CONFIG = {
    "n_estimators": 500,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "max_features": 1.0
}


# ============================================================
# HELPER
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def make_interaction(df, a, b, new_name):
    """
    Numeric x numeric compatibility interaction.
    Missing values remain missing and are handled later.
    """
    if a in df.columns and b in df.columns:
        df[new_name] = (
            pd.to_numeric(df[a], errors="coerce")
            * pd.to_numeric(df[b], errors="coerce")
        )
    else:
        df[new_name] = np.nan


def make_categorical_interaction(df, a, b, new_name):
    """
    Categorical compatibility interaction.
    Example:
        Attack + 4-3-3 Attacking
        Midfield + 4-2-3-1
    """
    if a in df.columns and b in df.columns:
        df[new_name] = (
            df[a].fillna("Unknown").astype(str)
            + "__"
            + df[b].fillna("Unknown").astype(str)
        )
    else:
        df[new_name] = "Unknown"


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

print(f"Transfers: {len(transfers):,}")
print(f"Target rows: {len(target):,}")
print(f"Team context rows: {len(team):,}")
print(f"Coach context rows: {len(coach):,}")


# ============================================================
# TRANSFER ID
# ============================================================

if "transfer_row_id" not in transfers.columns:

    transfers["transfer_row_id"] = np.arange(
        1,
        len(transfers) + 1
    )

    print(
        f"Created transfer_row_id: "
        f"1 to {len(transfers)}"
    )

else:

    print("transfer_row_id already exists.")


# ============================================================
# NORMALIZE IDs
# ============================================================

for df in [transfers, target, team, coach]:

    if "transfer_row_id" in df.columns:

        df["transfer_row_id"] = pd.to_numeric(
            df["transfer_row_id"],
            errors="coerce"
        )


# ============================================================
# MERGE DATASETS
# ============================================================

print("\n" + "=" * 70)
print("MERGING DATA")
print("=" * 70)

df = transfers.copy()

df = df.merge(
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

print(f"Merged rows: {len(df):,}")


# ============================================================
# TARGET CHECK
# ============================================================

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)


# ============================================================
# BUILD EXACT COMMON COHORT
# ============================================================

print("\n" + "=" * 70)
print("BUILDING COMMON COHORT")
print("=" * 70)

required_cohort_columns = [
    "eligible_450min",
    "transfer_date",
    TARGET
]

missing_cohort = [
    c for c in required_cohort_columns
    if c not in df.columns
]

if missing_cohort:
    raise ValueError(
        "Missing cohort columns: "
        + ", ".join(missing_cohort)
    )

df = df[
    df["eligible_450min"] == 1
].copy()

df = df.dropna(
    subset=[TARGET, "transfer_date"]
)

df = df.drop_duplicates(
    subset=["transfer_row_id"]
)

df["transfer_date"] = pd.to_datetime(
    df["transfer_date"],
    errors="coerce"
)

df = df.dropna(
    subset=["transfer_date"]
)

print(
    f"Exact C+ cohort: {len(df):,}"
)

print(
    f"Target: {TARGET}"
)


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
# CHECK BASIC FEATURES
# ============================================================

all_required = (
    player_numeric
    + player_categorical
    + team_features
    + coach_features
)

missing_features = [
    c for c in all_required
    if c not in df.columns
]

if missing_features:

    print("\nMISSING FEATURES")

    for feature in missing_features:
        print(" -", feature)

    raise ValueError(
        "One or more required features are missing."
    )


# ============================================================
# FIND FORMATION COLUMN
# ============================================================

formation_candidates = [
    "dominant_formation10",
    "dominant_formation",
    "rich_team_dominant_formation10"
]

formation_column = None

for candidate in formation_candidates:

    if candidate in df.columns:
        formation_column = candidate
        break

if formation_column is None:

    print(
        "\nWARNING: No formation column found."
    )

    df["_formation_for_compatibility"] = "Unknown"

    formation_column = "_formation_for_compatibility"

else:

    print(
        f"\nFormation column: {formation_column}"
    )


# ============================================================
# FIND POSITION GROUP
# ============================================================

position_group_candidates = [
    "target_position_group",
    "position_group"
]

position_group_column = None

for candidate in position_group_candidates:

    if candidate in df.columns:
        position_group_column = candidate
        break

if position_group_column is None:

    print(
        "\nWARNING: No position-group column found."
    )

    df["_position_group_for_compatibility"] = (
        df["position"]
        .map({
            "Goalkeeper": "Goalkeeper",

            "Centre-Back": "Defender",
            "Left-Back": "Defender",
            "Right-Back": "Defender",
            "Defender": "Defender",

            "Defensive Midfield": "Midfield",
            "Central Midfield": "Midfield",
            "Attacking Midfield": "Midfield",
            "Left Midfield": "Midfield",
            "Right Midfield": "Midfield",
            "Midfield": "Midfield",

            "Left Winger": "Attack",
            "Right Winger": "Attack",
            "Centre-Forward": "Attack",
            "Second Striker": "Attack",
            "Attack": "Attack"
        })
        .fillna("Unknown")
    )

    position_group_column = (
        "_position_group_for_compatibility"
    )

else:

    print(
        f"Position group column: "
        f"{position_group_column}"
    )


# ============================================================
# BUILD COMPATIBILITY FEATURES
# ============================================================

print("\n" + "=" * 70)
print("BUILDING PLAYER–TEAM COMPATIBILITY FEATURES")
print("=" * 70)


# ------------------------------------------------------------
# 1. PLAYER ATTACKING OUTPUT × TEAM ATTACKING ENVIRONMENT
# ------------------------------------------------------------

make_interaction(
    df,
    "goals_per90_pre",
    "rich_team_goals_for_avg10",
    "compat_goals90_x_team_goals"
)

make_interaction(
    df,
    "assists_per90_pre",
    "rich_team_goals_for_avg10",
    "compat_assists90_x_team_goals"
)

make_interaction(
    df,
    "goals_per90_pre",
    "rich_team_goal_difference_avg10",
    "compat_goals90_x_team_goal_diff"
)

make_interaction(
    df,
    "assists_per90_pre",
    "rich_team_goal_difference_avg10",
    "compat_assists90_x_team_goal_diff"
)


# ------------------------------------------------------------
# 2. PLAYER PRODUCTION × TEAM SUCCESS
# ------------------------------------------------------------

make_interaction(
    df,
    "goals_per90_pre",
    "rich_team_points_per_game10",
    "compat_goals90_x_team_ppg"
)

make_interaction(
    df,
    "assists_per90_pre",
    "rich_team_points_per_game10",
    "compat_assists90_x_team_ppg"
)


# ------------------------------------------------------------
# 3. PLAYER POSITION × DESTINATION FORMATION
# ------------------------------------------------------------

make_categorical_interaction(
    df,
    position_group_column,
    formation_column,
    "compat_position_group_x_formation"
)


# ------------------------------------------------------------
# 4. PLAYER POSITION × COACH USAGE
# ------------------------------------------------------------

make_interaction(
    df,
    "coach_position_group_minutes_share",
    "goals_per90_pre",
    "compat_coach_usage_x_goals90"
)

make_interaction(
    df,
    "coach_position_group_minutes_share",
    "assists_per90_pre",
    "compat_coach_usage_x_assists90"
)

make_interaction(
    df,
    "coach_position_group_start_rate",
    "goals_per90_pre",
    "compat_coach_start_rate_x_goals90"
)

make_interaction(
    df,
    "coach_position_group_start_rate",
    "assists_per90_pre",
    "compat_coach_start_rate_x_assists90"
)


# ------------------------------------------------------------
# 5. PLAYER POSITION × SQUAD COMPETITION
# ------------------------------------------------------------

make_interaction(
    df,
    "coach_position_group_minutes_share",
    "coach_position_competitor_minutes",
    "compat_coach_usage_x_competitor_minutes"
)

make_interaction(
    df,
    "coach_position_group_start_rate",
    "coach_position_competitor_count",
    "compat_coach_start_x_competitor_count"
)

make_interaction(
    df,
    "coach_position_rotation_hhi",
    "coach_position_competitor_minutes",
    "compat_rotation_x_competition"
)


# ------------------------------------------------------------
# 6. PLAYER PRE-TRANSFER EXPERIENCE × COACH OPPORTUNITY
# ------------------------------------------------------------

make_interaction(
    df,
    "minutes_pre",
    "coach_position_group_minutes_share",
    "compat_minutes_pre_x_coach_usage"
)

make_interaction(
    df,
    "minutes_pre",
    "coach_position_group_start_rate",
    "compat_minutes_pre_x_coach_start"
)


# ------------------------------------------------------------
# 7. PLAYER PRODUCTION × COACH OPPORTUNITY
# ------------------------------------------------------------

make_interaction(
    df,
    "goals_pre",
    "coach_position_group_minutes_share",
    "compat_goals_pre_x_coach_usage"
)

make_interaction(
    df,
    "assists_pre",
    "coach_position_group_minutes_share",
    "compat_assists_pre_x_coach_usage"
)


# ============================================================
# COMPATIBILITY FEATURE LIST
# ============================================================

compatibility_numeric = [
    "compat_goals90_x_team_goals",
    "compat_assists90_x_team_goals",
    "compat_goals90_x_team_goal_diff",
    "compat_assists90_x_team_goal_diff",

    "compat_goals90_x_team_ppg",
    "compat_assists90_x_team_ppg",

    "compat_coach_usage_x_goals90",
    "compat_coach_usage_x_assists90",

    "compat_coach_start_rate_x_goals90",
    "compat_coach_start_rate_x_assists90",

    "compat_coach_usage_x_competitor_minutes",
    "compat_coach_start_x_competitor_count",

    "compat_rotation_x_competition",

    "compat_minutes_pre_x_coach_usage",
    "compat_minutes_pre_x_coach_start",

    "compat_goals_pre_x_coach_usage",
    "compat_assists_pre_x_coach_usage"
]

compatibility_categorical = [
    "compat_position_group_x_formation"
]


# ============================================================
# SUMMARY
# ============================================================

print("\nCompatibility features created:")

for feature in (
    compatibility_numeric
    + compatibility_categorical
):
    print(" -", feature)


# ============================================================
# FEATURE COUNTS
# ============================================================

print("\n" + "=" * 70)
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
    f"Compatibility features: "
    f"{len(compatibility_numeric) + len(compatibility_categorical)}"
)

print(
    f"Total before encoding: "
    f"{len(player_numeric) + len(player_categorical) + len(team_features) + len(coach_features) + len(compatibility_numeric) + len(compatibility_categorical)}"
)


# ============================================================
# COACH COVERAGE
# ============================================================

if "coach_known_pretransfer" in df.columns:

    known = (
        df["coach_known_pretransfer"]
        .fillna(0)
        .astype(float)
        .sum()
    )

    unknown = len(df) - known

    print("\n" + "=" * 70)
    print("COACH COVERAGE")
    print("=" * 70)

    print(
        f"Coach known:   {int(known):,} "
        f"({known / len(df) * 100:.2f}%)"
    )

    print(
        f"Coach unknown: {int(unknown):,} "
        f"({unknown / len(df) * 100:.2f}%)"
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

numeric_features = (
    player_numeric
    + team_features
    + coach_features
    + compatibility_numeric
)

categorical_features = (
    player_categorical
    + compatibility_categorical
)


X = df[
    numeric_features + categorical_features
].copy()

y = df[TARGET].copy()


# ============================================================
# CHRONOLOGICAL SORT
# ============================================================

df["_row_order"] = np.arange(len(df))

order = np.argsort(
    df["transfer_date"].values
)

X = X.iloc[order].reset_index(drop=True)
y = y.iloc[order].reset_index(drop=True)
dates = (
    df["transfer_date"]
    .iloc[order]
    .reset_index(drop=True)
)


# ============================================================
# 60 / 20 / 20 CHRONOLOGICAL SPLIT
# ============================================================

n = len(X)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
X_test = X.iloc[val_end:]

y_train = y.iloc[:train_end]
y_val = y.iloc[train_end:val_end]
y_test = y.iloc[val_end:]

print("\n" + "=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print(
    f"Train:      {len(X_train):,}"
)

print(
    f"Validation: {len(X_val):,}"
)

print(
    f"Test:       {len(X_test):,}"
)

print(
    f"\nTrain dates: "
    f"{dates.iloc[0].date()} -> "
    f"{dates.iloc[train_end - 1].date()}"
)

print(
    f"Validation dates: "
    f"{dates.iloc[train_end].date()} -> "
    f"{dates.iloc[val_end - 1].date()}"
)

print(
    f"Test dates: "
    f"{dates.iloc[val_end].date()} -> "
    f"{dates.iloc[-1].date()}"
)


# ============================================================
# PREPROCESSING
# ============================================================

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
                handle_unknown="ignore"
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
# MODEL
# ============================================================

model = RandomForestRegressor(
    **RF_CONFIG
)

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
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL C+ RANDOM FOREST")
print("=" * 70)

pipeline.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, X_split, y_split):

    pred = pipeline.predict(X_split)

    metrics = {
        "split": name,
        "MAE": mean_absolute_error(
            y_split,
            pred
        ),
        "RMSE": rmse(
            y_split,
            pred
        ),
        "R2": r2_score(
            y_split,
            pred
        )
    }

    print("\n" + name.upper())
    print("-" * 40)

    print(
        f"MAE :  {metrics['MAE']:.6f}"
    )

    print(
        f"RMSE:  {metrics['RMSE']:.6f}"
    )

    print(
        f"R²  :  {metrics['R2']:.6f}"
    )

    return metrics, pred


train_metrics, train_pred = evaluate(
    "Train",
    X_train,
    y_train
)

val_metrics, val_pred = evaluate(
    "Validation",
    X_val,
    y_val
)

test_metrics, test_pred = evaluate(
    "Test",
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
    OUTPUT_DIR,
    "model_c_compatibility_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_predictions = pd.DataFrame({
    "transfer_date": dates.iloc[val_end:].values,
    "actual_post_GA90": y_test.values,
    "predicted_post_GA90": test_pred
})

predictions_path = os.path.join(
    OUTPUT_DIR,
    "model_c_compatibility_test_predictions.csv"
)

test_predictions.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING FEATURE IMPORTANCE")
print("=" * 70)

rf = pipeline.named_steps["model"]

feature_names = (
    pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

importance_path = os.path.join(
    OUTPUT_DIR,
    "model_c_compatibility_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)


# ============================================================
# TOP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP 30 FEATURES")
print("=" * 70)

print(
    importance_df.head(30).to_string(
        index=False
    )
)


# ============================================================
# COMPATIBILITY IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("COMPATIBILITY FEATURE IMPORTANCE")
print("=" * 70)

compat_importance = (
    importance_df[
        importance_df["feature"].str.contains(
            "compat_",
            case=False,
            na=False
        )
    ]
    .copy()
)

if len(compat_importance) > 0:

    print(
        compat_importance.to_string(
            index=False
        )
    )

    total_compat_importance = (
        compat_importance["importance"]
        .sum()
    )

    print(
        f"\nTotal compatibility importance: "
        f"{total_compat_importance:.6f}"
    )

else:

    print(
        "No compatibility features found "
        "in feature importance output."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("C+ COMPATIBILITY MODEL COMPLETE")
print("=" * 70)

print(
    f"Cohort: {len(df):,}"
)

print(
    f"Train: {len(X_train):,}"
)

print(
    f"Validation: {len(X_val):,}"
)

print(
    f"Test: {len(X_test):,}"
)

print("\nTEST RESULTS")

print(
    f"MAE : {test_metrics['MAE']:.6f}"
)

print(
    f"RMSE: {test_metrics['RMSE']:.6f}"
)

print(
    f"R²  : {test_metrics['R2']:.6f}"
)

print("\nFiles saved:")

print(metrics_path)
print(predictions_path)
print(importance_path)

print("\nDONE.")
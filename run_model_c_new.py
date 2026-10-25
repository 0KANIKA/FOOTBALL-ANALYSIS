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

BASE_DIR = r"C:\Users\rsuba\Downloads\R football analytics"

TRANSFER_FILE = os.path.join(
    BASE_DIR, "professional_transfer_dataset.csv"
)

TARGET_FILE = os.path.join(
    BASE_DIR, "post_transfer_performance.csv"
)

RICH_FILE = os.path.join(
    BASE_DIR, "rich_team_context_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "model_c_new_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("MODEL C-NEW - PLAYER + RICH DESTINATION CONTEXT")
print("=" * 60)

print("\nLoading datasets...")

transfers = pd.read_csv(TRANSFER_FILE)
target = pd.read_csv(TARGET_FILE)
rich = pd.read_csv(RICH_FILE)

print(f"Transfers: {len(transfers):,}")
print(f"Target:    {len(target):,}")
print(f"Rich:      {len(rich):,}")


# ============================================================
# TRANSFER IDs
# ============================================================

if "transfer_row_id" not in transfers.columns:
    transfers["transfer_row_id"] = np.arange(
        1, len(transfers) + 1
    )

if "transfer_row_id" not in target.columns:
    raise ValueError(
        "Target dataset has no transfer_row_id"
    )

if "transfer_row_id" not in rich.columns:
    raise ValueError(
        "Rich dataset has no transfer_row_id"
    )


# ============================================================
# BUILD EXACT COMMON COHORT
# ============================================================

print("\nBuilding exact common A/B/C cohort...")

# Target eligibility
target_check = target.copy()

if "complete_365d_window" in target_check.columns:
    target_check = target_check[
        target_check["complete_365d_window"].astype(bool)
    ]

if "eligible_450min" in target_check.columns:
    target_check = target_check[
        target_check["eligible_450min"].astype(bool)
    ]
elif "post_minutes" in target_check.columns:
    target_check = target_check[
        target_check["post_minutes"] >= 450
    ]
elif "minutes_post" in target_check.columns:
    target_check = target_check[
        target_check["minutes_post"] >= 450
    ]
else:
    raise ValueError(
        "Cannot identify post-transfer 450-minute eligibility."
    )

target_check = target_check[
    target_check["post_GA90"].notna()
].copy()

target_ids = set(
    target_check["transfer_row_id"].astype(int)
)

rich_ids = set(
    rich["transfer_row_id"].astype(int)
)

transfer_ids = set(
    transfers["transfer_row_id"].astype(int)
)

common_ids = (
    target_ids
    & rich_ids
    & transfer_ids
)

print(
    f"Eligible target IDs: {len(target_ids):,}"
)

print(
    f"Rich context IDs:    {len(rich_ids):,}"
)

print(
    f"Transfer IDs:        {len(transfer_ids):,}"
)

print(
    f"Exact common cohort:  {len(common_ids):,}"
)

if len(common_ids) != 6108:
    print(
        "\nWARNING: Expected 6,108 common records."
    )
    print(
        "Check the cohort construction before using final results."
    )


# ============================================================
# MERGE ONLY COMMON IDs
# ============================================================

common_ids = sorted(common_ids)

df = transfers[
    transfers["transfer_row_id"].isin(common_ids)
].copy()

df = df.merge(
    target[
        target["transfer_row_id"].isin(common_ids)
    ],
    on="transfer_row_id",
    how="inner",
    suffixes=("", "_target")
)

df = df.merge(
    rich[
        rich["transfer_row_id"].isin(common_ids)
    ],
    on="transfer_row_id",
    how="inner",
    suffixes=("", "_rich")
)

print(
    f"Merged common records: {len(df):,}"
)


# ============================================================
# TARGET
# ============================================================

TARGET = "post_GA90"

if TARGET not in df.columns:
    raise ValueError(
        f"{TARGET} not found in merged dataset."
    )

df = df[
    df[TARGET].notna()
    & np.isfinite(df[TARGET])
].copy()


# ============================================================
# TRANSFER DATE
# ============================================================

if "transfer_date" not in df.columns:
    raise ValueError(
        "transfer_date not found."
    )

df["transfer_date"] = pd.to_datetime(
    df["transfer_date"],
    errors="coerce"
)

df = df[
    df["transfer_date"].notna()
].copy()

# IMPORTANT:
# Exact same chronological ordering used by A-NEW/B-NEW
df = df.sort_values(
    "transfer_date"
).reset_index(drop=True)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

# ------------------------------------------------------------
# PLAYER FEATURES
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

    # Manager / coach context
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
# CHECK FEATURES
# ============================================================

numeric_features = [
    c
    for c in player_numeric + rich_numeric
    if c in df.columns
]

categorical_features = [
    c
    for c in player_categorical + rich_categorical
    if c in df.columns
]

missing_player = [
    c
    for c in player_numeric + player_categorical
    if c not in df.columns
]

missing_rich = [
    c
    for c in rich_numeric + rich_categorical
    if c not in df.columns
]

print("\nPlayer features used:")
for c in player_numeric + player_categorical:
    if c in df.columns:
        print(" -", c)

print("\nRich context features used:")
for c in rich_numeric + rich_categorical:
    if c in df.columns:
        print(" -", c)

if missing_player:
    print("\nWARNING - Missing player features:")
    for c in missing_player:
        print(" -", c)

if missing_rich:
    print("\nWARNING - Missing rich context features:")
    for c in missing_rich:
        print(" -", c)

print(
    f"\nTotal numeric features: "
    f"{len(numeric_features)}"
)

print(
    f"Total categorical features: "
    f"{len(categorical_features)}"
)

print(
    f"Total pre-encoding features: "
    f"{len(numeric_features) + len(categorical_features)}"
)


# ============================================================
# VERIFY COACH FEATURES
# ============================================================

print("\n========== COACH FEATURE CHECK ==========")

coach_features = [
    "manager_matches_last10",
    "unique_managers_last10"
]

for feature in coach_features:

    if feature not in df.columns:
        print(
            f"{feature}: MISSING"
        )
        continue

    missing = df[feature].isna().sum()
    unique = df[feature].nunique()

    print(
        f"{feature}: "
        f"missing={missing:,}, "
        f"unique_values={unique:,}"
    )


# ============================================================
# X / y
# ============================================================

X = df[
    numeric_features + categorical_features
].copy()

y = df[TARGET].astype(float)


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


# ============================================================
# SPLIT REPORT
# ============================================================

print("\n" + "=" * 60)
print("TEMPORAL SPLIT")
print("=" * 60)

print(
    f"Train:      {len(X_train):,}"
)

print(
    f"Validation: {len(X_val):,}"
)

print(
    f"Test:       {len(X_test):,}"
)

print("\nDate ranges:")

print(
    f"Train:      "
    f"{dates_train.min().date()} "
    f"to "
    f"{dates_train.max().date()}"
)

print(
    f"Validation: "
    f"{dates_val.min().date()} "
    f"to "
    f"{dates_val.max().date()}"
)

print(
    f"Test:       "
    f"{dates_test.min().date()} "
    f"to "
    f"{dates_test.max().date()}"
)


# ============================================================
# PREPROCESSOR
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

print("\n" + "=" * 60)
print("TRAINING MODEL C-NEW")
print("=" * 60)

pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    name,
    X_split,
    y_split
):

    pred = pipeline.predict(
        X_split
    )

    mae = mean_absolute_error(
        y_split,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_split,
            pred
        )
    )

    r2 = r2_score(
        y_split,
        pred
    )

    print(
        f"{name:<12}"
        f"MAE={mae:.6f}  "
        f"RMSE={rmse:.6f}  "
        f"R²={r2:.6f}"
    )

    return (
        pred,
        mae,
        rmse,
        r2
    )


print("\n========== MODEL C-NEW RESULTS ==========")

train_pred, train_mae, train_rmse, train_r2 = evaluate(
    "Train",
    X_train,
    y_train
)

val_pred, val_mae, val_rmse, val_r2 = evaluate(
    "Validation",
    X_val,
    y_val
)

test_pred, test_mae, test_rmse, test_r2 = evaluate(
    "Test",
    X_test,
    y_test
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_results = df.iloc[val_end:].copy()

test_results["actual"] = y_test.values
test_results["predicted"] = test_pred

test_results["residual"] = (
    test_results["actual"]
    - test_results["predicted"]
)

test_results["absolute_error"] = (
    test_results["residual"]
).abs()

test_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_c_new_test_predictions.csv"
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
        "model_c_new_metrics.csv"
    ),
    index=False
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nCalculating feature importance...")

try:

    feature_names = (
        pipeline
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importances = (
        pipeline
        .named_steps["model"]
        .feature_importances_
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    importance_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "model_c_new_feature_importance.csv"
        ),
        index=False
    )

    print("\nTop 25 features:")

    print(
        importance_df.head(25).to_string(
            index=False
        )
    )

except Exception as e:

    print(
        "\nFeature importance calculation failed:"
    )

    print(e)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL C-NEW COMPLETE")
print("=" * 60)

print(
    f"Cohort:      {len(df):,}"
)

print(
    f"Train:       {len(X_train):,}"
)

print(
    f"Validation:  {len(X_val):,}"
)

print(
    f"Test:        {len(X_test):,}"
)

print("\nFinal test performance:")

print(
    f"MAE  : {test_mae:.6f}"
)

print(
    f"RMSE : {test_rmse:.6f}"
)

print(
    f"R²   : {test_r2:.6f}"
)

print("\nSaved:")

print(
    os.path.join(
        OUTPUT_DIR,
        "model_c_new_metrics.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "model_c_new_test_predictions.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "model_c_new_feature_importance.csv"
    )
)

print("\nDone.")
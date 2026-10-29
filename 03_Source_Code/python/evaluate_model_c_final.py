import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "04_Datasets", "derived")

TRANSFER_FILE = os.path.join(DATA_DIR, "professional_transfer_dataset.csv")
TARGET_FILE = os.path.join(DATA_DIR, "post_transfer_performance.csv")
RICH_FILE = os.path.join(DATA_DIR, "rich_team_context_dataset.csv")

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "06_Results_and_Demo_Evidence", "research_results", "model_c_final_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# LOAD
# ============================================================

transfers = pd.read_csv(TRANSFER_FILE)
target = pd.read_csv(TARGET_FILE)
rich = pd.read_csv(RICH_FILE)

if "transfer_row_id" not in transfers.columns:
    transfers["transfer_row_id"] = np.arange(
        1, len(transfers) + 1
    )


# ============================================================
# EXACT COMMON COHORT
# ============================================================

target_check = target.copy()

if "complete_365_day_window" in target_check.columns:
    target_check = target_check[
        target_check["complete_365_day_window"].astype(bool)
    ]
elif "complete_365d_window" in target_check.columns:
    target_check = target_check[
        target_check["complete_365d_window"].astype(bool)
    ]

target_check = target_check[
    target_check["eligible_450min"].astype(bool)
]

target_check = target_check[
    target_check["post_GA90"].notna()
]

common_ids = (
    set(transfers["transfer_row_id"].astype(int))
    & set(target_check["transfer_row_id"].astype(int))
    & set(rich["transfer_row_id"].astype(int))
)

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

df["transfer_date"] = pd.to_datetime(
    df["transfer_date"],
    errors="coerce"
)

df = df[
    df["transfer_date"].notna()
    & df["post_GA90"].notna()
    & np.isfinite(df["post_GA90"])
].copy()

df = df.sort_values(
    "transfer_date"
).reset_index(drop=True)


# ============================================================
# FEATURES — EXACT MODEL C
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

rich_numeric = [
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

rich_categorical = [
    "dominant_formation10",
    "target_position_group"
]

numeric_features = [
    c for c in player_numeric + rich_numeric
    if c in df.columns
]

categorical_features = [
    c for c in player_categorical + rich_categorical
    if c in df.columns
]

X = df[
    numeric_features + categorical_features
].copy()

y = df["post_GA90"].astype(float)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

X_train = X.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
X_test = X.iloc[val_end:]

y_train = y.iloc[:train_end]
y_val = y.iloc[train_end:val_end]
y_test = y.iloc[val_end:]

dates_train = df["transfer_date"].iloc[:train_end]
dates_val = df["transfer_date"].iloc[train_end:val_end]
dates_test = df["transfer_date"].iloc[val_end:]


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
# FINAL SELECTED MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# ============================================================
# TRAIN
# ============================================================

print("=" * 90)
print("FINAL MODEL C — REGULARIZED RANDOM FOREST")
print("=" * 90)

print("\nSelected using validation performance:")
print("max_depth        : 15")
print("min_samples_leaf : 5")
print("max_features     : sqrt")
print("n_estimators     : 300")

print("\nCohort            :", len(df))
print("Train             :", len(X_train))
print("Validation        :", len(X_val))
print("Test              :", len(X_test))

print("\nTraining final selected model...")

pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# EVALUATE
# ============================================================

def metrics(y_true, pred):

    mae = mean_absolute_error(
        y_true,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            pred
        )
    )

    r2 = r2_score(
        y_true,
        pred
    )

    return mae, rmse, r2


train_pred = pipeline.predict(X_train)
val_pred = pipeline.predict(X_val)
test_pred = pipeline.predict(X_test)

train_mae, train_rmse, train_r2 = metrics(
    y_train,
    train_pred
)

val_mae, val_rmse, val_r2 = metrics(
    y_val,
    val_pred
)

test_mae, test_rmse, test_r2 = metrics(
    y_test,
    test_pred
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 90)
print("FINAL MODEL PERFORMANCE")
print("=" * 90)

print(
    f'{"Split":<15}'
    f'{"N":<10}'
    f'{"MAE":<15}'
    f'{"RMSE":<15}'
    f'{"R²":<15}'
)

print("-" * 90)

print(
    f'{"Train":<15}'
    f'{len(y_train):<10}'
    f'{train_mae:<15.6f}'
    f'{train_rmse:<15.6f}'
    f'{train_r2:<15.6f}'
)

print(
    f'{"Validation":<15}'
    f'{len(y_val):<10}'
    f'{val_mae:<15.6f}'
    f'{val_rmse:<15.6f}'
    f'{val_r2:<15.6f}'
)

print(
    f'{"Test":<15}'
    f'{len(y_test):<10}'
    f'{test_mae:<15.6f}'
    f'{test_rmse:<15.6f}'
    f'{test_r2:<15.6f}'
)

print("\nTest period:")
print(
    dates_test.min().date(),
    "to",
    dates_test.max().date()
)

print("\nTest set was NOT used for model selection.")

print("=" * 90)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({
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

metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_model_c_metrics.csv"
    ),
    index=False
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
        "final_model_c_test_predictions.csv"
    ),
    index=False
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

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
}).sort_values(
    "importance",
    ascending=False
)

importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_model_c_feature_importance.csv"
    ),
    index=False
)

print("\nSaved:")
print(
    os.path.join(
        OUTPUT_DIR,
        "final_model_c_metrics.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "final_model_c_test_predictions.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "final_model_c_feature_importance.csv"
    )
)

print("\nDone.")

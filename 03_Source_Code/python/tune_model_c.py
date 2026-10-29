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

TRANSFER_FILE = os.path.join(
    DATA_DIR, "professional_transfer_dataset.csv"
)

TARGET_FILE = os.path.join(
    DATA_DIR, "post_transfer_performance.csv"
)

RICH_FILE = os.path.join(
    DATA_DIR, "rich_team_context_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "06_Results_and_Demo_Evidence", "research_results", "model_c_tuning_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# LOAD
# ============================================================

print("=" * 90)
print("MODEL C — CONTROLLED RANDOM FOREST TUNING")
print("=" * 90)

transfers = pd.read_csv(TRANSFER_FILE)
target = pd.read_csv(TARGET_FILE)
rich = pd.read_csv(RICH_FILE)

if "transfer_row_id" not in transfers.columns:
    transfers["transfer_row_id"] = np.arange(1, len(transfers) + 1)

target_check = target.copy()

if "complete_365_day_window" in target_check.columns:
    target_check = target_check[
        target_check["complete_365_day_window"].astype(bool)
    ]
elif "complete_365d_window" in target_check.columns:
    target_check = target_check[
        target_check["complete_365d_window"].astype(bool)
    ]

if "eligible_450min" in target_check.columns:
    target_check = target_check[
        target_check["eligible_450min"].astype(bool)
    ]
else:
    target_check = target_check[
        target_check["post_minutes"] >= 450
    ]

target_check = target_check[
    target_check["post_GA90"].notna()
].copy()

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

print(f"\nCommon cohort: {len(df):,}")


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
# EXACT 60 / 20 / 20 CHRONOLOGICAL SPLIT
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
# CONTROLLED CONFIGURATIONS
# ============================================================

configs = [

    {
        "name": "Baseline",
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    },

    {
        "name": "Depth15_Leaf2",
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": "sqrt"
    },

    {
        "name": "Depth12_Leaf3",
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_split": 2,
        "min_samples_leaf": 3,
        "max_features": "sqrt"
    },

    {
        "name": "Depth10_Leaf5",
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 5,
        "max_features": "sqrt"
    },

    {
        "name": "Depth15_Leaf5",
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 5,
        "max_features": "sqrt"
    },

    {
        "name": "Depth12_Leaf5_Split5",
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 5,
        "max_features": "sqrt"
    },

    {
        "name": "Depth15_Leaf3_70pct",
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 3,
        "max_features": 0.7
    }

]


# ============================================================
# EVALUATION
# ============================================================

results = []

print("\n" + "=" * 90)
print("RUNNING CONTROLLED EXPERIMENT")
print("=" * 90)

for cfg in configs:

    print(f"\nRunning: {cfg['name']}")

    model = RandomForestRegressor(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_split=cfg["min_samples_split"],
        min_samples_leaf=cfg["min_samples_leaf"],
        max_features=cfg["max_features"],
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    train_pred = pipeline.predict(X_train)
    val_pred = pipeline.predict(X_val)

    train_mae = mean_absolute_error(
        y_train,
        train_pred
    )

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_pred
        )
    )

    train_r2 = r2_score(
        y_train,
        train_pred
    )

    val_mae = mean_absolute_error(
        y_val,
        val_pred
    )

    val_rmse = np.sqrt(
        mean_squared_error(
            y_val,
            val_pred
        )
    )

    val_r2 = r2_score(
        y_val,
        val_pred
    )

    gap = train_r2 - val_r2

    results.append({
        "configuration": cfg["name"],
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "validation_mae": val_mae,
        "validation_rmse": val_rmse,
        "validation_r2": val_r2,
        "r2_gap": gap,
        "max_depth": cfg["max_depth"],
        "min_samples_split": cfg["min_samples_split"],
        "min_samples_leaf": cfg["min_samples_leaf"],
        "max_features": cfg["max_features"]
    })

    print(
        f"Train R²={train_r2:.4f} | "
        f"Validation R²={val_r2:.4f} | "
        f"Validation MAE={val_mae:.4f}"
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "validation_r2",
    ascending=False
).reset_index(drop=True)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_c_tuning_results.csv"
    ),
    index=False
)


# ============================================================
# CLEAN TERMINAL TABLE
# ============================================================

print("\n" + "=" * 120)
print("MODEL C — VALIDATION-BASED HYPERPARAMETER COMPARISON")
print("=" * 120)

display_df = results_df[
    [
        "configuration",
        "train_r2",
        "validation_r2",
        "validation_mae",
        "validation_rmse",
        "r2_gap"
    ]
].copy()

display_df.columns = [
    "Configuration",
    "Train R²",
    "Val R²",
    "Val MAE",
    "Val RMSE",
    "R² Gap"
]

print(
    display_df.to_string(
        index=False,
        formatters={
            "Train R²": "{:.4f}".format,
            "Val R²": "{:.4f}".format,
            "Val MAE": "{:.4f}".format,
            "Val RMSE": "{:.4f}".format,
            "R² Gap": "{:.4f}".format
        }
    )
)


# ============================================================
# BEST CONFIGURATION
# ============================================================

best = results_df.iloc[0]

print("\n" + "=" * 90)
print("BEST CONFIGURATION — SELECTED USING VALIDATION SET")
print("=" * 90)

print("Configuration :", best["configuration"])
print("Validation R² :", f"{best['validation_r2']:.6f}")
print("Validation MAE:", f"{best['validation_mae']:.6f}")
print("Validation RMSE:", f"{best['validation_rmse']:.6f}")
print("Train R²      :", f"{best['train_r2']:.6f}")
print("R² gap        :", f"{best['r2_gap']:.6f}")

print("\nTest set has NOT been used for model selection.")

print("\nSaved:")
print(
    os.path.join(
        OUTPUT_DIR,
        "model_c_tuning_results.csv"
    )
)

print("=" * 90)

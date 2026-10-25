import pandas as pd
import numpy as np
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE = r"C:\Users\rsuba\Downloads\R football analytics"

TRANSFER_FILE = BASE + r"\professional_transfer_dataset.csv"
TARGET_FILE = BASE + r"\post_transfer_performance.csv"
TEAM_FILE = BASE + r"\rich_team_context_dataset.csv"


print("=" * 100)
print("MODEL C-REDUCED — FEATURE REDUCTION + REGULARIZATION SEARCH")
print("=" * 100)


# ============================================================
# LOAD
# ============================================================

transfers = pd.read_csv(TRANSFER_FILE)
target = pd.read_csv(TARGET_FILE)
team = pd.read_csv(TEAM_FILE)

print(f"Transfers: {len(transfers)}")
print(f"Target:    {len(target)}")
print(f"Team:      {len(team)}")


# ============================================================
# RECREATE TRANSFER ID
# ============================================================

transfers = transfers.reset_index(drop=True)
transfers["transfer_row_id"] = transfers.index + 1


# ============================================================
# ELIGIBLE COHORT
# ============================================================

eligible_target = target[
    (target["complete_365_day_window"] == True) &
    (target["eligible_450min"] == True)
].copy()

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

target_base = eligible_target[
    [
        "transfer_row_id",
        "post_GA90",
        "complete_365_day_window",
        "eligible_450min"
    ]
].copy()

df = transfers.merge(
    target_base,
    on="transfer_row_id",
    how="inner"
)

df = df.merge(
    team,
    on="transfer_row_id",
    how="inner",
    suffixes=("", "_team")
)

print(f"Common cohort after merge: {len(df)}")


# ============================================================
# REDUCED FEATURE SET
# ============================================================

# ------------------------------------------------------------
# PLAYER FEATURES
# ------------------------------------------------------------

player_numeric = [
    "age_at_transfer",
    "height_in_cm",
    "minutes_pre",
    "goals_pre",
    "assists_pre",
    "goals_per90_pre",
    "assists_per90_pre",
    "market_value_pre",
    "transfer_fee"
]

player_categorical = [
    "position",
    "sub_position",
    "foot"
]


# ------------------------------------------------------------
# DESTINATION TEAM FEATURES
# ------------------------------------------------------------

team_numeric = [
    "rich_team_goals_for_avg10",
    "rich_team_goals_against_avg10",
    "rich_team_goal_difference_avg10",
    "rich_team_win_rate10",
    "rich_team_points_per_game10",

    "dominant_formation_frequency10",
    "formation_diversity10",

    "same_position_competitor_count",
    "same_position_competitor_minutes",
    "same_position_top_competitor_minutes",
    "same_position_avg_market_value",

    "position_minutes_share"
]

team_categorical = [
    "dominant_formation10"
]


numeric_features = player_numeric + team_numeric
categorical_features = player_categorical + team_categorical

features = numeric_features + categorical_features


# ============================================================
# FEATURE CHECK
# ============================================================

missing_features = [
    f for f in features
    if f not in df.columns
]

if missing_features:
    print("\nERROR — Missing features:")
    print(missing_features)
    raise SystemExit(1)

print(f"Reduced numeric features     : {len(numeric_features)}")
print(f"Reduced categorical features : {len(categorical_features)}")
print(f"Total raw features           : {len(features)}")


# ============================================================
# CHRONOLOGICAL ORDER
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
# PREPROCESS
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
# REGULARIZATION SEARCH
# ============================================================

configs = [

    {
        "name": "Depth8_Leaf10",
        "max_depth": 8,
        "min_samples_leaf": 10
    },

    {
        "name": "Depth10_Leaf10",
        "max_depth": 10,
        "min_samples_leaf": 10
    },

    {
        "name": "Depth12_Leaf10",
        "max_depth": 12,
        "min_samples_leaf": 10
    },

    {
        "name": "Depth15_Leaf10",
        "max_depth": 15,
        "min_samples_leaf": 10
    },

    {
        "name": "Depth10_Leaf15",
        "max_depth": 10,
        "min_samples_leaf": 15
    },

    {
        "name": "Depth12_Leaf15",
        "max_depth": 12,
        "min_samples_leaf": 15
    },

    {
        "name": "Depth15_Leaf15",
        "max_depth": 15,
        "min_samples_leaf": 15
    },

    {
        "name": "Depth12_Leaf20",
        "max_depth": 12,
        "min_samples_leaf": 20
    }
]


# ============================================================
# SEARCH
# ============================================================

results = []

for cfg in configs:

    print("\n" + "-" * 100)
    print(cfg["name"])

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=cfg["max_depth"],
        min_samples_split=2,
        min_samples_leaf=cfg["min_samples_leaf"],
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_processed,
        y_train
    )

    train_pred = model.predict(
        X_train_processed
    )

    val_pred = model.predict(
        X_val_processed
    )

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

    print(f"Train R²       : {train_r2:.6f}")
    print(f"Validation R²  : {val_r2:.6f}")
    print(f"Validation MAE : {val_mae:.6f}")
    print(f"Validation RMSE: {val_rmse:.6f}")
    print(f"R² Gap         : {gap:.6f}")

    results.append({
        "configuration": cfg["name"],
        "max_depth": cfg["max_depth"],
        "min_samples_leaf": cfg["min_samples_leaf"],
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
print("C-REDUCED SEARCH RESULTS")
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

OUTPUT_DIR = BASE + r"\model_c_reduced_results"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

output_file = (
    OUTPUT_DIR +
    r"\model_c_reduced_search.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# BEST MODEL
# ============================================================

best = results_df.iloc[0]

print("\n" + "=" * 100)
print("BEST C-REDUCED CONFIGURATION")
print("=" * 100)

print(
    f"Configuration   : "
    f"{best['configuration']}"
)

print(
    f"Max depth       : "
    f"{best['max_depth']}"
)

print(
    f"Min leaf        : "
    f"{best['min_samples_leaf']}"
)

print(
    f"Validation R²   : "
    f"{best['validation_r2']:.6f}"
)

print(
    f"Validation MAE  : "
    f"{best['validation_mae']:.6f}"
)

print(
    f"Validation RMSE : "
    f"{best['validation_rmse']:.6f}"
)

print(
    f"Train R²        : "
    f"{best['train_r2']:.6f}"
)

print(
    f"R² gap          : "
    f"{best['r2_gap']:.6f}"
)

print("\nSaved:")
print(output_file)

print("\nIMPORTANT:")
print("The test set was NOT used during feature selection or tuning.")
print("=" * 100)
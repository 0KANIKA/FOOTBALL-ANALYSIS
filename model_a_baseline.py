import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = r"C:\Users\rsuba\Downloads\post_transfer_performance.csv"

OUTPUT_DIR = r"C:\Users\rsuba\Downloads\R football analytics\model_a_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("MODEL A - PLAYER ONLY BASELINE")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("Total records:", len(df))


# ============================================================
# DATE PROCESSING
# ============================================================

df["transfer_date"] = pd.to_datetime(
    df["transfer_date"],
    errors="coerce"
)


# ============================================================
# SELECT ELIGIBLE COHORT
# ============================================================

print("\nFiltering eligible modelling cohort...")

df = df[
    (df["complete_365_day_window"] == True) &
    (df["eligible_450min"] == True)
].copy()

print("Eligible records:", len(df))


# ============================================================
# TARGET
# ============================================================

TARGET = "post_GA90"

df = df[
    df[TARGET].notna()
].copy()

print("Records with valid target:", len(df))


# ============================================================
# PLAYER-ONLY FEATURES
# ============================================================

features = [
    "age_at_transfer",
    "position",
    "sub_position",
    "height_in_cm",
    "foot",
    "appearances_pre",
    "minutes_pre",
    "goals_pre",
    "assists_pre",
    "goals_per90_pre",
    "assists_per90_pre",
    "yellow_cards_pre",
    "red_cards_pre",
    "transfer_market_value",
    "market_value_pre"
]

features = [f for f in features if f in df.columns]

print("\nFeatures used:")
for f in features:
    print(" -", f)


X = df[features].copy()
y = df[TARGET].copy()


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

print("\nCreating chronological train/validation/test split...")

df_sorted = df.sort_values("transfer_date").copy()

n = len(df_sorted)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

train_df = df_sorted.iloc[:train_end].copy()
val_df = df_sorted.iloc[train_end:val_end].copy()
test_df = df_sorted.iloc[val_end:].copy()

print("\nDataset split:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nDate ranges:")

print(
    "Train:",
    train_df["transfer_date"].min().date(),
    "to",
    train_df["transfer_date"].max().date()
)

print(
    "Validation:",
    val_df["transfer_date"].min().date(),
    "to",
    val_df["transfer_date"].max().date()
)

print(
    "Test:",
    test_df["transfer_date"].min().date(),
    "to",
    test_df["transfer_date"].max().date()
)


X_train = train_df[features]
y_train = train_df[TARGET]

X_val = val_df[features]
y_val = val_df[TARGET]

X_test = test_df[features]
y_test = test_df[TARGET]


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_features = [
    f for f in [
        "position",
        "sub_position",
        "foot"
    ]
    if f in features
]

numeric_features = [
    f for f in features
    if f not in categorical_features
]


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

preprocessor = ColumnTransformer([
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
])


# ============================================================
# MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
}


if XGBOOST_AVAILABLE:

    models["XGBoost"] = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

else:

    print("\nWARNING:")
    print("XGBoost is not installed.")
    print("Only Linear Regression and Random Forest will run.")


# ============================================================
# TRAIN + EVALUATE
# ============================================================

results = []
prediction_frames = []


for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    pipeline = Pipeline([
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            model
        )
    ])

    print("Training...")

    pipeline.fit(
        X_train,
        y_train
    )

    # ----------------------------
    # Predictions
    # ----------------------------

    train_pred = pipeline.predict(X_train)
    val_pred = pipeline.predict(X_val)
    test_pred = pipeline.predict(X_test)

    # ----------------------------
    # Metrics
    # ----------------------------

    train_mae = mean_absolute_error(
        y_train,
        train_pred
    )

    val_mae = mean_absolute_error(
        y_val,
        val_pred
    )

    test_mae = mean_absolute_error(
        y_test,
        test_pred
    )

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_pred
        )
    )

    val_rmse = np.sqrt(
        mean_squared_error(
            y_val,
            val_pred
        )
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_pred
        )
    )

    train_r2 = r2_score(
        y_train,
        train_pred
    )

    val_r2 = r2_score(
        y_val,
        val_pred
    )

    test_r2 = r2_score(
        y_test,
        test_pred
    )

    # ----------------------------
    # Print results
    # ----------------------------

    print("\nRESULTS")

    print("\nTrain:")
    print(" MAE :", round(train_mae, 4))
    print(" RMSE:", round(train_rmse, 4))
    print(" R²  :", round(train_r2, 4))

    print("\nValidation:")
    print(" MAE :", round(val_mae, 4))
    print(" RMSE:", round(val_rmse, 4))
    print(" R²  :", round(val_r2, 4))

    print("\nTest:")
    print(" MAE :", round(test_mae, 4))
    print(" RMSE:", round(test_rmse, 4))
    print(" R²  :", round(test_r2, 4))

    # ----------------------------
    # Store metrics
    # ----------------------------

    results.append({
        "Model": model_name,

        "Train_MAE": train_mae,
        "Train_RMSE": train_rmse,
        "Train_R2": train_r2,

        "Validation_MAE": val_mae,
        "Validation_RMSE": val_rmse,
        "Validation_R2": val_r2,

        "Test_MAE": test_mae,
        "Test_RMSE": test_rmse,
        "Test_R2": test_r2
    })

    # ----------------------------
    # Store test predictions
    # ----------------------------

    temp = test_df[
        [
            "player_id",
            "transfer_date",
            "from_club_name",
            "to_club_name",
            TARGET
        ]
    ].copy()

    temp["predicted_GA90"] = test_pred

    temp["error"] = (
        temp[TARGET] -
        temp["predicted_GA90"]
    )

    temp["absolute_error"] = abs(
        temp["error"]
    )

    temp["Model"] = model_name

    prediction_frames.append(temp)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_file = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


predictions_df = pd.concat(
    prediction_frames,
    ignore_index=True
)

predictions_file = os.path.join(
    OUTPUT_DIR,
    "test_predictions.csv"
)

predictions_df.to_csv(
    predictions_file,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("MODEL A COMPLETE")
print("=" * 60)

print("\nModel comparison:")

print(
    results_df[
        [
            "Model",
            "Test_MAE",
            "Test_RMSE",
            "Test_R2"
        ]
    ].to_string(index=False)
)

print("\nSaved files:")

print(
    "1.",
    results_file
)

print(
    "2.",
    predictions_file
)

print("\n")
print("IMPORTANT:")
print("These are PRELIMINARY Model A results.")
print("Do not call them final research results yet.")

print("\nDone.")
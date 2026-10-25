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

BASE_DIR = r"C:\Users\rsuba\Downloads\R football analytics"

TRANSFER_FILE = os.path.join(
    BASE_DIR,
    "professional_transfer_dataset.csv"
)

TARGET_FILE = os.path.join(
    BASE_DIR,
    "post_transfer_performance.csv"
)

RICH_FILE = os.path.join(
    BASE_DIR,
    "rich_team_context_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "model_a_new_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("MODEL A-NEW - PLAYER ONLY BASELINE")
print("=" * 60)

print("\nLoading datasets...")

transfers = pd.read_csv(TRANSFER_FILE)
target = pd.read_csv(TARGET_FILE)
rich = pd.read_csv(RICH_FILE)

print("Transfers:", len(transfers))
print("Target:", len(target))
print("Rich context:", len(rich))


# ============================================================
# TRANSFER IDs
# ============================================================

if "transfer_row_id" not in transfers.columns:
    transfers["transfer_row_id"] = np.arange(
        1,
        len(transfers) + 1
    )

if "transfer_row_id" not in target.columns:
    raise ValueError(
        "Target dataset does not contain transfer_row_id."
    )

if "transfer_row_id" not in rich.columns:
    raise ValueError(
        "Rich context dataset does not contain transfer_row_id."
    )


# ============================================================
# BUILD COMMON COHORT
# ============================================================

print("\nBuilding common A/B cohort...")

# We use the rich dataset ONLY to identify the same
# records available to B-new.
#
# IMPORTANT:
# No rich/team features are used as Model A predictors.

common_ids = rich[
    ["transfer_row_id"]
].drop_duplicates()

print(
    "Available transfer IDs in rich dataset:",
    len(common_ids)
)


# ============================================================
# MERGE TRANSFER + TARGET
# ============================================================

df = transfers.merge(
    target,
    on="transfer_row_id",
    how="inner",
    suffixes=("", "_target")
)

df = df.merge(
    common_ids,
    on="transfer_row_id",
    how="inner"
)

print(
    "Merged transfer-target records:",
    len(df)
)


# ============================================================
# TARGET
# ============================================================

TARGET = "post_GA90"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

print(
    "Target column:",
    TARGET
)


# ============================================================
# ELIGIBILITY
# ============================================================

conditions = pd.Series(
    True,
    index=df.index
)

if "complete_365_day_window" in df.columns:

    conditions &= (
        df["complete_365_day_window"]
        .astype(bool)
    )

elif "complete_365d_window" in df.columns:

    conditions &= (
        df["complete_365d_window"]
        .astype(bool)
    )

else:

    raise ValueError(
        "Could not find complete 365-day window column."
    )


if "eligible_450min" in df.columns:

    conditions &= (
        df["eligible_450min"]
        .astype(bool)
    )

elif "post_minutes" in df.columns:

    conditions &= (
        df["post_minutes"] >= 450
    )

elif "minutes_post" in df.columns:

    conditions &= (
        df["minutes_post"] >= 450
    )

else:

    raise ValueError(
        "Could not find 450-minute eligibility column."
    )


df = df[
    conditions
].copy()


# Valid target

df = df[
    df[TARGET].notna()
].copy()

df = df[
    np.isfinite(
        df[TARGET].astype(float)
    )
].copy()


print(
    "Eligible common cohort:",
    len(df)
)


# ============================================================
# DATE PROCESSING
# ============================================================

df["transfer_date"] = pd.to_datetime(
    df["transfer_date"],
    errors="coerce"
)

df = df[
    df["transfer_date"].notna()
].copy()


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

    "market_value_pre",

    "transfer_fee"
]


features = [
    f for f in features
    if f in df.columns
]


print("\nPlayer-only features:")

for feature in features:

    print(
        " -",
        feature
    )


# ============================================================
# CHRONOLOGICAL SORT
# ============================================================

df = df.sort_values(
    "transfer_date"
).reset_index(drop=True)


# ============================================================
# X / y
# ============================================================

X = df[
    features
].copy()

y = df[
    TARGET
].astype(float)


# ============================================================
# 60 / 20 / 20 TEMPORAL SPLIT
# ============================================================

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


print("\n========== SPLIT ==========")

print(
    "Train:",
    len(train_df)
)

print(
    "Validation:",
    len(val_df)
)

print(
    "Test:",
    len(test_df)
)


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


# ============================================================
# SPLIT FEATURES
# ============================================================

X_train = train_df[
    features
]

y_train = train_df[
    TARGET
]

X_val = val_df[
    features
]

y_val = val_df[
    TARGET
]

X_test = test_df[
    features
]

y_test = test_df[
    TARGET
]


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

        SimpleImputer(
            strategy="median"
        )
    )

])


categorical_pipeline = Pipeline([

    (
        "imputer",

        SimpleImputer(
            strategy="most_frequent"
        )
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

            max_features="sqrt",

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

    print(
        "\nWARNING: XGBoost is not installed."
    )


# ============================================================
# TRAIN + EVALUATE
# ============================================================

results = []

prediction_frames = []


for model_name, model in models.items():

    print(
        "\n" + "=" * 60
    )

    print(
        "Training:",
        model_name
    )

    print(
        "=" * 60
    )


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


    print(
        "Training..."
    )


    pipeline.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    train_pred = pipeline.predict(
        X_train
    )

    val_pred = pipeline.predict(
        X_val
    )

    test_pred = pipeline.predict(
        X_test
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("\nRESULTS")

    print("\nTrain:")

    print(
        " MAE :",
        round(train_mae, 6)
    )

    print(
        " RMSE:",
        round(train_rmse, 6)
    )

    print(
        " R²  :",
        round(train_r2, 6)
    )


    print("\nValidation:")

    print(
        " MAE :",
        round(val_mae, 6)
    )

    print(
        " RMSE:",
        round(val_rmse, 6)
    )

    print(
        " R²  :",
        round(val_r2, 6)
    )


    print("\nTest:")

    print(
        " MAE :",
        round(test_mae, 6)
    )

    print(
        " RMSE:",
        round(test_rmse, 6)
    )

    print(
        " R²  :",
        round(test_r2, 6)
    )


    # --------------------------------------------------------
    # STORE METRICS
    # --------------------------------------------------------

    results.append({

        "Model":
        model_name,

        "Train_N":
        len(y_train),

        "Validation_N":
        len(y_val),

        "Test_N":
        len(y_test),

        "Train_MAE":
        train_mae,

        "Train_RMSE":
        train_rmse,

        "Train_R2":
        train_r2,

        "Validation_MAE":
        val_mae,

        "Validation_RMSE":
        val_rmse,

        "Validation_R2":
        val_r2,

        "Test_MAE":
        test_mae,

        "Test_RMSE":
        test_rmse,

        "Test_R2":
        test_r2

    })


    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    temp = test_df[
        [
            "transfer_row_id",
            "player_id",
            "transfer_date",
            "from_club_name",
            "to_club_name",
            TARGET
        ]
    ].copy()


    temp[
        "predicted_GA90"
    ] = test_pred


    temp[
        "error"
    ] = (

        temp[TARGET]
        -
        temp["predicted_GA90"]

    )


    temp[
        "absolute_error"
    ] = temp[
        "error"
    ].abs()


    temp[
        "Model"
    ] = model_name


    prediction_frames.append(
        temp
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


results_file = os.path.join(

    OUTPUT_DIR,

    "model_a_new_comparison.csv"
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

    "model_a_new_test_predictions.csv"
)


predictions_df.to_csv(

    predictions_file,

    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "MODEL A-NEW COMPLETE"
)

print(
    "=" * 60
)


print(
    "\nCohort:",
    len(df)
)

print(
    "Train:",
    len(train_df)
)

print(
    "Validation:",
    len(val_df)
)

print(
    "Test:",
    len(test_df)
)


print(
    "\nModel comparison:"
)


print(

    results_df[
        [
            "Model",
            "Test_MAE",
            "Test_RMSE",
            "Test_R2"
        ]
    ].to_string(
        index=False
    )

)


print(
    "\nSaved:"
)

print(
    results_file
)

print(
    predictions_file
)


print(
    "\nDone."
)
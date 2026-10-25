import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("=" * 60)
print("MODEL B - PLAYER + DESTINATION TEAM CONTEXT")
print("=" * 60)

BASE = Path(r"C:\Users\rsuba\Downloads\R football analytics")

DATA_FILE = BASE / "team_context_dataset.csv"
OUTPUT_DIR = BASE / "model_b_results"

OUTPUT_DIR.mkdir(exist_ok=True)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

df["transfer_date"] = pd.to_datetime(
    df["transfer_date"],
    errors="coerce"
)

print(f"Total records: {len(df)}")


# ---------------------------------------------------------
# SAME ELIGIBILITY AS MODEL A
# ---------------------------------------------------------

df = df[
    (df["complete_365_day_window"] == True) &
    (df["eligible_450min"] == True)
].copy()

# Require team context
df = df[
    df["team_matches_last10"] > 0
].copy()

print(f"Eligible records with team context: {len(df)}")


# ---------------------------------------------------------
# TARGET
# ---------------------------------------------------------

TARGET = "post_GA90"

df = df.dropna(subset=[TARGET])

print(f"Records with target: {len(df)}")


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

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
    "market_value_pre"
]

player_categorical = [
    "position",
    "sub_position",
    "foot"
]

team_features = [
    "team_matches_last10",
    "team_goals_for_avg10",
    "team_goals_against_avg10",
    "team_win_rate10",
    "team_points_per_game10",
    "team_goal_difference_avg10"
]

FEATURES = (
    player_numeric +
    player_categorical +
    team_features
)

X = df[FEATURES]
y = df[TARGET]


# ---------------------------------------------------------
# CHRONOLOGICAL SPLIT
# SAME LOGIC AS MODEL A
# ---------------------------------------------------------

df = df.sort_values("transfer_date")

n = len(df)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

train_idx = df.index[:train_end]
val_idx = df.index[train_end:val_end]
test_idx = df.index[val_end:]

X_train = X.loc[train_idx]
y_train = y.loc[train_idx]

X_val = X.loc[val_idx]
y_val = y.loc[val_idx]

X_test = X.loc[test_idx]
y_test = y.loc[test_idx]

print("\nChronological split:")
print(f"Train: {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test: {len(X_test)}")

print(
    f"\nTrain dates: "
    f"{df.loc[train_idx, 'transfer_date'].min().date()} "
    f"to "
    f"{df.loc[train_idx, 'transfer_date'].max().date()}"
)

print(
    f"Validation dates: "
    f"{df.loc[val_idx, 'transfer_date'].min().date()} "
    f"to "
    f"{df.loc[val_idx, 'transfer_date'].max().date()}"
)

print(
    f"Test dates: "
    f"{df.loc[test_idx, 'transfer_date'].min().date()} "
    f"to "
    f"{df.loc[test_idx, 'transfer_date'].max().date()}"
)


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

numeric_features = player_numeric + team_features

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, player_categorical)
])


# ---------------------------------------------------------
# RANDOM FOREST
# SAME BASELINE MODEL TYPE AS MODEL A
# ---------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------

print("\nTraining Model B...")

pipeline.fit(X_train, y_train)

print("Training complete.")


# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------

train_pred = pipeline.predict(X_train)
val_pred = pipeline.predict(X_val)
test_pred = pipeline.predict(X_test)


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

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


# ---------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL B RESULTS")
print("=" * 60)

print("\nTrain:")
print(f"MAE  : {train_mae:.4f}")
print(f"RMSE : {train_rmse:.4f}")
print(f"R²   : {train_r2:.4f}")

print("\nValidation:")
print(f"MAE  : {val_mae:.4f}")
print(f"RMSE : {val_rmse:.4f}")
print(f"R²   : {val_r2:.4f}")

print("\nTest:")
print(f"MAE  : {test_mae:.4f}")
print(f"RMSE : {test_rmse:.4f}")
print(f"R²   : {test_r2:.4f}")


# ---------------------------------------------------------
# SAVE MODEL COMPARISON
# ---------------------------------------------------------

comparison = pd.DataFrame([
    {
        "model": "Model B - Random Forest",
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "validation_mae": val_mae,
        "validation_rmse": val_rmse,
        "validation_r2": val_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2
    }
])

comparison.to_csv(
    OUTPUT_DIR / "model_b_results.csv",
    index=False
)


# ---------------------------------------------------------
# SAVE TEST PREDICTIONS
# ---------------------------------------------------------

predictions = df.loc[test_idx].copy()

predictions["actual_GA90"] = y_test.values
predictions["predicted_GA90"] = test_pred
predictions["residual"] = (
    predictions["actual_GA90"] -
    predictions["predicted_GA90"]
)

predictions.to_csv(
    OUTPUT_DIR / "model_b_test_predictions.csv",
    index=False
)


print("\nFiles created:")
print(OUTPUT_DIR / "model_b_results.csv")
print(OUTPUT_DIR / "model_b_test_predictions.csv")

print("\n" + "=" * 60)
print("MODEL B COMPLETE")
print("=" * 60)
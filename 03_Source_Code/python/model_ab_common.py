import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "04_Datasets" / "derived" / "team_context_dataset.csv"
OUT = BASE / "06_Results_and_Demo_Evidence" / "research_results" / "model_ab_common_results"
OUT.mkdir(exist_ok=True)

print("=" * 65)
print("COMMON-COHORT MODEL A vs MODEL B")
print("=" * 65)

df = pd.read_csv(DATA)
df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")

# EXACT COMMON COHORT
df = df[
    (df["complete_365_day_window"] == True) &
    (df["eligible_450min"] == True) &
    (df["team_matches_last10"] > 0)
].dropna(subset=["post_GA90", "transfer_date"]).copy()

df = df.sort_values("transfer_date").reset_index(drop=True)

print(f"\nCommon cohort: {len(df)}")

# ---------------- FEATURES ----------------

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

features_a = player_numeric + player_categorical
features_b = features_a + team_features

TARGET = "post_GA90"

# ---------------- SAME CHRONOLOGICAL SPLIT ----------------

n = len(df)
train_end = int(n * 0.60)
val_end = int(n * 0.80)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

print("\nSPLIT")
print(f"Train: {len(train)}")
print(f"Validation: {len(val)}")
print(f"Test: {len(test)}")

# ---------------- PIPELINE ----------------

def make_pipeline(numeric_features, categorical_features):

    prep = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numeric_features
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ))
            ]),
            categorical_features
        )
    ])

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt"
    )

    return Pipeline([
        ("preprocessor", prep),
        ("model", rf)
    ])


def evaluate(model, X, y):
    pred = model.predict(X)

    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred)
    }, pred


# ==========================================================
# MODEL A
# ==========================================================

print("\nTraining Model A: PLAYER ONLY")

model_a = make_pipeline(
    player_numeric,
    player_categorical
)

model_a.fit(
    train[features_a],
    train[TARGET]
)

a_train, _ = evaluate(model_a, train[features_a], train[TARGET])
a_val, _ = evaluate(model_a, val[features_a], val[TARGET])
a_test, a_test_pred = evaluate(model_a, test[features_a], test[TARGET])

# ==========================================================
# MODEL B
# ==========================================================

print("Training Model B: PLAYER + TEAM CONTEXT")

model_b = make_pipeline(
    player_numeric + team_features,
    player_categorical
)

model_b.fit(
    train[features_b],
    train[TARGET]
)

b_train, _ = evaluate(model_b, train[features_b], train[TARGET])
b_val, _ = evaluate(model_b, val[features_b], val[TARGET])
b_test, b_test_pred = evaluate(model_b, test[features_b], test[TARGET])


# ==========================================================
# RESULTS
# ==========================================================

results = pd.DataFrame([
    ["Model A", "Train", a_train["MAE"], a_train["RMSE"], a_train["R2"]],
    ["Model A", "Validation", a_val["MAE"], a_val["RMSE"], a_val["R2"]],
    ["Model A", "Test", a_test["MAE"], a_test["RMSE"], a_test["R2"]],
    ["Model B", "Train", b_train["MAE"], b_train["RMSE"], b_train["R2"]],
    ["Model B", "Validation", b_val["MAE"], b_val["RMSE"], b_val["R2"]],
    ["Model B", "Test", b_test["MAE"], b_test["RMSE"], b_test["R2"]],
], columns=["Model", "Split", "MAE", "RMSE", "R2"])

print("\n" + "=" * 65)
print("COMMON-COHORT RESULTS")
print("=" * 65)
print(results.to_string(index=False))

# ---------------- IMPROVEMENT ----------------

mae_change = b_test["MAE"] - a_test["MAE"]
rmse_change = b_test["RMSE"] - a_test["RMSE"]
r2_change = b_test["R2"] - a_test["R2"]

mae_pct = ((a_test["MAE"] - b_test["MAE"]) / a_test["MAE"]) * 100
rmse_pct = ((a_test["RMSE"] - b_test["RMSE"]) / a_test["RMSE"]) * 100

print("\n" + "=" * 65)
print("TEST-SET CHANGE: MODEL A → MODEL B")
print("=" * 65)

print(f"MAE change : {mae_change:+.4f}")
print(f"MAE change : {mae_pct:+.2f}%")
print(f"RMSE change: {rmse_change:+.4f}")
print(f"RMSE change: {rmse_pct:+.2f}%")
print(f"R² change  : {r2_change:+.4f}")

# ---------------- SAVE ----------------

results.to_csv(
    OUT / "common_cohort_comparison.csv",
    index=False
)

predictions = test.copy()
predictions["actual_GA90"] = test[TARGET].values
predictions["model_a_prediction"] = a_test_pred
predictions["model_b_prediction"] = b_test_pred
predictions["model_a_residual"] = (
    predictions["actual_GA90"] -
    predictions["model_a_prediction"]
)
predictions["model_b_residual"] = (
    predictions["actual_GA90"] -
    predictions["model_b_prediction"]
)

predictions.to_csv(
    OUT / "common_cohort_test_predictions.csv",
    index=False
)

print("\nSaved:")
print(OUT / "common_cohort_comparison.csv")
print(OUT / "common_cohort_test_predictions.csv")

print("\nDONE.")

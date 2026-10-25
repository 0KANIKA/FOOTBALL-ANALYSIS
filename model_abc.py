import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(r"C:\Users\rsuba\Downloads\R football analytics")

TEAM_FILE = BASE / "team_context_dataset.csv"
COACH_FILE = BASE / "team_coach_context_dataset.csv"

OUT = BASE / "model_abc_results"
OUT.mkdir(exist_ok=True)

print("=" * 65)
print("MODEL A vs B vs C")
print("PLAYER | PLAYER+TEAM | PLAYER+TEAM+COACH")
print("=" * 65)

# ---------------------------------------------------------
# LOAD BOTH FILES
# ---------------------------------------------------------

print("\nLoading team context dataset...")
team = pd.read_csv(TEAM_FILE)

print("Loading coach context dataset...")
coach = pd.read_csv(COACH_FILE)

print(f"Team dataset:  {len(team)}")
print(f"Coach dataset: {len(coach)}")

# ---------------------------------------------------------
# MERGE TEAM + COACH DATA
# ---------------------------------------------------------

# Coach file was created from the original transfer dataset,
# so merge the team context onto it using the transfer identity.

keys = [
    "player_id",
    "transfer_date",
    "from_club_id",
    "to_club_id"
]

team["transfer_date"] = pd.to_datetime(
    team["transfer_date"], errors="coerce"
)

coach["transfer_date"] = pd.to_datetime(
    coach["transfer_date"], errors="coerce"
)

team_cols = keys + [
    "team_matches_last10",
    "team_goals_for_avg10",
    "team_goals_against_avg10",
    "team_win_rate10",
    "team_points_per_game10",
    "team_goal_difference_avg10"
]

team_small = team[team_cols].copy()

df = coach.merge(
    team_small,
    on=keys,
    how="left"
)

print(f"Merged dataset: {len(df)}")

# ---------------------------------------------------------
# COMMON COHORT
# ---------------------------------------------------------

df = df[
    (df["complete_365_day_window"] == True) &
    (df["eligible_450min"] == True) &
    (df["team_matches_last10"] > 0) &
    (df["coach_name"].notna())
].dropna(
    subset=["post_GA90", "transfer_date"]
).copy()

df = df.sort_values("transfer_date").reset_index(drop=True)

print(f"\nCommon A/B/C cohort: {len(df)}")

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

coach_features = [
    "coach_name"
]

features_a = player_numeric + player_categorical
features_b = features_a + team_features
features_c = features_b + coach_features

TARGET = "post_GA90"

# ---------------------------------------------------------
# CHRONOLOGICAL SPLIT
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------

def make_pipeline(numeric_features, categorical_features):

    preprocessor = ColumnTransformer([
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

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt"
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def evaluate(model, X, y):

    pred = model.predict(X)

    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred)
    }, pred


# =========================================================
# MODEL A
# =========================================================

print("\nTraining Model A...")

model_a = make_pipeline(
    player_numeric,
    player_categorical
)

model_a.fit(
    train[features_a],
    train[TARGET]
)

a_train, _ = evaluate(
    model_a,
    train[features_a],
    train[TARGET]
)

a_val, _ = evaluate(
    model_a,
    val[features_a],
    val[TARGET]
)

a_test, a_pred = evaluate(
    model_a,
    test[features_a],
    test[TARGET]
)


# =========================================================
# MODEL B
# =========================================================

print("Training Model B...")

model_b = make_pipeline(
    player_numeric + team_features,
    player_categorical
)

model_b.fit(
    train[features_b],
    train[TARGET]
)

b_train, _ = evaluate(
    model_b,
    train[features_b],
    train[TARGET]
)

b_val, _ = evaluate(
    model_b,
    val[features_b],
    val[TARGET]
)

b_test, b_pred = evaluate(
    model_b,
    test[features_b],
    test[TARGET]
)


# =========================================================
# MODEL C
# =========================================================

print("Training Model C...")

model_c = make_pipeline(
    player_numeric,
    player_categorical + team_features + coach_features
)

model_c.fit(
    train[features_c],
    train[TARGET]
)

c_train, _ = evaluate(
    model_c,
    train[features_c],
    train[TARGET]
)

c_val, _ = evaluate(
    model_c,
    val[features_c],
    val[TARGET]
)

c_test, c_pred = evaluate(
    model_c,
    test[features_c],
    test[TARGET]
)


# =========================================================
# RESULTS
# =========================================================

results = pd.DataFrame([
    ["Model A", "Train", a_train["MAE"], a_train["RMSE"], a_train["R2"]],
    ["Model A", "Validation", a_val["MAE"], a_val["RMSE"], a_val["R2"]],
    ["Model A", "Test", a_test["MAE"], a_test["RMSE"], a_test["R2"]],

    ["Model B", "Train", b_train["MAE"], b_train["RMSE"], b_train["R2"]],
    ["Model B", "Validation", b_val["MAE"], b_val["RMSE"], b_val["R2"]],
    ["Model B", "Test", b_test["MAE"], b_test["RMSE"], b_test["R2"]],

    ["Model C", "Train", c_train["MAE"], c_train["RMSE"], c_train["R2"]],
    ["Model C", "Validation", c_val["MAE"], c_val["RMSE"], c_val["R2"]],
    ["Model C", "Test", c_test["MAE"], c_test["RMSE"], c_test["R2"]],
], columns=[
    "Model",
    "Split",
    "MAE",
    "RMSE",
    "R2"
])

print("\n" + "=" * 65)
print("A / B / C RESULTS")
print("=" * 65)

print(results.to_string(index=False))


# =========================================================
# INCREMENTAL CHANGES
# =========================================================

print("\n" + "=" * 65)
print("INCREMENTAL TEST-SET CHANGES")
print("=" * 65)

print("\nA -> B  (Team Context)")
print(f"MAE : {a_test['MAE']:.4f} -> {b_test['MAE']:.4f}")
print(f"RMSE: {a_test['RMSE']:.4f} -> {b_test['RMSE']:.4f}")
print(f"R2  : {a_test['R2']:.4f} -> {b_test['R2']:.4f}")

print("\nB -> C  (Coach Context)")
print(f"MAE : {b_test['MAE']:.4f} -> {c_test['MAE']:.4f}")
print(f"RMSE: {b_test['RMSE']:.4f} -> {c_test['RMSE']:.4f}")
print(f"R2  : {b_test['R2']:.4f} -> {c_test['R2']:.4f}")

print("\nA -> C  (Team + Coach Context)")
print(f"MAE : {a_test['MAE']:.4f} -> {c_test['MAE']:.4f}")
print(f"RMSE: {a_test['RMSE']:.4f} -> {c_test['RMSE']:.4f}")
print(f"R2  : {a_test['R2']:.4f} -> {c_test['R2']:.4f}")


# =========================================================
# SAVE
# =========================================================

results.to_csv(
    OUT / "model_abc_comparison.csv",
    index=False
)

predictions = test.copy()

predictions["actual_GA90"] = test[TARGET].values
predictions["model_a_prediction"] = a_pred
predictions["model_b_prediction"] = b_pred
predictions["model_c_prediction"] = c_pred

predictions.to_csv(
    OUT / "model_abc_test_predictions.csv",
    index=False
)

print("\nSaved:")
print(OUT / "model_abc_comparison.csv")
print(OUT / "model_abc_test_predictions.csv")

print("\n" + "=" * 65)
print("MODEL A/B/C COMPLETE")
print("=" * 65)
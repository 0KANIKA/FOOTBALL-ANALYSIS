import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PATHS
# ============================================================

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE, "04_Datasets", "derived")

TEAM_FILE = os.path.join(DATA_DIR, "team_context_dataset.csv")
COACH_FILE = os.path.join(DATA_DIR, "team_coach_context_dataset.csv")

OUT_DIR = os.path.join(BASE, "06_Results_and_Demo_Evidence", "research_results", "model_bc_common_results")

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

team = pd.read_csv(TEAM_FILE)
coach = pd.read_csv(COACH_FILE)

print("Team context:", team.shape)
print("Coach context:", coach.shape)


# ============================================================
# MERGE
# ============================================================

keys = [
    "player_id",
    "transfer_date",
    "from_club_id",
    "to_club_id"
]

df = coach.merge(
    team[keys + [
        "team_matches_last10",
        "team_goals_for_avg10",
        "team_goals_against_avg10",
        "team_win_rate10",
        "team_points_per_game10",
        "team_goal_difference_avg10"
    ]],
    on=keys,
    how="inner"
)

print("Merged:", df.shape)


# ============================================================
# FILTER COMMON B/C COHORT
# ============================================================

df["transfer_date"] = pd.to_datetime(df["transfer_date"])

df = df[
    (df["complete_365_day_window"] == True) &
    (df["eligible_450min"] == True) &
    (df["team_matches_last10"] > 0) &
    (df["coach_name"].notna())
].copy()

df = df.sort_values("transfer_date").reset_index(drop=True)

print("\nCommon B/C cohort:", len(df))


# ============================================================
# FEATURES
# ============================================================

target = "post_GA90"

player_features = [
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


# ============================================================
# CHRONOLOGICAL 60 / 20 / 20 SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

train = df.iloc[:train_end]
val = df.iloc[train_end:val_end]
test = df.iloc[val_end:]

print("\nSplit:")
print("Train:", len(train))
print("Validation:", len(val))
print("Test:", len(test))

print("\nDate ranges:")
print("Train:", train["transfer_date"].min(), "to", train["transfer_date"].max())
print("Val  :", val["transfer_date"].min(), "to", val["transfer_date"].max())
print("Test :", test["transfer_date"].min(), "to", test["transfer_date"].max())


# ============================================================
# MODEL FUNCTION
# ============================================================

def build_model(features):

    categorical = [
        f for f in features
        if f in ["position", "sub_position", "foot", "coach_name"]
    ]

    numerical = [
        f for f in features
        if f not in categorical
    ]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numerical),
        ("cat", categorical_pipe, categorical)
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
    ])# ============================================================
# TRAIN + EVALUATE
# ============================================================

def evaluate(name, features):

    print(f"\n================ {name} ================")

    model = build_model(features)

    X_train = train[features]
    y_train = train[target]

    X_val = val[features]
    y_val = val[target]

    X_test = test[features]
    y_test = test[target]

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"Test MAE : {mae:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test R²  : {r2:.6f}")

    return model, predictions, {
        "model": name,
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2
    }


# ============================================================
# MODEL B
# Player + Team Context
# ============================================================

model_b, pred_b, result_b = evaluate(
    "Model B",
    player_features + team_features
)


# ============================================================
# MODEL C
# Player + Team + Coach Context
# ============================================================

model_c, pred_c, result_c = evaluate(
    "Model C",
    player_features + team_features + coach_features
)


# ============================================================
# COMPARISON
# ============================================================

results = pd.DataFrame([
    result_b,
    result_c
])

print("\n================ COMPARISON ================")
print(results.to_string(index=False))

b = results.iloc[0]
c = results.iloc[1]

print("\nB → C change:")

print(
    "MAE:",
    f"{b.test_mae:.6f} → {c.test_mae:.6f}",
    f"({(c.test_mae - b.test_mae) / b.test_mae * 100:+.2f}%)"
)

print(
    "RMSE:",
    f"{b.test_rmse:.6f} → {c.test_rmse:.6f}",
    f"({(c.test_rmse - b.test_rmse) / b.test_rmse * 100:+.2f}%)"
)

print(
    "R²:",
    f"{b.test_r2:.6f} → {c.test_r2:.6f}",
    f"({c.test_r2 - b.test_r2:+.6f})"
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUT_DIR + r"\model_bc_common_comparison.csv",
    index=False
)

predictions = pd.DataFrame({
    "player_id": test["player_id"].values,
    "transfer_date": test["transfer_date"].values,
    "actual_post_GA90": test[target].values,
    "model_b_prediction": pred_b,
    "model_c_prediction": pred_c
})

predictions.to_csv(
    OUT_DIR + r"\model_bc_common_predictions.csv",
    index=False
)

print("\nSaved results to:")
print(OUT_DIR)

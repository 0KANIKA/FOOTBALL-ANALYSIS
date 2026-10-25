import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


# ============================================================
# PATHS
# ============================================================

BASE = r"C:\Users\rsuba\Downloads\R football analytics"

TEAM_FILE = BASE + r"\team_context_dataset.csv"
OUT_DIR = BASE + r"\feature_importance_results"

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(TEAM_FILE)

df["transfer_date"] = pd.to_datetime(df["transfer_date"])


# ============================================================
# FILTER MODEL B COHORT
# ============================================================

df = df[
    (df["complete_365_day_window"] == True) &
    (df["eligible_450min"] == True) &
    (df["team_matches_last10"] > 0)
].copy()

df = df.sort_values("transfer_date").reset_index(drop=True)

print("Cohort:", len(df))


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

features = player_features + team_features


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

train = df.iloc[:train_end]
test = df.iloc[val_end:]

print("Train:", len(train))
print("Test :", len(test))

print(
    "Test dates:",
    test["transfer_date"].min(),
    "to",
    test["transfer_date"].max()
)


# ============================================================
# PREPROCESSING
# ============================================================

categorical = [
    "position",
    "sub_position",
    "foot"
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


# ============================================================
# RANDOM FOREST
# ============================================================

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf)
])


# ============================================================
# TRAIN
# ============================================================

model.fit(
    train[features],
    train[target]
)

print("\nModel trained.")


# ============================================================
# GET TRANSFORMED FEATURE NAMES
# ============================================================

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

rf_model = model.named_steps["model"]

importance = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print("\n========== TOP 20 FEATURES ==========")

print(
    importance_df.head(20).to_string(index=False)
)


# ============================================================
# SAVE RAW IMPORTANCE
# ============================================================

importance_df.to_csv(
    OUT_DIR + r"\rf_feature_importance.csv",
    index=False
)


# ============================================================
# GROUP ONE-HOT FEATURES BACK TO ORIGINAL FEATURES
# ============================================================

grouped = {}

for original_feature in features:

    matching = [
        i for i, name in enumerate(feature_names)
        if (
            name.endswith("__" + original_feature)
            or name.startswith("num__" + original_feature)
            or name.startswith("cat__" + original_feature + "_")
        )
    ]

    if matching:
        grouped[original_feature] = importance[matching].sum()


grouped_df = pd.DataFrame(
    list(grouped.items()),
    columns=["feature", "importance"]
)

grouped_df = grouped_df.sort_values(
    "importance",
    ascending=False
)

print("\n========== GROUPED FEATURE IMPORTANCE ==========")

print(
    grouped_df.to_string(index=False)
)

grouped_df.to_csv(
    OUT_DIR + r"\grouped_feature_importance.csv",
    index=False
)


# ============================================================
# TEAM CONTEXT CONTRIBUTION
# ============================================================

team_importance = grouped_df[
    grouped_df["feature"].isin(team_features)
].copy()

team_importance["percentage_of_total"] = (
    team_importance["importance"]
    / grouped_df["importance"].sum()
    * 100
)

print("\n========== TEAM CONTEXT IMPORTANCE ==========")

print(
    team_importance.to_string(index=False)
)

team_importance.to_csv(
    OUT_DIR + r"\team_context_importance.csv",
    index=False
)


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

print("\nCalculating permutation importance...")
print("This may take a little while.")

X_test = test[features]
y_test = test[target]

perm = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

perm_df = pd.DataFrame({
    "feature": features,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
})

perm_df = perm_df.sort_values(
    "importance_mean",
    ascending=False
)

print("\n========== PERMUTATION IMPORTANCE ==========")

print(
    perm_df.to_string(index=False)
)

perm_df.to_csv(
    OUT_DIR + r"\permutation_importance.csv",
    index=False
)


# ============================================================
# SAVE TOP 10
# ============================================================

grouped_df.head(10).to_csv(
    OUT_DIR + r"\top_10_features.csv",
    index=False
)

print("\n==============================================")
print("Feature importance analysis complete.")
print("Results saved to:")
print(OUT_DIR)
print("==============================================")
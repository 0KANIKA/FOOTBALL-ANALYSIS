import os
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIG
# ============================================================

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE, "04_Datasets", "derived")
DATA_FILE = os.path.join(DATA_DIR, "rich_team_context_dataset.csv")
TARGET_FILE = os.path.join(DATA_DIR, "post_transfer_performance.csv")

OUTPUT_DIR = os.path.join(BASE, "06_Results_and_Demo_Evidence", "research_results", "model_c_rich_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MODEL C — RICH DESTINATION CONTEXT")
print("=" * 70)

print("\nLoading datasets...")

df = pd.read_csv(DATA_FILE)
target = pd.read_csv(TARGET_FILE)

df["transfer_date"] = pd.to_datetime(
    df["transfer_date"],
    errors="coerce"
)

target["transfer_date"] = pd.to_datetime(
    target["transfer_date"],
    errors="coerce"
)

print(f"Rich context rows: {len(df):,}")
print(f"Target rows:       {len(target):,}")


# ============================================================
# IDENTIFY TARGET COLUMN
# ============================================================

print("\nTarget columns:")
print(target.columns.tolist())

possible_targets = [
    "post_goals_assists_per90",
    "goals_assists_per90_post",
    "ga_per90_post",
    "post_ga_per90",
    "goals_plus_assists_per90",
    "g_plus_a_per90",
]

target_col = None

for col in possible_targets:
    if col in target.columns:
        target_col = col
        break

if target_col is None:
    raise ValueError(
        "Could not identify the post-transfer G+A/90 target. "
        f"Available columns: {target.columns.tolist()}"
    )

print(f"\nUsing target: {target_col}")


# ============================================================
# MERGE TARGET
# ============================================================

# Use transfer_row_id where available because it uniquely
# identifies the transfer observation in our dataset.

if "transfer_row_id" in df.columns and "transfer_row_id" in target.columns:

    target_small = target[
        [
            "transfer_row_id",
            target_col
        ]
    ].drop_duplicates("transfer_row_id")

    df = df.merge(
        target_small,
        on="transfer_row_id",
        how="inner"
    )

else:

    # Fallback
    merge_cols = [
        "player_id",
        "transfer_date",
        "to_club_id"
    ]

    target_small = target[
        merge_cols + [target_col]
    ].drop_duplicates(merge_cols)

    df = df.merge(
        target_small,
        on=merge_cols,
        how="inner"
    )


print(f"Merged rows: {len(df):,}")


# ============================================================
# TARGET CLEANING
# ============================================================

df[target_col] = pd.to_numeric(
    df[target_col],
    errors="coerce"
)

df = df[df[target_col].notna()].copy()


# ============================================================
# ELIGIBILITY
#
# Same primary research cohort:
#   - complete 365-day post-transfer window
#   - >= 450 post-transfer minutes
#   - destination team has usable historical matches
#
# We use the actual post-transfer dataset columns if present.
# ============================================================

eligibility_cols = target.columns.tolist()

print("\nChecking target eligibility columns...")

complete_candidates = [
    "complete_365d_window",
    "complete_window",
    "window_complete",
]

minutes_candidates = [
    "minutes_post",
    "post_minutes",
    "total_minutes_post",
]

complete_col = next(
    (c for c in complete_candidates if c in target.columns),
    None
)

minutes_col = next(
    (c for c in minutes_candidates if c in target.columns),
    None
)

# Re-merge eligibility information if available
eligibility_merge_cols = ["transfer_row_id"]

if (
    "transfer_row_id" in df.columns
    and "transfer_row_id" in target.columns
):

    extra_cols = ["transfer_row_id"]

    if complete_col:
        extra_cols.append(complete_col)

    if minutes_col:
        extra_cols.append(minutes_col)

    extra = target[extra_cols].drop_duplicates(
        "transfer_row_id"
    )

    df = df.drop(
        columns=[
            c for c in [complete_col, minutes_col]
            if c and c in df.columns
        ],
        errors="ignore"
    )

    df = df.merge(
        extra,
        on="transfer_row_id",
        how="left"
    )


# Apply complete-window requirement if available
if complete_col:
    before = len(df)

    df = df[
        df[complete_col].fillna(False).astype(bool)
    ].copy()

    print(
        f"Complete 365d window: "
        f"{len(df):,} / {before:,}"
    )
else:
    print(
        "WARNING: complete-window flag not found. "
        "Proceeding without this filter."
    )


# Apply >=450 post-transfer minute requirement
if minutes_col:
    df[minutes_col] = pd.to_numeric(
        df[minutes_col],
        errors="coerce"
    )

    before = len(df)

    df = df[
        df[minutes_col] >= 450
    ].copy()

    print(
        f">=450 post-transfer minutes: "
        f"{len(df):,} / {before:,}"
    )
else:
    print(
        "WARNING: post-transfer minutes column not found. "
        "Proceeding without this filter."
    )


# Require usable destination-team history
if "rich_team_matches_last10" in df.columns:

    before = len(df)

    df = df[
        df["rich_team_matches_last10"] > 0
    ].copy()

    print(
        f"Usable destination-team history: "
        f"{len(df):,} / {before:,}"
    )


# ============================================================
# REMOVE INVALID TARGETS
# ============================================================

df = df[
    np.isfinite(df[target_col])
].copy()


print("\nFinal Model C cohort:", len(df))


# ============================================================
# FEATURE GROUPS
# ============================================================

PLAYER_NUMERIC = [
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
]

PLAYER_CATEGORICAL = [
    "position",
    "sub_position",
    "foot",
]

TEAM_NUMERIC = [
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
]

TEAM_CATEGORICAL = [
    "dominant_formation10",
]


FEATURES = (
    PLAYER_NUMERIC
    + PLAYER_CATEGORICAL
    + TEAM_NUMERIC
    + TEAM_CATEGORICAL
)

FEATURES = [
    f for f in FEATURES
    if f in df.columns
]


print("\nFeatures used:")
for f in FEATURES:
    print("  ", f)

print(f"\nTotal features before encoding: {len(FEATURES)}")


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    "transfer_date"
).reset_index(drop=True)


# ============================================================
# CHRONOLOGICAL 60 / 20 / 20 SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

train = df.iloc[:train_end].copy()
val = df.iloc[train_end:val_end].copy()
test = df.iloc[val_end:].copy()

print("\nChronological split:")
print(
    f"Train: {len(train):,} "
    f"({train['transfer_date'].min().date()} → "
    f"{train['transfer_date'].max().date()})"
)

print(
    f"Validation: {len(val):,} "
    f"({val['transfer_date'].min().date()} → "
    f"{val['transfer_date'].max().date()})"
)

print(
    f"Test: {len(test):,} "
    f"({test['transfer_date'].min().date()} → "
    f"{test['transfer_date'].max().date()})"
)


# ============================================================
# X / y
# ============================================================

X_train = train[FEATURES]
y_train = train[target_col]

X_val = val[FEATURES]
y_val = val[target_col]

X_test = test[FEATURES]
y_test = test[target_col]


# ============================================================
# PREPROCESSING
#
# IMPORTANT:
# Numeric imputer is fitted ONLY on training data.
# Categorical imputer is fitted ONLY on training data.
# This avoids preprocessing leakage.
# ============================================================

numeric_features = [
    f for f in FEATURES
    if f in PLAYER_NUMERIC + TEAM_NUMERIC
]

categorical_features = [
    f for f in FEATURES
    if f in PLAYER_CATEGORICAL + TEAM_CATEGORICAL
]


numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
    ],
    remainder="drop"
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    max_features="sqrt"
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining Random Forest...")

pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, X, y):

    pred = pipeline.predict(X)

    mae = mean_absolute_error(
        y,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            pred
        )
    )

    r2 = r2_score(
        y,
        pred
    )

    print(
        f"{name:<12} "
        f"MAE={mae:.6f}  "
        f"RMSE={rmse:.6f}  "
        f"R²={r2:.6f}"
    )

    return pred, {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


print("\n" + "=" * 70)
print("MODEL C RESULTS")
print("=" * 70)

train_pred, train_metrics = evaluate(
    "Train",
    X_train,
    y_train
)

val_pred, val_metrics = evaluate(
    "Validation",
    X_val,
    y_val
)

test_pred, test_metrics = evaluate(
    "Test",
    X_test,
    y_test
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": "Model C - Rich Destination Context",
    "algorithm": "Random Forest",
    "target": target_col,
    "cohort_size": int(len(df)),
    "train_size": int(len(train)),
    "validation_size": int(len(val)),
    "test_size": int(len(test)),
    "train_start": str(train["transfer_date"].min().date()),
    "train_end": str(train["transfer_date"].max().date()),
    "validation_start": str(val["transfer_date"].min().date()),
    "validation_end": str(val["transfer_date"].max().date()),
    "test_start": str(test["transfer_date"].min().date()),
    "test_end": str(test["transfer_date"].max().date()),
    "train": train_metrics,
    "validation": val_metrics,
    "test": test_metrics,
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "metrics.json"
    ),
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

predictions = test[
    [
        "transfer_row_id",
        "player_id",
        "transfer_date",
        "from_club_name",
        "to_club_name",
    ]
].copy()

predictions["actual"] = y_test.values
predictions["predicted"] = test_pred

predictions["residual"] = (
    predictions["actual"]
    - predictions["predicted"]
)

predictions["absolute_error"] = (
    predictions["residual"]
    .abs()
)

predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_predictions.csv"
    ),
    index=False
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nCalculating feature importance...")

rf = pipeline.named_steps["model"]

feature_names = (
    pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_importance.csv"
    ),
    index=False
)


print("\nTop 30 encoded features:")

print(
    importance.head(30).to_string(
        index=False
    )
)


# ============================================================
# GROUPED IMPORTANCE
# ============================================================

def assign_group(feature_name):

    f = feature_name.lower()

    if "position" in f or "sub_position" in f:
        return "Player Role"

    for x in PLAYER_NUMERIC:
        if x.lower() in f:
            return "Player Characteristics"

    for x in TEAM_NUMERIC:
        if x.lower() in f:
            return "Destination Team Context"

    if "formation" in f:
        return "Destination Team Context"

    return "Other"


importance["group"] = importance[
    "feature"
].apply(assign_group)

grouped = (
    importance
    .groupby("group")["importance"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

grouped["percentage"] = (
    grouped["importance"]
    / grouped["importance"].sum()
    * 100
)

grouped.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "grouped_feature_importance.csv"
    ),
    index=False
)

print("\nGrouped feature importance:")

print(
    grouped.to_string(
        index=False
    )
)


# ============================================================
# POSITION PERFORMANCE
# ============================================================

if "position" in test.columns:

    position_results = []

    temp = test.copy()

    temp["prediction"] = test_pred

    for position, group in temp.groupby(
        "position",
        dropna=False
    ):

        position_results.append({
            "position": position,
            "n": len(group),
            "MAE": mean_absolute_error(
                group[target_col],
                group["prediction"]
            ),
            "RMSE": np.sqrt(
                mean_squared_error(
                    group[target_col],
                    group["prediction"]
                )
            ),
            "R2": r2_score(
                group[target_col],
                group["prediction"]
            )
            if len(group) > 1
            else np.nan,
        })

    position_results = pd.DataFrame(
        position_results
    )

    position_results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "position_results.csv"
        ),
        index=False
    )

    print("\nPosition-level test results:")
    print(
        position_results.to_string(
            index=False
        )
    )


# ============================================================
# YEAR PERFORMANCE
# ============================================================

test_year = test.copy()

test_year["prediction"] = test_pred
test_year["year"] = (
    test_year["transfer_date"]
    .dt.year
)

year_results = []

for year, group in test_year.groupby(
    "year"
):

    year_results.append({
        "year": int(year),
        "n": len(group),
        "MAE": mean_absolute_error(
            group[target_col],
            group["prediction"]
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                group[target_col],
                group["prediction"]
            )
        ),
        "R2": r2_score(
            group[target_col],
            group["prediction"]
        )
        if len(group) > 1
        else np.nan,
    })

year_results = pd.DataFrame(
    year_results
)

year_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "year_results.csv"
    ),
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL C COMPLETE")
print("=" * 70)

print(
    f"\nTest MAE :  {test_metrics['MAE']:.6f}"
)

print(
    f"Test RMSE:  {test_metrics['RMSE']:.6f}"
)

print(
    f"Test R²  :  {test_metrics['R2']:.6f}"
)

print(
    f"\nResults saved to:\n{OUTPUT_DIR}"
)

print("\nFiles created:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):
    print("  ", filename)

print("\nDone.")

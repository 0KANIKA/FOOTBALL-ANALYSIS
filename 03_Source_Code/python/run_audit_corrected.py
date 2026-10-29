from pathlib import Path
import itertools
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Locate the project data folder. You can override it with FOOTBALL_DATA_ROOT.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
candidate_roots = []
if os.environ.get("FOOTBALL_DATA_ROOT"):
    candidate_roots.append(Path(os.environ["FOOTBALL_DATA_ROOT"]).expanduser().resolve())
candidate_roots.extend([
    PROJECT_ROOT / "04_Datasets" / "derived",
    PROJECT_ROOT,
    SCRIPT_DIR,
    SCRIPT_DIR.parent,
])
ROOT = next(
    (p for p in candidate_roots if (p / "post_transfer_performance.csv").exists()),
    None,
)
if ROOT is None:
    raise FileNotFoundError(
        "Could not find post_transfer_performance.csv. Set FOOTBALL_DATA_ROOT "
        "to the folder containing the three source CSVs."
    )

OUT = PROJECT_ROOT / "06_Results_and_Demo_Evidence" / "audit_results" / "outputs_corrected"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
TUNE_TREES = 25
FINAL_TREES = 100
TARGETS = [
    "post_GA90", "post_goals_per90", "post_assists_per90",
    "post_minutes", "post_appearances"
]

CORE_NUM = [
    "age_at_transfer", "height_in_cm", "appearances_pre", "minutes_pre",
    "goals_pre", "assists_pre", "goals_per90_pre", "assists_per90_pre",
    "yellow_cards_pre", "red_cards_pre", "transfer_market_value",
    "market_value_pre", "transfer_fee"
]
CORE_CAT = ["position", "sub_position", "foot"]
TEAM = [
    "rich_team_matches_last10", "rich_team_goals_for_avg10",
    "rich_team_goals_against_avg10", "rich_team_goal_difference_avg10",
    "rich_team_win_rate10", "rich_team_points_per_game10"
]
TACTIC_NUM = [
    "dominant_formation_frequency10", "formation_diversity10",
    "manager_matches_last10", "unique_managers_last10"
]
TACTIC_CAT = ["dominant_formation10", "dominant_manager10"]
SQUAD = [
    "same_position_competitor_count", "same_position_competitor_minutes",
    "same_position_avg_market_value", "same_position_top_competitor_minutes",
    "same_sub_position_competitor_count", "same_sub_position_competitor_minutes",
    "same_sub_position_top_minutes", "position_minutes_share"
]
COACH = [
    "coach_known_pretransfer", "coach_position_group_minutes_share",
    "coach_position_group_start_rate", "coach_exact_position_minutes_share",
    "coach_exact_position_start_rate", "coach_position_competitor_count",
    "coach_position_competitor_minutes", "coach_position_rotation_hhi",
    "coach_formation_stability", "coach_observed_tenure_days"
]


GROUPS = {
    # Basic player information only
    "core": (
        CORE_NUM,
        CORE_CAT
    ),

    # Mode A: player + team form + tactics + squad competition
    # Coach features are excluded
    "mode_A": (
        CORE_NUM + TEAM + TACTIC_NUM + SQUAD,
        CORE_CAT + TACTIC_CAT
    ),

    # Original full model, retained for comparison
    "full_enhanced": (
        CORE_NUM + TEAM + TACTIC_NUM + SQUAD + COACH,
        CORE_CAT + TACTIC_CAT
    ),

    # Comparison: exclude tactical/formation features
    "without_tactical_manager": (
        CORE_NUM + TEAM + SQUAD + COACH,
        CORE_CAT
    ),

    # Comparison: exclude squad competition features
    "without_squad_competition": (
        CORE_NUM + TEAM + TACTIC_NUM + COACH,
        CORE_CAT + TACTIC_CAT
    ),

    # Same feature groups as the updated Mode A
    "without_coach": (
        CORE_NUM + TEAM + TACTIC_NUM + SQUAD,
        CORE_CAT + TACTIC_CAT
    ),
}



def read(name):
    return pd.read_csv(ROOT / name, low_memory=False)

base = read("post_transfer_performance.csv")
rich = read("rich_team_context_dataset.csv")
coach = read("coach_tactical_context_v21.csv")

# Validate unique keys. Unique transfer_row_id alone does not prove semantic alignment,
# so compare transfer identity fields wherever both source tables contain them.
for label, frame in [("target", base), ("rich", rich), ("coach", coach)]:
    if "transfer_row_id" not in frame:
        raise ValueError(f"{label}: missing transfer_row_id")
    dup = int(frame["transfer_row_id"].duplicated().sum())
    if dup:
        raise ValueError(f"{label}: duplicate transfer_row_id={dup}; refusing many-to-many merge")

identity_cols = ["player_id", "transfer_date", "from_club_id", "to_club_id"]
for label, frame in [("rich", rich), ("coach", coach)]:
    common = [c for c in identity_cols if c in base.columns and c in frame.columns]
    if len(common) == len(identity_cols):
        left = base[["transfer_row_id"] + common].copy()
        right = frame[["transfer_row_id"] + common].copy()
        left["transfer_date"] = pd.to_datetime(left["transfer_date"], errors="coerce")
        right["transfer_date"] = pd.to_datetime(right["transfer_date"], errors="coerce")
        check = left.merge(right, on="transfer_row_id", suffixes=("_base", "_ctx"),
                           how="inner", validate="one_to_one")
        mismatch = np.zeros(len(check), dtype=bool)
        for col in common:
            a, b = check[f"{col}_base"], check[f"{col}_ctx"]
            both_missing = a.isna() & b.isna()
            mismatch |= ~(a.eq(b) | both_missing)
        if mismatch.any():
            raise ValueError(
                f"{label}: {int(mismatch.sum())} transfer_row_id values do not match "
                "the base transfer identity fields."
            )
    else:
        warnings.warn(
            f"{label}: cannot verify transfer_row_id semantic alignment because "
            f"not all identity fields are present. Confirm IDs were preserved from "
            "the same master table, not independently generated by row order."
        )

rich_features = [c for c in TEAM + TACTIC_NUM + TACTIC_CAT + SQUAD if c in rich.columns]
df = base.merge(
    rich[["transfer_row_id"] + rich_features],
    on="transfer_row_id", how="left", validate="one_to_one"
)
coach_features = [c for c in COACH if c in coach.columns]
df = df.merge(
    coach[["transfer_row_id"] + coach_features],
    on="transfer_row_id", how="left", validate="one_to_one"
)
df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")

initial_rows = len(df)
df = df[
    (df["complete_365_day_window"] == 1)
    & (df["eligible_450min"] == 1)
    & df["transfer_date"].notna()
].copy()
eligible_rows = len(df)

identity = ["player_id", "transfer_date", "from_club_id", "to_club_id"]
exact_duplicates = int(df.duplicated(identity, keep="first").sum())
df = (
    df.drop_duplicates(identity, keep="first")
      .sort_values(["transfer_date", "transfer_row_id"], kind="mergesort")
      .reset_index(drop=True)
)

if "post_goals_per90" not in df:
    df["post_goals_per90"] = np.where(
        df["post_minutes"] > 0,
        df["post_goals"] / df["post_minutes"] * 90,
        np.nan,
    )
if "post_assists_per90" not in df:
    df["post_assists_per90"] = np.where(
        df["post_minutes"] > 0,
        df["post_assists"] / df["post_minutes"] * 90,
        np.nan,
    )

df = df[np.isfinite(pd.to_numeric(df["post_GA90"], errors="coerce"))].reset_index(drop=True)
n = len(df)
if n < 10:
    raise ValueError(f"Too few eligible rows for modeling: {n}")

# Split at whole-date boundaries nearest the 60% and 80% cumulative targets.
# Constrain candidates so train, validation, and test all contain at least one date.
daily_counts = df.groupby("transfer_date", sort=True).size()
if len(daily_counts) < 3:
    raise ValueError("Need at least three distinct transfer dates for chronological splitting.")
cum_counts = daily_counts.cumsum().to_numpy()
date_values = daily_counts.index.to_numpy()
train_candidates = np.arange(0, len(date_values) - 2)
train_cut_idx = int(train_candidates[np.argmin(np.abs(cum_counts[train_candidates] - 0.60 * n))])
val_candidates = np.arange(train_cut_idx + 1, len(date_values) - 1)
val_cut_idx = int(val_candidates[np.argmin(np.abs(cum_counts[val_candidates] - 0.80 * n))])
date_train_end = pd.Timestamp(date_values[train_cut_idx])
date_val_end = pd.Timestamp(date_values[val_cut_idx])

parts = {
    "train": df[df["transfer_date"] <= date_train_end].copy(),
    "validation": df[
        (df["transfer_date"] > date_train_end)
        & (df["transfer_date"] <= date_val_end)
    ].copy(),
    "test": df[df["transfer_date"] > date_val_end].copy(),
}
if any(len(v) == 0 for v in parts.values()):
    raise ValueError("Chronological split produced an empty partition.")

# Fail fast if a predictor is missing or any outcome/admin column is used as a predictor.
for name, (nums, cats) in GROUPS.items():
    missing = [c for c in nums + cats if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    forbidden = set(TARGETS + [
        "post_goals", "post_assists", "post_minutes", "post_appearances",
        "window_end", "complete_365_day_window", "eligible_450min",
        "transfer_row_id", "dataset_split", "has_target", "clean_final_cohort"
    ])
    overlap = forbidden.intersection(nums + cats)
    if overlap:
        raise ValueError(f"Outcome/administrative leakage in {name}: {sorted(overlap)}")

def make_pipeline(nums, cats, max_features, min_leaf, n_trees):
    prep = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median", add_indicator=True), nums),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                    sparse_output=True
                )),
            ]), cats),
        ],
        remainder="drop",
    )
    rf = RandomForestRegressor(
        n_estimators=n_trees,
        max_features=max_features,
        min_samples_leaf=min_leaf,
        random_state=SEED,
        n_jobs=-1,
    )
    return Pipeline([("prep", prep), ("rf", rf)])

def add_metrics(target, model_name, split_name, actual, predicted, maxf, leaf):
    return {
        "target": target,
        "model": model_name,
        "split": split_name,
        "n": int(len(actual)),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(mean_squared_error(actual, predicted) ** 0.5),
        "R2": float(r2_score(actual, predicted)),
        "max_features": str(maxf),
        "min_samples_leaf": int(leaf),
    }

# Full 2 x 2 grid (four configurations), not only two paired configurations.
search = list(itertools.product(["sqrt", 0.5], [5, 10]))
metrics, preds, tuning = [], [], []

for target in TARGETS:
    yall = pd.to_numeric(df[target], errors="coerce")
    valid = np.isfinite(yall.to_numpy())
    split_mask = {k: valid[v.index.to_numpy()] for k, v in parts.items()}
    tr = parts["train"].loc[split_mask["train"]]
    va = parts["validation"].loc[split_mask["validation"]]
    te = parts["test"].loc[split_mask["test"]]

    for model_name, (nums, cats) in GROUPS.items():
        best = None
        for maxf, leaf in search:
            candidate = make_pipeline(nums, cats, maxf, leaf, TUNE_TREES)
            candidate.fit(tr[nums + cats], yall.loc[tr.index])
            val_pred = candidate.predict(va[nums + cats])
            val_mae = mean_absolute_error(yall.loc[va.index], val_pred)
            tuning.append({
                "target": target,
                "model": model_name,
                "max_features": str(maxf),
                "min_samples_leaf": leaf,
                "validation_MAE": float(val_mae),
                "n_train": len(tr),
                "n_validation": len(va),
                "tuning_trees": TUNE_TREES,
            })
            if best is None or val_mae < best[0]:
                best = (val_mae, maxf, leaf)

        _, best_maxf, best_leaf = best

        # Correct validation evaluation: fit on TRAIN only and predict validation.
        # Validation has been used for hyperparameter selection, so this score is
        # a selection-set score, not an independent final generalization estimate.
        selected_train_model = make_pipeline(
            nums, cats, best_maxf, best_leaf, FINAL_TREES
        )
        
        selected_train_model.fit(tr[nums + cats], yall.loc[tr.index])

        # Training metrics
        train_actual = yall.loc[tr.index].to_numpy()
        train_pred = selected_train_model.predict(tr[nums + cats])
        metrics.append(add_metrics(
            target, model_name, "train", train_actual, train_pred,
            best_maxf, best_leaf
        ))

        # Validation metrics
        val_actual = yall.loc[va.index].to_numpy()
        val_pred = selected_train_model.predict(va[nums + cats])
        metrics.append(add_metrics(
            target, model_name, "validation", val_actual, val_pred,
            best_maxf, best_leaf
        ))

        if target == "post_GA90":
            preds.append(pd.DataFrame({
                "transfer_row_id": va["transfer_row_id"].to_numpy(),
                "transfer_date": va["transfer_date"].to_numpy(),
                "player_id": va["player_id"].to_numpy(),
                "actual": val_actual,
                "prediction": val_pred,
                "model": model_name,
                "split": "validation",
            }))

        # Final test model: refit selected configuration on TRAIN + VALIDATION.
        fit = pd.concat([tr, va]).sort_index()
        final_model = make_pipeline(nums, cats, best_maxf, best_leaf, FINAL_TREES)
        final_model.fit(fit[nums + cats], yall.loc[fit.index])
        test_actual = yall.loc[te.index].to_numpy()
        test_pred = final_model.predict(te[nums + cats])
        metrics.append(add_metrics(
            target, model_name, "test", test_actual, test_pred, best_maxf, best_leaf
        ))
        if target == "post_GA90":
            preds.append(pd.DataFrame({
                "transfer_row_id": te["transfer_row_id"].to_numpy(),
                "transfer_date": te["transfer_date"].to_numpy(),
                "player_id": te["player_id"].to_numpy(),
                "actual": test_actual,
                "prediction": test_pred,
                "model": model_name,
                "split": "test",
            }))

pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
pd.DataFrame(tuning).to_csv(OUT / "validation_tuning.csv", index=False)
pd.concat(preds, ignore_index=True).to_csv(OUT / "post_GA90_predictions.csv", index=False)

split_rows = []
for split_name, part in parts.items():
    split_rows.append({
        "split": split_name,
        "n_primary_cohort": len(part),
        "min_date": part["transfer_date"].min(),
        "max_date": part["transfer_date"].max(),
        "missing_post_minutes": int(part["post_minutes"].isna().sum()),
        "zero_post_minutes": int((part["post_minutes"] == 0).sum()),
        "missing_post_GA90": int(part["post_GA90"].isna().sum()),
    })
pd.DataFrame(split_rows).to_csv(OUT / "split_audit.csv", index=False)

feature_list = []
for nums, cats in GROUPS.values():
    feature_list.extend(nums + cats)
coverage = pd.DataFrame([
    {
        "feature": col,
        "missing_n": int(df[col].isna().sum()),
        "missing_pct": float(df[col].isna().mean() * 100),
        "unique_n": int(df[col].nunique(dropna=True)),
    }
    for col in dict.fromkeys(feature_list) if col in df.columns
])
coverage.to_csv(OUT / "feature_coverage.csv", index=False)

metadata = {
    "seed": SEED,
    "chronological_split_requested": [0.60, 0.20, 0.20],
    "split_boundary": "whole transfer dates; closest cumulative counts to 60% and 80%, constrained to non-empty splits",
    "initial_joined_rows": int(initial_rows),
    "eligible_full_window_rows": int(eligible_rows),
    "primary_common_cohort_rows": int(len(df)),
    "exact_duplicate_transfer_rows_removed": int(exact_duplicates),
    "n_train": int(len(parts["train"])),
    "n_validation": int(len(parts["validation"])),
    "n_test": int(len(parts["test"])),
    "tuning_rule": "full 2x2 grid; select by validation MAE; test not used for selection",
    "validation_reporting": "fit on train only; validation is a hyperparameter-selection set and is not an independent unbiased estimate",
    "test_fit_rule": "refit selected configuration on train+validation before one held-out test evaluation",
    "tuning_trees": TUNE_TREES,
    "final_trees": FINAL_TREES,
    "feature_identity_validation": "identity fields compared when available; warning emitted when they are not available in context sources",
}
(OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

print("Data root:", ROOT)
print("Outputs:", OUT)
print(pd.DataFrame(metrics).query("target == 'post_GA90'").to_string(index=False))

"""
Stage 1 Data Pipeline: Football Transfer Opportunity Dataset
=============================================================
Builds the Stage 1 historical transfer dataset strictly from local data:
- Filters for transfers with complete 365-day follow-up (N = 13,645).
- Constructs binary hurdle target: target_450 = I(post_minutes >= 450).
- Merges pre-transfer player features and destination-team context.
- Implements strict chronological splitting (60% Train, 20% Val, 20% Test).
- Validates that zero post-transfer or administrative leakage columns are used as predictors.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

# Directory setup
ROOT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT_DIR / "STAGE1_OUTPUTS"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Feature definitions (strictly available at transfer date t)
NUMERIC_FEATURES = [
    # Player Career & Physical Characteristics (Pre-transfer)
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
    "transfer_fee",
    # Destination Team Recent Form (Prior 10 matches before transfer date)
    "rich_team_matches_last10",
    "rich_team_goals_for_avg10",
    "rich_team_goals_against_avg10",
    "rich_team_goal_difference_avg10",
    "rich_team_win_rate10",
    "rich_team_points_per_game10",
    # Destination Squad Positional Competition & Congestion
    "same_position_competitor_count",
    "same_position_competitor_minutes",
    "same_position_avg_market_value",
    "same_position_top_competitor_minutes",
    "same_sub_position_competitor_count",
    "same_sub_position_competitor_minutes",
    "same_sub_position_top_minutes",
    "position_minutes_share",
    "previous_destination_appearances",
    "previous_destination_minutes",
    # Destination Manager / Tactical Context
    "formation_diversity10",
    "dominant_formation_frequency10",
    "manager_matches_last10",
    "unique_managers_last10",
]

CATEGORICAL_FEATURES = [
    "position",
    "sub_position",
    "foot",
]

ALL_PREDICTORS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FORBIDDEN_LEAKAGE_COLUMNS = {
    "post_minutes",
    "post_appearances",
    "post_goals",
    "post_assists",
    "post_GA90",
    "post_goals_per90",
    "post_assists_per90",
    "eligible_450min",
    "complete_365_day_window",
    "window_start",
    "window_end",
    "has_target",
    "clean_final_cohort",
}


def load_and_prepare_stage1_data():
    """
    Loads base transfer outcomes and rich team context, applies filters,
    validates chronological ordering, and splits into train, validation, and test.
    """
    base_path = ROOT_DIR / "post_transfer_performance.csv"
    rich_path = ROOT_DIR / "rich_team_context_dataset.csv"

    if not base_path.exists():
        raise FileNotFoundError(f"Missing {base_path}")
    if not rich_path.exists():
        raise FileNotFoundError(f"Missing {rich_path}")

    base_df = pd.read_csv(base_path, low_memory=False)
    rich_df = pd.read_csv(rich_path, low_memory=False)

    # 1. Filter for complete 365-day follow-up window
    complete_mask = base_df["complete_365_day_window"] == 1
    df = base_df[complete_mask].copy()

    # 2. Merge rich destination team context on transfer_row_id
    rich_cols = [c for c in rich_df.columns if c not in df.columns or c == "transfer_row_id"]
    df = df.merge(rich_df[rich_cols], on="transfer_row_id", how="left")

    # 3. Create target: post_minutes >= 450
    df["target_450"] = (df["post_minutes"] >= 450).astype(int)

    # 4. Parse transfer dates and sort chronologically
    df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")
    if df["transfer_date"].isna().any():
        raise ValueError("Missing transfer dates detected in complete window transfers.")
    
    df = df.sort_values(["transfer_date", "transfer_row_id"], kind="mergesort").reset_index(drop=True)

    # 5. Leakage check
    leakage = FORBIDDEN_LEAKAGE_COLUMNS.intersection(ALL_PREDICTORS)
    if leakage:
        raise ValueError(f"CRITICAL LEAKAGE DETECTED in predictors: {sorted(leakage)}")

    # 6. Chronological split (whole date cutoffs nearest 60% and 80%)
    n = len(df)
    daily_counts = df.groupby("transfer_date", sort=True).size()
    cum_counts = daily_counts.cumsum().to_numpy()
    date_values = daily_counts.index.to_numpy()

    train_candidates = np.arange(0, len(date_values) - 2)
    train_cut_idx = int(train_candidates[np.argmin(np.abs(cum_counts[train_candidates] - 0.60 * n))])
    val_candidates = np.arange(train_cut_idx + 1, len(date_values) - 1)
    val_cut_idx = int(val_candidates[np.argmin(np.abs(cum_counts[val_candidates] - 0.80 * n))])

    date_train_end = pd.Timestamp(date_values[train_cut_idx])
    date_val_end = pd.Timestamp(date_values[val_cut_idx])

    train_mask = df["transfer_date"] <= date_train_end
    val_mask = (df["transfer_date"] > date_train_end) & (df["transfer_date"] <= date_val_end)
    test_mask = df["transfer_date"] > date_val_end

    df["split"] = np.where(train_mask, "train", np.where(val_mask, "validation", "test"))

    split_summary = []
    for split_name, mask in [("train", train_mask), ("validation", val_mask), ("test", test_mask)]:
        sub = df[mask]
        split_summary.append({
            "split": split_name,
            "count": int(len(sub)),
            "share": float(len(sub) / n),
            "start_date": str(sub["transfer_date"].min().date()),
            "end_date": str(sub["transfer_date"].max().date()),
            "target_positive_count": int(sub["target_450"].sum()),
            "target_negative_count": int((sub["target_450"] == 0).sum()),
            "target_positive_rate": float(sub["target_450"].mean()),
            "unique_players": int(sub["player_id"].nunique()),
            "unique_dest_clubs": int(sub["to_club_id"].nunique()),
        })

    split_summary_df = pd.DataFrame(split_summary)
    split_summary_df.to_csv(OUTPUTS_DIR / "stage1_split_summary.csv", index=False)

    return df, split_summary_df, NUMERIC_FEATURES, CATEGORICAL_FEATURES


if __name__ == "__main__":
    df, summary, num_feats, cat_feats = load_and_prepare_stage1_data()
    print("Stage 1 Dataset Loaded Successfully!")
    print(f"Total transfers: {len(df):,}")
    print(f"Numeric features: {len(num_feats)}, Categorical features: {len(cat_feats)}")
    print("\nSplit Summary:")
    print(summary.to_string(index=False))

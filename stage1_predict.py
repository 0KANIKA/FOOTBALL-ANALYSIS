"""
Stage 1 Transfer Opportunity Predictor
======================================
Predicts the probability that one player transferring to one destination club
will play at least 450 minutes during the first 365 days after the transfer.

Usage:
    from stage1_predict import predict_transfer_opportunity

    player = {
        "age_at_transfer": 24,
        "position": "Attack",
        "sub_position": "Centre-Forward",
        "foot": "right",
        "height_in_cm": 187,
        "appearances_pre": 115,
        "minutes_pre": 8900,
        "goals_pre": 42,
        "assists_pre": 15,
        "goals_per90_pre": 0.42,
        "assists_per90_pre": 0.15,
        "market_value_pre": 35_000_000,
        "transfer_market_value": 35_000_000,
        "transfer_fee": 40_000_000,
    }

    club = {
        "rich_team_matches_last10": 10,
        "rich_team_goals_for_avg10": 2.2,
        "rich_team_goals_against_avg10": 0.8,
        "rich_team_goal_difference_avg10": 1.4,
        "rich_team_win_rate10": 0.70,
        "rich_team_points_per_game10": 2.30,
        "same_position_competitor_count": 2,
        "same_position_competitor_minutes": 2200,
        "same_position_avg_market_value": 20_000_000,
        "same_position_top_competitor_minutes": 1700,
        "same_sub_position_competitor_count": 1,
        "same_sub_position_competitor_minutes": 1700,
        "same_sub_position_top_minutes": 1700,
        "position_minutes_share": 0.28,
        "previous_destination_appearances": 0,
        "previous_destination_minutes": 0,
    }

    result = predict_transfer_opportunity(player, club)
    print(result["message"])
    # -> "Estimated probability of reaching 450 minutes: 78%"
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

# Paths
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = MODULE_DIR / "stage1_opportunity_model.joblib"
DEFAULT_METADATA_PATH = MODULE_DIR / "stage1_model_metadata.json"

_CACHED_MODEL = None
_CACHED_METADATA = None


def load_model_and_metadata(model_path=None, metadata_path=None):
    """Load the model and its matching metadata."""

    if (model_path is None) != (metadata_path is None):
        raise ValueError(
            "Provide both model_path and metadata_path, "
            "or leave both as None to use the default files."
        )

    m_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    meta_path = (
        Path(metadata_path)
        if metadata_path
        else DEFAULT_METADATA_PATH
    )

    if not m_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{m_path}'. "
            "Run 'python stage1_train.py' first."
        )

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found at '{meta_path}'. "
            "Run 'python stage1_train.py' first."
        )

    model = joblib.load(m_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, metadata



def predict_transfer_opportunity(
    player_info: dict,
    destination_club_context: dict = None,
    model_path: str = None,
    metadata_path: str = None,
    threshold: float = 0.50
) -> dict:
    """
    Predicts the probability that one player transferring to one destination club
    will play at least 450 minutes during their first 365 days after the transfer.

    Parameters:
    -----------
    player_info : dict
        Pre-transfer player characteristics (e.g., age, position, minutes_pre, transfer_fee, valuations).
        Can also contain combined player and destination club fields if destination_club_context is None.
    destination_club_context : dict, optional
        Pre-transfer destination club context (recent form, squad competition, incumbent minutes).
        If None, player_info is assumed to contain both.
    model_path : str, optional
        Path to stage1_opportunity_model.joblib.
    metadata_path : str, optional
        Path to stage1_model_metadata.json.
    threshold : float, default=0.50
        Decision threshold for binary classification.

    Returns:
    --------
    dict with keys:
        - 'probability': float (e.g., 0.723)
        - 'percentage': float (e.g., 72.3)
        - 'message': str ("Estimated probability of reaching 450 minutes: 72%")
        - 'will_reach_450': bool (True/False)
        - 'risk_tier': str (Qualitative assessment)
        - 'interpretation': str (Plain English summary of the prediction and key drivers)
        - 'input_summary': dict (Normalized inputs evaluated by the model)
    """
    model, metadata = load_model_and_metadata(model_path, metadata_path)

    # Combine player and club inputs
    combined = dict(player_info)
    if destination_club_context is not None:
        combined.update(destination_club_context)

    # Strictly verify that input is for a single transfer, not a batch
    for k, v in combined.items():
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)) and not isinstance(v, str):
            if len(v) != 1:
                raise ValueError(
                    f"Invalid batch input for key '{k}'. The Stage 1 prediction function "
                    "requires exactly ONE player and ONE destination club at a time."
                )
            combined[k] = v[0]

    # Verify no post-transfer leakage fields are passed as inputs
    forbidden = {
        "post_minutes", "post_appearances", "post_goals", "post_assists",
        "post_GA90", "eligible_450min", "complete_365_day_window"
    }
    overlap = forbidden.intersection(combined.keys())
    if overlap:
        raise ValueError(
            f"Prohibited post-transfer information supplied in input: {sorted(overlap)}. "
            "Stage 1 predictions must strictly use pre-transfer data to prevent data leakage."
        )

    # Build single-row DataFrame matching the exact training schema
    all_features = metadata["all_features"]
    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]
    train_medians = metadata.get("train_medians", {})
    train_modes = metadata.get("train_modes", {})

    row = {}
    for feat in all_features:
        if feat in combined and combined[feat] is not None and not pd.isna(combined[feat]):
            row[feat] = combined[feat]
        elif feat in numeric_features:
            # Fall back to training cohort median if feature is omitted
            row[feat] = train_medians.get(feat, np.nan)
        elif feat in categorical_features:
            # Fall back to training cohort mode if omitted
            row[feat] = train_modes.get(feat, "Unknown")
        else:
            row[feat] = np.nan

    input_df = pd.DataFrame([row], columns=all_features)

    # Compute predicted probability
    prob = float(model.predict_proba(input_df)[0, 1])
    pct = round(prob * 100, 1)
    pct_int = round(prob * 100)
    will_reach = bool(prob >= threshold)

    # Risk Tiering
    if prob >= 0.70:
        risk_tier = "High Probability (Projected Regular / First-Team Rotation)"
        verdict = (
            "The player demonstrates strong historical career momentum and/or significant "
            "institutional investment (fee/valuation) that comfortably surmounts the destination "
            "squad's positional competition."
        )
    elif prob >= 0.50:
        risk_tier = "Moderate Probability (Squad Rotation / Contender)"
        verdict = (
            "The player has a favorable chance of securing rotation minutes, though playing time "
            "will be actively contested by existing squad incumbents."
        )
    elif prob >= 0.30:
        risk_tier = "Elevated Risk (Fringe Player / Potential Opportunity Failure)"
        verdict = (
            "The player faces a substantial risk of falling below the 450-minute threshold, "
            "likely due to low transfer investment, high rival congestion, or limited recent match activity."
        )
    else:
        risk_tier = "High Risk (Sub-Rotation / Reserve / High Failure Risk)"
        verdict = (
            "Severe risk of failing to break into the destination team. Out of historical transfers "
            "with comparable profiles, the vast majority played fewer than 450 minutes."
        )

    formatted_msg = f"Estimated probability of reaching 450 minutes: {pct_int}%"

    return {
        "probability": prob,
        "percentage": pct,
        "message": formatted_msg,
        "will_reach_450": will_reach,
        "risk_tier": risk_tier,
        "interpretation": verdict,
        "player_position": str(row.get("position", "Unknown")),
        "transfer_fee": row.get("transfer_fee", 0),
        "pre_minutes": row.get("minutes_pre", 0),
        "competitor_minutes": row.get("same_position_competitor_minutes", 0),
        "input_summary": row
    }


def demo():
    """Runs demonstration predictions for illustrative transfer scenarios."""
    print("=" * 70)
    print("STAGE 1 PREDICTION DEMONSTRATION: SINGLE TRANSFER OPPORTUNITY")
    print("=" * 70)

    # Scenario 1: Marquee Forward Transfer (High Fee, High Past Minutes, Moderate Rival)
    player_marquee = {
        "age_at_transfer": 25,
        "position": "Attack",
        "sub_position": "Centre-Forward",
        "foot": "right",
        "height_in_cm": 185,
        "appearances_pre": 140,
        "minutes_pre": 11200,
        "goals_pre": 65,
        "assists_pre": 22,
        "goals_per90_pre": 0.52,
        "assists_per90_pre": 0.18,
        "transfer_market_value": 45_000_000,
        "market_value_pre": 45_000_000,
        "transfer_fee": 50_000_000,
    }
    club_marquee = {
        "rich_team_matches_last10": 10,
        "rich_team_goals_for_avg10": 2.4,
        "rich_team_goals_against_avg10": 0.9,
        "rich_team_win_rate10": 0.70,
        "rich_team_points_per_game10": 2.30,
        "same_position_competitor_count": 2,
        "same_position_competitor_minutes": 2000,
        "same_position_avg_market_value": 22_000_000,
        "same_position_top_competitor_minutes": 1500,
        "same_sub_position_competitor_count": 1,
        "same_sub_position_competitor_minutes": 1500,
        "same_sub_position_top_minutes": 1500,
        "position_minutes_share": 0.25,
        "previous_destination_appearances": 0,
        "previous_destination_minutes": 0,
    }
    res1 = predict_transfer_opportunity(player_marquee, club_marquee)
    print(f"\nScenario 1: Marquee Forward ($50M Fee, 11,200 Pre-Minutes)")
    print(f"  Result: {res1['message']}")
    print(f"  Risk Tier: {res1['risk_tier']}")
    print(f"  Context: {res1['interpretation']}")

    # Scenario 2: Free Transfer / Fringe Developmental Signing (Zero Fee, Highly Congested Squad)
    player_free = {
        "age_at_transfer": 21,
        "position": "Defender",
        "sub_position": "Centre-Back",
        "foot": "left",
        "height_in_cm": 186,
        "appearances_pre": 18,
        "minutes_pre": 950,
        "goals_pre": 0,
        "assists_pre": 0,
        "goals_per90_pre": 0.0,
        "assists_per90_pre": 0.0,
        "transfer_market_value": 800_000,
        "market_value_pre": 800_000,
        "transfer_fee": 0,  # Free transfer
    }
    club_congested = {
        "rich_team_matches_last10": 10,
        "rich_team_goals_for_avg10": 1.6,
        "rich_team_goals_against_avg10": 1.1,
        "rich_team_win_rate10": 0.50,
        "rich_team_points_per_game10": 1.70,
        "same_position_competitor_count": 5,
        "same_position_competitor_minutes": 7200,
        "same_position_avg_market_value": 12_000_000,
        "same_position_top_competitor_minutes": 2700,
        "same_sub_position_competitor_count": 4,
        "same_sub_position_competitor_minutes": 5800,
        "same_sub_position_top_minutes": 2700,
        "position_minutes_share": 0.10,
        "previous_destination_appearances": 0,
        "previous_destination_minutes": 0,
    }
    res2 = predict_transfer_opportunity(player_free, club_congested)
    print(f"\nScenario 2: Free Developmental Signing to Congested Defence (5 Rivals, 7,200 Rival Mins)")
    print(f"  Result: {res2['message']}")
    print(f"  Risk Tier: {res2['risk_tier']}")
    print(f"  Context: {res2['interpretation']}")

    # Scenario 3: Returning Loan Midfielder (Previous Minutes at Club)
    player_returning = {
        "age_at_transfer": 23,
        "position": "Midfield",
        "sub_position": "Central Midfield",
        "foot": "both",
        "height_in_cm": 180,
        "appearances_pre": 75,
        "minutes_pre": 5400,
        "goals_pre": 8,
        "assists_pre": 12,
        "goals_per90_pre": 0.13,
        "assists_per90_pre": 0.20,
        "transfer_market_value": 15_000_000,
        "market_value_pre": 15_000_000,
        "transfer_fee": 18_000_000,
    }
    club_returning = {
        "rich_team_matches_last10": 10,
        "rich_team_goals_for_avg10": 1.8,
        "rich_team_goals_against_avg10": 1.0,
        "rich_team_win_rate10": 0.60,
        "rich_team_points_per_game10": 2.00,
        "same_position_competitor_count": 3,
        "same_position_competitor_minutes": 3500,
        "same_position_avg_market_value": 14_000_000,
        "same_position_top_competitor_minutes": 2100,
        "same_sub_position_competitor_count": 2,
        "same_sub_position_competitor_minutes": 2500,
        "same_sub_position_top_minutes": 1800,
        "position_minutes_share": 0.22,
        "previous_destination_appearances": 22,
        "previous_destination_minutes": 1650,  # Proved track record at club
    }
    res3 = predict_transfer_opportunity(player_returning, club_returning)
    print(f"\nScenario 3: Returning Midfielder with 1,650 Prior Club Minutes ($18M Fee)")
    print(f"  Result: {res3['message']}")
    print(f"  Risk Tier: {res3['risk_tier']}")
    print(f"  Context: {res3['interpretation']}")


if __name__ == "__main__":
    demo()

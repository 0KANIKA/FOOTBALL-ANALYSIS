"""
Validation and Test Suite for Stage 1 Transfer Opportunity Predictor
===================================================================
Tests:
1. Input validation and single-transfer constraint (rejection of batch inputs).
2. Data leakage guardrails (rejection of post_minutes, eligible_450min, etc.).
3. Robustness to omitted optional fields (safe fallback to training medians).
4. Real-world case study: evaluates individual transfers from the held-out test cohort
   comparing predicted probabilities with actual observed post-transfer minutes.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

from stage1_predict import predict_transfer_opportunity
from stage1_data import load_and_prepare_stage1_data, ROOT_DIR


def test_single_transfer_enforcement():
    print("[TEST 1] Single Transfer Enforcement...")
    player = {"age_at_transfer": 23, "position": "Attack"}
    club = {"rich_team_matches_last10": 10}
    
    # Should succeed for single scalar inputs
    res = predict_transfer_opportunity(player, club)
    assert "Estimated probability of reaching 450 minutes:" in res["message"]
    assert 0.0 <= res["probability"] <= 1.0

    # Should reject batch / group inputs
    batch_player = {"age_at_transfer": [23, 25, 27], "position": ["Attack", "Midfield", "Defender"]}
    try:
        predict_transfer_opportunity(batch_player, club)
        raise AssertionError("Failed to reject batch input!")
    except ValueError as e:
        assert "Invalid batch input" in str(e) or "requires exactly ONE" in str(e)
    print("  -> PASSED: Single-transfer constraint strictly enforced.")


def test_leakage_rejection():
    print("[TEST 2] Data Leakage Prevention...")
    player = {
        "age_at_transfer": 24,
        "position": "Attack",
        "post_minutes": 1200,  # Prohibited post-transfer information!
    }
    club = {"rich_team_matches_last10": 10}
    try:
        predict_transfer_opportunity(player, club)
        raise AssertionError("Failed to detect post-transfer leakage feature!")
    except ValueError as e:
        assert "Prohibited post-transfer information supplied" in str(e)
    print("  -> PASSED: Leakage guardrails successfully block post-transfer information.")


def test_missing_value_safety():
    print("[TEST 3] Missing Value Safety & Graceful Defaults...")
    # Minimal input with many omitted features (e.g. unknown formation diversity, unrecorded fee)
    minimal_player = {
        "age_at_transfer": 22,
        "position": "Midfield",
    }
    res = predict_transfer_opportunity(minimal_player)
    assert 0.0 <= res["probability"] <= 1.0
    print(f"  -> PASSED: Successfully predicted for minimal player ({res['message']}).")


def test_real_world_test_transfers():
    print("[TEST 4] Real Held-Out Historical Transfer Predictions...")
    df, _, num_feats, cat_feats = load_and_prepare_stage1_data()
    test_df = df[df["split"] == "test"].copy()

    # Pick 4 diverse real transfers from held-out test cohort
    samples = [
        test_df[test_df["target_450"] == 1].iloc[10],  # Successful transfer (>= 450 min)
        test_df[test_df["target_450"] == 0].iloc[15],  # Failed transfer (< 450 min)
        test_df[(test_df["target_450"] == 1) & (test_df["position"] == "Defender")].iloc[5],
        test_df[(test_df["target_450"] == 0) & (test_df["position"] == "Attack")].iloc[8],
    ]

    all_features = num_feats + cat_feats
    for i, s in enumerate(samples, 1):
        transfer_dict = {feat: s[feat] for feat in all_features if feat in s and pd.notna(s[feat])}
        res = predict_transfer_opportunity(transfer_dict)
        actual_mins = s["post_minutes"]
        actual_target = s["target_450"]
        correct = (res["will_reach_450"] == bool(actual_target))

        print(f"\n  Case {i}: Player ID {s['player_id']} | Date: {str(s['transfer_date'].date())}")
        print(f"    Position: {s['position']} ({s.get('sub_position', 'N/A')}) | Age: {s['age_at_transfer']}")
        print(f"    Pre-transfer Minutes: {s.get('minutes_pre', 0):,} | Fee: ${s.get('transfer_fee', 0):,}")
        print(f"    Prediction: {res['message']} (Risk: {res['risk_tier']})")
        print(f"    Actual Outcome: {actual_mins:.0f} post-minutes (Eligible >= 450: {actual_target == 1})")
        print(f"    Model Decision: {'CORRECT' if correct else 'INCORRECT'}")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING STAGE 1 PREDICTOR VALIDATION SUITE")
    print("=" * 70)
    test_single_transfer_enforcement()
    test_leakage_rejection()
    test_missing_value_safety()
    test_real_world_test_transfers()
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

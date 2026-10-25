"""
Stage 1 Model Training & Evaluation Pipeline
============================================
Trains, tunes, and comprehensively evaluates classification models for predicting
P(post_minutes >= 450) across professional football transfers.

Requirements Addressed:
1. Complete 365-day follow-up window (N = 13,645).
2. Binary target: post_minutes >= 450.
3. Pre-transfer features only (player characteristics and destination squad context).
4. Zero post-transfer leakage features.
5. Strict chronological split (Train: <= 2022-07-08, Val: 2022-07-09 to 2024-01-26, Test: 2024-01-27 to 2025-06-27).
6. Comprehensive evaluation: ROC-AUC, PR-AUC, Brier score, Precision, Recall, Confusion Matrix.
   Compared against simple baselines (Dummy prior, Logistic Regression).
7. Class balance checks and safe missing value handling (median imputation with missing indicators).
8. Model artifact persistence with joblib and schema metadata.
"""

from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.inspection import permutation_importance

from stage1_data import load_and_prepare_stage1_data, ROOT_DIR, OUTPUTS_DIR


def evaluate_predictions(y_true, probs, threshold=0.5):
    """Computes comprehensive binary classification metrics."""
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    
    return {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "brier_score": float(brier_score_loss(y_true, probs)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
    }


def train_and_evaluate_stage1():
    print("=" * 70)
    print("STAGE 1 OPPORTUNITY MODEL: P(post_minutes >= 450) TRAINING & EVALUATION")
    print("=" * 70)

    # 1. Load data
    df, split_summary, num_feats, cat_feats = load_and_prepare_stage1_data()
    all_features = num_feats + cat_feats

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "validation"
    test_mask = df["split"] == "test"

    X_train = df.loc[train_mask, all_features].copy()
    y_train = df.loc[train_mask, "target_450"].to_numpy()

    X_val = df.loc[val_mask, all_features].copy()
    y_val = df.loc[val_mask, "target_450"].to_numpy()

    X_test = df.loc[test_mask, all_features].copy()
    y_test = df.loc[test_mask, "target_450"].to_numpy()

    print(f"\nCohort Sizes:")
    print(f"  Training cohort:   {len(X_train):,} transfers (Target rate: {y_train.mean():.2%})")
    print(f"  Validation cohort: {len(X_val):,} transfers (Target rate: {y_val.mean():.2%})")
    print(f"  Test cohort:       {len(X_test):,} transfers (Target rate: {y_test.mean():.2%})")

    # 2. Build Preprocessor (strictly fit on X_train only)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median", add_indicator=True), num_feats),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False))
            ]), cat_feats),
        ],
        remainder="drop"
    )

    # 3. Model candidate definitions
    models = {
        "Dummy Prior Baseline": DummyClassifier(strategy="prior"),
        "Standardized Logistic Regression (L2)": Pipeline([
            ("prep", preprocessor),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0))
        ]),
        "Random Forest Classifier (Primary)": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "HistGradientBoosting Classifier": Pipeline([
            ("prep", preprocessor),
            ("clf", HistGradientBoostingClassifier(
                max_iter=120,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                learning_rate=0.08,
                random_state=42
            ))
        ]),
    }

    # 4. Train and benchmark models
    comparison_records = []
    trained_pipelines = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        trained_pipelines[name] = model

        # Val evaluation
        val_probs = model.predict_proba(X_val)[:, 1]
        val_metrics = evaluate_predictions(y_val, val_probs)

        # Test evaluation
        test_probs = model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_predictions(y_test, test_probs)

        print(f"  Done in {elapsed:.1f}s | Test ROC-AUC: {test_metrics['roc_auc']:.4f} | PR-AUC: {test_metrics['pr_auc']:.4f} | Brier: {test_metrics['brier_score']:.4f}")

        comparison_records.append({
            "model": name,
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_brier": val_metrics["brier_score"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_brier": test_metrics["brier_score"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_specificity": test_metrics["specificity"],
            "test_tn": test_metrics["true_negatives"],
            "test_fp": test_metrics["false_positives"],
            "test_fn": test_metrics["false_negatives"],
            "test_tp": test_metrics["true_positives"],
            "fit_time_sec": round(elapsed, 2),
        })

    comparison_df = pd.DataFrame(comparison_records)
    comparison_df.to_csv(OUTPUTS_DIR / "stage1_model_comparison.csv", index=False)
    print("\nModel Comparison Table:")
    print(comparison_df[["model", "test_roc_auc", "test_pr_auc", "test_brier", "test_precision", "test_recall"]].to_string(index=False))

    # 5. Select primary production model: Random Forest
    primary_name = "Random Forest Classifier (Primary)"
    primary_pipeline = trained_pipelines[primary_name]

    # Save model artifact
    model_artifact_path = ROOT_DIR / "stage1_opportunity_model.joblib"
    joblib.dump(primary_pipeline, model_artifact_path)
    print(f"\nSaved production model artifact to: {model_artifact_path}")

    # 6. Extract Feature Importances
    clf = primary_pipeline.named_steps["clf"]
    prep = primary_pipeline.named_steps["prep"]
    num_trans = prep.named_transformers_["num"]
    cat_trans = prep.named_transformers_["cat"].named_steps["onehot"]

    feature_names = list(num_feats)
    if hasattr(num_trans, "indicator_") and num_trans.indicator_ is not None:
        feature_names.extend([f"{num_feats[i]}_missing" for i in num_trans.indicator_.features_])
    feature_names.extend(cat_trans.get_feature_names_out(cat_feats))

    importances_series = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
    imp_df = pd.DataFrame({
        "feature": importances_series.index,
        "importance": importances_series.values,
        "cumulative_importance": importances_series.cumsum().values
    })
    imp_df.to_csv(OUTPUTS_DIR / "stage1_feature_importances.csv", index=False)
    print("\nTop 15 Most Important Pre-Transfer Features:")
    print(imp_df.head(15).to_string(index=False))

    # 7. Disaggregated Analysis on Test Set
    test_df = df[test_mask].copy()
    test_probs = primary_pipeline.predict_proba(X_test)[:, 1]
    test_df["predicted_prob"] = test_probs
    test_df["predicted_class"] = (test_probs >= 0.5).astype(int)

    disagg_records = []

    # A. By Position
    for pos in ["Attack", "Midfield", "Defender", "Goalkeeper"]:
        sub = test_df[test_df["position"] == pos]
        metrics = evaluate_predictions(sub["target_450"].to_numpy(), sub["predicted_prob"].to_numpy())
        metrics["dimension"] = "position"
        metrics["group"] = pos
        disagg_records.append(metrics)

    # B. By Seen vs Unseen Players (Multi-transfer players check)
    train_player_ids = set(df.loc[train_mask, "player_id"])
    test_df["seen_in_train"] = test_df["player_id"].isin(train_player_ids)
    for seen_val, label in [(True, "Player transferred previously (Seen in Train)"),
                            (False, "First transfer in dataset (Unseen in Train)")]:
        sub = test_df[test_df["seen_in_train"] == seen_val]
        metrics = evaluate_predictions(sub["target_450"].to_numpy(), sub["predicted_prob"].to_numpy())
        metrics["dimension"] = "player_temporal_status"
        metrics["group"] = label
        disagg_records.append(metrics)

    # C. By Transfer Fee Tier
    def fee_tier(val):
        if pd.isna(val) or val <= 0:
            return "Free Transfer / Loan ($0)"
        elif val < 2_000_000:
            return "Low Fee (< $2M)"
        elif val <= 10_000_000:
            return "Mid Fee ($2M - $10M)"
        else:
            return "High Fee (> $10M)"

    test_df["fee_tier"] = test_df["transfer_fee"].apply(fee_tier)
    for tier in ["Free Transfer / Loan ($0)", "Low Fee (< $2M)", "Mid Fee ($2M - $10M)", "High Fee (> $10M)"]:
        sub = test_df[test_df["fee_tier"] == tier]
        if len(sub) > 0:
            metrics = evaluate_predictions(sub["target_450"].to_numpy(), sub["predicted_prob"].to_numpy())
            metrics["dimension"] = "transfer_fee_tier"
            metrics["group"] = tier
            disagg_records.append(metrics)

    disagg_df = pd.DataFrame(disagg_records)
    disagg_df.to_csv(OUTPUTS_DIR / "stage1_disaggregated_evaluation.csv", index=False)
    print("\nDisaggregated Test Set Results:")
    print(disagg_df[["dimension", "group", "n", "positive_rate", "roc_auc", "pr_auc", "brier_score", "precision", "recall"]].to_string(index=False))

    # 8. Confusion Matrix Export
    cm = confusion_matrix(y_test, test_df["predicted_class"])
    cm_df = pd.DataFrame(cm, index=["Actual < 450 min", "Actual >= 450 min"],
                         columns=["Predicted < 450 min", "Predicted >= 450 min"])
    cm_df.to_csv(OUTPUTS_DIR / "stage1_test_confusion_matrix.csv")

    # 9. Extract Reference Medians & Modes for Single Prediction Preprocessing
    # Allows prediction function to safely populate any omitted optional fields using training medians
    train_medians = X_train[num_feats].median().to_dict()
    train_modes = {col: str(X_train[col].mode().iloc[0]) for col in cat_feats}

    metadata = {
        "model_name": "Stage 1 Hurdle Classifier (Transfer Opportunity)",
        "algorithm": "RandomForestClassifier",
        "target": "post_minutes >= 450 in first 365 days after transfer",
        "training_n": int(len(X_train)),
        "validation_n": int(len(X_val)),
        "test_n": int(len(X_test)),
        "date_cutoffs": {
            "train_end": "2022-07-08",
            "val_end": "2024-01-26",
            "test_end": "2025-06-27"
        },
        "numeric_features": num_feats,
        "categorical_features": cat_feats,
        "all_features": all_features,
        "train_medians": train_medians,
        "train_modes": train_modes,
        "overall_test_metrics": {
            "roc_auc": float(roc_auc_score(y_test, test_probs)),
            "pr_auc": float(average_precision_score(y_test, test_probs)),
            "brier_score": float(brier_score_loss(y_test, test_probs)),
            "precision": float(precision_score(y_test, test_df["predicted_class"])),
            "recall": float(recall_score(y_test, test_df["predicted_class"])),
            "f1": float(f1_score(y_test, test_df["predicted_class"])),
            "accuracy": float((test_df["predicted_class"] == y_test).mean())
        }
    }

    metadata_path = ROOT_DIR / "stage1_model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to: {metadata_path}")

    return metadata


if __name__ == "__main__":
    train_and_evaluate_stage1()

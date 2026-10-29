
import os
import joblib
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "06_Results_and_Demo_Evidence", "research_results", "model_a_new_results")

model_path = os.path.join(
    RESULTS_DIR, "random_forest_pipeline.joblib"
)

# Load the saved dictionary and extract the pipeline
saved = joblib.load(model_path)
pipeline = saved["pipeline"]
features = saved["features"]

# Get the trained Random Forest
rf = pipeline.named_steps["model"]

# Obtain feature names after preprocessing
preprocessor = pipeline.named_steps["preprocessing"]

try:
    transformed_names = preprocessor.get_feature_names_out()
    importances = rf.feature_importances_

    if len(transformed_names) == len(importances):
        feature_names = transformed_names
    else:
        feature_names = [f"feature_{i}" for i in range(len(importances))]
except Exception:
    feature_names = [f"feature_{i}" for i in range(len(rf.feature_importances_))]
    importances = rf.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

output_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
importance_df.to_csv(output_path, index=False)

print("\nTop 15 important features:")
print(importance_df.head(15).to_string(index=False))
print(f"\nSaved feature importance to: {output_path}")


import os
import joblib
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "06_Results_and_Demo_Evidence", "research_results", "model_a_new_results",
    "random_forest_pipeline.joblib"
)

saved = joblib.load(MODEL_PATH)
pipeline = saved["pipeline"]
features = saved["features"]

# Example input — replace these values with real player data.
player = {
    "age_at_transfer": 24,
    "position": "Attack",
    "sub_position": "Centre-Forward",
    "height_in_cm": 194,
    "foot": "left",
    "appearances_pre": 100,
    "minutes_pre": 7500,
    "goals_pre": 60,
    "assists_pre": 15,
    "goals_per90_pre": 0.72,
    "assists_per90_pre": 0.18,
    "yellow_cards_pre": 10,
    "red_cards_pre": 0,
    "transfer_market_value": 150000000,
    "market_value_pre": 150000000,
    "transfer_fee": 75000000
}

X_new = pd.DataFrame([player], columns=features)
prediction = pipeline.predict(X_new)[0]

print("\nPLAYER PERFORMANCE PREDICTION")
print("-----------------------------")
print(f"Predicted post-transfer G+A per 90 minutes: {prediction:.3f}")
print("This is a model estimate, not a guaranteed outcome.")

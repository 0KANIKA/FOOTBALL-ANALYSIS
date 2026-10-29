
import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "06_Results_and_Demo_Evidence", "research_results", "model_a_new_results")

csv_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
output_path = os.path.join(RESULTS_DIR, "feature_importance.png")

df = pd.read_csv(csv_path)
df = df.head(15).sort_values("importance", ascending=True)

# Make feature names easier to read
df["feature"] = (
    df["feature"]
    .str.replace("numeric__", "", regex=False)
    .str.replace("categorical__", "", regex=False)
    .str.replace("_", " ", regex=False)
)

plt.figure(figsize=(11, 7))
plt.barh(df["feature"], df["importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features — Random Forest Model")
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Chart saved to: {output_path}")

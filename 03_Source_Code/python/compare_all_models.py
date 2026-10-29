
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DATA_DIR = PROJECT_ROOT / "04_Datasets" / "derived"

print("=" * 75)
print("FOOTBALL TRANSFER MODEL COMPARISON AUDIT")
print("=" * 75)
print("Project folder:", ROOT)

# --------------------------------------------------
# 1. Original Model A / B / C results
# --------------------------------------------------
original = pd.DataFrame([
    ["Model A", 1194, 0.123159, 0.172344, 0.551043],
    ["Model B", 1194, 0.121047, 0.169027, 0.568156],
    ["Model C", 1194, 0.121325, 0.169764, 0.564382],
], columns=["Model", "Test_N", "MAE", "RMSE", "R2"])

original["MAE_improvement_vs_A_pct"] = (
    (original.loc[0, "MAE"] - original["MAE"])
    / original.loc[0, "MAE"] * 100
)

print("\n[1] ORIGINAL MODEL A / B / C")
print(original.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# --------------------------------------------------
# 2. Corrected audit results (separate experiment)
# --------------------------------------------------
corrected = pd.DataFrame([
    ["Core", 1223, 0.120966, 0.170774, 0.554984],
    ["Mode A", 1223, 0.119102, 0.167612, 0.571309],
    ["Full enhanced", 1223, 0.118346, 0.166695, 0.575988],
    ["Without tactical/manager", 1223, 0.118345, 0.166290, 0.578044],
    ["Without squad competition", 1223, 0.118494, 0.166298, 0.578005],
    ["Without coach", 1223, 0.119102, 0.167612, 0.571309],
], columns=["Variant", "Test_N", "MAE", "RMSE", "R2"])

corrected["MAE_improvement_vs_core_pct"] = (
    (corrected.loc[0, "MAE"] - corrected["MAE"])
    / corrected.loc[0, "MAE"] * 100
)

print("\n[2] CORRECTED AUDIT (SEPARATE TEST SET)")
print(corrected.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# --------------------------------------------------
# 3. Inspect the actual project files
# --------------------------------------------------
print("\n[3] DATASET ID ALIGNMENT")

files = {
    "main": DATA_DIR / "post_transfer_performance.csv",
    "team": DATA_DIR / "rich_team_context_dataset.csv",
    "coach": DATA_DIR / "coach_tactical_context_v21.csv",
}

datasets = {}

for name, path in files.items():
    if not path.exists():
        print(f"{name}: FILE NOT FOUND - {path.name}")
        continue

    cols = ["transfer_row_id"]
    if name != "coach":
        cols += [
            "player_id", "transfer_date",
            "from_club_id", "to_club_id"
        ]

    df = pd.read_csv(path, usecols=cols, low_memory=False)
    datasets[name] = df

    print(
        f"{name}: rows={len(df)}, "
        f"unique transfer IDs={df['transfer_row_id'].nunique()}, "
        f"duplicate IDs={df['transfer_row_id'].duplicated().sum()}"
    )

if all(k in datasets for k in ["main", "team", "coach"]):
    main_ids = set(datasets["main"]["transfer_row_id"])
    team_ids = set(datasets["team"]["transfer_row_id"])
    coach_ids = set(datasets["coach"]["transfer_row_id"])

    print("Main IDs missing from team:", len(main_ids - team_ids))
    print("Team IDs missing from main:", len(team_ids - main_ids))
    print("Main IDs missing from coach:", len(main_ids - coach_ids))
    print("Coach IDs missing from main:", len(coach_ids - main_ids))

    identity = [
        "transfer_row_id", "player_id", "transfer_date",
        "from_club_id", "to_club_id"
    ]
    matched = datasets["main"].merge(
        datasets["team"],
        on=identity,
        how="inner"
    )
    print("Exact main/team identity matches:", len(matched))

# --------------------------------------------------
# 4. Inspect feature definitions in the audit script
# --------------------------------------------------
print("\n[4] CORRECTED AUDIT FEATURE GROUPS")
audit_script = ROOT / "run_audit_corrected.py"

if audit_script.exists():
    lines = audit_script.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()

    terms = [
        "CORE_NUM", "CORE_CAT", "TEAM",
        "TACTIC_NUM", "TACTIC_CAT", "SQUAD", "COACH",
        "mode_A", "full_enhanced", "without_coach"
    ]

    for i, line in enumerate(lines):
        if any(term in line for term in terms):
            print(f"{i + 1}: {line.strip()}")
else:
    print("run_audit_corrected.py not found.")

print("\n[5] AGREED PROJECT STRUCTURE")
print("Model A         = player data only")
print("Model B         = player data + destination-team context")
print("Mode A          = player + team + tactics/manager + squad competition")
print("Extended Mode A = Mode A + additional validated pre-transfer features")
print("Model C         = richer coach-related extension in original ABC experiment")

# --------------------------------------------------
# 5. Save comparison charts and CSVs
# --------------------------------------------------
out = ROOT / "model_comparison_report"
out.mkdir(exist_ok=True)

original.to_csv(out / "original_ABC_metrics.csv", index=False)
corrected.to_csv(out / "corrected_audit_metrics.csv", index=False)

plt.figure(figsize=(8, 5))
plt.bar(original["Model"], original["MAE"])
plt.ylabel("MAE (lower is better)")
plt.title("Original Model A / B / C: Test MAE")
for i, value in enumerate(original["MAE"]):
    plt.text(i, value, f"{value:.5f}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig(out / "original_ABC_MAE.png", dpi=200)
plt.close()

plt.figure(figsize=(9, 6))
ordered = corrected.sort_values("MAE")
plt.barh(ordered["Variant"], ordered["MAE"])
plt.xlabel("MAE (lower is better)")
plt.title("Corrected Audit: Test MAE (separate experiment)")
plt.tight_layout()
plt.savefig(out / "corrected_audit_MAE.png", dpi=200)
plt.close()

print("\n[6] OUTPUT FILES")
print("Folder:", out)
for path in sorted(out.iterdir()):
    print(" -", path.name)

print("\nAUDIT FINISHED")
print("Note: this script reports existing metrics; it does not retrain models.")
print("Original ABC and corrected-audit scores use different test cohorts.")

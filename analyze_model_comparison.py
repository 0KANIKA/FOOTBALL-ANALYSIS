import pandas as pd

# ============================================================
# LOCKED TEST RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": [
        "A - Player Only",
        "B - Player + Team Performance",
        "C - Player + Rich Team Context"
    ],
    "MAE": [
        0.122038,
        0.120192,
        0.119643
    ],
    "RMSE": [
        0.171886,
        0.168063,
        0.168064
    ],
    "R2": [
        0.549471,
        0.569293,
        0.569287
    ]
})


A = results.iloc[0]
B = results.iloc[1]
C = results.iloc[2]


def pct_change(old, new):
    return ((new - old) / old) * 100


print("=" * 90)
print("LOCKED MODEL A / B / C TEST COMPARISON")
print("=" * 90)

print("\n")
print(results.to_string(index=False))


# ============================================================
# A -> B
# ============================================================

print("\n" + "=" * 90)
print("A -> B : DESTINATION-TEAM PERFORMANCE CONTRIBUTION")
print("=" * 90)

print(
    f"MAE change  : {B['MAE'] - A['MAE']:.6f} "
    f"({pct_change(A['MAE'], B['MAE']):.3f}%)"
)

print(
    f"RMSE change : {B['RMSE'] - A['RMSE']:.6f} "
    f"({pct_change(A['RMSE'], B['RMSE']):.3f}%)"
)

print(
    f"R² change   : {B['R2'] - A['R2']:.6f}"
)

print(
    f"R² relative : "
    f"{pct_change(A['R2'], B['R2']):.3f}%"
)


# ============================================================
# B -> C
# ============================================================

print("\n" + "=" * 90)
print("B -> C : RICH CONTEXT CONTRIBUTION")
print("=" * 90)

print(
    f"MAE change  : {C['MAE'] - B['MAE']:.6f} "
    f"({pct_change(B['MAE'], C['MAE']):.3f}%)"
)

print(
    f"RMSE change : {C['RMSE'] - B['RMSE']:.6f} "
    f"({pct_change(B['RMSE'], C['RMSE']):.3f}%)"
)

print(
    f"R² change   : {C['R2'] - B['R2']:.6f}"
)

print(
    f"R² relative : "
    f"{pct_change(B['R2'], C['R2']):.3f}%"
)


# ============================================================
# A -> C
# ============================================================

print("\n" + "=" * 90)
print("A -> C : TOTAL CONTEXT CONTRIBUTION")
print("=" * 90)

print(
    f"MAE change  : {C['MAE'] - A['MAE']:.6f} "
    f"({pct_change(A['MAE'], C['MAE']):.3f}%)"
)

print(
    f"RMSE change : {C['RMSE'] - A['RMSE']:.6f} "
    f"({pct_change(A['RMSE'], C['RMSE']):.3f}%)"
)

print(
    f"R² change   : {C['R2'] - A['R2']:.6f}"
)

print(
    f"R² relative : "
    f"{pct_change(A['R2'], C['R2']):.3f}%"
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    "model_abc_locked_test_comparison.csv",
    index=False
)

print("\nSaved:")
print("model_abc_locked_test_comparison.csv")

print("\nIMPORTANT:")
print("These values are the locked test results.")
print("Do not use the test set for further model tuning.")
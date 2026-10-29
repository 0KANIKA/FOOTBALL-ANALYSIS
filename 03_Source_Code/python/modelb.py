# Model B predictions
pred_B = model_B.predict(X_test_B)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, pred_B, alpha=0.45, s=20)

min_val = min(y_test.min(), pred_B.min())
max_val = max(y_test.max(), pred_B.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.xlabel("Actual Post-Transfer G+A/90")
plt.ylabel("Predicted Post-Transfer G+A/90")
plt.title("Model B — Actual vs Predicted")

plt.tight_layout()
plt.savefig(
    "ML_OUTPUT/model_B_actual_vs_predicted.png",
    dpi=300
)

plt.show()
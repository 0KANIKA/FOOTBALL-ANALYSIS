# ============================================================
# ACTUAL vs PREDICTED - MODEL A
# ============================================================

pred_A <- predict_model(
  model_A,
  test_df
)

plot_A <- data.frame(
  Actual = test_df$post_GA90,
  Predicted = pred_A
)

ggplot(
  plot_A,
  aes(
    x = Actual,
    y = Predicted
  )
) +
  geom_point(
    alpha = 0.45,
    size = 2
  ) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title = "Model A: Actual vs Predicted Post-Transfer G+A/90",
    subtitle = "Player characteristics only",
    x = "Actual Post-Transfer G+A/90",
    y = "Predicted Post-Transfer G+A/90"
  ) +
  theme_minimal()

ggsave(
  file.path(
    OUTPUT_DIR,
    "Model_A_Actual_vs_Predicted.png"
  ),
  width = 8,
  height = 6,
  dpi = 300
)
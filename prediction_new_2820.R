# ============================================================
# FOOTBALL TRANSFER PERFORMANCE PREDICTION
# ============================================================
#
# Research:
# Predicting Post-Transfer Football Player Performance
# Through Player-Team Contextual Compatibility Using ML
#
# Model A = Player characteristics only
# Model B = Player + destination-team context
# Model C = Player + richer destination context
#
# Target:
# post-transfer G+A/90 over the first 365 days
#
# IMPORTANT:
# The model is GENERAL.
# Cunha is ONLY a demonstration input.
#
# ============================================================


# ============================================================
# 1. PACKAGES
# ============================================================

required_packages <- c(
  "dplyr",
  "tidyr",
  "readr",
  "ranger",
  "ggplot2"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(dplyr)
library(tidyr)
library(readr)
library(ranger)
library(ggplot2)


# ============================================================
# 2. PATHS
# ============================================================

DATA_PATH <- "C:/Users/rsuba/Downloads/R football analytics/FINAL_MODEL_C_CLEAN.csv"

OUTPUT_DIR <- "C:/Users/rsuba/Downloads/R football analytics/ML_OUTPUT"

if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}


# ============================================================
# 3. LOAD DATA
# ============================================================

cat("\n========================================\n")
cat("LOADING DATA\n")
cat("========================================\n")

df <- read_csv(
  DATA_PATH,
  show_col_types = FALSE
)

cat("Rows:", nrow(df), "\n")
cat("Columns:", ncol(df), "\n")


# ============================================================
# 4. BASIC CLEANING
# ============================================================

df <- df %>%
  mutate(
    transfer_date = as.Date(transfer_date),
    post_GA90 = as.numeric(post_GA90)
  ) %>%
  filter(
    !is.na(post_GA90),
    !is.na(transfer_date)
  )


# ============================================================
# 5. DEFINE FEATURES
# ============================================================

# ----------------------------
# MODEL A
# Player characteristics
# ----------------------------

features_A <- c(
  "age_at_transfer",
  "height_in_cm",
  "appearances_pre",
  "minutes_pre",
  "goals_pre",
  "assists_pre",
  "goals_per90_pre",
  "assists_per90_pre",
  "yellow_cards_pre",
  "red_cards_pre",
  "transfer_market_value",
  "market_value_pre",
  "transfer_fee",
  "position",
  "sub_position",
  "foot"
)


# ----------------------------
# MODEL B
# Player + basic team context
# ----------------------------

features_B <- c(
  features_A,

  "rich_team_goals_for_avg10",
  "rich_team_goals_against_avg10",
  "rich_team_goal_difference_avg10",
  "rich_team_win_rate10",
  "rich_team_points_per_game10",
  "rich_team_matches_last10"
)


# ----------------------------
# MODEL C
# Richer contextual model
# ----------------------------

features_C <- c(
  features_B,

  "dominant_formation10",
  "dominant_formation_frequency10",
  "formation_diversity10",
  "manager_matches_last10",
  "unique_managers_last10",
  "same_position_competitor_count",
  "same_position_competitor_minutes",
  "same_position_avg_market_value",
  "same_position_top_competitor_minutes",
  "same_sub_position_competitor_count",
  "same_sub_position_competitor_minutes",
  "same_sub_position_top_minutes",
  "position_minutes_share",
  "previous_destination_appearances",
  "previous_destination_minutes",
  "coach_position_group_minutes_share",
  "coach_position_group_start_rate",
  "coach_exact_position_minutes_share",
  "coach_exact_position_start_rate",
  "coach_position_competitor_count",
  "coach_position_competitor_minutes",
  "coach_position_rotation_hhi",
  "coach_formation_stability",
  "coach_observed_tenure_days"
)


# ============================================================
# 6. VERIFY FEATURES
# ============================================================

cat("\n========================================\n")
cat("FEATURE CHECK\n")
cat("========================================\n")

missing_A <- setdiff(features_A, names(df))
missing_B <- setdiff(features_B, names(df))
missing_C <- setdiff(features_C, names(df))

if (length(missing_A) > 0) {
  stop(
    paste(
      "Model A missing columns:",
      paste(missing_A, collapse = ", ")
    )
  )
}

if (length(missing_B) > 0) {
  stop(
    paste(
      "Model B missing columns:",
      paste(missing_B, collapse = ", ")
    )
  )
}

if (length(missing_C) > 0) {
  stop(
    paste(
      "Model C missing columns:",
      paste(missing_C, collapse = ", ")
    )
  )
}

cat("Model A features:", length(features_A), "\n")
cat("Model B features:", length(features_B), "\n")
cat("Model C features:", length(features_C), "\n")


# ============================================================
# 7. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

cat("\n========================================\n")
cat("CHRONOLOGICAL SPLIT\n")
cat("========================================\n")

df <- df %>%
  arrange(transfer_date)

n <- nrow(df)

train_end <- floor(0.60 * n)
val_end <- floor(0.80 * n)

train_df <- df[1:train_end, ]
val_df <- df[(train_end + 1):val_end, ]
test_df <- df[(val_end + 1):n, ]

cat("Train:", nrow(train_df), "\n")
cat("Validation:", nrow(val_df), "\n")
cat("Test:", nrow(test_df), "\n")

cat("\nTrain dates:\n")
cat(min(train_df$transfer_date), "->", max(train_df$transfer_date), "\n")

cat("\nValidation dates:\n")
cat(min(val_df$transfer_date), "->", max(val_df$transfer_date), "\n")

cat("\nTest dates:\n")
cat(min(test_df$transfer_date), "->", max(test_df$transfer_date), "\n")


# ============================================================
# 8. MEDIAN IMPUTATION
# ============================================================

# Imputation values are calculated ONLY from training data.
# This avoids using validation/test information during training.

prepare_features <- function(train_data, new_data, feature_names) {

  result_train <- train_data[, feature_names, drop = FALSE]
  result_new <- new_data[, feature_names, drop = FALSE]

  numeric_cols <- names(result_train)[
    sapply(result_train, is.numeric)
  ]

  categorical_cols <- names(result_train)[
    sapply(result_train, function(x) is.character(x) ||
             is.factor(x))
  ]

  # ----------------------------
  # Numeric imputation
  # ----------------------------

  medians <- list()

  for (col in numeric_cols) {

    med <- median(
      result_train[[col]],
      na.rm = TRUE
    )

    if (is.na(med)) {
      med <- 0
    }

    medians[[col]] <- med

    result_train[[col]][
      is.na(result_train[[col]])
    ] <- med

    result_new[[col]][
      is.na(result_new[[col]])
    ] <- med
  }


  # ----------------------------
  # Categorical imputation
  # ----------------------------

  categorical_values <- list()

  for (col in categorical_cols) {

    train_values <- as.character(result_train[[col]])

    mode_value <- names(
      sort(
        table(train_values),
        decreasing = TRUE
      )
    )[1]

    if (is.na(mode_value) || length(mode_value) == 0) {
      mode_value <- "Unknown"
    }

    categorical_values[[col]] <- mode_value

    result_train[[col]] <- as.character(
      result_train[[col]]
    )

    result_new[[col]] <- as.character(
      result_new[[col]]
    )

    result_train[[col]][
      is.na(result_train[[col]]) |
        result_train[[col]] == ""
    ] <- mode_value

    result_new[[col]][
      is.na(result_new[[col]]) |
        result_new[[col]] == ""
    ] <- mode_value
  }


  # ----------------------------
  # Factor levels
  # ----------------------------

  for (col in categorical_cols) {

    all_levels <- unique(
      c(
        as.character(result_train[[col]]),
        as.character(result_new[[col]])
      )
    )

    result_train[[col]] <- factor(
      result_train[[col]],
      levels = all_levels
    )

    result_new[[col]] <- factor(
      result_new[[col]],
      levels = all_levels
    )
  }


  list(
    train = result_train,
    new = result_new,
    medians = medians,
    categorical_values = categorical_values
  )
}


# ============================================================
# 9. MODEL TRAINING FUNCTION
# ============================================================

train_rf_model <- function(
    train_data,
    feature_names,
    target = "post_GA90"
) {

  prepared <- prepare_features(
    train_data,
    train_data,
    feature_names
  )

  X <- prepared$train
  y <- train_data[[target]]

  # sqrt(number of predictors)
  mtry_value <- max(
    1,
    floor(sqrt(length(feature_names)))
  )

  model <- ranger(
    dependent.variable.name = target,
    data = cbind(
      X,
      post_GA90 = y
    ),
    num.trees = 300,
    mtry = mtry_value,
    min.node.size = 10,
    max.depth = 12,
    importance = "permutation",
    seed = 42
  )

  list(
    model = model,
    feature_names = feature_names,
    medians = prepared$medians,
    categorical_values = prepared$categorical_values
  )
}


# ============================================================
# 10. PREDICTION FUNCTION
# ============================================================

predict_model <- function(
    trained_object,
    new_data
) {

  feature_names <- trained_object$feature_names

  temp <- new_data[, feature_names, drop = FALSE]

  # Numeric imputation using training medians
  for (col in names(trained_object$medians)) {

    med <- trained_object$medians[[col]]

    temp[[col]][
      is.na(temp[[col]])
    ] <- med
  }


  # Categorical imputation
  for (col in names(trained_object$categorical_values)) {

    mode_value <-
      trained_object$categorical_values[[col]]

    temp[[col]] <- as.character(temp[[col]])

    temp[[col]][
      is.na(temp[[col]]) |
        temp[[col]] == ""
    ] <- mode_value

    temp[[col]] <- factor(
      temp[[col]]
    )
  }


  prediction <- predict(
    trained_object$model,
    data = temp
  )$predictions

  prediction
}


# ============================================================
# 11. TRAIN ALL THREE MODELS
# ============================================================

cat("\n========================================\n")
cat("TRAINING MODEL A\n")
cat("========================================\n")

model_A <- train_rf_model(
  train_df,
  features_A
)

cat("Model A trained.\n")


cat("\n========================================\n")
cat("TRAINING MODEL B\n")
cat("========================================\n")

model_B <- train_rf_model(
  train_df,
  features_B
)

cat("Model B trained.\n")


cat("\n========================================\n")
cat("TRAINING MODEL C\n")
cat("========================================\n")

model_C <- train_rf_model(
  train_df,
  features_C
)

cat("Model C trained.\n")


# ============================================================
# 12. EVALUATION FUNCTION
# ============================================================

evaluate_model <- function(
    trained_model,
    data
) {

  predictions <- predict_model(
    trained_model,
    data
  )

  actual <- data$post_GA90

  mae <- mean(
    abs(predictions - actual)
  )

  rmse <- sqrt(
    mean((predictions - actual)^2)
  )

  r2 <- 1 -
    sum((actual - predictions)^2) /
    sum((actual - mean(actual))^2)

  data.frame(
    MAE = mae,
    RMSE = rmse,
    R2 = r2
  )
}


# ============================================================
# 13. EVALUATE MODELS
# ============================================================

cat("\n========================================\n")
cat("MODEL PERFORMANCE\n")
cat("========================================\n")

results_A_train <- evaluate_model(model_A, train_df)
results_A_val   <- evaluate_model(model_A, val_df)
results_A_test  <- evaluate_model(model_A, test_df)

results_B_train <- evaluate_model(model_B, train_df)
results_B_val   <- evaluate_model(model_B, val_df)
results_B_test  <- evaluate_model(model_B, test_df)

results_C_train <- evaluate_model(model_C, train_df)
results_C_val   <- evaluate_model(model_C, val_df)
results_C_test  <- evaluate_model(model_C, test_df)


results <- data.frame(

  Model = c(
    "A - Player Only",
    "B - Player + Team",
    "C - Rich Context"
  ),

  Train_MAE = c(
    results_A_train$MAE,
    results_B_train$MAE,
    results_C_train$MAE
  ),

  Validation_MAE = c(
    results_A_val$MAE,
    results_B_val$MAE,
    results_C_val$MAE
  ),

  Test_MAE = c(
    results_A_test$MAE,
    results_B_test$MAE,
    results_C_test$MAE
  ),

  Train_RMSE = c(
    results_A_train$RMSE,
    results_B_train$RMSE,
    results_C_train$RMSE
  ),

  Validation_RMSE = c(
    results_B_val$RMSE,
    results_C_val$RMSE,
    results_C_val$RMSE
  ),

  Test_RMSE = c(
    results_A_test$RMSE,
    results_B_test$RMSE,
    results_C_test$RMSE
  ),

  Train_R2 = c(
    results_A_train$R2,
    results_B_train$R2,
    results_C_train$R2
  ),

  Validation_R2 = c(
    results_A_val$R2,
    results_B_val$R2,
    results_C_val$R2
  ),

  Test_R2 = c(
    results_A_test$R2,
    results_B_test$R2,
    results_C_test$R2
  )
)

# Print only numeric columns rounded
results_display <- results %>%
  mutate(
    across(
      where(is.numeric),
      ~ round(.x, 4)
    )
  )

print(results_display)


# Save full-precision results
write_csv(
  results,
  file.path(
    OUTPUT_DIR,
    "model_comparison.csv"
  )
)
# ============================================================
# 14. SAVE RESULTS
# ============================================================

write_csv(
  results,
  file.path(
    OUTPUT_DIR,
    "model_comparison.csv"
  )
)


# ============================================================
# 15. MODEL B FEATURE IMPORTANCE
# ============================================================

cat("\n========================================\n")
cat("MODEL B FEATURE IMPORTANCE\n")
cat("========================================\n")

importance_B <- data.frame(
  Feature = names(
    model_B$model$variable.importance
  ),
  Importance = as.numeric(
    model_B$model$variable.importance
  )
) %>%
  arrange(
    desc(Importance)
  )

print(
  head(
    importance_B,
    20
  )
)

write_csv(
  importance_B,
  file.path(
    OUTPUT_DIR,
    "model_B_feature_importance.csv"
  )
)


# ============================================================
# 16. SAVE MODEL B
# ============================================================

saveRDS(
  model_B,
  file.path(
    OUTPUT_DIR,
    "football_transfer_prediction_model_B.rds"
  )
)

cat("\nModel B saved.\n")


# ============================================================
# 17. GENERAL PREDICTION FUNCTION
# ============================================================

predict_transfer <- function(

  age_at_transfer,
  height_in_cm,
  appearances_pre,
  minutes_pre,
  goals_pre,
  assists_pre,
  goals_per90_pre,
  assists_per90_pre,
  yellow_cards_pre,
  red_cards_pre,
  transfer_market_value,
  market_value_pre,
  transfer_fee,

  position,
  sub_position,
  foot,

  destination_goals_for_avg10,
  destination_goals_against_avg10,
  destination_goal_difference_avg10,
  destination_win_rate10,
  destination_points_per_game10,
  destination_matches_last10

) {

  input <- data.frame(

    age_at_transfer = age_at_transfer,
    height_in_cm = height_in_cm,
    appearances_pre = appearances_pre,
    minutes_pre = minutes_pre,
    goals_pre = goals_pre,
    assists_pre = assists_pre,
    goals_per90_pre = goals_per90_pre,
    assists_per90_pre = assists_per90_pre,
    yellow_cards_pre = yellow_cards_pre,
    red_cards_pre = red_cards_pre,
    transfer_market_value = transfer_market_value,
    market_value_pre = market_value_pre,
    transfer_fee = transfer_fee,

    position = position,
    sub_position = sub_position,
    foot = foot,

    rich_team_goals_for_avg10 =
      destination_goals_for_avg10,

    rich_team_goals_against_avg10 =
      destination_goals_against_avg10,

    rich_team_goal_difference_avg10 =
      destination_goal_difference_avg10,

    rich_team_win_rate10 =
      destination_win_rate10,

    rich_team_points_per_game10 =
      destination_points_per_game10,

    rich_team_matches_last10 =
      destination_matches_last10
  )

  prediction <- predict_model(
    model_B,
    input
  )

  return(
    as.numeric(prediction)
  )
}


# ============================================================
# 18. SEARCH FOR CUNHA IN YOUR DATASET
# ============================================================

cat("\n========================================\n")
cat("CUNHA DEMONSTRATION\n")
cat("========================================\n")


cunha <- df %>%
  filter(
    grepl(
      "cunha",
      tolower(
        paste(
          player_id,
          from_club_name,
          to_club_name,
          sep = " "
        )
      )
    )
  )


# ============================================================
# 19. IF CUNHA EXISTS, USE HIS ACTUAL DATA
# ============================================================

if (nrow(cunha) > 0) {

  cat("\nCunha found in dataset!\n\n")

  print(
    cunha %>%
      select(
        player_id,
        transfer_date,
        from_club_name,
        to_club_name,
        age_at_transfer,
        position,
        sub_position,
        foot,
        post_GA90
      )
  )


  # Use the first matching transfer
  cunha_row <- cunha[1, ]


  cunha_prediction <- predict_transfer(

    age_at_transfer =
      cunha_row$age_at_transfer,

    height_in_cm =
      cunha_row$height_in_cm,

    appearances_pre =
      cunha_row$appearances_pre,

    minutes_pre =
      cunha_row$minutes_pre,

    goals_pre =
      cunha_row$goals_pre,

    assists_pre =
      cunha_row$assists_pre,

    goals_per90_pre =
      cunha_row$goals_per90_pre,

    assists_per90_pre =
      cunha_row$assists_per90_pre,

    yellow_cards_pre =
      cunha_row$yellow_cards_pre,

    red_cards_pre =
      cunha_row$red_cards_pre,

    transfer_market_value =
      cunha_row$transfer_market_value,

    market_value_pre =
      cunha_row$market_value_pre,

    transfer_fee =
      cunha_row$transfer_fee,

    position =
      cunha_row$position,

    sub_position =
      cunha_row$sub_position,

    foot =
      cunha_row$foot,

    destination_goals_for_avg10 =
      cunha_row$rich_team_goals_for_avg10,

    destination_goals_against_avg10 =
      cunha_row$rich_team_goals_against_avg10,

    destination_goal_difference_avg10 =
      cunha_row$rich_team_goal_difference_avg10,

    destination_win_rate10 =
      cunha_row$rich_team_win_rate10,

    destination_points_per_game10 =
      cunha_row$rich_team_points_per_game10,

    destination_matches_last10 =
      cunha_row$rich_team_matches_last10
  )


  cat("\n----------------------------------------\n")
  cat("CUNHA PREDICTION\n")
  cat("----------------------------------------\n")

  cat(
    "Predicted post-transfer G+A/90:",
    round(cunha_prediction, 4),
    "\n"
  )


  # Pre-transfer performance
  cunha_pre_ga90 <-
    (
      cunha_row$goals_pre +
        cunha_row$assists_pre
    ) /
    cunha_row$minutes_pre *
    90


  cat(
    "Pre-transfer G+A/90:",
    round(cunha_pre_ga90, 4),
    "\n"
  )


  # Actual value if available
  if (
    "post_GA90" %in% names(cunha_row) &&
    !is.na(cunha_row$post_GA90)
  ) {

    cat(
      "Actual post-transfer G+A/90:",
      round(
        cunha_row$post_GA90,
        4
      ),
      "\n"
    )

    cat(
      "Absolute prediction error:",
      round(
        abs(
          cunha_prediction -
            cunha_row$post_GA90
        ),
        4
      ),
      "\n"
    )
  }


  # Save demonstration
  cunha_output <- data.frame(

    Player = "Matheus Cunha",

    Transfer_Date =
      cunha_row$transfer_date,

    From =
      cunha_row$from_club_name,

    To =
      cunha_row$to_club_name,

    Pre_Transfer_GA90 =
      cunha_pre_ga90,

    Predicted_Post_Transfer_GA90 =
      cunha_prediction,

    Actual_Post_Transfer_GA90 =
      cunha_row$post_GA90
  )


  write_csv(
    cunha_output,
    file.path(
      OUTPUT_DIR,
      "cunha_prediction_demo.csv"
    )
  )


} else {

  # ==========================================================
  # 20. MANUAL CUNHA DEMONSTRATION
  # ==========================================================

  cat("\n")
  cat("Cunha was not found in the dataset.\n")
  cat("Running the model with a demonstration input.\n")
  cat("\n")

  # ----------------------------------------------------------
  # IMPORTANT:
  # These are DEMONSTRATION values.
  #
  # They are NOT used during model training.
  # ----------------------------------------------------------

  cunha_demo <- data.frame(

    age_at_transfer = 26,

    height_in_cm = 183,

    appearances_pre = 33,

    minutes_pre = 2800,

    goals_pre = 15,

    assists_pre = 6,

    goals_per90_pre =
      15 / 2800 * 90,

    assists_per90_pre =
      6 / 2800 * 90,

    yellow_cards_pre = 5,

    red_cards_pre = 0,

    transfer_market_value = 0,

    market_value_pre = 0,

    transfer_fee = 0,

    position = "Attack",

    sub_position = "Centre-Forward",

    foot = "Right",

    rich_team_goals_for_avg10 = 1.1,

    rich_team_goals_against_avg10 = 1.6,

    rich_team_goal_difference_avg10 =
      1.1 - 1.6,

    rich_team_win_rate10 = 0.2,

    rich_team_points_per_game10 = 0.8,

    rich_team_matches_last10 = 10
  )


  cunha_prediction <- predict_model(
    model_B,
    cunha_demo
  )


  cat("\n----------------------------------------\n")
  cat("CUNHA DEMONSTRATION PREDICTION\n")
  cat("----------------------------------------\n")

  cat(
    "Predicted post-transfer G+A/90:",
    round(
      cunha_prediction,
      4
    ),
    "\n"
  )


  cunha_pre_ga90 <-
    (
      cunha_demo$goals_pre +
        cunha_demo$assists_pre
    ) /
    cunha_demo$minutes_pre *
    90


  cat(
    "Input pre-transfer G+A/90:",
    round(
      cunha_pre_ga90,
      4
    ),
    "\n"
  )


  cat("\nNOTE:\n")
  cat(
    "This is a demonstration input because\n",
    "Cunha was not present in the dataset.\n"
  )


  cunha_output <- data.frame(

    Player = "Matheus Cunha",

    Pre_Transfer_GA90 =
      cunha_pre_ga90,

    Predicted_Post_Transfer_GA90 =
      as.numeric(cunha_prediction),

    Demonstration = TRUE
  )


  write_csv(
    cunha_output,
    file.path(
      OUTPUT_DIR,
      "cunha_prediction_demo.csv"
    )
  )
}


# ============================================================
# 21. PREDICTION EXAMPLE FOR ANY PLAYER
# ============================================================

cat("\n========================================\n")
cat("GENERAL PREDICTION EXAMPLE\n")
cat("========================================\n")

cat(
  "The same model can predict ANY player\n",
  "provided the required input features are supplied.\n"
)


# ============================================================
# 22. FINAL OUTPUT
# ============================================================

cat("\n========================================\n")
cat("PIPELINE COMPLETE\n")
cat("========================================\n")

cat("\nFiles generated in:\n")
cat(OUTPUT_DIR, "\n\n")

cat("1. model_comparison.csv\n")
cat("2. model_B_feature_importance.csv\n")
cat("3. football_transfer_prediction_model_B.rds\n")
cat("4. cunha_prediction_demo.csv\n")

cat("\n========================================\n")
cat("RESEARCH INTERPRETATION\n")
cat("========================================\n")

cat(
  "\nModel A: Player characteristics only.\n",
  "Model B: Player + destination-team performance.\n",
  "Model C: Player + richer contextual features.\n\n"
)

cat(
  "The primary research comparison is whether\n",
  "destination-team context improves prediction\n",
  "beyond player characteristics alone.\n"
)

cat("\nDone.\n")
# ============================================================
# CUNHA - DIRECT PREDICTION
# ============================================================

cat("\n========================================\n")
cat("MATHEUS CUNHA PREDICTION\n")
cat("========================================\n\n")

# Cunha demonstration input
cunha_demo <- data.frame(

  age_at_transfer = 26,

  height_in_cm = 183,

  appearances_pre = 33,

  minutes_pre = 2800,

  goals_pre = 15,

  assists_pre = 6,

  goals_per90_pre = 15 / 2800 * 90,

  assists_per90_pre = 6 / 2800 * 90,

  yellow_cards_pre = 5,

  red_cards_pre = 0,

  transfer_market_value = 0,

  market_value_pre = 0,

  transfer_fee = 0,

  position = "Attack",

  sub_position = "Centre-Forward",

  foot = "Right",

  # Manchester United destination context
  rich_team_goals_for_avg10 = 1.1,

  rich_team_goals_against_avg10 = 1.6,

  rich_team_goal_difference_avg10 = -0.5,

  rich_team_win_rate10 = 0.2,

  rich_team_points_per_game10 = 0.8,

  rich_team_matches_last10 = 10
)


# Make prediction
cunha_prediction <- predict_model(
  model_B,
  cunha_demo
)


# Calculate input performance
cunha_pre_ga90 <-
  (
    cunha_demo$goals_pre +
      cunha_demo$assists_pre
  ) /
  cunha_demo$minutes_pre *
  90


# ============================================================
# DISPLAY RESULT
# ============================================================

cat("PLAYER: Matheus Cunha\n")
cat("DESTINATION: Manchester United\n\n")

cat(
  "Pre-transfer G+A/90:",
  round(cunha_pre_ga90, 3),
  "\n"
)

cat(
  "Predicted post-transfer G+A/90:",
  round(
    as.numeric(cunha_prediction),
    3
  ),
  "\n\n"
)

cat("========================================\n")
cat("PREDICTION COMPLETE\n")
cat("========================================\n")


# Updated football transfer models
# Run from the project root.
# Outputs go to a new folder; existing audit outputs are preserved.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ranger)
})

set.seed(20261028)

ROOT <- getwd()
OUT <- file.path(ROOT, "outputs_updated_models")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

main_file <- file.path(ROOT, "post_transfer_performance.csv")
team_file <- file.path(ROOT, "rich_team_context_dataset.csv")
coach_file <- file.path(ROOT, "coach_tactical_context_v21.csv")
extended_file <- file.path(
  ROOT, "outputs_mode_a_features", "mode_a_pretransfer_features.csv"
)

required_files <- c(main_file, team_file, coach_file, extended_file)
if (!all(file.exists(required_files))) {
  stop("Missing input file(s): ",
       paste(required_files[!file.exists(required_files)], collapse = ", "))
}

main <- read_csv(main_file, show_col_types = FALSE)
team <- read_csv(team_file, show_col_types = FALSE)
coach <- read_csv(coach_file, show_col_types = FALSE)
extra <- read_csv(extended_file, show_col_types = FALSE)

check_key <- function(x, label) {
  if (!"transfer_row_id" %in% names(x)) {
    stop(label, " has no transfer_row_id")
  }
  if (anyNA(x$transfer_row_id) || anyDuplicated(x$transfer_row_id)) {
    stop(label, " has missing or duplicated transfer_row_id values")
  }
}

check_key(main, "main")
check_key(team, "team")
check_key(coach, "coach")
check_key(extra, "extra")

# Validate shared identity fields before joining.
identity_fields <- c(
  "player_id", "transfer_date", "to_club_id",
  "from_club_id", "to_club_name"
)

for (label in c("team", "extra")) {
  other <- if (label == "team") team else extra
  shared <- intersect(identity_fields, intersect(names(main), names(other)))

  chk <- main |>
    select(transfer_row_id, all_of(shared)) |>
    inner_join(
      other |> select(transfer_row_id, all_of(shared)),
      by = "transfer_row_id",
      suffix = c("_main", "_other")
    )

  for (field in shared) {
    left <- chk[[paste0(field, "_main")]]
    right <- chk[[paste0(field, "_other")]]

    # Compare dates as dates and other values as character.
    if (field == "transfer_date") {
      mismatch <- as.character(as.Date(left)) !=
        as.character(as.Date(right))
    } else {
      mismatch <- as.character(left) != as.character(right)
    }
    mismatch[is.na(mismatch)] <- TRUE

    if (any(mismatch)) {
      stop(label, " identity mismatch in ", field,
           ": ", sum(mismatch), " rows")
    }
  }
}

# One-to-one joins only; never allow row multiplication.
team_features <- setdiff(names(team), names(main))
team_features <- setdiff(team_features, "transfer_row_id")

coach_features <- setdiff(names(coach), "transfer_row_id")
extra_features <- setdiff(
  names(extra),
  c("transfer_row_id", "player_id", "transfer_date", "to_club_id")
)

# Prefix potential overlaps rather than silently replacing columns.
team <- team |>
  select(transfer_row_id, all_of(team_features)) |>
  rename_with(~ paste0("team_", .x), -transfer_row_id)

coach <- coach |>
  select(transfer_row_id, all_of(coach_features))

extra <- extra |>
  select(transfer_row_id, all_of(extra_features))

dat <- main |>
  inner_join(team, by = "transfer_row_id", relationship = "one-to-one") |>
  left_join(coach, by = "transfer_row_id", relationship = "one-to-one") |>
  left_join(extra, by = "transfer_row_id", relationship = "one-to-one")

if (nrow(dat) != nrow(main)) {
  stop("Row count changed during joins; refusing to continue.")
}

# Keep the eligible cohort used in the corrected audit.
dat <- dat |>
  mutate(
    transfer_date = as.Date(transfer_date),
    post_GA90 = as.numeric(post_GA90)
  ) |>
  filter(
    !is.na(transfer_date),
    complete_365_day_window == 1,
    eligible_450min == 1,
    is.finite(post_GA90)
  ) |>
  arrange(transfer_date, transfer_row_id)

if (nrow(dat) < 100) stop("Unexpectedly small eligible cohort.")

# Chronological split by whole date: 60% / 20% / 20%.
dates <- sort(unique(dat$transfer_date))
if (length(dates) < 10) stop("Too few distinct dates to split safely.")

cut1 <- dates[max(1, floor(length(dates) * 0.60))]
cut2 <- dates[max(2, floor(length(dates) * 0.80))]

train <- dat |> filter(transfer_date <= cut1)
valid <- dat |> filter(transfer_date > cut1, transfer_date <= cut2)
test  <- dat |> filter(transfer_date > cut2)

if (nrow(train) == 0 || nrow(valid) == 0 || nrow(test) == 0) {
  stop("A chronological split is empty; inspect split dates.")
}

# Baseline player-only predictors.
core_num <- c(
  "age_at_transfer", "height_in_cm", "appearances_pre",
  "minutes_pre", "goals_pre", "assists_pre",
  "goals_per90_pre", "assists_per90_pre",
  "yellow_cards_pre", "red_cards_pre",
  "transfer_market_value", "market_value_pre", "transfer_fee"
)
core_cat <- c("position", "sub_position", "foot")

# Team and tactical context from rich_team_context_dataset.csv.
team_context <- paste0("team_", c(
  "rich_team_matches_last10",
  "rich_team_goals_for_avg10",
  "rich_team_goals_against_avg10",
  "rich_team_goal_difference_avg10",
  "rich_team_win_rate10",
  "rich_team_points_per_game10",
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
  "position_minutes_share"
))
team_cat <- paste0("team_", c("dominant_formation10", "dominant_manager10"))

coach_cols <- c(
  "coach_known_pretransfer",
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

# Additional pre-transfer candidate features.
extra_cols <- c(
  "pre_apps_365d", "pre_minutes_365d", "pre_goals_365d",
  "pre_assists_365d", "pre_yellow_cards_365d", "pre_red_cards_365d",
  "pre_goals_per90_365d", "pre_assists_per90_365d",
  "pre_yellows_per90_365d", "pre_reds_per90_365d",
  "pre_apps_90d", "pre_minutes_90d",
  "days_since_last_pre_transfer_appearance",
  "market_value_latest_pre_eur", "market_value_prior_365d_eur",
  "market_value_change_365d_pct",
  "market_value_latest_observation_age_days"
)

# Only use features that actually exist.
core_num <- intersect(core_num, names(dat))
core_cat <- intersect(core_cat, names(dat))
team_context <- intersect(team_context, names(dat))
team_cat <- intersect(team_cat, names(dat))
coach_cols <- intersect(coach_cols, names(dat))
extra_cols <- intersect(extra_cols, names(dat))

# Do not train on post-transfer outcomes or identifiers.
forbidden <- c(
  "post_GA90", "post_goals", "post_assists", "post_minutes",
  "post_appearances", "complete_365_day_window", "eligible_450min",
  "player_id", "transfer_row_id", "transfer_date",
  "transfer_season", "from_club_id", "to_club_id",
  "from_club_name", "to_club_name", "window_start", "window_end"
)

make_features <- function(groups) {
  unique(setdiff(unlist(groups, use.names = FALSE), forbidden))
}

# Mode A includes team, tactical/manager, and squad competition context.
# Extended Mode A adds validated pre-transfer history/valuation candidates.
specs <- list(
  Model_A_player_only = list(
    num = core_num, cat = core_cat
  ),
  Mode_A = list(
    num = c(core_num, team_context, coach_cols),
    cat = c(core_cat, team_cat)
  ),
  Extended_Mode_A = list(
    num = c(core_num, team_context, coach_cols, extra_cols),
    cat = c(core_cat, team_cat)
  )
)

# Use training-only medians/modes; unseen test categories get an explicit level.
prepare_splits <- function(train, valid, test, features) {
  features <- intersect(features, names(train))
  if (!length(features)) stop("No predictors available for a model.")

  tr <- train[, features, drop = FALSE]
  va <- valid[, features, drop = FALSE]
  te <- test[, features, drop = FALSE]

  for (f in features) {
    if (is.character(tr[[f]]) || is.factor(tr[[f]])) {
      vals <- as.character(tr[[f]])
      vals[is.na(vals) | vals == ""] <- "__MISSING__"
      lev <- unique(c(vals, "__UNSEEN__"))
      convert <- function(x) {
        x <- as.character(x)
        x[is.na(x) | x == ""] <- "__MISSING__"
        x[!x %in% lev] <- "__UNSEEN__"
        factor(x, levels = lev)
      }
      tr[[f]] <- convert(tr[[f]])
      va[[f]] <- convert(va[[f]])
      te[[f]] <- convert(te[[f]])
    } else {
      tr[[f]] <- suppressWarnings(as.numeric(tr[[f]]))
      va[[f]] <- suppressWarnings(as.numeric(va[[f]]))
      te[[f]] <- suppressWarnings(as.numeric(te[[f]]))
      med <- median(tr[[f]][is.finite(tr[[f]])], na.rm = TRUE)
      if (!is.finite(med)) med <- 0
      for (nm in c("tr", "va", "te")) {
        obj <- get(nm)
        obj[[f]][!is.finite(obj[[f]]) | is.na(obj[[f]])] <- med
        assign(nm, obj)
      }
    }
  }

  list(train = tr, valid = va, test = te, features = features)
}

metric_row <- function(model, split, actual, pred) {
  e <- actual - pred
  data.frame(
    model = model, split = split, n = length(actual),
    MAE = mean(abs(e)),
    RMSE = sqrt(mean(e^2)),
    R2 = if (stats::var(actual) > 0)
      1 - sum(e^2) / sum((actual - mean(actual))^2) else NA_real_
  )
}

all_metrics <- list()
all_predictions <- list()
all_importance <- list()

for (model_name in names(specs)) {
  cat("\nTraining", model_name, "\n")
  spec <- specs[[model_name]]
  features <- make_features(spec)
  prep <- prepare_splits(train, valid, test, features)

  # Small validation-only grid.
  grid <- expand.grid(
    mtry_fraction = c(0.25, 0.5, 0.75),
    min_node_size = c(5, 10),
    KEEP.OUT.ATTRS = FALSE
  )

  best <- NULL
  best_mae <- Inf

  for (i in seq_len(nrow(grid))) {
    mtry <- max(1, min(
      ncol(prep$train),
      floor(ncol(prep$train) * grid$mtry_fraction[i])
    ))

    fit <- ranger(
      x = prep$train,
      y = train$post_GA90,
      num.trees = 300,
      mtry = mtry,
      min.node.size = grid$min_node_size[i],
      seed = 20261028,
      num.threads = 1
    )

    pred <- predict(fit, data = prep$valid)$predictions
    mae <- mean(abs(valid$post_GA90 - pred))

    if (mae < best_mae) {
      best_mae <- mae
      best <- list(
        mtry = mtry,
        min_node_size = grid$min_node_size[i]
      )
    }
  }

  # Refit on train + validation after choosing hyperparameters.
  combined <- bind_rows(train, valid)
  prep_final <- prepare_splits(combined, test, test, features)
  # prepare_splits expects three inputs; the first is train, second validation,
  # third test. Here validation and test are both the untouched test frame.
  final_fit <- ranger(
    x = prep_final$train,
    y = combined$post_GA90,
    num.trees = 500,
    mtry = best$mtry,
    min.node.size = best$min_node_size,
    seed = 20261028,
    importance = "permutation",
    num.threads = 1
  )

  pred_train <- predict(final_fit, data = prep_final$train)$predictions
  pred_test <- predict(final_fit, data = prep_final$test)$predictions

  all_metrics[[model_name]] <- bind_rows(
    metric_row(model_name, "train_plus_validation",
               combined$post_GA90, pred_train),
    metric_row(model_name, "test",
               test$post_GA90, pred_test)
  )

  all_predictions[[model_name]] <- data.frame(
    transfer_row_id = test$transfer_row_id,
    transfer_date = test$transfer_date,
    actual_GA90 = test$post_GA90,
    prediction = pred_test,
    model = model_name
  )

  imp <- data.frame(
    model = model_name,
    feature = names(final_fit$variable.importance),
    importance = as.numeric(final_fit$variable.importance)
  )
  all_importance[[model_name]] <- imp

  write_csv(
    data.frame(
      model = model_name,
      selected_mtry = best$mtry,
      selected_min_node_size = best$min_node_size,
      validation_MAE = best_mae,
      features_used = length(prep$features)
    ),
    file.path(OUT, paste0(model_name, "_tuning.csv"))
  )
}

metrics <- bind_rows(all_metrics)
predictions <- bind_rows(all_predictions)
importance <- bind_rows(all_importance)

write_csv(metrics, file.path(OUT, "metrics.csv"))
write_csv(predictions, file.path(OUT, "test_predictions.csv"))
write_csv(importance, file.path(OUT, "feature_importance.csv"))

cat("\n=== TEST RESULTS ===\n")
print(metrics |> filter(split == "test") |> arrange(MAE))
cat("\nSaved results to:", OUT, "\n")

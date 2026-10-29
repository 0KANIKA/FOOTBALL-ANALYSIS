# Train reusable Models A, B and C for the new-player Shiny app.
# Run from this directory: Rscript train_models.R

suppressPackageStartupMessages({
  library(ranger)
})

ROOT <- normalizePath(file.path(getwd(), ".."), mustWork = TRUE)
APP <- normalizePath(getwd(), mustWork = TRUE)
DATA <- file.path(ROOT, "FINAL_MODEL_C_CLEAN.csv")
if (!file.exists(DATA)) stop("Missing training data: ", DATA)

models_dir <- file.path(APP, "models")
dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)

features_A <- c(
  "age_at_transfer", "height_in_cm", "appearances_pre", "minutes_pre",
  "goals_pre", "assists_pre", "goals_per90_pre", "assists_per90_pre",
  "yellow_cards_pre", "red_cards_pre", "transfer_market_value",
  "market_value_pre", "transfer_fee", "position", "sub_position", "foot"
)
features_B <- c(features_A, "rich_team_goals_for_avg10",
  "rich_team_goals_against_avg10", "rich_team_goal_difference_avg10",
  "rich_team_win_rate10", "rich_team_points_per_game10",
  "rich_team_matches_last10")
features_C <- c(features_B, "dominant_formation10",
  "dominant_formation_frequency10", "formation_diversity10",
  "manager_matches_last10", "unique_managers_last10",
  "same_position_competitor_count", "same_position_competitor_minutes",
  "same_position_avg_market_value", "same_position_top_competitor_minutes",
  "same_sub_position_competitor_count", "same_sub_position_competitor_minutes",
  "same_sub_position_top_minutes", "position_minutes_share",
  "previous_destination_appearances", "previous_destination_minutes",
  "coach_position_group_minutes_share", "coach_position_group_start_rate",
  "coach_exact_position_minutes_share", "coach_exact_position_start_rate",
  "coach_position_competitor_count", "coach_position_competitor_minutes",
  "coach_position_rotation_hhi", "coach_formation_stability",
  "coach_observed_tenure_days")

dat <- read.csv(DATA, stringsAsFactors = FALSE, check.names = FALSE)
needed <- unique(c(features_C, "post_GA90", "transfer_date"))
missing <- setdiff(needed, names(dat))
if (length(missing)) stop("Training data is missing: ", paste(missing, collapse = ", "))
dat$transfer_date <- as.Date(dat$transfer_date)
dat$post_GA90 <- as.numeric(dat$post_GA90)
dat <- dat[is.finite(dat$post_GA90) & !is.na(dat$transfer_date), , drop = FALSE]
dat <- dat[order(dat$transfer_date), , drop = FALSE]
train_n <- max(1, floor(.8 * nrow(dat)))
train <- dat[seq_len(train_n), , drop = FALSE]
test <- dat[seq.int(train_n + 1, nrow(dat)), , drop = FALSE]

prepare <- function(x, features, schema = NULL) {
  out <- x[, features, drop = FALSE]
  if (is.null(schema)) {
    schema <- list(medians = list(), levels = list())
    for (f in features) {
      if (is.character(out[[f]]) || is.factor(out[[f]])) {
        vals <- as.character(out[[f]]); vals[is.na(vals) | vals == ""] <- "__MISSING__"
        schema$levels[[f]] <- unique(c(vals, "__UNSEEN__"))
      } else {
        z <- suppressWarnings(as.numeric(out[[f]]))
        m <- median(z[is.finite(z)], na.rm = TRUE); if (!is.finite(m)) m <- 0
        schema$medians[[f]] <- m
      }
    }
  }
  for (f in features) {
    if (f %in% names(schema$levels)) {
      z <- as.character(out[[f]]); z[is.na(z) | z == ""] <- "__MISSING__"
      z[!z %in% schema$levels[[f]]] <- "__UNSEEN__"
      out[[f]] <- factor(z, levels = schema$levels[[f]])
    } else {
      z <- suppressWarnings(as.numeric(out[[f]])); z[!is.finite(z) | is.na(z)] <- schema$medians[[f]]
      out[[f]] <- z
    }
  }
  list(data = out, schema = schema)
}

fit_one <- function(name, features) {
  p <- prepare(train, features)
  fit <- ranger(x = p$data, y = train$post_GA90, num.trees = 500,
                mtry = max(1, floor(sqrt(length(features)))),
                min.node.size = 10, seed = 20261028,
                importance = "permutation", num.threads = 1)
  saveRDS(list(name = name, target = "post_GA90", features = features,
               preprocessing = p$schema, model = fit),
          file.path(models_dir, paste0(name, ".rds")))
  test_p <- prepare(test, features, p$schema)
  test_pred <- as.numeric(predict(fit, data = test_p$data)$predictions)
  err <- test$post_GA90 - test_pred
  metrics <- data.frame(model = name, split = "chronological_holdout", n = nrow(test),
    MAE = mean(abs(err)), RMSE = sqrt(mean(err^2)),
    R2 = if (stats::var(test$post_GA90) > 0) 1 - sum(err^2) / sum((test$post_GA90 - mean(test$post_GA90))^2) else NA_real_)
  write.csv(metrics, file.path(models_dir, paste0(name, "_metrics.csv")), row.names = FALSE)
}

fit_one("model_A", features_A)
fit_one("model_B", features_B)
fit_one("model_C", features_C)
saveRDS(list(features_A = features_A, features_B = features_B,
             features_C = features_C, target = "post_GA90",
             training_rows = nrow(train), trained_at = Sys.time()),
        file.path(models_dir, "manifest.rds"))
cat("Saved Models A, B and C to ", models_dir, "\n", sep = "")

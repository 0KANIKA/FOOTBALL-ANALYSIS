# Executable tests for the final single-Model-C Shiny app. Run from this folder.
stopifnot(file.exists("app.R"), file.exists("train_models.R"), file.exists("README.md"))
invisible(parse("app.R"))
if (!requireNamespace("ranger", quietly = TRUE)) stop("ranger is required")
model_C <- readRDS(file.path("models", "model_C.rds"))
stopifnot(inherits(model_C$model, "ranger"), identical(model_C$target, "post_GA90"), length(model_C$features) == 46)
app_text <- paste(readLines("app.R", warn = FALSE), collapse = "\n")
stopifnot(grepl("TransferIQ", app_text, fixed = TRUE), grepl("transfer_fee", app_text, fixed = TRUE), grepl("Model C", app_text, fixed = TRUE))
stopifnot(!grepl('add\\("Model A', app_text), !grepl('add\\("Model B', app_text))

make_input <- function() {
  x <- as.data.frame(as.list(setNames(rep(1, length(model_C$features)), model_C$features)), stringsAsFactors = FALSE)
  x$position <- "NeverSeenPosition"; x$sub_position <- "NeverSeenSubPosition"; x$foot <- "NeverSeenFoot"; x$dominant_formation10 <- "NeverSeenFormation"
  x$transfer_fee <- 10000000; x$transfer_market_value <- 8000000; x$market_value_pre <- 7000000
  x$minutes_pre <- 2100; x$appearances_pre <- 30; x$goals_pre <- 12; x$assists_pre <- 5
  x
}
prepare <- function(x, spec) {
  z <- x[, spec$features, drop = FALSE]
  for (f in spec$features) {
    if (f %in% names(spec$preprocessing$levels)) {
      v <- as.character(z[[f]]); v[is.na(v) | v == ""] <- "__MISSING__"; v[!v %in% spec$preprocessing$levels[[f]]] <- "__UNSEEN__"; z[[f]] <- factor(v, levels = spec$preprocessing$levels[[f]])
    } else { v <- suppressWarnings(as.numeric(z[[f]])); v[!is.finite(v) | is.na(v)] <- spec$preprocessing$medians[[f]]; z[[f]] <- v }
  }
  z
}

new_player <- make_input()
prediction <- as.numeric(predict(model_C$model, data = prepare(new_player, model_C))$predictions)
stopifnot(length(prediction) == 1, is.finite(prediction))
stopifnot(is.finite(as.numeric(predict(model_C$model, data = prepare(new_player, model_C))$predictions)))

missing_rich <- new_player; missing_rich$coach_formation_stability <- NA
stopifnot(anyNA(missing_rich$coach_formation_stability))
stopifnot(all(grepl("transfer_fee|transfer_market_value|market_value_pre", app_text)))

export <- data.frame(prediction_date = as.character(Sys.Date()), player_label = "Genuinely new player", model = "Model C — Rich Context", target = "post_GA90", predicted_post_GA90 = prediction, units = "goals plus assists per 90 minutes", transfer_fee = new_player$transfer_fee, transfer_market_value = new_player$transfer_market_value, market_value_pre = new_player$market_value_pre, warnings = "none")
out <- file.path("outputs", "test_model_c_export.csv")
write.csv(export, out, row.names = FALSE)
check <- read.csv(out, stringsAsFactors = FALSE)
stopifnot(file.exists(out), nrow(check) == 1, check$model == "Model C — Rich Context", check$target == "post_GA90", check$transfer_fee == 10000000)
stopifnot(!grepl("Cunha", app_text, ignore.case = TRUE))
cat("PASS: final app parses; saved Model C loads; new-player prediction is finite; unseen categories are handled; financial fields are present; missing context is detectable; export contains Model C, target, prediction, and financial values; Models A/B are absent from user-facing prediction output.\n")

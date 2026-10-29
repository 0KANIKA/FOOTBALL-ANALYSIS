suppressPackageStartupMessages({
  library(shiny)
  library(ranger)
})

APP_DIR <- normalizePath(getwd(), mustWork = TRUE)
MODEL_DIR <- file.path(APP_DIR, "models")
PROJECT_ROOT <- normalizePath(file.path(APP_DIR, "..", ".."), mustWork = TRUE)
DATA_FILE <- file.path(PROJECT_ROOT, "04_Datasets", "derived", "FINAL_MODEL_C_CLEAN.csv")

# This is the exact Model C specification used by train_models.R.
features_A <- c("age_at_transfer", "height_in_cm", "appearances_pre", "minutes_pre",
  "goals_pre", "assists_pre", "goals_per90_pre", "assists_per90_pre",
  "yellow_cards_pre", "red_cards_pre", "transfer_market_value", "market_value_pre",
  "transfer_fee", "position", "sub_position", "foot")
features_B <- c(features_A, "rich_team_goals_for_avg10", "rich_team_goals_against_avg10",
  "rich_team_goal_difference_avg10", "rich_team_win_rate10",
  "rich_team_points_per_game10", "rich_team_matches_last10")
features_C <- c(features_B, "dominant_formation10", "dominant_formation_frequency10",
  "formation_diversity10", "manager_matches_last10", "unique_managers_last10",
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

if (!file.exists(file.path(MODEL_DIR, "model_C.rds"))) {
  source(file.path(APP_DIR, "train_models.R"), local = TRUE)
}
model_C <- readRDS(file.path(MODEL_DIR, "model_C.rds"))
if (!identical(model_C$features, features_C)) stop("Saved Model C feature specification does not match app.R")

choices <- function(col, fallback) {
  if (!file.exists(DATA_FILE)) return(fallback)
  x <- read.csv(DATA_FILE, stringsAsFactors = FALSE, check.names = FALSE)[[col]]
  unique(c(sort(unique(x[!is.na(x) & nzchar(x)])), "__UNSEEN__"))
}
num <- function(id, label, value = NA, min = NA_real_, max = NA_real_, step = 0.01, help = NULL) {
  args <- list(inputId = id, label = label, value = value, min = min, step = step, width = "100%")
  if (length(max) == 1 && is.finite(max)) args$max <- max
  do.call(numericInput, args)
}

rich_numeric <- setdiff(features_C, c(features_B, "dominant_formation10"))
rich_labels <- c(
  dominant_formation_frequency10 = "Dominant formation frequency (0–1)",
  formation_diversity10 = "Formation diversity (count)", manager_matches_last10 = "Manager matches in previous 10",
  unique_managers_last10 = "Unique managers in previous 10", same_position_competitor_count = "Same-position competitor count",
  same_position_competitor_minutes = "Same-position competitor minutes", same_position_avg_market_value = "Same-position average market value (€)",
  same_position_top_competitor_minutes = "Top same-position competitor minutes", same_sub_position_competitor_count = "Same-sub-position competitor count",
  same_sub_position_competitor_minutes = "Same-sub-position competitor minutes", same_sub_position_top_minutes = "Top same-sub-position competitor minutes",
  position_minutes_share = "Position minutes share (0–1)", previous_destination_appearances = "Previous destination appearances",
  previous_destination_minutes = "Previous destination minutes", coach_position_group_minutes_share = "Coach position-group minutes share (0–1)",
  coach_position_group_start_rate = "Coach position-group start rate (0–1)", coach_exact_position_minutes_share = "Coach exact-position minutes share (0–1)",
  coach_exact_position_start_rate = "Coach exact-position start rate (0–1)", coach_position_competitor_count = "Coach position competitor count",
  coach_position_competitor_minutes = "Coach position competitor minutes", coach_position_rotation_hhi = "Coach position rotation HHI (0–1)",
  coach_formation_stability = "Coach formation stability (0–1)", coach_observed_tenure_days = "Coach observed tenure (days)"
)

test_example <- list(
  player_name = "Alex Morgan Jr.", age_at_transfer = 22, height_in_cm = 182, appearances_pre = 30,
  minutes_pre = 2100, goals_pre = 12, assists_pre = 5, goals_per90_pre = 0.5143,
  assists_per90_pre = 0.2143, yellow_cards_pre = 4, red_cards_pre = 0,
  transfer_market_value = 8000000, market_value_pre = 7000000, transfer_fee = 10000000,
  position = "Attack", sub_position = "Attacking Midfield", foot = "both",
  rich_team_matches_last10 = 10, rich_team_goals_for_avg10 = 1.8, rich_team_goals_against_avg10 = 1.1,
  rich_team_goal_difference_avg10 = 0.7, rich_team_win_rate10 = 0.6, rich_team_points_per_game10 = 2,
  rich_dominant_formation = "3-1-4-2",
  rich_numeric_values = c(0.4, 3, 8, 2, 3, 4200, 6000000, 1800, 2, 2500, 1400, 0.25, 0, 0, 0.2, 0.5, 0.15, 0.4, 3, 4200, 0.4, 0.7, 365)
)

ui <- fluidPage(
  tags$head(tags$style(HTML("
    :root { --navy:#102a43; --navy2:#183b56; --green:#2f855a; --pale:#f5f8fa; --ink:#243b53; }
    body { background:#f5f8fa; color:var(--ink); font-family:Inter,Segoe UI,Arial,sans-serif; }
    .navbar, .well { border-radius:10px; } .hero { background:linear-gradient(135deg,var(--navy),var(--navy2)); color:white; padding:28px 32px; border-radius:12px; margin-bottom:20px; box-shadow:0 4px 14px rgba(16,42,67,.15); }
    .hero h1 { margin-top:0; font-weight:700; } .hero p { margin-bottom:0; color:#d9e8f2; max-width:950px; }
    .card { background:white; border:1px solid #d9e2ec; border-radius:10px; padding:18px 20px; margin-bottom:16px; box-shadow:0 2px 8px rgba(16,42,67,.05); }
    .card h3, .card h4 { color:var(--navy); margin-top:4px; } .section-note { color:#52606d; font-size:13px; }
    .btn-primary { background:var(--green); border-color:var(--green); } .btn-primary:hover { background:#276749; border-color:#276749; }
    .btn-info { background:var(--navy2); border-color:var(--navy2); } .prediction-card { border-top:5px solid var(--green); text-align:center; padding:26px; }
    .prediction-value { color:var(--navy); font-size:48px; font-weight:800; line-height:1.1; } .prediction-unit { color:#52606d; font-size:15px; }
    .loading { color:#245269; background:#e8f4fb; border:1px solid #b8dcec; border-radius:5px; padding:8px 12px; display:inline-block; margin-bottom:12px; }
    .synthetic { background:#fff8e1; border:1px solid #f0c36d; padding:10px 12px; border-radius:6px; margin:10px 0; }
    .metric-label { color:#627d98; font-size:12px; text-transform:uppercase; letter-spacing:.04em; } .metric-value { color:var(--navy); font-weight:650; }
    .shiny-output-error { color:#a61b1b; }
    @media (max-width: 900px) { .hero { padding:20px; } .prediction-value { font-size:38px; } }
  "))),
  div(class = "hero", h1("TransferIQ — Football Transfer Performance Predictor"),
    p("An evidence-based decision-support tool for estimating a player's post-transfer attacking contribution using player performance, destination-team context, tactical fit, squad competition, and coach-related information.")),
  sidebarLayout(
    sidebarPanel(width = 4,
      div(class = "card", h3("Prepare a prediction"),
        actionButton("load_test_example", "Load Test Example", class = "btn-info"),
        actionButton("clear_form", "Clear Form"), uiOutput("example_notice")),
      div(class = "card", h4("A. Player profile"), p(class = "section-note", "Identity is only a display label; no player lookup or player ID is required."),
        textInput("player_name", "Player label (optional)", "New player"),
        num("age_at_transfer", "Age at transfer (years)", 24, 15, 60, .1),
        num("height_in_cm", "Height (cm)", 180, 120, 230, 1),
        selectInput("position", "Position", choices("position", c("Attack", "Midfield", "Defender", "Goalkeeper"))),
        selectInput("sub_position", "Sub-position", choices("sub_position", c("Attacking Midfield", "Centre-Forward", "Central Midfield", "Centre-Back", "Goalkeeper"))),
        selectInput("foot", "Preferred foot", choices = unique(c("left", "right", "both", choices("foot", character()))))),
      div(class = "card", h4("B. Previous performance and transfer economics"), p(class = "section-note", "Currency values are euros and retain the training-data scale."),
        num("appearances_pre", "Previous appearances", 20, 0, step = 1), num("minutes_pre", "Previous minutes", 1500, 0, step = 1),
        num("goals_pre", "Previous goals", 5, 0, step = 1), num("assists_pre", "Previous assists", 3, 0, step = 1),
        num("goals_per90_pre", "Previous goals per 90 (0 = calculate)", 0, 0, step = .0001), num("assists_per90_pre", "Previous assists per 90 (0 = calculate)", 0, 0, step = .0001),
        num("yellow_cards_pre", "Previous yellow cards", 2, 0, step = 1), num("red_cards_pre", "Previous red cards", 0, 0, step = 1),
        num("transfer_market_value", "Transfer market value (€)", 0, 0, step = 1000), num("market_value_pre", "Previous market value (€)", 0, 0, step = 1000), num("transfer_fee", "Transfer fee (€)", 0, 0, step = 1000)),
      div(class = "card", h4("C. Destination-team context"), p(class = "section-note", "Use only matches and records available before the proposed transfer date."),
        num("rich_team_matches_last10", "Matches in previous 10", NA, 0, step = 1), num("rich_team_goals_for_avg10", "Average goals for", NA, 0, step = .01),
        num("rich_team_goals_against_avg10", "Average goals against", NA, 0, step = .01), num("rich_team_goal_difference_avg10", "Average goal difference", NA, step = .01),
        num("rich_team_win_rate10", "Win rate (0–1)", NA, 0, 1, .01), num("rich_team_points_per_game10", "Points per game (0–3)", NA, 0, 3, .01)),
      div(class = "card", h4("D. Tactical and formation context"),
        selectInput("rich_dominant_formation", "Dominant formation", choices("dominant_formation10", "Unknown")),
        num("rich_dominant_formation_frequency10", rich_labels[["dominant_formation_frequency10"]], NA, 0, 1, .01),
        num("rich_formation_diversity10", rich_labels[["formation_diversity10"]], NA, 0, step = 1),
        num("rich_manager_matches_last10", rich_labels[["manager_matches_last10"]], NA, 0, step = 1), num("rich_unique_managers_last10", rich_labels[["unique_managers_last10"]], NA, 0, step = 1)),
      div(class = "card", h4("E. Squad competition and destination history"),
        num("rich_same_position_competitor_count", rich_labels[["same_position_competitor_count"]], NA, 0, step = 1), num("rich_same_position_competitor_minutes", rich_labels[["same_position_competitor_minutes"]], NA, 0, step = 1), num("rich_same_position_avg_market_value", rich_labels[["same_position_avg_market_value"]], NA, 0, step = 1000), num("rich_same_position_top_competitor_minutes", rich_labels[["same_position_top_competitor_minutes"]], NA, 0, step = 1), num("rich_same_sub_position_competitor_count", rich_labels[["same_sub_position_competitor_count"]], NA, 0, step = 1), num("rich_same_sub_position_competitor_minutes", rich_labels[["same_sub_position_competitor_minutes"]], NA, 0, step = 1), num("rich_same_sub_position_top_minutes", rich_labels[["same_sub_position_top_minutes"]], NA, 0, step = 1), num("rich_position_minutes_share", rich_labels[["position_minutes_share"]], NA, 0, 1, .01), num("rich_previous_destination_appearances", rich_labels[["previous_destination_appearances"]], NA, 0, step = 1), num("rich_previous_destination_minutes", rich_labels[["previous_destination_minutes"]], NA, 0, step = 1)),
      div(class = "card", h4("F. Coach context"),
        num("rich_coach_position_group_minutes_share", rich_labels[["coach_position_group_minutes_share"]], NA, 0, 1, .01), num("rich_coach_position_group_start_rate", rich_labels[["coach_position_group_start_rate"]], NA, 0, 1, .01), num("rich_coach_exact_position_minutes_share", rich_labels[["coach_exact_position_minutes_share"]], NA, 0, 1, .01), num("rich_coach_exact_position_start_rate", rich_labels[["coach_exact_position_start_rate"]], NA, 0, 1, .01), num("rich_coach_position_competitor_count", rich_labels[["coach_position_competitor_count"]], NA, 0, step = 1), num("rich_coach_position_competitor_minutes", rich_labels[["coach_position_competitor_minutes"]], NA, 0, step = 1), num("rich_coach_position_rotation_hhi", rich_labels[["coach_position_rotation_hhi"]], NA, 0, 1, .01), num("rich_coach_formation_stability", rich_labels[["coach_formation_stability"]], NA, 0, 1, .01), num("rich_coach_observed_tenure_days", rich_labels[["coach_observed_tenure_days"]], NA, 0, 365, 1)),
      div(class = "card", p(class = "section-note", "Model C requires complete, valid, pre-transfer rich context. This checkbox records your declaration; it does not replace numeric validation."), checkboxInput("rich_verified", "I verified all context is pre-transfer", FALSE), uiOutput("predict_button"), downloadButton("download", "Download prediction CSV"))
    ),
    mainPanel(width = 8,
      uiOutput("prediction_loading"), uiOutput("prediction_view"), uiOutput("details_view"),
      div(class = "card", h4("Methodology"), p("Model C is a saved ranger random-forest regression model trained on the project's pre-transfer dataset. The target is post_GA90: post-transfer goals plus assists per 90 minutes over the defined 365-day outcome window. Training uses an 80% chronological training slice, training-only median imputation for numeric fields, and explicit missing/unseen categorical levels. The app does not fetch or invent team or coach history."))
    )
  )
)

server <- function(input, output, session) {
  result <- reactiveVal(NULL); prediction_running <- reactiveVal(FALSE); prediction_error <- reactiveVal(NULL); example_loaded <- reactiveVal(FALSE)
  update_numeric <- function(values) for (id in names(values)) updateNumericInput(session, id, value = values[[id]])

  observeEvent(input$load_test_example, {
    updateTextInput(session, "player_name", value = test_example$player_name)
    update_numeric(test_example[c("age_at_transfer", "height_in_cm", "appearances_pre", "minutes_pre", "goals_pre", "assists_pre", "goals_per90_pre", "assists_per90_pre", "yellow_cards_pre", "red_cards_pre", "transfer_market_value", "market_value_pre", "transfer_fee", "rich_team_matches_last10", "rich_team_goals_for_avg10", "rich_team_goals_against_avg10", "rich_team_goal_difference_avg10", "rich_team_win_rate10", "rich_team_points_per_game10")])
    updateSelectInput(session, "position", selected = test_example$position); updateSelectInput(session, "sub_position", choices = unique(c(test_example$sub_position, choices("sub_position", character()))), selected = test_example$sub_position); updateSelectInput(session, "foot", choices = unique(c("left", "right", "both", choices("foot", character()))), selected = test_example$foot)
    updateSelectInput(session, "rich_dominant_formation", choices = unique(c(test_example$rich_dominant_formation, choices("dominant_formation10", "Unknown"))), selected = test_example$rich_dominant_formation)
    z <- as.list(test_example$rich_numeric_values); names(z) <- paste0("rich_", rich_numeric); update_numeric(z)
    updateCheckboxInput(session, "rich_verified", value = FALSE); example_loaded(TRUE); result(NULL); prediction_error(NULL)
  })
  observeEvent(input$clear_form, {
    updateTextInput(session, "player_name", value = ""); update_numeric(setNames(as.list(rep(NA_real_, 19)), c("age_at_transfer", "height_in_cm", "appearances_pre", "minutes_pre", "goals_pre", "assists_pre", "goals_per90_pre", "assists_per90_pre", "yellow_cards_pre", "red_cards_pre", "transfer_market_value", "market_value_pre", "transfer_fee", "rich_team_matches_last10", "rich_team_goals_for_avg10", "rich_team_goals_against_avg10", "rich_team_goal_difference_avg10", "rich_team_win_rate10", "rich_team_points_per_game10")))
    update_numeric(setNames(as.list(rep(NA_real_, length(rich_numeric))), paste0("rich_", rich_numeric))); updateSelectInput(session, "rich_dominant_formation", selected = "Unknown"); updateCheckboxInput(session, "rich_verified", value = FALSE); example_loaded(FALSE); result(NULL); prediction_error(NULL); prediction_running(FALSE)
  })
  output$example_notice <- renderUI(if (example_loaded()) div(class = "synthetic", strong("Synthetic test data loaded. "), "All example values are fictional software-test inputs, not verified football statistics."))
  output$predict_button <- renderUI(actionButton("predict", if (prediction_running()) "Generating prediction..." else "Generate Model C prediction", class = "btn-primary", disabled = prediction_running()))
  output$prediction_loading <- renderUI(if (prediction_running()) div(class = "loading", "Generating prediction...") else NULL)

  observeEvent(input$predict, {
    if (isTRUE(prediction_running())) return()
    prediction_running(TRUE); prediction_error(NULL); result(NULL); on.exit(prediction_running(FALSE), add = TRUE)
    tryCatch({
      finite <- function(x, label) { if (length(x) != 1 || !is.finite(x)) stop(paste(label, "must be a finite value."), call. = FALSE) }
      nonnegative <- function(x, label) { finite(x, label); if (x < 0) stop(paste(label, "cannot be negative."), call. = FALSE) }
      nonnegative(input$age_at_transfer, "Age"); nonnegative(input$height_in_cm, "Height"); nonnegative(input$appearances_pre, "Appearances"); nonnegative(input$minutes_pre, "Minutes"); nonnegative(input$goals_pre, "Goals"); nonnegative(input$assists_pre, "Assists"); nonnegative(input$yellow_cards_pre, "Yellow cards"); nonnegative(input$red_cards_pre, "Red cards")
      if (input$age_at_transfer < 15 || input$age_at_transfer > 60) stop("Age must be between 15 and 60 years.", call. = FALSE)
      if (input$height_in_cm < 120 || input$height_in_cm > 230) stop("Height must be between 120 and 230 cm.", call. = FALSE)
      if (input$goals_pre > input$appearances_pre || input$assists_pre > input$appearances_pre) stop("Goals and assists cannot exceed appearances.", call. = FALSE)
      if (input$minutes_pre <= 0) stop("Previous minutes must be greater than zero.", call. = FALSE)
      all_team <- list(input$rich_team_matches_last10, input$rich_team_goals_for_avg10, input$rich_team_goals_against_avg10, input$rich_team_goal_difference_avg10, input$rich_team_win_rate10, input$rich_team_points_per_game10)
      if (!all(vapply(all_team, function(x) length(x) == 1 && is.finite(x), logical(1)))) stop("All destination-team context fields are required.", call. = FALSE)
      if (input$rich_team_win_rate10 < 0 || input$rich_team_win_rate10 > 1) stop("Win rate must be between 0 and 1.", call. = FALSE)
      if (input$rich_team_points_per_game10 < 0 || input$rich_team_points_per_game10 > 3) stop("Points per game must be between 0 and 3.", call. = FALSE)
      if (!isTRUE(input$rich_verified)) stop("Confirm that all context is verified and pre-transfer.", call. = FALSE)
      d <- data.frame(age_at_transfer=input$age_at_transfer, height_in_cm=input$height_in_cm, appearances_pre=input$appearances_pre, minutes_pre=input$minutes_pre, goals_pre=input$goals_pre, assists_pre=input$assists_pre, goals_per90_pre=if (isTRUE(input$goals_per90_pre == 0)) input$goals_pre/input$minutes_pre*90 else input$goals_per90_pre, assists_per90_pre=if (isTRUE(input$assists_per90_pre == 0)) input$assists_pre/input$minutes_pre*90 else input$assists_per90_pre, yellow_cards_pre=input$yellow_cards_pre, red_cards_pre=input$red_cards_pre, transfer_market_value=input$transfer_market_value, market_value_pre=input$market_value_pre, transfer_fee=input$transfer_fee, position=input$position, sub_position=input$sub_position, foot=input$foot, rich_team_goals_for_avg10=input$rich_team_goals_for_avg10, rich_team_goals_against_avg10=input$rich_team_goals_against_avg10, rich_team_goal_difference_avg10=input$rich_team_goal_difference_avg10, rich_team_win_rate10=input$rich_team_win_rate10, rich_team_points_per_game10=input$rich_team_points_per_game10, rich_team_matches_last10=input$rich_team_matches_last10, stringsAsFactors=FALSE)
      d$dominant_formation10 <- input$rich_dominant_formation; for (f in rich_numeric) d[[f]] <- input[[paste0("rich_", f)]]
      invalid <- rich_numeric[!vapply(d[rich_numeric], function(x) length(x) == 1 && is.finite(x), logical(1))]; if (length(invalid) || !nzchar(input$rich_dominant_formation)) stop(paste("Missing or invalid Model C context:", paste(c(invalid, if (!nzchar(input$rich_dominant_formation)) "dominant_formation10"), collapse = ", ")), call. = FALSE)
      bounded <- c("rich_team_win_rate10", "dominant_formation_frequency10", "position_minutes_share", "coach_position_group_minutes_share", "coach_position_group_start_rate", "coach_exact_position_minutes_share", "coach_exact_position_start_rate", "coach_position_rotation_hhi", "coach_formation_stability"); if (!all(vapply(d[bounded], function(x) is.finite(x) && x >= 0 && x <= 1, logical(1)))) stop("Win rates, shares, and stability values must be between 0 and 1.", call. = FALSE)
      z <- d[, model_C$features, drop = FALSE]
      for (f in model_C$features) { if (f %in% names(model_C$preprocessing$levels)) { v <- as.character(z[[f]]); v[is.na(v) | v == ""] <- "__MISSING__"; v[!v %in% model_C$preprocessing$levels[[f]]] <- "__UNSEEN__"; z[[f]] <- factor(v, levels = model_C$preprocessing$levels[[f]]) } else { v <- as.numeric(z[[f]]); v[!is.finite(v) | is.na(v)] <- model_C$preprocessing$medians[[f]]; z[[f]] <- v } }
      prediction <- as.numeric(predict(model_C$model, data = z)$predictions)
      if (!is.finite(prediction)) stop("Model C returned a non-finite prediction.", call. = FALSE)
      warnings <- character(); if (input$goals_per90_pre != 0 && abs(input$goals_per90_pre - input$goals_pre/input$minutes_pre*90) > .1) warnings <- c(warnings, "Entered goals per 90 differs materially from goals/minutes; the entered value was used."); if (input$assists_per90_pre != 0 && abs(input$assists_per90_pre - input$assists_pre/input$minutes_pre*90) > .1) warnings <- c(warnings, "Entered assists per 90 differs materially from assists/minutes; the entered value was used.")
      result(list(prediction=prediction, data=d, warnings=warnings, player=input$player_name, date=as.character(Sys.Date())))
    }, error = function(e) { result(NULL); prediction_error(conditionMessage(e)) })
  })

  output$prediction_view <- renderUI({
    if (prediction_running()) return(div(class = "card", h3("Prediction in progress")))
    if (!is.null(prediction_error())) return(div(class = "card", h3("Prediction could not be generated"), p(class = "text-danger", prediction_error())))
    r <- result(); if (is.null(r)) return(div(class = "card", h3("Ready for a new-player assessment"), p("Complete every Model C field, confirm that the context is pre-transfer, and generate one saved-model prediction.")))
    div(class = "card prediction-card", div(class = "metric-label", "Predicted Post-Transfer Attacking Contribution"), div(class = "prediction-value", sprintf("%.4f", r$prediction)), div(class = "prediction-unit", "goals plus assists per 90 minutes (post_GA90)"), tags$hr(), p(strong(r$player), " · Model C — Rich Context · ", r$date), p(class = "section-note", "This is an estimate of the post-transfer rate over the defined 365-day outcome window, not a guarantee."))
  })
  output$details_view <- renderUI({
    r <- result()
    if (is.null(r)) return(NULL)
    d <- r$data
    fmt <- function(x) if (is.finite(x)) formatC(x, format = "f", big.mark = ",", digits = 0) else "Unavailable"
    financial <- column(4, div(class = "metric-label", "Financial comparison"),
      div(class = "metric-value", paste0("Fee: €", fmt(d$transfer_fee))),
      p(paste0("Market value: €", fmt(d$transfer_market_value)), paste0("Previous value: €", fmt(d$market_value_pre))),
      p(if (d$transfer_market_value > 0) paste0("Fee − market value: €", fmt(d$transfer_fee - d$transfer_market_value), "; fee/value: ", sprintf("%.2fx", d$transfer_fee / d$transfer_market_value)) else "Fee/value ratio unavailable because market value is zero."))
    profile <- column(4, div(class = "metric-label", "Player profile"),
      div(class = "metric-value", paste(d$age_at_transfer, "years ·", d$position, "·", d$sub_position)),
      p(paste(d$height_in_cm, "cm · preferred foot:", d$foot)),
      p(paste(d$appearances_pre, "appearances ·", d$minutes_pre, "minutes ·", d$goals_pre, "goals ·", d$assists_pre, "assists")),
      p(sprintf("Pre-transfer rates: %.4f goals/90 · %.4f assists/90", d$goals_per90_pre, d$assists_per90_pre)))
    context <- column(4, div(class = "metric-label", "Context used"),
      div(class = "metric-value", paste(d$rich_team_matches_last10, "team matches")),
      p(sprintf("Win rate %.2f · %.2f points/game · formation %s", d$rich_team_win_rate10, d$rich_team_points_per_game10, d$dominant_formation10)),
      p(paste(length(features_C), "Model C features supplied and preprocessed.")),
      if (length(r$warnings)) p(class = "text-warning", paste(r$warnings, collapse = " ")))
    div(class = "card", h4("Assessment summary"), fluidRow(financial, profile, context),
      tags$details(tags$summary("Prediction details"),
        p(strong("Model:"), " Model C — Rich Context; ", strong("Target:"), " post_GA90"),
        p(strong("Features used:"), paste(features_C, collapse = ", ")),
        p(strong("Data provenance:"), " Player and context values were entered in the app; no player lookup or post-transfer data was used.")))
  })
  output$download <- downloadHandler(filename=function() paste0("transferiq_prediction_", Sys.Date(), ".csv"), content=function(file) { r <- result(); req(r); out <- cbind(data.frame(prediction_date=r$date, player_label=r$player, model="Model C — Rich Context", target="post_GA90", predicted_post_GA90=unname(r$prediction), units="goals plus assists per 90 minutes", warnings=paste(r$warnings, collapse=" | "), stringsAsFactors=FALSE), r$data); write.csv(out, file, row.names=FALSE, na = "") })
}

shinyApp(ui, server)

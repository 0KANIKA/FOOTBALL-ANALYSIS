# Football Transfer Performance Prediction

This repository investigates whether a player's post-transfer attacking performance can be predicted from pre-transfer player statistics, destination-team context, tactical information, squad competition, and coach-related context. It contains the original A/B/C research workflow, the corrected audit workflow, feature builders, results, and a Shiny application for assessing a genuinely new player.

The main application target is `post_GA90`: post-transfer goals plus assists per 90 minutes over the defined 365-day outcome window. It is a rate estimate, not total goals, a probability, or a guarantee.

## Objectives

- Build leakage-aware pre-transfer player features.
- Add destination-team form, formation/manager context, squad competition, and coach context.
- Compare player-only, team/tactical, and richer Model A/B/C specifications.
- Evaluate models with chronological splits and preserve the original experiments separately from the corrected audit.
- Provide a new-player prediction workflow in `new_player_transfer_app/`.

## Repository structure

| Path | Purpose |
| --- | --- |
| `build_mode_a_features.py` | Builds leakage-safe pre-transfer Mode A features and window diagnostics. |
| `build_rich_team_context.py` | Builds destination-team, tactical, and squad-competition context. |
| `archive/build_coach_tactical_context_v21.py` | Builds v21 coach/tactical context from raw match, lineup, and appearance data. |
| `run_audit_corrected.py` | Runs the corrected chronological audit across core, Mode A, rich, and coach feature groups. |
| `02_train_models_ABC.py`, `03_model_analysis.py`, `04_visualizations.py` | Original A/B/C training, analysis, and visualization workflow. |
| `run_model_a_new.py`, `run_model_b_new.py`, `run_model_c_new.py`, `run_model_c_rich.py` | Newer model variants and application-oriented experiments. |
| `new_player_transfer_app/app.R` | Shiny UI and Model C prediction workflow. |
| `new_player_transfer_app/train_models.R` | Re-trains the app's Model A/B/C artifacts from the final clean dataset. |
| `new_player_transfer_app/test_app.R` | Lightweight app/model integration test. |
| `predict_player.py` | Loads the saved Python Model A-new pipeline for a prediction. |
| `requirements.txt` | Python package requirements for the scripts. |
| `FINAL_RESEARCH_RESULTS/`, `model_abc_results/`, `outputs_corrected/` | Existing research and audit outputs; the experiments are not interchangeable. |

The existing literature, result files, and tracked source datasets are retained. Large historical files already present in Git history were not deleted or rewritten.

## Datasets

| Dataset | Approx. size / rows | Purpose | Main consumers | Type |
| --- | ---: | --- | --- | --- |
| `post_transfer_performance.csv` | 3.8 MB / 16,862 | Transfer cohort with pre-transfer player fields and 365-day outcomes, including `post_GA90`, `post_minutes`, `complete_365_day_window`, and `eligible_450min`. | `build_mode_a_features.py`, `run_audit_corrected.py`, `build_performance_analysis.py` | Derived target/cohort |
| `professional_transfer_dataset.csv` | 2.8 MB / 16,862 | Professional-player cohort with pre-transfer player and transfer economics fields. | `build_rich_team_context.py`, coach builder, model scripts | Derived source cohort |
| `rich_team_context_dataset.csv` | 5.0 MB / 16,862 | Destination-team form, formation/manager context, position-group and sub-position squad competition, and destination history. | `run_audit_corrected.py`, `clean_final_model_c.py`, rich model scripts | Derived feature table |
| `coach_tactical_context_v21.csv` | 1.3 MB / 16,862 | Coach-known, position usage, rotation, tenure, and formation-stability features keyed by `transfer_row_id`. | `run_audit_corrected.py`, `clean_final_model_c.py` | Derived feature table |
| `outputs_mode_a_features/mode_a_pretransfer_features.csv` | 2.8 MB / 16,862 | Strictly pre-transfer appearance, recent-form, and valuation features. | Mode A inspection and downstream joins | Derived feature table |
| `FINAL_MODEL_C_CLEAN.csv` | 3.0 MB / 5,969 | Final clean Model C cohort with target, rich context, coach features, and chronological split labels. | `02_train_models_ABC.py`, app training | Derived final modeling table |
| `preliminary_real_ml_dataset.csv` | 21.4 MB / 174,665 | Earlier transfer cohort used to form the professional subset. | `filter_professional_transfer_dataset.py` | Earlier derived table |
| `archive/appearances.csv` | ~142 MB | Player appearances, goals, assists, minutes, cards, and club IDs. | Mode A, rich context, performance builders | Original source |
| `archive/games.csv` | ~23.9 MB | Match dates, seasons, teams, managers, formations, and scores. | Rich team and coach/tactical builders | Original source |
| `archive/game_lineups.csv` | ~336 MB | Lineup types used to identify starts and tactical participation. | Mode A and coach/tactical builders | Original source |
| `archive/player_valuations.csv` | ~30.5 MB | Time-stamped market valuations. | Mode A, rich context, performance builders | Original source |
| `archive/transfers.csv`, `archive/players.csv`, `archive/clubs.csv` | varies | Transfer, player, and club reference data used by broader rebuild scripts. | `archive/build_post_transfer.py`, inspection scripts | Original sources |

`transfer_row_id` is the join key used by the corrected audit. Core predictors include age, position, pre-transfer minutes and attacking rates, market values, and transfer fee. Rich features include destination-team form, formations, managers, and squad competition. Coach features remain missing when unavailable; they are not silently converted to zero.

## Requirements

Use Python 3.10 or newer:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`xgboost` is optional for `run_model_a_new.py`; the script has a fallback path when it is not installed.

The Shiny application uses R with `shiny` and `ranger`. Legacy R experiments additionally use `dplyr`, `tidyr`, `readr`, and `ggplot2` where applicable:

```powershell
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" -e "install.packages(c('shiny','ranger','dplyr','tidyr','readr','ggplot2'), repos='https://cloud.r-project.org')"
```

Replace the executable path if R is installed elsewhere. Git and a web browser are needed to clone the repository and launch Shiny.

## Windows installation and workflow

PowerShell:

```powershell
git clone https://github.com/0KANIKA/FOOTBALL-ANALYSIS.git
Set-Location FOOTBALL-ANALYSIS
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

From the repository root, build the context tables and run the corrected audit:

```powershell
python .\build_rich_team_context.py
python .\archive\build_coach_tactical_context_v21.py
python .\build_mode_a_features.py --project-dir (Get-Location).Path
python .\run_audit_corrected.py
```

The builders write to `rich_team_context_dataset.csv`, `coach_tactical_context_v21.csv`, and `outputs_mode_a_features/`; the audit writes diagnostics to `outputs_corrected/`. The checked-in `FINAL_MODEL_C_CLEAN.csv` is provided so the app and existing research can run without replaying every historical merge stage.

Original A/B/C research scripts:

```powershell
python .\02_train_models_ABC.py
python .\03_model_analysis.py
python .\04_visualizations.py
```

Train the Shiny model artifacts:

```powershell
Set-Location .\new_player_transfer_app
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" .\train_models.R
```

The optional Python `predict_player.py` helper expects `model_a_new_results/random_forest_pipeline.joblib`; create that local artifact first with `python ..\run_model_a_new.py`. It is intentionally ignored because it is a generated model file rather than a required input to the Shiny application.

CMD equivalents:

```cmd
git clone https://github.com/0KANIKA/FOOTBALL-ANALYSIS.git
cd /d FOOTBALL-ANALYSIS
py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

## Running the application

From `new_player_transfer_app/`:

```powershell
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" -e "shiny::runApp('.', launch.browser=TRUE)"
```

The form accepts a new player's profile and pre-transfer statistics, transfer economics, destination-team form, tactical/formation context, squad competition, destination history, and coach context. Complete the fields, confirm that the context is genuinely pre-transfer, select **Generate Model C prediction**, and use **Download prediction CSV** to export the result. The saved `new_player_transfer_app/models/model_C.rds` artifact is included for immediate startup; `train_models.R` can regenerate it.

`Load Test Example` uses fictional values for software testing only. The app does not fetch or invent verified football data, does not provide confidence intervals, and cannot establish causality or guarantee transfer success.

## Research methodology and results

The original A/B/C experiment, corrected audit experiment, and newer app workflow are separate analyses:

- The original A/B/C scripts compare progressively richer feature sets using `model_abc_results/` and `FINAL_RESEARCH_RESULTS/`.
- `run_audit_corrected.py` performs a separate leakage-conscious audit using the target, rich-context, and coach-context tables, with chronological train/validation/test boundaries. Its metrics are in `outputs_corrected/`.
- The app's Model C is trained by `new_player_transfer_app/train_models.R` on `FINAL_MODEL_C_CLEAN.csv`, using an 80% chronological training slice and a final 20% holdout. The checked-in app metric is MAE 0.1239, RMSE 0.1721, and R² 0.5528 on 1,194 holdout rows, recorded in `new_player_transfer_app/models/model_C_metrics.csv`.

These metrics come from different cohorts, feature definitions, and scripts. They must not be merged into one headline result.

## Reproducibility and limitations

- Pre-transfer builders use dates and windows to reduce temporal leakage, but audit outputs should be inspected before making causal claims.
- `post_GA90` is defined by the project's 365-day outcome and eligibility conventions; changing the window changes the target.
- Raw source tables are large and may contain missing, ambiguous, or provider-specific records.
- The application relies on user-supplied rich context for a new player and does not automatically calculate team or coach history from external services.
- Saved RDS and CSV results are artifacts for this project snapshot, not a substitute for retraining after data or code changes.

## Troubleshooting

- **Missing CSV:** run commands from the repository root and confirm the documented relative path. Set `FOOTBALL_DATA_ROOT` for `run_audit_corrected.py` when data lives elsewhere.
- **Missing Python package:** activate `.venv`, then run `python -m pip install -r requirements.txt`.
- **Missing R package:** run the `install.packages(...)` command with the same R executable used to start the app.
- **Missing app model:** from `new_player_transfer_app/`, run `train_models.R`; it reads `..\FINAL_MODEL_C_CLEAN.csv` and writes under `models/`.
- **Shiny startup error:** verify the working directory is `new_player_transfer_app/`, `shiny` and `ranger` are installed, and the R executable path is correct.
- **PowerShell activation blocked:** run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate `.venv` again.
- **Large-file warning:** do not add new raw exports or generated artifacts blindly. Future source snapshots above GitHub's normal file limit should use Git LFS or an approved external data location.

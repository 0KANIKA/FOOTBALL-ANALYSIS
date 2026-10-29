# Predictive Football Analytics Using Machine Learning

This project studies post-transfer football performance using player statistics, destination-team form, tactical and managerial context, squad competition, and coach-related context. It contains the original Model A/B/C research experiments, a separate corrected audit workflow, an updated R workflow, and an R Shiny dashboard for estimating a genuinely new player's post-transfer attacking contribution.

The main target is post_GA90: post-transfer goals plus assists per 90 minutes over the defined 365-day outcome window. It is an observed-rate estimate, not a transfer-success probability, market-value forecast, or guarantee.

## Folder structure

~~~text
01_Final_Report/
02_Presentation_and_Abstract/            presentation PDF; abstract intentionally empty
03_Source_Code/python/                   Python builders, experiments, audits, and tests
03_Source_Code/r/                        R research workflows
03_Source_Code/new_player_transfer_app/  Shiny app, training code, tests, and RDS models
04_Datasets/derived/                     required derived and modeling datasets
04_Datasets/raw_transfer_data/            raw transfer data and excluded external archives
05_Literature_and_References/             literature review and research papers
06_Results_and_Demo_Evidence/            metrics, audits, figures, and test outputs
07_Installation_and_User_Guide/           installation guide and dashboard manual
~~~

See 07_Installation_and_User_Guide/INSTALLATION_GUIDE.md and USER_MANUAL.md for detailed instructions.

## Separate workflows

1. Original Model A/B/C research: 01_final_dataset_check.py through 07_final_results_summary.py.
2. Corrected audit: run_audit_corrected.py, using transfer_row_id to merge target, rich-team, and coach-context tables.
3. Updated R workflow: run_updated_models.R, using target/context tables and Mode A features.
4. New-player dashboard: 03_Source_Code/new_player_transfer_app/, loading or training the ranger Model C artifact.

These workflows use different cohorts, feature definitions, and evaluation conventions. Their metrics must not be merged into one result.

## Important datasets

| Dataset | Location | Role |
|---|---|---|
| post_transfer_performance.csv | 04_Datasets/derived/ | Transfer cohort and 365-day outcomes including post_GA90. |
| professional_transfer_dataset.csv | 04_Datasets/derived/ | Professional-player transfer cohort. |
| preliminary_real_ml_dataset.csv | 04_Datasets/derived/ | Earlier cohort used to form the professional subset. |
| rich_team_context_dataset.csv | 04_Datasets/derived/ | Destination-team, tactical, manager, and squad context. |
| coach_tactical_context_v21.csv | 04_Datasets/derived/ | Coach usage, rotation, formation stability, and tenure features. |
| FINAL_MODEL_C_CLEAN.csv | 04_Datasets/derived/ | Final clean Model C training cohort. |
| mode_a_pretransfer_features.csv | 04_Datasets/derived/outputs_mode_a_features/ | Leakage-aware pre-transfer features. |
| raw_transfer_data/archive/*.csv | 04_Datasets/raw_transfer_data/archive/ | Raw appearances, games, lineups, valuations, transfers, players, and clubs. |

The application target is post_GA90. transfer_row_id is the context-join key. Dataset schemas and values were preserved.

## Prerequisites and installation

Use Windows, Python 3.10 or newer, R 4.6.1 or compatible, Git, and a web browser. Python dependencies are in 03_Source_Code/requirements.txt. The app requires shiny and ranger; the updated R workflow additionally uses readr and dplyr.

~~~powershell
py -3 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\\03_Source_Code\\requirements.txt
& "C:\\Program Files\\R\\R-4.6.1\\bin\\Rscript.exe" -e "install.packages(c('shiny','ranger','readr','dplyr','tidyr','ggplot2'), repos='https://cloud.r-project.org')"
~~~

## Build features and run research

From 03_Source_Code/python/:

~~~powershell
Set-Location .\\03_Source_Code\\python
python .\\build_rich_team_context.py
python .\\build_coach_tactical_context_v21.py
python .\\build_mode_a_features.py --project-dir (Resolve-Path ..\\..).Path
python .\\run_audit_corrected.py
~~~

Run the original experiment scripts when a full regeneration is intended:

~~~powershell
python .\\01_final_dataset_check.py
python .\\02_train_models_ABC.py
python .\\03_model_analysis.py
python .\\04_visualizations.py
python .\\05_sensitivity_analysis.py
python .\\06_statistical_robustness.py
python .\\07_final_results_summary.py
~~~

## Launch the dashboard

PowerShell:

~~~powershell
Set-Location .\\03_Source_Code\\new_player_transfer_app
& "C:\\Program Files\\R\\R-4.6.1\\bin\\Rscript.exe" -e "shiny::runApp('.', launch.browser=TRUE)"
~~~

CMD:

~~~cmd
cd /d "FOOTBALL-ANALYSIS\\03_Source_Code\\new_player_transfer_app"
"C:\\Program Files\\R\\R-4.6.1\\bin\\Rscript.exe" -e "shiny::runApp('.', launch.browser=TRUE)"
~~~

Keep the terminal open while Shiny runs. Stop it with Ctrl+C. The app loads models/model_C.rds; if it is missing, run train_models.R from the app directory. The script reads 04_Datasets/derived/FINAL_MODEL_C_CLEAN.csv.

Enter a new player's profile, pre-transfer performance, transfer economics, destination-team form, tactical context, squad competition, destination history, and coach context. Select Generate Model C prediction, then use Download prediction CSV. The test example is fictional only.

## Submission, data, and limitations

The final report is in 01_Final_Report/. The presentation PDF is in 02_Presentation_and_Abstract/. The abstract directory is intentionally empty. No IEEE research paper was created because no source file was supplied. Existing papers and literature-review files are in 05_Literature_and_References/.

Chronological splits and pre-transfer windows reduce leakage, but the work is predictive rather than causal. Dashboard predictions depend on the supplied context and do not forecast transfer fee, market value, or guaranteed success.

The 15.4 GB open-data-master directory and large ZIP archives are excluded from Git staging; see 04_Datasets/external_data_notes.md.

## Validation and troubleshooting

Run the app test from the app directory:

~~~powershell
& "C:\\Program Files\\R\\R-4.6.1\\bin\\Rscript.exe" .\\test_app.R
~~~

If Python packages are missing, activate .venv and reinstall 03_Source_Code/requirements.txt. If R packages are missing, install them using the full Rscript path. If the app cannot find data, verify 04_Datasets/derived/FINAL_MODEL_C_CLEAN.csv and run from 03_Source_Code/new_player_transfer_app/.

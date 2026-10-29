# Installation and execution guide

## Clone and install

PowerShell:

~~~powershell
git clone https://github.com/0KANIKA/FOOTBALL-ANALYSIS.git
Set-Location .\FOOTBALL-ANALYSIS
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\03_Source_Code\requirements.txt
~~~

CMD:

~~~cmd
git clone https://github.com/0KANIKA/FOOTBALL-ANALYSIS.git
cd /d FOOTBALL-ANALYSIS
py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r .\03_Source_Code\requirements.txt
~~~

If PowerShell blocks activation, run Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass.

## R packages

PowerShell:

~~~powershell
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" -e "install.packages(c('shiny','ranger','readr','dplyr','tidyr','ggplot2'), repos='https://cloud.r-project.org')"
~~~

CMD:

~~~cmd
"C:\Program Files\R\R-4.6.1\bin\Rscript.exe" -e "install.packages(c('shiny','ranger','readr','dplyr','tidyr','ggplot2'), repos='https://cloud.r-project.org')"
~~~

## Expected inputs

The dashboard requires 04_Datasets/derived/FINAL_MODEL_C_CLEAN.csv and 03_Source_Code/new_player_transfer_app/models/model_C.rds. Feature builders use the derived transfer/context tables and raw files under 04_Datasets/raw_transfer_data/archive/. Large external data is documented in 04_Datasets/external_data_notes.md.

## Build features and run the audit

From PowerShell:

~~~powershell
Set-Location .\03_Source_Code\python
python .\build_rich_team_context.py
python .\build_coach_tactical_context_v21.py
python .\build_mode_a_features.py --project-dir (Resolve-Path ..\..).Path
python .\run_audit_corrected.py
~~~

Derived tables are written under 04_Datasets/derived/ and corrected-audit outputs under 06_Results_and_Demo_Evidence/audit_results/.

## Train the dashboard models

~~~powershell
Set-Location ..\new_player_transfer_app
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" .\train_models.R
~~~

## Launch the dashboard

PowerShell:

~~~powershell
Set-Location .\03_Source_Code\new_player_transfer_app
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" -e "shiny::runApp('.', launch.browser=TRUE)"
~~~

CMD:

~~~cmd
cd /d "FOOTBALL-ANALYSIS\03_Source_Code\new_player_transfer_app"
"C:\Program Files\R\R-4.6.1\bin\Rscript.exe" -e "shiny::runApp('.', launch.browser=TRUE)"
~~~

Keep the terminal open while Shiny runs. Stop it intentionally with Ctrl+C.

## Test

~~~powershell
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" .\test_app.R
~~~

Python syntax validation:

~~~powershell
Get-ChildItem .\03_Source_Code\python -File -Filter *.py | ForEach-Object { python -m py_compile $_.FullName }
~~~

Some historical scripts are optional experiments and may require additional packages or external data. Do not regenerate historical result tables unless you intend to rerun that experiment.

## Troubleshooting

- Rscript not found: use the full executable path above.
- Missing shiny or ranger: install them with the R command above.
- Missing FINAL_MODEL_C_CLEAN.csv: verify 04_Datasets/derived/ and the working directory.
- Model C error: run train_models.R from the app directory, then rerun test_app.R.
- External-data error: review 04_Datasets/external_data_notes.md.

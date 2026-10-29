@echo off
cd /d "%~dp0\..\python"
python -c "import pandas as pd; f='..\\..\\04_Datasets\\raw_transfer_data\\archive\\game_events.csv'; print(pd.read_csv(f,nrows=5).columns.tolist())"

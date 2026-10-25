cd /d "C:\Users\rsuba\Downloads\R football analytics"
python -c "import pandas as pd; f='archive/game_events.csv'; print(pd.read_csv(f,nrows=5).columns.tolist())"
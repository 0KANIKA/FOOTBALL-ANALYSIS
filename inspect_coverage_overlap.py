from pathlib import Path
import json
import csv
from collections import Counter, defaultdict

BASE = Path(r"C:\Users\rsuba\Downloads\R football analytics")
STATSBOMB = BASE / "open-data-master" / "open-data-master" / "data"
TM = BASE / "archive"

def read_csv_rows(path, wanted=None):
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if wanted is None:
                yield row
            else:
                yield {k: row.get(k, "") for k in wanted}

def load_json(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)

print("="*90)
print("FOOTBALL DATASET OVERLAP / COVERAGE INSPECTION")
print("="*90)

# ============================================================
# 1. STATSBOMB COVERAGE
# ============================================================
sb_comp_path = STATSBOMB / "competitions.json"
competitions = load_json(sb_comp_path)

sb_comp_names = {}
for c in competitions:
    sb_comp_names[c.get("competition_id")] = c.get("competition_name", "")

print("\nSTATSBOMB COMPETITIONS")
print("-"*90)
for c in competitions:
    print(f"ID={c.get('competition_id'):>4} | {c.get('competition_name')} | country={c.get('country_name')}")

# StatsBomb match files are arranged as matches/<competition_id>/<season_id>.json
sb_match_files = list((STATSBOMB / "matches").rglob("*.json"))

sb_matches = []
for path in sb_match_files:
    try:
        data = load_json(path)
        if not isinstance(data, list):
            continue
        for m in data:
            sb_matches.append({
                "competition_id": m.get("competition", {}).get("competition_id"),
                "competition_name": m.get("competition", {}).get("competition_name"),
                "season_id": m.get("season", {}).get("season_id"),
                "season_name": m.get("season", {}).get("season_name"),
                "match_id": m.get("match_id"),
                "date": m.get("match_date"),
                "home": m.get("home_team", {}).get("home_team_name"),
                "away": m.get("away_team", {}).get("away_team_name"),
            })
    except Exception as e:
        print("Could not read:", path, "|", e)

print("\nSTATSBOMB MATCH COVERAGE")
print("-"*90)
print("Total matches:", len(sb_matches))

sb_by_comp = Counter((m["competition_name"], m["season_name"]) for m in sb_matches)
for (comp, season), n in sorted(sb_by_comp.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
    print(f"{comp} | {season} | {n} matches")

# Unique teams
sb_teams = Counter()
for m in sb_matches:
    if m["home"]: sb_teams[m["home"]] += 1
    if m["away"]: sb_teams[m["away"]] += 1

print("\nSTATSBOMB UNIQUE TEAMS:", len(sb_teams))

# ============================================================
# 2. TRANSFERMARKT COVERAGE
# ============================================================
print("\n" + "="*90)
print("TRANSFERMARKT COVERAGE")
print("="*90)

tm_comp_path = TM / "competitions.csv"
tm_comps = list(read_csv_rows(tm_comp_path))
tm_comp_name = {r["competition_id"]: r["name"] for r in tm_comps}

tm_games = list(read_csv_rows(TM / "games.csv",
                              ["game_id","competition_id","season","date","home_club_id",
                               "away_club_id","home_club_name","away_club_name"]))

tm_games_by_comp_season = Counter(
    (tm_comp_name.get(g["competition_id"], g["competition_id"]), g["season"])
    for g in tm_games
)

print("\nTRANSFERMARKT GAME COVERAGE")
print("-"*90)
for (comp, season), n in sorted(tm_games_by_comp_season.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
    print(f"{comp} | {season} | {n} games")

# Transfer years
transfer_rows = list(read_csv_rows(TM / "transfers.csv",
                                   ["player_id","transfer_date","transfer_season",
                                    "from_club_id","to_club_id","from_club_name",
                                    "to_club_name","transfer_fee","market_value_in_eur",
                                    "player_name"]))

transfer_years = Counter()
for r in transfer_rows:
    date = r["transfer_date"]
    if date:
        transfer_years[date[:4]] += 1

print("\nTRANSFERS BY YEAR")
print("-"*90)
for year, n in sorted(transfer_years.items()):
    print(year, n)

# ============================================================
# 3. COMMON COMPETITIONS / SEASONS
# ============================================================
print("\n" + "="*90)
print("PRELIMINARY OVERLAP")
print("="*90)

sb_comp_seasons = set(
    (m["competition_name"], m["season_name"])
    for m in sb_matches
)

tm_comp_seasons = set(
    (tm_comp_name.get(g["competition_id"], g["competition_id"]), g["season"])
    for g in tm_games
)

# Exact text overlap may fail because provider naming differs.
exact_overlap = sb_comp_seasons & tm_comp_seasons

print("\nExact competition-season overlaps:")
if exact_overlap:
    for x in sorted(exact_overlap):
        print(x)
else:
    print("None by exact name/season text. This does NOT mean there is no overlap; provider names may differ.")

# Print likely major-league StatsBomb seasons separately
print("\nStatsBomb competition names available:")
for comp in sorted(set(m["competition_name"] for m in sb_matches if m["competition_name"])):
    print(" -", comp)

# ============================================================
# 4. TEAM NAME SAMPLES FOR MAPPING
# ============================================================
print("\n" + "="*90)
print("TEAM MAPPING PREVIEW")
print("="*90)

tm_clubs = list(read_csv_rows(TM / "clubs.csv", ["club_id","name","domestic_competition_id"]))

print("\nStatsBomb teams (first 100):")
for x in sorted(sb_teams)[:100]:
    print("SB:", x)

print("\nTransfermarkt clubs (first 100):")
for r in tm_clubs[:100]:
    print("TM:", r["club_id"], "|", r["name"], "|", r["domestic_competition_id"])

# ============================================================
# 5. SAVE A SMALL MACHINE-READABLE COVERAGE FILE
# ============================================================
out = BASE / "coverage_summary.csv"
with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source","competition","season","matches_or_games"])
    for (comp, season), n in sorted(sb_by_comp.items(), key=lambda x: str(x[0])):
        w.writerow(["StatsBomb", comp, season, n])
    for (comp, season), n in sorted(tm_games_by_comp_season.items(), key=lambda x: str(x[0])):
        w.writerow(["Transfermarkt", comp, season, n])

print("\nSaved:", out)
print("\nIMPORTANT: This script only inspects coverage. It does NOT alter your raw datasets.")

from pathlib import Path
import json
import csv
from collections import Counter

STATSBOMB = Path(r"C:\Users\rsuba\Downloads\R football analytics\open-data-master\open-data-master\data")
TRANSFERMARKT = Path(r"C:\Users\rsuba\Downloads\R football analytics\archive")
REPORT_FILE = Path(r"C:\Users\rsuba\Downloads\R football analytics\dataset_inspection_report.txt")

def human(n):
    if n >= 1_000_000: return f"{n:,} ({n/1_000_000:.2f}M)"
    if n >= 1_000: return f"{n:,} ({n/1_000:.2f}K)"
    return f"{n:,}"

def csv_info(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        sample = f.read(100000)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, [])
        samples = [next(reader, []) for _ in range(3)]
    with open(path, "rb") as f:
        rows = sum(block.count(b"\n") for block in iter(lambda: f.read(8*1024*1024), b""))
    return header, samples, max(0, rows-1)

def json_info(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        if isinstance(data, list):
            return type(data).__name__, len(data), list(data[0].keys()) if data and isinstance(data[0], dict) else []
        return type(data).__name__, None, list(data.keys()) if isinstance(data, dict) else []
    except Exception as e:
        return "ERROR", None, [repr(e)]

lines = []
lines += ["FOOTBALL ANALYTICS DATASET INSPECTION", "="*80]
lines += [f"StatsBomb: {STATSBOMB}", f"Transfermarkt: {TRANSFERMARKT}"]

for label, p in [("StatsBomb", STATSBOMB), ("Transfermarkt", TRANSFERMARKT)]:
    lines += ["", f"{label} exists: {p.exists()} | directory: {p.is_dir()}"]

# Transfermarkt
lines += ["", "="*80, "TRANSFERMARKT / CSV FILES", "="*80]
csv_files = sorted(TRANSFERMARKT.rglob("*.csv"))
priority = ["players.csv","transfers.csv","clubs.csv","games.csv","appearances.csv",
            "game_events.csv","game_lineups.csv","club_games.csv","competitions.csv",
            "countries.csv","national_teams.csv","player_valuations.csv"]
ordered = [p for name in priority for p in csv_files if p.name.lower() == name.lower()]
ordered += [p for p in csv_files if p not in ordered]

for p in ordered:
    try:
        header, samples, rows = csv_info(p)
        lines += [f"\nFILE: {p.name}",
                  f"SIZE: {p.stat().st_size/(1024*1024):.2f} MB",
                  f"ROWS: {human(rows)}",
                  f"COLUMNS ({len(header)}): {', '.join(header)}",
                  "SAMPLE: " + " | ".join(str(x)[:80] for x in (samples[0] if samples else []))]
    except Exception as e:
        lines += [f"\nFILE: {p.name}", f"ERROR: {e}"]

# StatsBomb
lines += ["", "="*80, "STATSBOMB / JSON FILES", "="*80]
json_files = sorted(STATSBOMB.rglob("*.json"))
lines.append(f"TOTAL JSON FILES: {len(json_files)}")

counts = Counter(str(p.parent.relative_to(STATSBOMB)).replace("\\","/") for p in json_files)
lines.append("\nFILES BY DIRECTORY:")
for d, n in sorted(counts.items()):
    lines.append(f"  {d}: {n}")

for rel in ["", "matches", "events", "lineups", "three-sixty"]:
    files = [p for p in json_files if str(p.parent.relative_to(STATSBOMB)).replace("\\","/") == rel]
    if not files: continue
    lines += ["", "-"*80, f"STATSBOMB DIRECTORY: {rel or '[root]'}", "-"*80]
    for p in files[:3]:
        typ, length, keys = json_info(p)
        lines.append(f"{p.relative_to(STATSBOMB)} | {p.stat().st_size/(1024*1024):.2f} MB | {typ} | length={length} | keys={keys}")

lines += ["", "="*80, "NEXT STEP", "="*80,
          "Do not merge yet. We will first inspect seasons, competitions, IDs, dates, and player/team matching.",
          "StatsBomb events will be processed file-by-file to avoid loading the massive dataset into RAM.",
          "three-sixty remains optional until its usefulness is established."]

report = "\n".join(lines)
print(report)
REPORT_FILE.write_text(report, encoding="utf-8")
print(f"\nReport saved to: {REPORT_FILE}")

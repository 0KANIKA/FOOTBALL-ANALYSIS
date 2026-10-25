"""Run the separate, limited-coverage StatsBomb enrichment layer."""
import json
import pandas as pd
from build_performance_analysis import OUT, build_statsbomb, read_csv

players = read_csv("players.csv", usecols=["player_id", "name"])
metrics, audit = build_statsbomb(players)
print(json.dumps({
    "matched_identity_rows": int((audit["match_status"] == "unique_exact").sum()) if not audit.empty else 0,
    "ambiguous_identity_rows": int((audit["match_status"] == "ambiguous").sum()) if not audit.empty else 0,
    "unmatched_identity_rows": int((audit["match_status"] == "unmatched").sum()) if not audit.empty else 0,
    "metric_player_rows": int(len(metrics)),
    "output": str(OUT / "statsbomb_enrichment" / "statsbomb_player_event_metrics.csv"),
}, indent=2))

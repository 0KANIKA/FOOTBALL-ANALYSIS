"""Finalize coverage and documentation from an already-built core layer."""
from pathlib import Path
import json
import pandas as pd
from build_performance_analysis import OUT, coverage_rows

ROOT = Path(__file__).resolve().parent
profiles = pd.read_csv(OUT / "player_performance_profiles.csv", low_memory=False)
season = pd.read_csv(OUT / "player_season_performance.csv", low_memory=False)
transfers = pd.read_csv(OUT / "player_transfer_performance.csv", low_memory=False)
sb_path = OUT / "statsbomb_enrichment" / "statsbomb_player_event_metrics.csv"
sb = pd.read_csv(sb_path, low_memory=False) if sb_path.exists() else pd.DataFrame()

coverage_rows(profiles, season, transfers, sb, 0).to_csv(OUT / "metric_coverage.csv", index=False)

dictionary = pd.DataFrame([
    {"metric": "goals", "definition": "Sum of recorded goals", "source_columns": "appearances.csv:goals", "unit": "goals", "grain": "player/window", "window": "observed, season, or transfer window"},
    {"metric": "assists", "definition": "Sum of recorded assists", "source_columns": "appearances.csv:assists", "unit": "assists", "grain": "player/window", "window": "observed, season, or transfer window"},
    {"metric": "goal_contributions_per90", "definition": "(goals + assists) * 90 / minutes", "source_columns": "goals, assists, minutes_played", "unit": "per 90 minutes", "grain": "player/window", "window": "missing when minutes <= 0"},
    {"metric": "minutes_per_appearance", "definition": "minutes / appearances", "source_columns": "minutes_played, game_id", "unit": "minutes per appearance", "grain": "player/window", "window": "missing when appearances <= 0"},
    {"metric": "starts", "definition": "Unique games with lineup type starting_lineup", "source_columns": "game_lineups.csv:game_id,player_id,type", "unit": "starts", "grain": "player/observed history", "window": "full observed history"},
    {"metric": "substitute_appearances", "definition": "Appearance records minus verified starts", "source_columns": "appearances.csv and game_lineups.csv", "unit": "appearances", "grain": "player/observed history", "window": "not bench listings"},
    {"metric": "yellow_cards_per90", "definition": "yellow cards * 90 / minutes", "source_columns": "yellow_cards, minutes_played", "unit": "per 90 minutes", "grain": "player/window", "window": "missing when minutes <= 0"},
    {"metric": "red_cards_per90", "definition": "red cards * 90 / minutes", "source_columns": "red_cards, minutes_played", "unit": "per 90 minutes", "grain": "player/window", "window": "missing when minutes <= 0"},
    {"metric": "market_value_change_365", "definition": "Latest valuation within one year after transfer minus latest valuation on/before transfer", "source_columns": "player_valuations.csv:date,market_value_in_eur", "unit": "EUR", "grain": "transfer", "window": "transfer date to +365 days"},
    {"metric": "fee_to_market_value_ratio", "definition": "positive transfer fee / observed pre-transfer market value", "source_columns": "transfers.csv:transfer_fee and player_valuations.csv", "unit": "ratio", "grain": "transfer", "window": "descriptive only"},
    {"metric": "percentile_*", "definition": "Within-position or within-sub-position percentile among reliable peers", "source_columns": "derived rates, position, sub_position", "unit": "percentile 0-100", "grain": "player profile", "window": "full observed history"},
    {"metric": "season_id", "definition": "Verified season field joined from games.csv by game_id", "source_columns": "games.csv:game_id,season", "unit": "source season identifier", "grain": "player-season", "window": "football season, not calendar year"},
    {"metric": "statsbomb_event_metrics", "definition": "Event counts for exact unique normalized-name matches only", "source_columns": "StatsBomb events JSON", "unit": "event counts", "grain": "matched player", "window": "available StatsBomb matches only"},
])
dictionary.to_csv(OUT / "data_dictionary.csv", index=False)
profiles.sort_values(["minutes", "goal_contributions"], ascending=False).head(10).to_csv(OUT / "example_player_profiles.csv", index=False)

metadata = {
    "output_grains": {
        "player_performance_profiles.csv": "one row per player over full observed appearance history",
        "player_season_performance.csv": "one row per player per games.season value",
        "player_transfer_performance.csv": "one row per validated transfer_row_id",
        "metric_coverage.csv": "one row per metric/coverage check",
    },
    "reliability_thresholds": {"minimum_minutes": 450, "minimum_appearances": 5, "minimum_peer_group": 10},
    "natural_transfer_key": "player_id|transfer_date|from_club_id|to_club_id|transfer_season|transfer_fee",
    "statsbomb_match_method": "exact normalized player name, unique Transfermarkt target only",
    "source_rows": {"player_profiles": len(profiles), "player_seasons": len(season), "player_transfers": len(transfers), "statsbomb_matched_players": len(sb)},
}
(OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
print(json.dumps(metadata, indent=2))

#!/usr/bin/env python3
"""
Coverage-first audit for a football transfer prediction project.

Default project folder:
C:\\Users\\rsuba\\Downloads\\R football analytics (4)\\R football analytics

Run:
  python source_coverage_audit.py
  python source_coverage_audit.py --project-dir "C:\\path\\to\\R football analytics"
  python source_coverage_audit.py --project-dir "C:\\path\\to\\project" --candidate-dir "C:\\path\\to\\external_dataset"

Writes:
  outputs_source_audit/source_coverage_summary.csv
  outputs_source_audit/player_match_audit.csv
  outputs_source_audit/target_availability.csv
  outputs_source_audit/integration_recommendation.txt

This script only audits coverage. It does not merge new features into the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

DEFAULT_PROJECT = Path(
    r"C:\Users\rsuba\Downloads\R football analytics (4)\R football analytics"
)

TARGET_COLUMNS = [
    "post_GA90", "post_goals", "post_assists", "post_minutes",
    "post_appearances", "complete_365_day_window", "eligible_450min",
]
PLAYER_SOURCE_FILES = {
    "players": ["archive/players.csv", "players.csv"],
    "appearances": ["archive/appearances.csv", "appearances.csv"],
    "games": ["archive/games.csv", "games.csv"],
    "valuations": ["archive/player_valuations.csv", "player_valuations.csv"],
    "events": ["archive/game_events.csv", "game_events.csv"],
    "lineups": ["archive/game_lineups.csv", "game_lineups.csv"],
}

def read_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"WARNING: Could not read {path}: {exc}")
        return None

def first_existing(root: Path, options: list[str]) -> Path | None:
    for item in options:
        path = root / item
        if path.exists():
            return path
    return None

def clean_id(series: pd.Series) -> pd.Series:
    # IDs sometimes arrive as floats (e.g. 123.0) when a CSV has missing values.
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()

def parse_dates(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    return pd.to_datetime(df[column], errors="coerce", utc=True).dt.tz_convert(None)

def pct(n: int, denom: int) -> float:
    return round(100.0 * n / denom, 2) if denom else 0.0

def main() -> int:
    parser = argparse.ArgumentParser(description="Audit transfer-cohort coverage in local football datasets.")
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument(
        "--candidate-dir", type=Path, default=None,
        help="Optional folder containing additional downloaded CSV files to audit."
    )
    args = parser.parse_args()

    root = args.project_dir.expanduser()
    out_dir = root / "outputs_source_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort_path = root / "post_transfer_performance.csv"
    cohort = read_csv_safe(cohort_path)
    if cohort is None:
        print(f"ERROR: Could not find/read master cohort: {cohort_path}")
        print("Pass the project folder with --project-dir.")
        return 2
    if "player_id" not in cohort.columns:
        print("ERROR: post_transfer_performance.csv must contain player_id.")
        return 2

    cohort = cohort.copy()
    cohort["player_id_audit"] = clean_id(cohort["player_id"])
    cohort["transfer_date_audit"] = parse_dates(cohort, "transfer_date")
    if "to_club_id" in cohort.columns:
        cohort["to_club_id_audit"] = clean_id(cohort["to_club_id"])
    else:
        cohort["to_club_id_audit"] = pd.NA

    n_rows = len(cohort)
    cohort_ids = set(cohort["player_id_audit"].dropna())
    cohort_ids.discard("")
    source_frames: dict[str, pd.DataFrame | None] = {}
    source_paths: dict[str, str] = {}
    summary_rows = []

    def add_summary(source_name: str, path_label: str, df: pd.DataFrame | None,
                    matched_rows: int | None = None, matched_ids: int | None = None,
                    note: str = "") -> None:
        summary_rows.append({
            "source": source_name,
            "file": path_label,
            "file_found_readable": df is not None,
            "source_rows": len(df) if df is not None else 0,
            "cohort_rows_with_match": matched_rows if matched_rows is not None else 0,
            "cohort_row_match_pct": pct(matched_rows or 0, n_rows),
            "unique_cohort_players_matched": matched_ids if matched_ids is not None else 0,
            "unique_cohort_players_match_pct": pct(matched_ids or 0, len(cohort_ids)),
            "notes": note,
        })

    # Audit known local archive files. Print progress because some CSVs are large.
    for source_name, candidates in PLAYER_SOURCE_FILES.items():
        path = first_existing(root, candidates)
        print(f"Loading {source_name}: {path if path else '(not found)'}", flush=True)
        df = read_csv_safe(path) if path else None
        if df is not None:
            print(f"  Loaded {len(df):,} rows x {len(df.columns):,} columns", flush=True)
        source_frames[source_name] = df
        source_paths[source_name] = str(path) if path else "(not found)"
        if df is None:
            add_summary(source_name, "(not found)", None, note="File not found or unreadable.")
            continue
        if "player_id" not in df.columns:
            add_summary(source_name, str(path), df, note="No player_id column; cannot match cohort.")
            continue
        ids = set(clean_id(df["player_id"]).dropna())
        ids.discard("")
        matched_ids = len(cohort_ids & ids)
        # Row-level match means this transfer's player occurs in source.
        row_match = cohort["player_id_audit"].isin(ids)
        add_summary(source_name, str(path), df, int(row_match.sum()), matched_ids,
                    note="Player-ID overlap only; not proof of date/club/metric coverage.")

    # Optional external/candidate CSV folder.
    candidate_files: list[Path] = []
    if args.candidate_dir:
        if args.candidate_dir.exists():
            candidate_files = sorted(p for p in args.candidate_dir.rglob("*.csv") if p.is_file())
        else:
            print(f"WARNING: candidate folder not found: {args.candidate_dir}")
    candidate_frames: dict[str, pd.DataFrame] = {}
    for path in candidate_files:
        df = read_csv_safe(path)
        if df is None:
            continue
        label = f"candidate:{path.name}"
        candidate_frames[str(path)] = df
        if "player_id" not in df.columns:
            add_summary(label, str(path), df, note="No player_id column; inspect manually.")
            continue
        ids = set(clean_id(df["player_id"]).dropna())
        ids.discard("")
        row_match = cohort["player_id_audit"].isin(ids)
        add_summary(label, str(path), df, int(row_match.sum()), len(cohort_ids & ids),
                    note="Player-ID overlap only; inspect date, competition, and metric fields.")

    # Player-level audit table: one row per transfer.
    audit = pd.DataFrame({
        "transfer_row_id": cohort["transfer_row_id"] if "transfer_row_id" in cohort else pd.NA,
        "player_id": cohort["player_id"] if "player_id" in cohort else pd.NA,
        "transfer_date": cohort["transfer_date"] if "transfer_date" in cohort else pd.NA,
        "from_club_id": cohort["from_club_id"] if "from_club_id" in cohort else pd.NA,
        "to_club_id": cohort["to_club_id"] if "to_club_id" in cohort else pd.NA,
        "from_club_name": cohort["from_club_name"] if "from_club_name" in cohort else pd.NA,
        "to_club_name": cohort["to_club_name"] if "to_club_name" in cohort else pd.NA,
        "player_id_audit": cohort["player_id_audit"],
    })

    for source_name, df in source_frames.items():
        if df is not None and "player_id" in df.columns:
            ids = set(clean_id(df["player_id"]).dropna())
            ids.discard("")
            audit[f"matched_{source_name}_player_id"] = audit["player_id_audit"].isin(ids)
        else:
            audit[f"matched_{source_name}_player_id"] = False

    # Appearance coverage by transfer timing and destination club, when fields allow.
    apps = source_frames.get("appearances")
    if apps is not None and "player_id" in apps.columns:
        apps = apps.copy()
        apps["player_id_audit"] = clean_id(apps["player_id"])
        apps["date_audit"] = parse_dates(apps, "date")
        app_player_ids = set(apps["player_id_audit"].dropna())
        app_player_ids.discard("")
        # Build a per-player lookup once. Filtering the entire appearances table
        # once per transfer is extremely slow on million-row files.
        print("Indexing appearances by player for fast transfer-window checks...", flush=True)
        club_col = "player_club_id" if "player_club_id" in apps.columns else (
            "player_current_club_id" if "player_current_club_id" in apps.columns else None
        )
        app_lookup = {}
        valid_apps = apps.dropna(subset=["player_id_audit", "date_audit"]).sort_values("date_audit")
        for pid, grp in valid_apps.groupby("player_id_audit", sort=False):
            dates = grp["date_audit"].to_numpy(dtype="datetime64[ns]")
            clubs = clean_id(grp[club_col]).to_numpy() if club_col else None
            app_lookup[pid] = (dates, clubs)
        pre_counts, post_counts, dest_post_counts = [], [], []
        import numpy as np
        total = len(cohort)
        for i, (_, tr) in enumerate(cohort.iterrows(), start=1):
            pid = tr["player_id_audit"]
            td = tr["transfer_date_audit"]
            if pd.isna(td) or pid not in app_lookup:
                pre_counts.append(0 if pd.notna(td) else pd.NA)
                post_counts.append(0 if pd.notna(td) else pd.NA)
                dest_post_counts.append(0 if (pd.notna(td) and club_col) else pd.NA)
                continue
            dates, clubs = app_lookup[pid]
            cut = np.searchsorted(dates, np.datetime64(td), side="left")
            pre_counts.append(int(cut))
            post_counts.append(int(len(dates) - cut))
            dest_id = tr["to_club_id_audit"]
            if clubs is not None and pd.notna(dest_id):
                dest_post_counts.append(int((clubs[cut:] == dest_id).sum()))
            else:
                dest_post_counts.append(pd.NA)
            if i % 1000 == 0 or i == total:
                print(f"  Transfer rows checked: {i:,}/{total:,}", flush=True)
        audit["archive_appearances_pre_transfer_count"] = pre_counts
        audit["archive_appearances_post_transfer_count"] = post_counts
        audit["archive_destination_post_appearance_count"] = dest_post_counts
    else:
        audit["archive_appearances_pre_transfer_count"] = pd.NA
        audit["archive_appearances_post_transfer_count"] = pd.NA
        audit["archive_destination_post_appearance_count"] = pd.NA

    # Candidate player-level matching, per file.
    for path_str, df in candidate_frames.items():
        safe_name = Path(path_str).stem[:40]
        if "player_id" in df.columns:
            ids = set(clean_id(df["player_id"]).dropna())
            ids.discard("")
            audit[f"candidate_{safe_name}_match"] = audit["player_id_audit"].isin(ids)

    audit.drop(columns=["player_id_audit"], errors="ignore").to_csv(
        out_dir / "player_match_audit.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(out_dir / "source_coverage_summary.csv", index=False)

    # Existing target availability plus possible derived metrics.
    target_rows = []
    for col in TARGET_COLUMNS:
        if col in cohort.columns:
            nonnull = int(cohort[col].notna().sum())
            target_rows.append({
                "target_or_metric": col, "status": "already in master cohort",
                "rows_available": nonnull, "coverage_pct": pct(nonnull, n_rows),
                "definition_or_caution": "Check eligibility and exact post-transfer observation window before modelling."
            })
        else:
            target_rows.append({
                "target_or_metric": col, "status": "not present in master cohort",
                "rows_available": 0, "coverage_pct": 0.0,
                "definition_or_caution": "Not currently available as a cohort column."
            })

    # Suggest candidate metrics based on known source columns.
    metric_specs = [
        ("goals_per90_post_derived", "appearances.csv: goals, minutes_played",
         "Sum post-transfer goals / post-transfer minutes * 90; confirm same 365-day window and destination club."),
        ("assists_per90_post_derived", "appearances.csv: assists, minutes_played",
         "Sum post-transfer assists / post-transfer minutes * 90; confirm same 365-day window and destination club."),
        ("yellow_cards_per90_post_derived", "appearances.csv: yellow_cards, minutes_played",
         "Sum post-transfer yellow cards / minutes * 90; low counts can be noisy."),
        ("red_cards_per90_post_derived", "appearances.csv: red_cards, minutes_played",
         "Sum post-transfer red cards / minutes * 90; sparse outcome, treat cautiously."),
        ("market_value_change_365d", "player_valuations.csv: date, market_value_in_eur, player_id",
         "Only derive if valuations exist near transfer date and around +365 days; report actual observation gap."),
        ("successful_tackles", "Not in listed local appearances/events schema",
         "Requires a source with defensive event or player-match statistics; do not infer from goals/cards."),
        ("pass_completion_pct", "Not in listed local appearances/events schema",
         "Requires completed and attempted passes or equivalent event data."),
        ("interceptions", "Not in listed local appearances/events schema",
         "Requires defensive event or player-match statistics."),
        ("duels_won_pct", "Not in listed local appearances/events schema",
         "Requires duel attempts and wins; define duel types consistently."),
    ]
    for name, source, caution in metric_specs:
        target_rows.append({
            "target_or_metric": name,
            "status": "candidate / needs validation",
            "rows_available": 0,
            "coverage_pct": 0.0,
            "definition_or_caution": f"Candidate source: {source}. {caution}"
        })
    pd.DataFrame(target_rows).to_csv(out_dir / "target_availability.csv", index=False)

    # Human-readable recommendations.
    summary = pd.DataFrame(summary_rows)
    lines = [
        "SOURCE COVERAGE AUDIT — RECOMMENDATIONS",
        "=" * 42,
        f"Project: {root}",
        f"Transfer cohort rows: {n_rows:,}",
        f"Unique cohort players: {len(cohort_ids):,}",
        "",
        "Interpretation:",
        "- Player-ID overlap is a screening metric, not proof that the source covers the correct transfer window, competition, club, or metric.",
        "- Keep Mode A unchanged until candidate coverage and feature timing are validated.",
        "- Never fill missing tackles/passing/interceptions/duels with zero; missing means unobserved unless proven otherwise.",
        "- Build post-transfer rates using the same window and eligibility rules as the existing target.",
        "- Use only pre-transfer observations as predictors; post-transfer stats belong only in targets.",
        "",
        "Best next checks:",
        "1. Review source_coverage_summary.csv and rank candidate files by unique-player overlap.",
        "2. Review player_match_audit.csv for unmatched players and missing transfer dates/club IDs.",
        "3. Review target_availability.csv before adding targets.",
        "4. For any promising source, validate a sample of matched players and dates manually.",
        "5. Re-run the same chronological evaluation and report performance on the same eligible rows.",
    ]
    if not args.candidate_dir:
        lines += [
            "",
            "No --candidate-dir was supplied. This run audits the project's existing archive only.",
            "To test an external free dataset, extract its CSV files into a separate folder and rerun with --candidate-dir."
        ]
    if len(summary):
        best = summary[summary["source"].str.startswith("candidate:")]
        if not best.empty:
            best = best.sort_values("unique_cohort_players_match_pct", ascending=False).head(5)
            lines += ["", "Top candidate CSVs by unique-player overlap:"]
            for _, row in best.iterrows():
                lines.append(
                    f"- {row['source']}: {row['unique_cohort_players_match_pct']}% of cohort players "
                    f"({row['unique_cohort_players_matched']:,} matched)"
                )
    (out_dir / "integration_recommendation.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nAudit complete.")
    print(f"Outputs folder: {out_dir}")
    print(f"- {out_dir / 'source_coverage_summary.csv'}")
    print(f"- {out_dir / 'player_match_audit.csv'}")
    print(f"- {out_dir / 'target_availability.csv'}")
    print(f"- {out_dir / 'integration_recommendation.txt'}")
    print("\nNote: this script audits source coverage; it does not change or retrain Mode A.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

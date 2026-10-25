#!/usr/bin/env python3
"""
Leakage-safe 365-day transfer audit + pre-transfer feature builder.

Reads the existing project files; does not modify the source data or train a model.

Run from the project folder:
    python -u build_mode_a_features.py

Or:
    python -u build_mode_a_features.py --project-dir "C:\\path\\to\\R football analytics"

Outputs in outputs_mode_a_features:
    transfer_window_audit.csv
    mode_a_pretransfer_features.csv
    lineup_type_counts.csv
    feature_build_summary.txt

Important:
- Pre-transfer features use appearances strictly before transfer_date.
- Post-transfer audit uses [transfer_date, transfer_date + 365 days) and destination-club
  appearances when player_club_id is available.
- The audit's computed post-transfer values are diagnostics, not replacements for
  the established target columns. Validate definitions before changing model targets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_PROJECT = Path(
    r"C:\Users\rsuba\Downloads\R football analytics (4)\R football analytics"
)

def read_csv(path: Path, usecols=None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, usecols=usecols, low_memory=False)

def clean_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()

def dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None)

def safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT)
    args = parser.parse_args()
    root = args.project_dir.expanduser()
    out = root / "outputs_mode_a_features"
    out.mkdir(parents=True, exist_ok=True)

    cohort_path = root / "post_transfer_performance.csv"
    apps_path = root / "archive" / "appearances.csv"
    vals_path = root / "archive" / "player_valuations.csv"
    lineups_path = root / "archive" / "game_lineups.csv"

    print(f"Reading transfer cohort: {cohort_path}", flush=True)
    cohort = read_csv(cohort_path)
    required = {"player_id", "transfer_date"}
    if not required.issubset(cohort.columns):
        raise ValueError(f"Cohort is missing required columns: {sorted(required - set(cohort.columns))}")
    cohort = cohort.copy()
    cohort["__row_number"] = np.arange(len(cohort))
    cohort["__pid"] = clean_id(cohort["player_id"])
    cohort["__transfer_date"] = dates(cohort["transfer_date"])
    cohort["__to_club"] = clean_id(cohort["to_club_id"]) if "to_club_id" in cohort else pd.NA

    if "transfer_row_id" not in cohort.columns:
        raise ValueError("transfer_row_id is missing; refusing to create a joinable feature table without it.")
    if cohort["transfer_row_id"].isna().any() or cohort["transfer_row_id"].duplicated().any():
        raise ValueError("transfer_row_id has missing or duplicate values; fix identity alignment before merging.")

    print(f"Reading appearances: {apps_path}", flush=True)
    apps = read_csv(apps_path)
    if not {"player_id", "date"}.issubset(apps.columns):
        raise ValueError("appearances.csv needs player_id and date columns.")
    apps["__pid"] = clean_id(apps["player_id"])
    apps["__date"] = dates(apps["date"])
    if "player_club_id" in apps.columns:
        apps["__club"] = clean_id(apps["player_club_id"])
    else:
        apps["__club"] = pd.NA
    for col in ["minutes_played", "goals", "assists", "yellow_cards", "red_cards"]:
        apps[f"__{col}"] = safe_num(apps, col)

    apps = apps.dropna(subset=["__pid", "__date"]).sort_values(["__pid", "__date"])
    app_groups = {pid: grp.reset_index(drop=True) for pid, grp in apps.groupby("__pid", sort=False)}
    print(f"Indexed {len(apps):,} appearance rows for {len(app_groups):,} players.", flush=True)

    feature_rows = []
    audit_rows = []
    n = len(cohort)
    for idx, tr in cohort.iterrows():
        pid = tr["__pid"]
        td = tr["__transfer_date"]
        rid = tr["transfer_row_id"]
        dest = tr["__to_club"]
        base = {
            "transfer_row_id": rid,
            "player_id": tr["player_id"],
            "transfer_date": tr["transfer_date"],
        }
        if "to_club_id" in cohort.columns:
            base["to_club_id"] = tr["to_club_id"]

        features = dict(base)
        if pd.isna(td) or pid not in app_groups:
            for name in [
                "pre_apps_365d", "pre_minutes_365d", "pre_goals_365d", "pre_assists_365d",
                "pre_yellow_cards_365d", "pre_red_cards_365d", "pre_goals_per90_365d",
                "pre_assists_per90_365d", "pre_yellows_per90_365d", "pre_reds_per90_365d",
                "pre_apps_90d", "pre_minutes_90d", "days_since_last_pre_transfer_appearance",
            ]:
                features[name] = np.nan
            audit = dict(base)
            audit.update({
                "valid_transfer_date": False,
                "player_found_in_appearances": pid in app_groups,
                "pre_apps_365d": np.nan,
                "post_apps_365d_any_club": np.nan,
                "post_apps_365d_destination": np.nan,
                "post_minutes_365d_destination": np.nan,
                "post_goals_365d_destination": np.nan,
                "post_assists_365d_destination": np.nan,
                "post_GA90_365d_destination": np.nan,
                "post_window_has_365d_calendar_span": False,
                "note": "Missing transfer date or no appearance records for player."
            })
            feature_rows.append(features)
            audit_rows.append(audit)
            continue

        pg = app_groups[pid]
        d = pg["__date"]
        pre365 = pg[(d < td) & (d >= td - pd.Timedelta(days=365))]
        pre90 = pg[(d < td) & (d >= td - pd.Timedelta(days=90))]
        post365_any = pg[(d >= td) & (d < td + pd.Timedelta(days=365))]
        if pd.notna(dest) and "__club" in pg.columns and pg["__club"].notna().any():
            post_dest = post365_any[post365_any["__club"] == dest]
            dest_filter_used = True
        else:
            post_dest = post365_any.iloc[0:0]
            dest_filter_used = False

        def total(frame, col):
            return float(frame[f"__{col}"].sum()) if len(frame) else 0.0

        pre_minutes = total(pre365, "minutes_played")
        pre_goals = total(pre365, "goals")
        pre_assists = total(pre365, "assists")
        pre_yellow = total(pre365, "yellow_cards")
        pre_red = total(pre365, "red_cards")
        post_minutes = total(post_dest, "minutes_played")
        post_goals = total(post_dest, "goals")
        post_assists = total(post_dest, "assists")

        features.update({
            "pre_apps_365d": int(len(pre365)),
            "pre_minutes_365d": pre_minutes,
            "pre_goals_365d": pre_goals,
            "pre_assists_365d": pre_assists,
            "pre_yellow_cards_365d": pre_yellow,
            "pre_red_cards_365d": pre_red,
            "pre_goals_per90_365d": (pre_goals * 90 / pre_minutes) if pre_minutes > 0 else np.nan,
            "pre_assists_per90_365d": (pre_assists * 90 / pre_minutes) if pre_minutes > 0 else np.nan,
            "pre_yellows_per90_365d": (pre_yellow * 90 / pre_minutes) if pre_minutes > 0 else np.nan,
            "pre_reds_per90_365d": (pre_red * 90 / pre_minutes) if pre_minutes > 0 else np.nan,
            "pre_apps_90d": int(len(pre90)),
            "pre_minutes_90d": total(pre90, "minutes_played"),
            "days_since_last_pre_transfer_appearance":
                int((td - pre365["__date"].max()).days) if len(pre365) else np.nan,
        })

        audit = dict(base)
        audit.update({
            "valid_transfer_date": True,
            "player_found_in_appearances": True,
            "pre_apps_365d": int(len(pre365)),
            "pre_minutes_365d": pre_minutes,
            "post_apps_365d_any_club": int(len(post365_any)),
            "destination_club_id_available": pd.notna(dest),
            "destination_club_filter_used": dest_filter_used,
            "post_apps_365d_destination": int(len(post_dest)) if dest_filter_used else np.nan,
            "post_minutes_365d_destination": post_minutes if dest_filter_used else np.nan,
            "post_goals_365d_destination": post_goals if dest_filter_used else np.nan,
            "post_assists_365d_destination": post_assists if dest_filter_used else np.nan,
            "post_GA90_365d_destination":
                ((post_goals + post_assists) * 90 / post_minutes)
                if dest_filter_used and post_minutes > 0 else np.nan,
            "post_window_has_365d_calendar_span": bool(td + pd.Timedelta(days=365) <= apps["__date"].max()),
            "note": "Destination metrics are archive-derived diagnostics; validate against target construction."
        })
        # Preserve original target columns for a direct diagnostic comparison.
        for col in ["post_appearances", "post_minutes", "post_goals", "post_assists", "post_GA90",
                    "complete_365_day_window", "eligible_450min"]:
            if col in tr.index:
                audit[f"master_{col}"] = tr[col]
        feature_rows.append(features)
        audit_rows.append(audit)

        if len(feature_rows) % 1000 == 0 or len(feature_rows) == n:
            print(f"Transfer rows processed: {len(feature_rows):,}/{n:,}", flush=True)

    features_df = pd.DataFrame(feature_rows)
    audit_df = pd.DataFrame(audit_rows)

    # Pre-transfer valuations: latest value at/before transfer and nearest value at/before
    # the one-year lookback boundary, only if observations exist.
    if vals_path.exists():
        print(f"Reading valuations: {vals_path}", flush=True)
        vals = read_csv(vals_path, usecols=lambda c: c in {"player_id", "date", "market_value_in_eur"})
        if {"player_id", "date", "market_value_in_eur"}.issubset(vals.columns):
            vals["__pid"] = clean_id(vals["player_id"])
            vals["__date"] = dates(vals["date"])
            vals["__value"] = pd.to_numeric(vals["market_value_in_eur"], errors="coerce")
            vals = vals.dropna(subset=["__pid", "__date", "__value"]).sort_values(["__pid", "__date"])
            val_groups = {pid: grp.reset_index(drop=True) for pid, grp in vals.groupby("__pid", sort=False)}
            latest_vals, prior_vals, latest_dates, prior_dates = [], [], [], []
            for j, tr in cohort.iterrows():
                pid, td = tr["__pid"], tr["__transfer_date"]
                grp = val_groups.get(pid)
                if grp is None or pd.isna(td):
                    latest_vals.append(np.nan); prior_vals.append(np.nan)
                    latest_dates.append(pd.NaT); prior_dates.append(pd.NaT)
                    continue
                eligible = grp[grp["__date"] <= td]
                latest = eligible.iloc[-1] if len(eligible) else None
                boundary = td - pd.Timedelta(days=365)
                older = grp[grp["__date"] <= boundary]
                prior = older.iloc[-1] if len(older) else None
                latest_vals.append(float(latest["__value"]) if latest is not None else np.nan)
                prior_vals.append(float(prior["__value"]) if prior is not None else np.nan)
                latest_dates.append(latest["__date"] if latest is not None else pd.NaT)
                prior_dates.append(prior["__date"] if prior is not None else pd.NaT)
            features_df["market_value_latest_pre_eur"] = latest_vals
            features_df["market_value_latest_pre_date"] = latest_dates
            features_df["market_value_prior_365d_eur"] = prior_vals
            features_df["market_value_prior_365d_date"] = prior_dates
            features_df["market_value_change_365d_pct"] = [
                ((a - b) / b * 100) if pd.notna(a) and pd.notna(b) and b > 0 else np.nan
                for a, b in zip(latest_vals, prior_vals)
            ]
            features_df["market_value_latest_observation_age_days"] = [
                (td - dt).days if pd.notna(td) and pd.notna(dt) else np.nan
                for td, dt in zip(cohort["__transfer_date"], latest_dates)
            ]
        else:
            print("WARNING: valuation file lacks expected columns; skipping valuation features.", flush=True)

    # Inspect lineup type labels only. Do not create start features until labels are validated.
    lineup_type_summary = pd.DataFrame(columns=["type", "rows"])
    if lineups_path.exists():
        print(f"Reading lineup types: {lineups_path}", flush=True)
        lineups = read_csv(lineups_path, usecols=lambda c: c in {"type"})
        if "type" in lineups:
            lineup_type_summary = (
                lineups["type"].fillna("(missing)").astype(str).value_counts()
                .rename_axis("type").reset_index(name="rows")
            )
            lineup_type_summary.to_csv(out / "lineup_type_counts.csv", index=False)

    # Save in stable transfer order; preserve exact transfer_row_id for downstream merge.
    features_df.to_csv(out / "mode_a_pretransfer_features.csv", index=False)
    audit_df.to_csv(out / "transfer_window_audit.csv", index=False)

    lines = [
        "MODE A PRE-TRANSFER FEATURE BUILD SUMMARY",
        "=========================================",
        f"Project: {root}",
        f"Transfer rows: {len(cohort):,}",
        f"Unique transfer_row_id values: {cohort['transfer_row_id'].nunique():,}",
        f"Rows with valid transfer date: {int(cohort['__transfer_date'].notna().sum()):,}",
        f"Rows with appearance player match: {int(cohort['__pid'].isin(set(apps['__pid'])).sum()):,}",
        "",
        "Files written:",
        "- mode_a_pretransfer_features.csv: predictor features from appearances strictly before transfer date, plus pre-transfer valuations when available. No post-transfer metrics are included in this file.",
        "- transfer_window_audit.csv: exact [transfer_date, transfer_date + 365 days) destination-club diagnostic counts and original target columns.",
        "- lineup_type_counts.csv: lineup type values and counts; inspect before engineering start/bench features.",
        "",
        "Safety notes:",
        "- Do not merge the post-transfer diagnostic columns into model predictors.",
        "- Pre-transfer rates with zero pre-transfer minutes are missing (NaN), not zero.",
        "- Destination-club post metrics are only populated when to_club_id and player_club_id are available.",
        "- A transfer window may not be fully observable near the end of the archive. Check post_window_has_365d_calendar_span and the master complete_365_day_window flag.",
        "- Compare archive-derived post metrics with existing master targets before replacing any target definition.",
        "- Keep the existing Mode A baseline and chronological split unchanged for fair comparison.",
    ]
    (out / "feature_build_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\nFeature build complete.", flush=True)
    print(f"Output folder: {out}", flush=True)
    print(f"- {out / 'mode_a_pretransfer_features.csv'}", flush=True)
    print(f"- {out / 'transfer_window_audit.csv'}", flush=True)
    print(f"- {out / 'lineup_type_counts.csv'}", flush=True)
    print(f"- {out / 'feature_build_summary.txt'}", flush=True)
    print("Review the audit and validate definitions before using any new features in model training.", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

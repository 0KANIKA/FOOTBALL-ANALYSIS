#!/usr/bin/env python3
"""Validate master transfer windows against archive appearances and build pre-transfer lineup features."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

DEFAULT_PROJECT = Path(__file__).resolve().parents[2]

def clean_id(s):
    return s.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()

def dt(s):
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None)

def read(path, **kwargs):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False, **kwargs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT)
    args = ap.parse_args()
    root = args.project_dir
    data_dir = root / "04_Datasets" / "derived"
    raw_dir = root / "04_Datasets" / "raw_transfer_data" / "archive"
    out = root / "06_Results_and_Demo_Evidence" / "audit_results" / "outputs_window_lineup_validation"
    out.mkdir(parents=True, exist_ok=True)

    master = read(data_dir / "post_transfer_performance.csv")
    needed = ["transfer_row_id", "player_id", "transfer_date", "to_club_id"]
    missing = [c for c in needed if c not in master]
    if missing:
        raise ValueError(f"Master cohort missing columns: {missing}")
    if master.transfer_row_id.isna().any() or master.transfer_row_id.duplicated().any():
        raise ValueError("transfer_row_id is missing or duplicated; stop before merging.")

    master = master.copy()
    master["__pid"] = clean_id(master.player_id)
    master["__club"] = clean_id(master.to_club_id)
    master["__transfer"] = dt(master.transfer_date)
    has_windows = {"window_start", "window_end"}.issubset(master.columns)
    if has_windows:
        master["__start"] = dt(master.window_start)
        master["__end"] = dt(master.window_end)
    else:
        master["__start"] = master["__transfer"]
        master["__end"] = master["__transfer"] + pd.Timedelta(days=365)

    print("Loading appearances...", flush=True)
    apps = read(raw_dir / "appearances.csv")
    apps["__pid"] = clean_id(apps.player_id)
    apps["__club"] = clean_id(apps.player_club_id) if "player_club_id" in apps else pd.NA
    apps["__date"] = dt(apps.date)
    for col in ["minutes_played", "goals", "assists"]:
        apps[f"__{col}"] = pd.to_numeric(apps[col], errors="coerce").fillna(0)
    apps = apps.dropna(subset=["__pid", "__date"]).sort_values(["__pid", "__date"])
    app_groups = {pid: grp.reset_index(drop=True) for pid, grp in apps.groupby("__pid", sort=False)}
    print(f"Indexed {len(apps):,} appearance rows for {len(app_groups):,} players.", flush=True)

    # Test four plausible endpoint conventions against existing targets.
    conventions = [
        ("start_inclusive_end_exclusive", True, False),
        ("start_inclusive_end_inclusive", True, True),
        ("start_exclusive_end_inclusive", False, True),
        ("start_exclusive_end_exclusive", False, False),
    ]
    comparisons = []
    detailed = None
    for name, start_inc, end_inc in conventions:
        print(f"Testing window convention: {name}", flush=True)
        rows = []
        for i, tr in master.iterrows():
            pid, club = tr["__pid"], tr["__club"]
            start, end = tr["__start"], tr["__end"]
            if pd.isna(start) or pd.isna(end):
                rows.append({"transfer_row_id": tr.transfer_row_id, "computed_apps": np.nan,
                             "computed_minutes": np.nan, "computed_goals": np.nan,
                             "computed_assists": np.nan, "computed_GA90": np.nan})
                continue
            p = app_groups.get(pid)
            if p is None:
                rows.append({"transfer_row_id": tr.transfer_row_id, "computed_apps": 0,
                             "computed_minutes": 0.0, "computed_goals": 0.0,
                             "computed_assists": 0.0, "computed_GA90": np.nan})
                continue
            p = p[p["__club"] == club]
            left = p["__date"] >= start if start_inc else p["__date"] > start
            right = p["__date"] <= end if end_inc else p["__date"] < end
            p = p[left & right]
            mins = float(p["__minutes_played"].sum())
            goals = float(p["__goals"].sum())
            assists = float(p["__assists"].sum())
            rows.append({
                "transfer_row_id": tr.transfer_row_id,
                "computed_apps": int(len(p)),
                "computed_minutes": mins,
                "computed_goals": goals,
                "computed_assists": assists,
                "computed_GA90": ((goals + assists) * 90 / mins) if mins > 0 else np.nan,
            })
            if (i + 1) % 5000 == 0:
                print(f"  Rows: {i+1:,}/{len(master):,}", flush=True)
        cand = pd.DataFrame(rows)
        merged = master[["transfer_row_id"] + [c for c in [
            "post_appearances", "post_minutes", "post_goals", "post_assists", "post_GA90",
            "complete_365_day_window", "eligible_450min"
        ] if c in master.columns]].merge(cand, on="transfer_row_id", how="left", validate="one_to_one")
        result = {"window_convention": name, "rows": len(merged)}
        for actual, computed in [
            ("post_appearances", "computed_apps"),
            ("post_minutes", "computed_minutes"),
            ("post_goals", "computed_goals"),
            ("post_assists", "computed_assists"),
            ("post_GA90", "computed_GA90"),
        ]:
            if actual in merged:
                valid = merged[actual].notna() & merged[computed].notna()
                diff = (pd.to_numeric(merged.loc[valid, actual], errors="coerce") -
                        pd.to_numeric(merged.loc[valid, computed], errors="coerce")).abs()
                result[f"{actual}_compared"] = int(valid.sum())
                result[f"{actual}_exact_matches"] = int((diff < 1e-8).sum())
                result[f"{actual}_mean_abs_error"] = float(diff.mean()) if len(diff) else np.nan
                result[f"{actual}_max_abs_error"] = float(diff.max()) if len(diff) else np.nan
        comparisons.append(result)
        if detailed is None or (
            result.get("post_appearances_exact_matches", -1) +
            result.get("post_minutes_exact_matches", -1) +
            result.get("post_goals_exact_matches", -1) +
            result.get("post_assists_exact_matches", -1)
        ) > (
            detailed[0].get("post_appearances_exact_matches", -1) +
            detailed[0].get("post_minutes_exact_matches", -1) +
            detailed[0].get("post_goals_exact_matches", -1) +
            detailed[0].get("post_assists_exact_matches", -1)
        ):
            detailed = (result, merged.copy())

    pd.DataFrame(comparisons).to_csv(out / "window_convention_comparison.csv", index=False)
    best_result, best_merged = detailed
    best_merged.to_csv(out / "best_window_target_comparison.csv", index=False)

    # Build pre-transfer lineup features, using dates strictly before transfer_date.
    print("Loading lineups for pre-transfer selection features...", flush=True)
    lineups = read(raw_dir / "game_lineups.csv",
                   usecols=lambda c: c in {"date", "player_id", "game_id", "type"})
    lineups["__pid"] = clean_id(lineups.player_id)
    lineups["__date"] = dt(lineups.date)
    lineups["__type"] = lineups["type"].astype("string").str.lower().str.strip()
    lineups = lineups.dropna(subset=["__pid", "__date"])
    groups = {pid: g.sort_values("__date").reset_index(drop=True)
              for pid, g in lineups.groupby("__pid", sort=False)}
    line_features = []
    for j, tr in master.iterrows():
        pid, transfer = tr["__pid"], tr["__transfer"]
        base = {"transfer_row_id": tr.transfer_row_id}
        if pd.isna(transfer) or pid not in groups:
            base.update({"pre_lineup_records_365d": np.nan, "pre_starts_365d": np.nan,
                         "pre_substitute_selections_365d": np.nan, "pre_start_rate_365d": np.nan,
                         "pre_lineup_games_90d": np.nan})
        else:
            g = groups[pid]
            g = g[(g["__date"] < transfer) & (g["__date"] >= transfer - pd.Timedelta(days=365))]
            starts = int((g["__type"] == "starting_lineup").sum())
            subs = int((g["__type"] == "substitutes").sum())
            total = starts + subs
            g90 = groups[pid]
            g90 = g90[(g90["__date"] < transfer) & (g90["__date"] >= transfer - pd.Timedelta(days=90))]
            base.update({
                "pre_lineup_records_365d": int(len(g)),
                "pre_starts_365d": starts,
                "pre_substitute_selections_365d": subs,
                "pre_start_rate_365d": starts / total if total else np.nan,
                "pre_lineup_games_90d": int(g90["game_id"].nunique()) if "game_id" in g90 else int(len(g90)),
            })
        line_features.append(base)
        if (j + 1) % 5000 == 0:
            print(f"  Transfer rows: {j+1:,}/{len(master):,}", flush=True)

    pd.DataFrame(line_features).to_csv(out / "mode_a_pretransfer_lineup_features.csv", index=False)
    summary = [
        "WINDOW AND LINEUP VALIDATION SUMMARY",
        "====================================",
        f"Project: {root}",
        f"Rows: {len(master):,}",
        f"Master contains window_start/window_end: {has_windows}",
        f"Best tested convention by exact-match sum: {best_result['window_convention']}",
        "",
        "Comparison metrics are in window_convention_comparison.csv.",
        "Detailed comparison for the best tested convention is in best_window_target_comparison.csv.",
        "Pre-transfer lineup features are in mode_a_pretransfer_lineup_features.csv.",
        "",
        "Cautions:",
        "- The best tested convention is a diagnostic, not automatic proof of the intended definition.",
        "- Review all metrics, especially GA/90, and verify how the master target was originally constructed.",
        "- Lineup records measure selection/lineup status, not minutes played; treat starting rate as a selection signal.",
        "- Lineup features use only records dated strictly before transfer_date.",
        "- Do not add computed post-transfer columns to model predictors.",
    ]
    (out / "validation_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\nDone. Outputs:", out, flush=True)
    for name in ["window_convention_comparison.csv", "best_window_target_comparison.csv",
                 "mode_a_pretransfer_lineup_features.csv", "validation_summary.txt"]:
        print(f"- {out / name}", flush=True)

if __name__ == "__main__":
    main()

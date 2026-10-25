# Audited football transfer prediction run

This folder contains a reproducible audited runner and its outputs. Source project files are preserved unchanged.

## Runtime correction

The workspace does not contain R scripts that fit models with `ranger`; its model-building sources are Python/scikit-learn, and the `.rds` artifacts cannot be reproduced or source-audited as R code here. `run_audit.py` is therefore the executable audited implementation. It compares Random Forests with one-hot categorical preprocessing, fitted only on training rows, and validation-only hyperparameter selection. No test metrics influence selection.

## Cohort and split

The runner joins `post_transfer_performance.csv`, `rich_team_context_dataset.csv`, and `coach_tactical_context_v21.csv` by `transfer_row_id`, rejecting duplicate keys. The common primary-target cohort requires a complete 365-day post-transfer observation window, `eligible_450min == 1`, a valid transfer date, and finite `post_GA90`. Exact duplicate player/date/from/to transfer rows are removed; repeated transfers on different dates remain. All model variants share this cohort and chronological 60/20/20 row boundaries. Secondary outcomes are evaluated on valid target values within these same boundaries; zero-minute records have no 450-minute eligibility and are excluded from the common cohort.

The target file defines `post_GA90` as `(post_goals + post_assists) / post_minutes * 90`; its observed zero-minute rows have a zero target, but are excluded by the 450-minute eligibility rule. Goals/90 and assists/90 are recomputed from post counts/minutes; post minutes and appearances are direct 365-day outcomes. These secondary target names do not exist as stored fields in source data.

## Feature provenance and risk audit

- Core characteristics and pre-transfer stats (age, height, position, foot, appearances/minutes/goals/assists, rates, cards, fee and valuations) originate in `professional_transfer_dataset.csv`; pre-transfer feature builder uses a 365-day history ending strictly before transfer, and valuation is backward-as-of transfer date. `transfer_fee` is transfer-time information if the recorded fee was known at prediction time; potential operational availability caveat.
- Mode A destination team form uses the preceding ten destination-club matches, strictly `game_date < transfer_date`, from `build_rich_team_context.py` / `build_team_context.py`.
- Enhanced formation/manager summaries use the same preceding ten games. Manager identity is retained as a categorical feature; unseen categories are ignored by the encoder. It is a high-cardinality, potentially unstable feature.
- Squad competition fields are calculated from destination-club appearances in the prior year and historical lineup position. Squad valuation aggregates use valuations no later than transfer date. `previous_destination_appearances/minutes` also appear in the rich file, but are omitted because the published group specification did not include them.
- Coach/tactical numeric columns come from archived `build_coach_tactical_context_v21.py`; it selects known coach and player/squad usage based on destination-club appearances in the prior year, strictly before transfer. Missing coach fields are median-imputed from training and `coach_known_pretransfer` preserves missingness.
- Confirmed leakage bug in the older `build_coach_context.py`: its coach name lookup prioritizes the first destination match on or after transfer (up to 60 days), while claiming to be transfer-time coach context. The audited model excludes this dataset and its post-transfer coach name fields.
- Potential leakage / validity issue: `rich_team_context_dataset.csv` includes `previous_destination_appearances/minutes`, which are pre-transfer and not outcome leakage, but may identify returning players; omitted from declared model groups. `dominant_manager10` / formation are strictly prior-game according to builder source.
- `post_*`, `window_end`, completeness/eligibility flags and transfer row ID are never predictors. Flags are cohort filters only.

## Changes made in audited runner

1. Uses keyed one-to-one joins and validates uniqueness, avoiding positional joins and accidental many-to-many expansion.
2. Creates a single complete-window eligible cohort and one chronological split for every model/target.
3. Fits numeric median imputation and categorical missing-value handling/encoding inside each pipeline; unseen categories are ignored and rare levels grouped using training data only.
4. Adds the core, Mode A, full enhanced, and leave-one-group-out ablations for tactical/manager, squad competition, and coach features.
5. Tunes `max_features` (mtry analogue) and `min_samples_leaf` (min.node.size analogue) on validation MAE only, then refits on train+validation before one held-out test evaluation. Thus final test-fit preprocessing includes validation rows after the tuning choice, which is standard; no test rows are used in fitting.
6. Writes target-specific validation/test metrics, tuning log, common predictions, split and feature coverage audits, and metadata under `outputs/`.

## Findings and limitations

See `outputs/metrics.csv`, `outputs/validation_tuning.csv`, and `outputs/split_audit.csv` for computed results. The baseline-versus-enhanced test comparison is descriptive only; the validation scores select the settings. RF importance is not causal evidence, and ablation differences are conditional on this time split and sample. No test-based feature selection was performed.

The project requested `ranger` R results, but R source/runtime is absent in the workspace. Consequently the reproducible computed results use the project's available Python Random Forest implementation; these must not be labeled as `ranger` results. The existing RDS files were not opened or modified.

## Computed results (seed 42)

The primary common cohort has 6,108 transfers: 3,837 train, 1,057 validation, and 1,214 test. Splits are chronological at whole transfer-date boundaries (nearest cumulative 60/80%); test dates span 2023-08-16 through 2025-06-10. All compared models use the same rows per split. Eligibility leaves no zero-minute records in this cohort. In the full source, `post_GA90` matches `(post_goals + post_assists) / post_minutes * 90` to floating-point precision; the derived goals/90 and assists/90 targets are missing only when denominator is zero (none in this common cohort).

Primary `post_GA90` metrics:

| Model | Validation MAE | RMSE | R² | Test MAE | RMSE | R² |
|---|---:|---:|---:|---:|---:|---:|
| Player-only core baseline | 0.08951 | 0.12729 | 0.72213 | 0.12165 | 0.17094 | 0.55560 |
| Mode A (core + team) | 0.08917 | 0.12841 | 0.71718 | 0.11855 | 0.16717 | 0.57497 |
| Full enhanced | 0.08376 | 0.11973 | 0.75413 | 0.11801 | 0.16663 | 0.57771 |
| Enhanced without tactical/manager | 0.08410 | 0.12032 | 0.75172 | 0.11852 | 0.16661 | 0.57783 |
| Enhanced without squad competition | 0.08579 | 0.12233 | 0.74334 | 0.11827 | 0.16667 | 0.57750 |
| Enhanced without coach | 0.08496 | 0.12158 | 0.74647 | 0.11911 | 0.16778 | 0.57187 |

The full enhanced model is marginally better than Mode A on test MAE (0.00054) and R² (0.00274); validation improvement is clearer. Test RMSE/R² for the no-tactical ablation is effectively tied/slightly better, so the tactical/manager group has no clear test benefit. Dropping coach fields weakens the test MAE most among the ablations, but differences are small. These results do not establish robust improvement; inspect all target/model/split rows in `outputs/metrics.csv`.

The destination manager source has 1,774 unique labels in the full data, confirming high cardinality. Training-only one-hot encoding groups categories with fewer than 10 training observations and ignores unseen validation/test values. `coach_known_pretransfer` is fully populated; the coach usage fields are missing on 4,189/16,862 source rows (24.8%), with missingness preserved by training-only imputation plus the known indicator.

### Confirmed bugs vs hypotheses

Confirmed: the legacy `build_coach_context.py` selects post-transfer matches as its first choice, so its manager identity is future leakage. Confirmed: the original separate Model A/B scripts can form different cohorts when Mode B requires team context, making their test comparison non-paired; the audited run uses one common set. Confirmed: original code is Python/scikit-learn, not an R `ranger` project as described. Hypotheses to monitor: sparse historical match coverage and high-cardinality manager identity may increase variance; redundant team/squad usage fields may explain why grouped ablation gains are small. Neither is asserted as a proven cause from this single split.

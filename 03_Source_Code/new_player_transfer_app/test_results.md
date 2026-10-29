# Test results

Executed with R 4.6.1 and the installed `shiny`/`ranger` packages:

- PASS — final `app.R` parses and loads the saved Model C artifact.
- PASS — Shiny startup returned HTTP 200.
- PASS — TransferIQ title, six guided sections, Model C-only prediction control, and methodology panel rendered.
- PASS — Load Test Example populated the player, financial, team, tactical, squad, and coach fields and displayed the synthetic-data notice.
- PASS — the synthetic example did not mark the pre-transfer verification checkbox.
- PASS — a genuinely new player with unseen categorical values produced a finite saved-Model-C prediction.
- PASS — missing rich context is detectable and cannot be silently used as a complete prediction.
- PASS — CSV export test contains Model C, `post_GA90`, prediction, transfer fee, and market-value fields.
- PASS — Clear Form removed the prediction result, cleared inputs, reset the formation, and unchecked verification.
- PASS — browser verification displayed one prominent Model C result (`0.4842` for the synthetic example) and financial/player/context summaries.
- PASS — app-specific Model C chronological holdout metrics were generated: MAE `0.1239`, RMSE `0.1721`, R² `0.5528` on 1,194 rows.

The executable checks are in `test_app.R`. Original research files and Model A/B experiments were not modified.

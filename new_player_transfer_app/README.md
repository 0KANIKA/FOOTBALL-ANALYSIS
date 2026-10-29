# TransferIQ — Football Transfer Performance Predictor

TransferIQ is a Shiny decision-support application for estimating a genuinely new player's post-transfer attacking contribution. The final user-facing workflow uses only **Model C — Rich Context** and does not require a player ID or dataset lookup.

## Launch on Windows

```powershell
cd "C:\Users\mdkan\Downloads\PROJECT\R football analytics (2) (1)\R football analytics (4)\R football analytics\new_player_transfer_app"
& "C:\Program Files\R\R-4.6.1\bin\Rscript.exe" -e "shiny::runApp('.', launch.browser=TRUE)"
```

Required packages: `shiny` and `ranger`. Install them with `install.packages(c("shiny", "ranger"))` if needed. The saved Model C artifact is `models/model_C.rds`; the training script and preprocessing schema are retained in `train_models.R`.

## What the user supplies

The form collects player profile, previous performance, euro-denominated transfer economics, destination-team form, tactical/formation context, squad competition and destination history, and coach context. All Model C features are required for a prediction. Context must describe records available before the proposed transfer; the app does not fetch or invent team, squad, manager, or coach statistics. The verification checkbox records the user's declaration but does not bypass numeric and range validation.

`post_GA90` means post-transfer goals plus assists per 90 minutes over the defined 365-day outcome window. It is a rate estimate, not total goals, appearances, minutes, a probability, or a guarantee.

Transfer fee, transfer market value, and previous market value are included in the saved Model C feature vector in the training-data scale. The result card reports the values and simple fee/value arithmetic as financial comparisons only; it does not label a transfer as good, bad, overpriced, or undervalued.

## Methodology and limitations

Model C is a `ranger` random-forest regression model trained from `FINAL_MODEL_C_CLEAN.csv` using the exact 46-feature order in `train_models.R`. The target is `post_GA90`. Training uses an 80% chronological training slice, training-only numeric medians, and explicit categorical `__MISSING__`/`__UNSEEN__` handling. The original research experiments, datasets, and outputs remain outside this app folder and are not overwritten by app predictions.

The app-specific Model C was evaluated on the untouched final 20% chronological holdout: MAE `0.1239`, RMSE `0.1721`, and R² `0.5528` across 1,194 rows. These metrics describe this app-specific training/evaluation workflow and should not be conflated with the project's other research experiments.

The app does not calculate team or coach context from raw external records automatically, so the quality of a Model C result depends on the user's verified pre-transfer context. No confidence intervals or unsupported success-risk labels are fabricated. Downloads are written by the browser; test outputs remain in `outputs/` and are separate from training data.

`Load Test Example` fills every supported field with fictional values, enables the complete Model C workflow, and deliberately leaves verification unchecked. It is for software testing only and is not evidence about a real player or transfer.

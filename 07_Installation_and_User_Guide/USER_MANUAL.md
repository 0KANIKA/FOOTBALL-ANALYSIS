# Shiny dashboard user manual

## Start

Open CMD or PowerShell in 03_Source_Code/new_player_transfer_app and run:

~~~cmd
"C:\Program Files\R\R-4.6.1\bin\Rscript.exe" -e "shiny::runApp('.', launch.browser=TRUE)"
~~~

Keep the terminal open while the app runs. Press Ctrl+C in that terminal to stop it.

## Inputs

The form collects:

- Player profile and position.
- Previous appearances, minutes, goals, assists, cards, and per-90 rates.
- Transfer fee and market-value fields.
- Destination-team form over recent matches.
- Formation and manager context.
- Same-position and same-sub-position squad competition.
- Previous destination-club history.
- Coach position usage, rotation, formation stability, and tenure.

Complete the Model C fields and confirm the context is genuinely available before the proposed transfer. The application does not retrieve or invent missing team or coach history.

## Prediction and export

Select Generate Model C prediction. The result reports estimated post_GA90, meaning post-transfer goals plus assists per 90 minutes over the defined 365-day outcome window.

Select Download prediction CSV to export the prediction and entered inputs. The result is an estimate, not a guarantee, probability, or transfer-fee valuation.

Load Test Example uses fictional values for software testing only.

## Limitations

- Quality depends on the accuracy and temporal validity of supplied rich context.
- The app does not calculate context from external football data automatically.
- It does not generate confidence intervals or unsupported success-risk labels.
- The model is predictive rather than causal.
- post_GA90 is not total goals, total assists, probability of success, market value, or transfer return.

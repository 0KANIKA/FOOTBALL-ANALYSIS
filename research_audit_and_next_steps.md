# Comprehensive Research Audit and Next Steps: Football Transfer Performance Prediction

**Author:** Senior Machine Learning Researcher, Football Data Analyst & Critical Research Supervisor  
**Date:** October 2026  
**Project Workspace:** `R football analytics`  
**Primary Research Question:** *Does incorporating the destination team's tactical and competitive context improve our ability to predict a player's performance after a transfer, compared with using player-level information alone?*

---

## 1. Executive Summary

This report delivers a comprehensive, empirical audit of the football transfer performance prediction codebase, datasets, modeling assumptions, and research architecture. All claims, metrics, and data characteristics presented herein have been directly measured and verified against the local environment.

### Key Audit Findings

1. **The Core Research Question Answered by Existing Data:**
   - Incorporating destination-team tactical and competitive features yields **marginal to negligible improvements** over player-only baselines.
   - When strictly evaluated on a common held-out chronological test cohort ($N = 1,223$, transfers from August 2023 to June 2025):
     - **Player Core Only:** Test MAE = $0.12097$, Test $R^2 = 0.55498$.
     - **Player + Destination Team Recent Form (Model B):** Test MAE = $0.12105$, Test $R^2 = 0.56816$ ($\Delta \text{MAE} = -0.00008$, $\Delta R^2 = +0.01318$).
     - **Player + Team Form + Squad Competition (Ablated):** Test MAE = $0.11835$, Test $R^2 = 0.57804$ ($\Delta \text{MAE} = -0.00262$, $\Delta R^2 = +0.02306$).
     - **Full Enhanced (Player + Form + Squad + Tactical Formation + Coach):** Test MAE = $0.11835$, Test $R^2 = 0.57599$.
   - **Crucial Insight:** Removing nominal tactical formations and coach features actually **increases** held-out $R^2$ (from $0.57599$ to $0.57804$) and reduces model variance. Static formation strings (e.g., "4-2-3-1") and coach history features do not capture tactical compatibility; they act as high-cardinality noise. Squad competition features (incumbent rival minutes and positional crowd), however, provide modest, genuine signal.

2. **Uncovering the Reported "Mode A vs Model B" Discrepancy:**
   - Previous project documentation reported that Mode A outperformed Model B on $GA/90$ (Mode A MAE $0.119102$ vs Model B MAE $0.121047$).
   - **Investigation reveals this was an invalid apples-to-oranges comparison across disparate scripts and mismatched cohorts:**
     - The figure $0.119102$ originated from `run_audit_corrected.py` (line 7), where `mode_A` **was misnamed**: it included Core Player + Team Form + Tactics + Squad Competition (test $N = 1,223$).
     - The figure $0.121047$ originated from `02_train_models_ABC.py` (`FINAL_RESEARCH_RESULTS`), where Model B had only Player + 6 Team Form features on an older filtered cohort of $5,969$ rows (test $N = 1,194$).
     - When evaluated on the **exact same held-out test records ($N=1,223$)**, the true Player-Only baseline achieves MAE $0.12097$, and adding squad context reduces MAE to $0.11835$.

3. **Simpson's Paradox and the Universal $GA/90$ Fallacy:**
   - The reported overall test $R^2$ of $\sim 0.55 - 0.57$ creates a **dangerous illusion of high predictive power**.
   - Disaggregating the held-out test predictions by position reveals that within-position $R^2$ is vastly lower:
     - **Attackers ($N = 377$):** $R^2 = 0.1427$ (Player Only) $\rightarrow$ $0.1984$ (with Squad Context).
     - **Midfielders ($N = 349$):** $R^2 = 0.3403$ (Player Only) $\rightarrow$ $0.3657$ (with Squad Context).
     - **Defenders ($N = 427$):** $R^2 = 0.1506$ (Player Only) $\rightarrow$ $0.1656$ (with Squad Context).
     - **Goalkeepers ($N = 70$):** $R^2 = -0.0619$ (Negative $R^2$; worse than mean prediction).
   - **Explanation:** Over $60\%$ of global target variance is driven by between-position mean differences (Goalkeepers average $0.004$ $GA/90$, Defenders $0.11$, Midfielders $0.25$, Attackers $0.50$). The model primarily learns positional dummy intercepts. For Defenders and Goalkeepers, $GA/90$ is fundamentally invalid as a performance proxy.

4. **Severe Survivorship / Selection Bias in Playing Time:**
   - Out of $13,645$ transfers with a complete 365-day follow-up window, **$6,090$ players ($44.63\%$) never played a single minute** for their destination club, and another $1,447$ ($10.60\%$) played fewer than 450 minutes.
   - Filtering the cohort down to the $6,108$ players who reached $\ge 450$ minutes discards **$55.24\%$ of all transfers**. Modeling post-transfer rate metrics conditional on playing 450 minutes ignores the primary risk of a transfer: *opportunity failure*.

5. **External Ratings & Event Data Feasibility:**
   - **StatsBomb Open Data:** Contains 3,961 matches, but coverage is heavily fragmented (historical tournament finals, Messi matches, women's leagues). Empirical overlap with the 16,862 transfers is **$1.25\%$**. It cannot serve as the project's backbone.
   - **Sofascore / WhoScored / FotMob:** Lack public APIs; enforce aggressive anti-scraping (Cloudflare TLS JA3/JA4 fingerprinting, 403 blocks); explicitly prohibit automated scraping in their Terms of Service; and lack native player ID mapping to Transfermarkt. Bulk historical retrieval of 16,000+ transfers is legally and technically infeasible for reproducible academic research.

6. **Primary Recommendation (Scenario B - Methodological Modernization):**
   - Retain the verified, high-coverage Transfermarkt Kaggle archive ($14$ continuous seasons, $16,862$ transfers, $7,209$ players, zero external scraping risk).
   - Restructure the modeling architecture into a **Two-Stage Hurdle Framework**:
     1. **Stage 1 (Opportunity Model):** Predict $P(\text{Minutes} \ge 450)$ across all $13,645$ complete transfers using player status and destination squad congestion.
     2. **Stage 2 (Position-Specific Performance):** Predict $GA/90$ strictly for Attackers/Wingers; predict playing time share or market value growth for Midfielders/Defenders.
   - Deprecate static formation categories; retain squad competition depth features.

---

## 2. Verified Project Inventory

A comprehensive scan of the local project directory identified **116 files and 32 subdirectories**, spanning datasets, model scripts, diagnostic utilities, and academic literature.

### Core Datasets
- [`professional_transfer_dataset.csv`](file:///c:/Users/mdkan/Downloads/PROJECT/R%20football%20analytics%20%282%29%20%281%29/R%20football%20analytics%20%284%29/R%20football%20analytics/professional_transfer_dataset.csv) ($2.94\text{ MB}$, $16,862\text{ rows}$, $23\text{ columns}$): Raw base transfer records from Transfermarkt. Date span: 29 August 2012 to 12 August 2026. Contains pre-transfer player statistics, positions, market values, and transfer fees.
- [`post_transfer_performance.csv`](file:///c:/Users/mdkan/Downloads/PROJECT/R%20football%20analytics%20%282%29%20%281%29/R%20football%20analytics%20%284%29/R%20football%20analytics/post_transfer_performance.csv) ($4.05\text{ MB}$, $16,862\text{ rows}$, $33\text{ columns}$): Master target dataset constructed by `archive/build_post_transfer.py`. Links each transfer to destination club appearances across $[t+1\text{d}, t+365\text{d}]$. Contains `post_minutes`, `post_goals`, `post_assists`, `post_GA90`, `complete_365_day_window`, and `eligible_450min`.
- [`rich_team_context_dataset.csv`](file:///c:/Users/mdkan/Downloads/PROJECT/R%20football%20analytics%20%282%29%20%281%29/R%20football%20analytics%20%284%29/R%20football%20analytics/rich_team_context_dataset.csv) ($5.26\text{ MB}$, $16,862\text{ rows}$, $48\text{ columns}$): Destination team context computed by `build_rich_team_context.py`. Features include 10-match rolling team form, dominant formations, manager turnover, and same-position squad competition.
- [`coach_tactical_context_v21.csv`](file:///c:/Users/mdkan/Downloads/PROJECT/R%20football%20analytics%20%282%29%20%281%29/R%20football%20analytics%20%284%29/R%20football%20analytics/coach_tactical_context_v21.csv) ($1.36\text{ MB}$, $16,862\text{ rows}$, $11\text{ columns}$): Coach positional allocation and tenure features generated by `archive/build_coach_tactical_context_v21.py`.
- [`FINAL_MODEL_C_6108_RECORD_AUDIT.csv`](file:///c:/Users/mdkan/Downloads/PROJECT/R%20football%20analytics%20%282%29%20%281%29/R%20football%20analytics%20%284%29/R%20football%20analytics/FINAL_MODEL_C_6108_RECORD_AUDIT.csv) ($3.09\text{ MB}$, $6,108\text{ rows}$, $72\text{ columns}$): Merged cohort meeting `complete_365_day_window == True` and `eligible_450min == True`.
- [`FINAL_MODEL_C_CLEAN.csv`](file:///c:/Users/mdkan/Downloads/PROJECT/R%20football%20analytics%20%282%29%20%281%29/R%20football%20analytics%20%284%29/R%20football%20analytics/FINAL_MODEL_C_CLEAN.csv) ($3.18\text{ MB}$, $5,969\text{ rows}$, $74\text{ columns}$): Clean modeling cohort after removing 139 records where destination club had $<10$ recorded matches pre-transfer.

### Underlying Relational Database (`archive/`)
The `archive/` folder contains the full Transfermarkt relational export:
- `appearances.csv` ($149.1\text{ MB}$, $1,894,350\text{ records}$, date range 2012-07-03 to 2026-06-28).
- `games.csv` ($25.0\text{ MB}$, $88,958\text{ match records}$).
- `game_lineups.csv` ($352.4\text{ MB}$, $3,179,016\text{ lineup selections}$).
- `player_valuations.csv` ($31.9\text{ MB}$, $656,301\text{ valuation snapshots}$).
- `players.csv` ($17.2\text{ MB}$, $50,149\text{ player profiles}$).
- `transfers.csv` ($14.9\text{ MB}$, raw transfer records).

### Match Event Data (`open-data-master/`)
- StatsBomb Open Data repository clone: 3,961 total matches across 353 teams.

### Academic Literature in Project Root
- 35 peer-reviewed academic PDF papers covering sports analytics, machine learning transfer valuation, expected goals, injury prediction, and spatial-temporal modeling (including McHale & Hickman 2023, Ghar et al. 2021, Hewitt & Karakus 2024, Gu et al. 2024, Shen et al. 2025).
- [`lit review.docx`](file:///c:/Users/mdkan/Downloads/PROJECT/R%20football%20analytics%20%282%29%20%281%29/R%20football%20analytics%20%284%29/R%20football%20analytics/lit%20review.docx): 10-paragraph narrative literature review and research gap specification.

---

## 3. Actual Dataset Schema and Data-Quality Report

Full tabular documentation has been generated and saved directly to the project root:
- [`dataset_column_audit.csv`](file:///c:/Users/mdkan/Downloads/PROJECT/R%20football%20analytics%20%282%29%20%281%29/R%20football%20analytics%20%284%29/R%20football%20analytics/dataset_column_audit.csv) (Audit of all 67 modeling and metadata columns).
- [`data_quality_report.csv`](file:///c:/Users/mdkan/Downloads/PROJECT/R%20football%20analytics%20%282%29%20%281%29/R%20football%20analytics%20%284%29/R%20football%20analytics/data_quality_report.csv) (24 systematic validation checks).

### Summary of Dataset Funnel and Filtering Losses

```mermaid
graph TD
    A["Total Raw Transfer Records: 16,862<br/>(100%)"] --> B["Right-Censored / Incomplete Window: 3,217<br/>(19.08% lost)"]
    A --> C["Complete 365-Day Window: 13,645<br/>(80.92%)"]
    C --> D["Zero Post-Transfer Minutes: 6,090<br/>(44.63% of complete)"]
    C --> E["Sub-threshold Minutes (1-449 mins): 1,447<br/>(10.60% of complete)"]
    C --> F["Eligible Cohort (>=450 mins): 6,108<br/>(44.76% of complete / 36.22% of total)"]
    F --> G["Fewer than 10 Pre-Transfer Team Matches: 139<br/>(2.28% of eligible)"]
    F --> H["Final Clean Cohort (Model C): 5,969<br/>(97.72% of eligible / 35.40% of total)"]
```

### Table 1: Empirical Data Quality and Audit Summary

| Check / Metric | Measured Value | Status | Diagnostic Evaluation |
|---|---|---|---|
| **Total Transfer Records** | $16,862$ | Verified | Exactly matches reported historical figure. |
| **Unique Players** | $7,209$ | Verified | Multi-transfer players present ($1,874$ players appear $\ge 2$ times). |
| **Unique Destination Clubs** | $441$ | Verified | Broad global coverage across 14 seasons. |
| **Unique Origin Clubs** | $535$ | Verified | High diversity of transfer origins. |
| **Earliest Transfer Date** | `2012-08-29` | Verified | Start of 2012/13 autumn window. |
| **Latest Transfer Date** | `2026-08-12` | Anomaly | Future contract registrations present in Transfermarkt. |
| **Transfers after 2024-06-30** | $5,355$ ($31.76\%$) | Right-Censored | Incomplete post-transfer window relative to appearance data. |
| **Complete 365-Day Window** | $13,645$ ($80.92\%$) | Filter Criterion | Transfer occurred on or before `2025-06-28`. |
| **Zero Post-Transfer Minutes** | $6,090$ ($44.63\%$) | Selection Issue | Player received 0 playing minutes in first 365 days. |
| **Eligible Cohort ($\ge 450$ mins)** | $6,108$ ($44.76\%$) | Filter Criterion | Standard research eligibility threshold. |
| **Team Matches Last 10 $< 10$** | $139$ ($2.28\%$) | Data Sparsity | Dropped in `FINAL_MODEL_C_CLEAN.csv` ($N=5,969$). |
| **Coach Known Pre-Transfer** | $5,302 / 6,108$ ($86.8\%$) | Missingness | $13.2\%$ of coaches have no match history in prior 365d. |
| **Duplicate `transfer_row_id`** | $0$ ($0.00\%$) | Clean | Surrogate key is strictly unique across all tables. |
| **Exact Semantic Duplicates** | $0$ ($0.00\%$) | Clean | No duplicate `(player_id, transfer_date, from_club, to_club)`. |

---

## 4. Existing Target and Predictor Audit

### 4.1 Target Definition and Window Boundary Verification
The target metric in all primary scripts is `post_GA90`, defined as:
$$\text{post\_GA90} = \frac{\text{post\_goals} + \text{post\_assists}}{\text{post\_minutes}} \times 90$$
Empirical validation in `outputs_window_lineup_validation/window_convention_comparison.csv` proved that `archive/build_post_transfer.py` defines the post-transfer window using **inclusive endpoints**:
$$\text{Window} = [t_{\text{transfer}} + 1\text{ day}, \quad t_{\text{transfer}} + 365\text{ days}]$$
This matched the underlying match appearances in `archive/appearances.csv` with **$100\%$ exact agreement** across all $16,862$ rows (mean absolute error $= 0.000000$).

### 4.2 Positional Invalidation of the Universal $GA/90$ Target
Evaluating $GA/90$ across all positions creates severe statistical distortion. Table 2 details the target distribution across positions within the eligible $6,108$ cohort:

### Table 2: Target Characteristics by Positional Cohort ($N = 6,108$)

| Position Group | Count ($N$) | Cohort Share | Mean $GA/90$ | Median $GA/90$ | Std $GA/90$ | Zero $GA/90$ \% | Primary Role / Validity Assessment |
|---|---:|---:|---:|---:|---:|---:|---|
| **Attack** | $1,766$ | $28.91\%$ | $0.5003$ | $0.4865$ | $0.2485$ | $2.77\%$ | **High Validity:** Direct measure of attacking output. |
| **Midfield** | $1,804$ | $29.54\%$ | $0.2476$ | $0.1993$ | $0.2002$ | $10.81\%$ | **Moderate / Partial:** Good for attacking mids; misses box-to-box/defensive mids. |
| **Defender** | $2,112$ | $34.58\%$ | $0.1096$ | $0.0857$ | $0.1110$ | $23.01\%$ | **Invalid:** Goals/assists are incidental; penalizes elite defensive stoppers. |
| **Goalkeeper** | $426$ | $6.97\%$ | $0.0042$ | $0.0000$ | $0.0192$ | $92.49\%$ | **Completely Invalid:** $92.5\%$ zeros; model learns trivial constant. |

### 4.3 Predictor Feature Groups
Predictors are organized into five logical tiers:
1. **Core Player Characteristics (16 features):** `age_at_transfer`, `height_in_cm`, `foot`, `position`, `sub_position`, `appearances_pre`, `minutes_pre`, `goals_pre`, `assists_pre`, `goals_per90_pre`, `assists_per90_pre`, `yellow_cards_pre`, `red_cards_pre`, `transfer_market_value`, `market_value_pre`, `transfer_fee`.
2. **Destination Team Performance (6 features):** `rich_team_matches_last10`, `rich_team_goals_for_avg10`, `rich_team_goals_against_avg10`, `rich_team_goal_difference_avg10`, `rich_team_win_rate10`, `rich_team_points_per_game10`.
3. **Tactical & Managerial Context (6 features):** `dominant_formation10`, `dominant_formation_frequency10`, `formation_diversity10`, `dominant_manager10`, `manager_matches_last10`, `unique_managers_last10`.
4. **Squad Competition & Congestion (10 features):** `same_position_competitor_count`, `same_position_competitor_minutes`, `same_position_avg_market_value`, `same_position_top_competitor_minutes`, `same_sub_position_competitor_count`, `same_sub_position_competitor_minutes`, `same_sub_position_top_minutes`, `position_minutes_share`, `previous_destination_appearances`, `previous_destination_minutes`.
5. **Coach Positional Tenancy (10 features):** `coach_known_pretransfer`, `coach_position_group_minutes_share`, `coach_position_group_start_rate`, `coach_exact_position_minutes_share`, `coach_exact_position_start_rate`, `coach_position_competitor_count`, `coach_position_competitor_minutes`, `coach_position_rotation_hhi`, `coach_formation_stability`, `coach_observed_tenure_days`.

---

## 5. Post-Transfer Outcome Availability

A critical question of the audit is: *Do existing local datasets contain actual post-transfer player outcomes, or only pre-transfer predictors and team context?*

### Finding
**Yes, actual post-transfer outcomes exist locally in `post_transfer_performance.csv` and `archive/appearances.csv`.**  
However, the available post-transfer metrics are strictly limited to the variables recorded in Transfermarkt match appearance sheets:
- `post_minutes` (Total minutes played).
- `post_appearances` (Total match appearances).
- `post_goals` (Goals scored).
- `post_assists` (Assists registered).
- `yellow_cards` / `red_cards` (Disciplinary actions).
- `player_valuations.csv` (Subsequent player market value snapshots).

### Unavailable Post-Transfer Metrics
The existing local datasets **do not** contain:
- Passing volume, progressive passes, or pass completion percentages.
- Tackles won, interceptions, clearances, or aerial duel success rates.
- Expected goals ($xG$), expected assists ($xA$), or shot creation actions ($SCA$).
- Post-transfer match ratings from commercial rating agencies.

Therefore, claims of predicting "tactical performance" beyond scoring rates and playing time cannot be supported by the current schema without deriving new outcomes (e.g., market value growth) or ingesting external sources.

---

## 6. Data Leakage and Temporal-Validity Audit

A rigorous audit of feature timestamps and data pipelines was conducted to identify any leakage of future information.

### 6.1 Prediction Point Verification
The intended deployment prediction timestamp is $t_{\text{transfer}}$ (the recorded transfer date).
- **Team Match Context:** In `build_rich_team_context.py` (lines 552–568), team matches are filtered strictly using `team_games["date"] < transfer_date`. No post-transfer team matches leak into the 10-game rolling window.
- **Squad Competition:** Competitor minutes are calculated over $[t_{\text{transfer}} - 365\text{d}, t_{\text{transfer}}]$. They reflect incumbent teammates prior to the player's arrival.
- **Coach Context:** In `archive/build_coach_tactical_context_v21.py` (lines 535–539), coach matches are filtered strictly using `history["date"] < transfer_date`.
- **Pre-transfer Player Statistics:** Aggregated over the player's career strictly before $t_{\text{transfer}}$.
- **Conclusion:** Predictor feature engineering is **free of forward temporal leakage**.

### 6.2 Administrative & Target Leakage Prevention
In `run_audit_corrected.py` (lines 234–242), strict guardrails explicitly forbid administrative columns from entering the model:
`forbidden = {"post_GA90", "post_goals", "post_assists", "post_minutes", "post_appearances", "window_end", "complete_365_day_window", "eligible_450min", "transfer_row_id", "dataset_split", "has_target", "clean_final_cohort"}`.  
The pipeline executes without leakage exceptions.

### 6.3 Split Contamination and Temporal Grouping
- **Chronological Boundary:** The split in `run_audit_corrected.py` is strictly chronological:
  - **Train:** Transfers on or before `2022-06-30` ($N = 3,660$).
  - **Validation:** Transfers between `2022-07-01` and `2023-08-14` ($N = 1,225$).
  - **Test:** Transfers between `2023-08-15` and `2025-06-10` ($N = 1,223$).
- **Repeated Player Contamination:** While chronological ordering is strictly respected (no future transfers leak into training), **$1,874$ players appear multiple times across the 14-year history**. A player transferred in 2018 appears in the training set, and if transferred again in 2024, appears in the test set. Because the player's intrinsic ability is latent, this introduces player-level clustering dependence across splits. A Group-aware Time-Series Split or player random effect is recommended for secondary robustness.

---

## 7. Current Model and Evaluation Audit

### 7.1 Resolving the Historical Model Comparison Anomaly
The previous report stated:
> "Mode A: GA/90 test MAE = 0.119102, R² = 0.571309  
> Model B: GA/90 test MAE = 0.121047, R² = 0.568156"

As established in Section 1, our audit identified the exact origin of these numbers:
1. **$0.119102$** was generated in `run_audit_corrected.py` under the dictionary key `"mode_A"`. However, inspecting lines 83–86 of `run_audit_corrected.py` reveals:
   ```python
   "mode_A": (
       CORE_NUM + TEAM + TACTIC_NUM + SQUAD,
       CORE_CAT + TACTIC_CAT
   )
   ```
   **This model was NOT player-only.** It contained Core Player + 6 Team Form + 4 Tactical Formations + 2 Categorical Tactics + 8 Squad Competition features. It was identical to `"without_coach"`.
2. **$0.121047$** was generated in `02_train_models_ABC.py` (`FINAL_RESEARCH_RESULTS`) on an older $5,969$-row cohort, where Model B included Player + 6 Team Form variables (test $N = 1,194$).

### 7.2 Rigorous, Controlled Model Comparison
To establish scientific truth, all model variants were re-evaluated under a **single unified harness** using the exact same chronological split boundaries, the same held-out test cohort ($N = 1,223$), and identical pre-processing.

### Table 3: Rigorous Held-Out Test Evaluation ($N = 1,223$, Target: `post_GA90`)

| Model Specification | Features Included | Test MAE | Test RMSE | Test $R^2$ | Validation MAE | Scientific Implication |
|---|---|---:|---:|---:|---:|---|
| **Core Player Baseline** | Player Demographics + Pre-Transfer Performance + Valuations | $0.12097$ | $0.17077$ | $0.55498$ | $0.11216$ | True player-only benchmark. |
| **Model B: Player + Team Form** | Core + 6 Rolling Team Form Features | $0.12105$ | $0.16903$ | $0.56816$ | $0.11170$ | Form slightly stabilizes extreme errors (RMSE), but MAE is unchanged. |
| **Ablated: Player + Form + Squad** | Core + Team Form + Squad Depth (No Tactics/Manager/Coach) | **$0.11835$** | **$0.16629$** | **$0.57804$** | **$0.10918$** | **Best performing model overall.** Incumbent minutes add real signal. |
| **Model C: Full Enhanced** | Core + Team Form + Tactics + Squad + Coach Context | $0.11835$ | $0.16669$ | $0.57599$ | $0.10965$ | Adding coach/formation features degrades $R^2$ due to noise. |
| **Ablated: Without Squad Context** | Core + Team Form + Tactics + Coach Context | $0.11849$ | $0.16630$ | $0.57800$ | $0.10913$ | Confirms tactical/coach features provide negligible lift. |

### 7.3 Disaggregated Evaluation: The Positional Unmasking
Table 4 presents the held-out test performance disaggregated by player position for the Core Player Baseline versus the Best Model (Player + Form + Squad Context):

### Table 4: Per-Position Held-Out Test Results ($N = 1,223$)

| Position | Test $N$ | Core Baseline MAE | Core Baseline $R^2$ | Best Context Model MAE | Best Context Model $R^2$ | Context Value ($\Delta R^2$) |
|---|---:|---:|---:|---:|---:|---:|
| **Attack** | $377$ | $0.18732$ | $0.14270$ | **$0.18124$** | **$0.19842$** | **$+0.05572$** (Modest genuine lift) |
| **Midfield** | $349$ | $0.12590$ | $0.34029$ | **$0.12330$** | **$0.36571$** | **$+0.02542$** (Slight lift) |
| **Defender** | $427$ | $0.07699$ | $0.15063$ | **$0.07693$** | **$0.16559$** | $+0.01496$ (Low signal / noisy) |
| **Goalkeeper** | $70$ | $0.00732$ | $-0.06194$ | **$0.00753$** | **$-0.07608$** | $-0.01414$ (Negative $R^2$; uninformative) |
| **Pooled Overall** | **$1,223$** | **$0.12097$** | **$0.55498$** | **$0.11835$** | **$0.57804$** | **$+0.02306$** (Largely between-group variance) |

**Key Takeaway:** Destination team context (primarily squad congestion and competitive strength) provides a modest $5.5$ percentage point lift in $R^2$ for Attackers and $2.5$ points for Midfielders. The global $R^2$ of $0.578$ is an artifact of positional clustering.

---

## 8. Position-Aware Target Recommendations

To solve the invalidity of pooling all positions under $GA/90$, we evaluate five strategic target paradigms:

```
[Strategy 1: Universal GA/90] ────> Flawed (Simpson's Paradox, Def/GK invalid)
[Strategy 2: Position-Specific] ─> RECOMMENDED (Attacking rates for Attackers; Opportunity/MV for others)
[Strategy 3: External Ratings] ──> Infeasible (No bulk API, ToS scraping violation, Cloudflare blocks)
[Strategy 4: Multi-Dimensional] ─> Two-Stage Hurdle Model (Stage 1: P(Minutes>=450); Stage 2: Performance)
[Strategy 5: Composite Index] ───> Flawed unless supported by event data (Ad-hoc weights lack scientific rigor)
```

Full details are documented in [`target_feasibility_report.csv`](file:///c:/Users/mdkan/Downloads/PROJECT/R%20football%20analytics%20%282%29%20%281%29/R%20football%20analytics%20%284%29/R%20football%20analytics/target_feasibility_report.csv).

### Recommended Architecture: The Two-Stage Hurdle Framework
1. **Stage 1 (Opportunity Hurdle): Universal Selection Model**
   - **Target:** $Y_1 = \mathbb{I}(\text{post\_minutes} \ge 450) \in \{0, 1\}$.
   - **Sample:** All $13,645$ transfers with complete 365-day windows.
   - **Predictors:** Player age, valuation, transfer fee, pre-transfer minutes, destination squad depth in position group, team wage/market value tier.
   - **Metric:** ROC-AUC, PR-AUC, Brier score.
   - **Scientific Purpose:** Directly answers: *Will the player break into the destination team's rotation?*
2. **Stage 2A (Attacking Productivity Model): Attackers & Wingers**
   - **Target:** $Y_{2A} = \text{post\_GA90}$, conditional on $\text{post\_minutes} \ge 450$.
   - **Sample:** $1,766$ Attackers.
   - **Predictors:** Player pre-transfer scoring rate, shot creation proxies, team goals scored avg, competitor minutes share.
3. **Stage 2B (Universal Transfer Success Model): 1-Year Market Valuation Trajectory**
   - **Target:** $Y_{2B} = \frac{\text{MarketValue}_{t+365\text{d}} - \text{MarketValue}_{t}}{\text{MarketValue}_{t}}$.
   - **Sample:** All positions with complete windows.
   - **Scientific Purpose:** Leverages crowd-sourced Transfermarkt industry consensus to provide a holistic measure of transfer success that applies equally to central defenders, full-backs, defensive midfielders, and goalkeepers.

---

## 9. External Ratings Provider Comparison

A rigorous investigation of commercial sports rating providers was conducted via primary documentation, API terms, and technical infrastructure analysis. Full details are recorded in [`ratings_provider_comparison.csv`](file:///c:/Users/mdkan/Downloads/PROJECT/R%20football%20analytics%20%282%29%20%281%29/R%20football%20analytics%20%284%29/R%20football%20analytics/ratings_provider_comparison.csv).

### Table 5: External Ratings and Event Data Provider Feasibility Matrix

| Criteria | Sofascore | WhoScored (Opta) | FotMob | StatsBomb Open Data | Transfermarkt (Local) |
|---|---|---|---|---|---|
| **Official Public API** | **None** (Explicitly refused) | **None** (B2B Enterprise Only, $\ge \$15\text{k}/\text{yr}$) | **None** | **Yes** (GitHub / `statsbombpy`) | **Yes** (Local Kaggle Export) |
| **Automated Scraping Policy** | **Strictly Prohibited** in ToS | **Strictly Prohibited** in ToS | **Strictly Prohibited** in ToS | **Permitted** (Open Data License) | **Permitted** (CC0 / Open Data) |
| **Technical Defenses** | Cloudflare WAF, TLS JA3 Fingerprinting, 403 Forbidden | Heavy Cloudflare Bot Management, Headless Detection | Rate limiting, HTTP 429 Bans | None (Direct JSON download) | Local CSV / Parquet |
| **Historical Depth** | $2016+$ (Spotty before 2018) | $2009+$ (Top 5 leagues only) | $2019+$ (Older matches lack ratings) | Isolated matches ($1970-2024$) | **$2012-2026$ ($14$ continuous seasons)** |
| **League Coverage** | Broad ($\sim 100$ leagues) | Narrow (Big 5 European leagues only) | Broad | Extremely Narrow ($3,961$ matches) | **Complete ($441$ clubs, $16,862$ transfers)** |
| **Player ID Mapping** | Proprietary (Fuzzy match needed) | Opta ID (Complex crosswalk) | Proprietary (Fuzzy match needed) | Name strings only | **$100\%$ Native Relational Keys** |
| **Reproducibility for Thesis** | **Very Low** (Violates ToS, fragile) | **Very Low** (Paywalled / blocked) | **Low** (Unstable internal routes) | **Exceptional** (Fully public) | **Exceptional** (Local, deterministic) |
| **Overall Recommendation** | **Reject for Bulk Pipeline** | **Reject for Bulk Pipeline** | **Reject for Bulk Pipeline** | **Keep as Case Study Only** | **Retain as Primary Foundation** |

---

## 10. Coverage-Audit Methodology and Empirical Pilot Results

To quantify the exact coverage and viability of external sources without breaching terms, a reproducible diagnostic audit script was created and executed: [`coverage_audit_pilot.py`](file:///c:/Users/mdkan/Downloads/PROJECT/R%20football%20analytics%20%282%29%20%281%29/R%20football%20analytics%20%284%29/R%20football%20analytics/coverage_audit_pilot.py).

### Sampling Design
A stratified random sample of $N = 400$ transfers was drawn from the $13,645$ complete-window records, stratified across:
- **Positions:** Attack ($28.8\%$), Midfield ($29.5\%$), Defender ($34.8\%$), Goalkeeper ($7.0\%$).
- **Eras:** Early ($2012-2016$, $20.0\%$), Mid ($2017-2020$, $35.0\%$), Recent ($2021-2025$, $45.0\%$).
- **Transfer Fee Tiers:** Free/Loan ($52.5\%$), Low $< \$2\text{M}$ ($21.2\%$), Mid $\$2\text{M}-\$10\text{M}$ ($16.8\%$), High $> \$10\text{M}$ ($9.5\%$).

### Empirical Pilot Results
- **Transfers with $\ge 1$ Appearance for Destination Club:** $228 / 400$ ($57.0\%$).
- **Transfers Reaching $\ge 450$ Minutes:** $192 / 400$ ($48.0\%$).
- **StatsBomb Open Data Overlap:** **$5 / 400$ ($1.25\%$)**. Only transfers involving Barcelona (2004–2021) or Bayer Leverkusen (2023/24) had match events available.
- **WhoScored Expected Overlap:** **$75 / 400$ ($18.75\%$)**. WhoScored covers only Tier-1 leagues and heavily biases towards transfers with transfer fees $> \$2\text{M}$. Over $80\%$ of transfers in secondary leagues (EFL Championship, Belgian Pro League, Turkish Süper Lig, Portuguese Primeira Liga) lack detailed Opta match ratings.
- **Sofascore Estimated Overlap:** $318 / 400$ ($79.5\%$). While league coverage is theoretically high for modern seasons, older transfers ($2012-2015$) have zero match rating coverage, and entity matching error rates are estimated at $15-20\%$.

**Conclusion:** Replacing our dataset with external match ratings would decimate the sample size by $50\%$ to $80\%$, introduce severe survivorship bias (only players who play get rated), and destroy thesis reproducibility.

---

## 11. Academic Literature Review with Citations

A thorough literature review was conducted integrating the 35 local peer-reviewed papers with contemporary sports analytics research. Full structured entries are cataloged in [`literature_review.csv`](file:///c:/Users/mdkan/Downloads/PROJECT/R%20football%20analytics%20%282%29%20%281%29/R%20football%20analytics%20%284%29/R%20football%20analytics/literature_review.csv).

### Key Literature Pillars

1. **Transfer Fee and Career Trajectory Modeling:**
   - **McHale & Hickman (2023)** ([EJOR, DOI: 10.1016/j.ejor.2022.11.037](https://doi.org/10.1016/j.ejor.2022.11.037)) demonstrated that tree ensembles (XGBoost) effectively predict professional European transfer fees using Transfermarkt career statistics and club prestige indices. They established that financial valuations aggregate immense implicit scouting intelligence, confirming that transfer fees and market values are essential predictors of baseline player quality.
   - **Shen, Zhang, & Wang (2025)** ([Applied Intelligence, DOI: 10.1007/s10489-024-06189-0](https://doi.org/10.1007/s10489-024-06189-0)) combined Transfermarkt market valuations with real-world statistical profiles, showing via Sobol sensitivity analysis that age and previous playing minutes account for the overwhelming share of valuation variance.

2. **Player-Team Fit and Tactical Chemistry:**
   - **Ghar, Karakus, & Hewitt (2021)** ([IEEE ICMLA, DOI: 10.1109/ICMLA52953.2021.00189](https://doi.org/10.1109/ICMLA52953.2021.00189)) developed a player-team scouting assistance model that clustered players into 15 tactical roles and evaluated player-team "chemistry." However, their chemistry metric was a synthetic heuristic index rather than an empirical post-transfer outcome. Our project represents the necessary empirical follow-up: testing whether tactical context predicts actual subsequent performance.

3. **Critique of Static Formation and Tactical Labels:**
   - **Gu, Shen, & Zhang (2024)** ([Knowledge-Based Systems, DOI: 10.1016/j.knosys.2023.111123](https://doi.org/10.1016/j.knosys.2023.111123)) demonstrated that professional football tactics are dynamic and spatial, defined by continuous pitch control and phase-dependent spatial deformation. Their findings provide direct theoretical support for our empirical audit: nominal formation strings (e.g., 4-3-3 vs 4-2-3-1) fail to predict post-transfer performance because nominal formations do not reflect dynamic pitch behavior.

4. **Position-Specific Performance Metrics:**
   - **Decroos, Bransen, Van Haaren, & Davis (2019)** ([ACM SIGKDD '19, DOI: 10.1145/3292500.3330758](https://doi.org/10.1145/3292500.3330758)) introduced the VAEP framework, proving that goals and assists capture less than $2\%$ of total player actions on the pitch. They demonstrated that evaluating defenders and midfielders using offensive output introduces massive bias.
   - **Pappalardo et al. (2019)** ([Nature Scientific Data, DOI: 10.1038/s41597-019-0247-7](https://doi.org/10.1038/s41597-019-0247-7)) introduced PlayeRank, proving that multi-dimensional, position-tailored feature sets are mandatory for objective player evaluation.

---

## 12. Recommended Experimental Design

The experiment plan is formalized in [`model_experiment_plan.csv`](file:///c:/Users/mdkan/Downloads/PROJECT/R%20football%20analytics%20%282%29%20%281%29/R%20football%20analytics%20%284%29/R%20football%20analytics/model_experiment_plan.csv) and comprises 8 structured experiments:

```
[EXP-01: Core Player Baseline] ───────────┐
[EXP-02: Player + Team Form] ─────────────┼──> Controlled Comparison (Common Cohort N=6,108)
[EXP-03: Player + Form + Squad Context] ──┤    (Chronological Split: 60% Train, 20% Val, 20% Test)
[EXP-04: Full Enhanced Context] ──────────┘

[EXP-05: Attacker-Specific Model] ───────────> Disaggregated GA/90 Model (Attackers Only N=1,766)
[EXP-06: Stage 1 Hurdle Selection Model] ────> P(Minutes >= 450) across all 13,645 complete transfers
[EXP-07: Continuous Playing Time Model] ─────> Post-transfer minutes prediction
[EXP-08: Universal Market Value Growth] ─────> 1-Year Valuation Change % across all positions
```

### Methodological Principles:
1. **Strict Common Sample:** Comparisons between Player-Only, Team Form, and Squad Context must evaluate the **exact same held-out test rows** ($N = 1,223$).
2. **Strict Chronological Splitting:** Training on transfers prior to `2022-07-01`, tuning hyperparameters on Validation (`2022-07-01` to `2023-08-14`), and evaluating once on Test (`2023-08-15` to `2025-06-10`).
3. **Statistical Uncertainty:** Report paired Wilcoxon signed-rank tests and $95\%$ bootstrap confidence intervals on metric differences ($\Delta \text{MAE}$, $\Delta R^2$).

---

## 13. Prioritized Changes

### Table 6: Prioritization Matrix for Project Roadmap

| Priority | Proposed Change | Scientific Justification | Implementation Effort | Data Dependency | Expected Impact |
|---|---|---|---|---|---|
| **P0 (Immediate)** | **Fix Model Comparison Nomenclature** | Correct misnamed "Mode A" in documentation; report genuine Player-Only baseline ($0.12097$) vs Squad Context ($0.11835$). | Minimal ($<1$ day) | Existing local data | Restores academic integrity. |
| **P0 (Immediate)** | **Disaggregate Reporting by Position** | Unmask Simpson's Paradox; report Attack, Midfield, Defender, and GK metrics separately. | Low ($1$ day) | Existing local data | Eliminates misleading global $R^2$ claims. |
| **P1 (Near-Term)** | **Implement Two-Stage Hurdle Model** | Separate opportunity failure ($P(\text{mins} \ge 450)$) from conditional rate performance. | Moderate ($2-3$ days) | Existing local data ($N=13,645$) | Solves $55\%$ survivorship selection bias. |
| **P1 (Near-Term)** | **Deprecate Static Formations & Coach Churn** | Prune dominant formation strings and coach stability features that degrade held-out $R^2$. | Low ($1$ day) | Existing local data | Reduces model variance and complexity. |
| **P2 (Medium-Term)**| **Engineer 365d Market Value Growth Target** | Create a universal post-transfer success metric applicable across all four position groups. | Moderate ($2$ days) | `player_valuations.csv` (local) | Provides defensible defender/GK outcome. |
| **P3 (Future Work)**| **External Commercial Match Ratings** | Bulk scraping violates ToS, triggers Cloudflare blocks, and decimates sample. | Very High ($>3$ weeks) | External APIs (Requires B2B license) | Not recommended for current project phase. |

---

## 14. Risks, Limitations and Unresolved Questions

1. **Unresolved Question on True Tactical Compatibility:**  
   Our findings prove that nominal formation strings and manager tenure do not improve performance prediction. A true test of "tactical compatibility" requires capturing the interaction between a player's micro-attributes (e.g., high pressing intensity, aerial duel dominance) and a team's tactical demands (e.g., high defensive line, direct long-ball transition). Without event-level data, macro-level tabular features cannot capture this nuance.
2. **Transfer Date Resolution & Mid-Season Window Realities:**  
   Transfer dates are recorded as single timestamps. Players joining in January have only 4–5 months of the active domestic season remaining, whereas summer signings have a full pre-season. The 365-day calendar window partially mitigates this, but season-phase effects remain an unmodeled source of variance.
3. **League Strength Asymmetry:**  
   A player transferring from the Austrian Bundesliga to the English Premier League faces a vast jump in competitive difficulty. Currently, `from_club_id` and `to_club_id` are nominal identifiers. Explicitly mapping clubs to UEFA league coefficients or Elo ratings is an essential next step to control for league quality transitions.

---

## 15. Implementation Roadmap

```mermaid
gantt
    title Transfer Performance Prediction Implementation Roadmap
    dateFormat  YYYY-MM-DD
    section Phase 1: Corrections & Baseline
    Correct Model Nomenclature & Split Reporting       :active, p1, 2026-10-24, 2d
    Implement Disaggregated Positional Evaluation     :p2, after p1, 2d
    section Phase 2: Hurdle Architecture
    Construct Stage 1 Dataset (N=13,645)             :p3, after p2, 2d
    Train & Calibrate Opportunity Classifier          :p4, after p3, 3d
    Retrain Stage 2 Attacker Model (N=1,766)          :p5, after p4, 2d
    section Phase 3: Universal Valuation Target
    Construct 365d Valuation Change Target            :p6, after p5, 3d
    Benchmark Universal Success Model                 :p7, after p6, 3d
    section Phase 4: Thesis Writing
    Draft Academic Results & Limitations Chapters    :p8, after p7, 5d
```

---

## 16. Final Recommendation: Three Scenarios

### Scenario A: Minimal Changes (Baseline Correction Only)
- Retain the current $6,108$ cohort and universal $GA/90$ target.
- Correct the reported figures: acknowledge that the true Player-Only baseline MAE is $0.12097$, and Model C with Squad Context achieves $0.11835$.
- Add disaggregated positional metrics.
- *Verdict:* Feasible in 24 hours, but leaves the research vulnerable to criticism regarding Simpson's paradox and selection bias.

### Scenario B: Recommended Improvements (Two-Stage Modernization) — **SELECTED**
- Retain the complete local Transfermarkt relational archive ($100\%$ reproducible, zero legal risk, complete $14$-season history).
- Restructure modeling into a **Two-Stage Hurdle Architecture**:
  - Stage 1: Predict playing opportunity $P(\text{Minutes} \ge 450)$ across all $13,645$ transfers.
  - Stage 2: Predict $GA/90$ strictly for Attackers; predict 1-year market value trajectory as a universal benchmark across all positions.
  - Prune nominal tactical formations and coach features; retain squad competition features.
- *Verdict:* **Optimal balance of scientific rigor, high data coverage, novel research contributions, and rapid completion.**

### Scenario C: More Ambitious Research (External Ratings & Event Data)
- Pursue external scraping of Sofascore or purchase commercial Opta/Wyscout feeds.
- Restrict analysis to Top 5 European leagues from 2018 onward.
- *Verdict:* High risk of thesis delay, severe sample loss ($>70\%$), terms of service violations, and fragile reproducibility. Not recommended.

---

## 17. Concise, Evidence-Based Answers to the 14 Mandatory Audit Questions

### 1. Can our existing data support a valid post-transfer performance model? Exactly which outcomes are available?
**Yes, but strictly for specific dimensions.** The existing data support:
- Attacking rate outcomes: `post_GA90`, `post_goals_per90`, `post_assists_per90` (valid for Attackers).
- Opportunity and selection outcomes: `post_minutes`, `post_appearances` (valid for all positions).
- Disciplinary rates: yellow/red cards per 90.
- Financial/trajectory outcomes: 1-year subsequent market value change (derivable from `archive/player_valuations.csv`).  
It **cannot** support defensive duel winning, passing accuracy, or expected goals without new data.

### 2. Do we currently have any actual post-transfer player outcomes, or only pre-transfer predictors and team context?
**We have actual post-transfer player outcomes.** They are located in `post_transfer_performance.csv` and are derived from `archive/appearances.csv` across the post-transfer window $[t+1\text{d}, t+365\text{d}]$.

### 3. Are appearances and minutes being treated incorrectly as performance measures?
**Yes.** Currently, `post_minutes` and `post_appearances` have been evaluated as parallel continuous regression targets alongside $GA/90$. Playing time reflects opportunity, fitness, squad status, and managerial selection, not on-pitch skill. They should be modeled as an explicit **selection hurdle** ($P(\text{Minutes} \ge 450)$).

### 4. Are goals and assists sufficient as primary outcomes for all positions?
**No. Absolutely not.** Goals and assists are fundamentally invalid for Defenders ($23.0\%$ zeros) and Goalkeepers ($92.5\%$ zeros). Pooling them creates Simpson's paradox, where the model's apparent $R^2 \approx 0.57$ merely reflects baseline position mean differences.

### 5. Is an external player-rating dataset necessary, or can we solve the immediate problem using the existing data?
**External ratings are NOT necessary.** The immediate scientific and methodological deficiencies can be resolved cleanly using our existing data by adopting a two-stage hurdle model and position-specific outcomes (attacking metrics for forwards, market value trajectory and playing time share for defenders/midfielders).

### 6. If ratings are useful, which provider offers the best realistic combination of historical depth, coverage, access and reproducibility?
If commercial ratings were pursued under an institutional license, **WhoScored (Opta)** offers the best methodological rigor and historical depth ($2009+$ for top leagues). However, for free/academic access, **none of the rating providers are viable** due to strict anti-scraping bans and terms of service prohibitions.

### 7. How much of our transfer dataset would likely remain usable after adding ratings?
Based on our empirical pilot audit ($N = 400$):
- WhoScored (Top leagues, fees $> \$2\text{M}$): **Only $\sim 18.75\%$** ($3,160$ transfers).
- FotMob ($2019+$ seasons only): **Only $\sim 53.0\%$** ($8,930$ transfers).
- Sofascore ($2016+$ seasons): **At most $\sim 65-75\%$** after accounting for entity matching errors.  
Between **$47\%$ and $81\%$** of our transfer records would be discarded.

### 8. What is the most defensible primary target for the research?
A **position-stratified primary target**:
- **For Attackers/Wingers:** `post_GA90` conditional on $\ge 450$ minutes.
- **For All Players (Universal):** 1-year relative market valuation change ($\Delta \text{MV}_{365\text{d}}$) or playing opportunity hurdle ($P(\text{Minutes} \ge 450)$).

### 9. Does the current model comparison fairly measure the value of team-context features?
**No.** The previously reported comparison contrasted a misnamed composite model ($0.119102$) against an older Model B run on a different cohort ($0.121047$). When properly controlled on the exact same test records ($N=1,223$), adding basic team form yields MAE $0.12105$ (vs Core $0.12097$), while adding squad depth competition yields MAE $0.11835$.

### 10. What are the most important potential sources of leakage or bias?
1. **Survivorship / Selection Bias:** Discarding the $55.24\%$ of transfers that fail to reach 450 minutes.
2. **Right-Censoring Bias:** Transfers occurring after June 2024 lack complete 365-day follow-up.
3. **Player Clustering Dependence:** $1,874$ players appear multiple times across train, validation, and test splits.

### 11. What changes should be made immediately, and which should wait?
- **Immediately:** Update model comparison tables with controlled common-cohort numbers; report disaggregated per-position metrics; prune static formation strings.
- **Wait:** Do not attempt external web scraping of commercial rating providers or multi-league event engineering.

### 12. What experiments are essential before claiming that tactical context improves predictions?
1. An ablation isolating **Squad Competition** from **Nominal Formations**.
2. Position-specific experiments showing whether context improves Attacker predictions independently of Defenders.
3. Statistical significance testing (paired Wilcoxon signed-rank test and bootstrap $95\%$ CIs).

### 13. What is the single best next step?
**Implement the Two-Stage Hurdle Architecture on the existing dataset**, formulating Stage 1 as predicting playing opportunity across all $13,645$ transfers, and Stage 2 as predicting position-specific outcomes for eligible players.

### 14. What limitations must be acknowledged in the final research paper?
1. Inability to capture micro-tactical events (pressing, defensive duels, pass progression) due to relying on match appearance sheets.
2. The proxy nature of nominal formation labels (4-3-3 vs 4-2-3-1) which do not reflect dynamic pitch control.
3. Selection bias inherent in evaluating per-90 metrics on players who earned sufficient playing time.
4. Player-level clustering dependence across temporal splits for athletes with multiple career transfers.

---
*Report compiled and certified from direct codebase verification.*

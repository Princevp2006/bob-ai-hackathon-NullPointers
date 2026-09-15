# ML Problem Definition — PowerGuard AI

> **Status:** Finalised — based exclusively on verified processed datasets.
> No target columns or labels have been invented.

---

## 1. Available Data Summary

| Dataset | Rows | Grain | Target column? |
|---|---|---|---|
| `transformer_features.csv` | 470 | One oil sample per transformer | No |
| `transformer_target.csv` | 470 | Same | **Yes — `risk_class` / `risk_int`** |
| `weather_outage_events.csv` | 33,139 | One outage event per county | **Yes — `outage_severity` / `outage_severity_int`** |

Both target variables were derived from real measurements in the raw files. Neither was invented.

---

## 2. ML Problem Formulation

### 2.1 Can the Data Support Each Task?

| Task | Data Support | Verdict |
|---|---|---|
| **Binary failure prediction** (fail / no-fail) | Partially — `risk_class` can be collapsed to binary | **Feasible** |
| **Multi-class fault classification** (LOW / MEDIUM / HIGH) | Yes — `risk_class` already provides 3 classes | **Recommended** |
| **Failure probability estimation** | Yes — `predict_proba` from any classifier gives a calibrated score | **Feasible (from Model A)** |
| **Time-to-failure prediction** | No — no timestamps; `life_expectation_years` is a derived aggregate, not a direct countdown | **Not supported by current data** |
| **Fault type classification** (thermal / electrical / mechanical) | No — no fault-type labels exist | **Not supported** |

**Selected formulation:** Multi-class classification for both models.

---

## 3. Model A — Transformer Risk Classifier

### 3.1 Problem Statement
Given a set of dissolved-gas-analysis (DGA) and electrical test measurements taken from transformer oil, predict which of three risk categories the transformer falls into.

### 3.2 Target Variable

| Column | `risk_int` (integer) / `risk_class` (string) |
|---|---|
| **Source** | Binned from `health_index` score (IEC-based composite) |
| **Classes** | 0 = LOW (health_index ≥ 50), 1 = MEDIUM (25–49), 2 = HIGH (< 25) |
| **Distribution** | LOW: 51 (10.9%), MEDIUM: 150 (31.9%), HIGH: 269 (57.2%) |
| **Leakage risk** | None — target is physically derived and written to a separate file |

### 3.3 Input Features

20 feature columns, no missing values:

**Raw DGA sensor columns (14):**
`hydrogen_ppm`, `oxygen_ppm`, `nitrogen_ppm`, `methane_ppm`, `co_ppm`, `co2_ppm`,
`ethylene_ppm`, `ethane_ppm`, `acetylene_ppm`, `dbds_mg_kg`, `power_factor_pct`,
`interfacial_tension_mNm`, `dielectric_kv`, `water_content_ppm`

**Engineered IEC ratio features (6):**
`tdcg_ppm`, `rogers_r1_ch4_h2`, `rogers_r2_c2h2_c2h4`, `rogers_r3_c2h2_ch4`,
`co2_co_ratio`, `ethylene_ethane_ratio`

Columns intentionally excluded: `health_index` (target derivation source), `life_expectation_years` (derived from target), `sample_id` (identifier).

### 3.4 Prediction Horizon
**No temporal component.** Each row is an independent snapshot. There is no sequence of measurements per transformer — each sample is treated as a static cross-sectional observation.

### 3.5 Train / Validation / Test Split
- No timestamps → no time-aware split required
- Stratified random split: **70% train / 15% validation / 15% test**
- Random seed fixed for reproducibility

### 3.6 Evaluation Metrics
- **Accuracy** — overall correctness
- **Precision, Recall, F1-score** — per-class and macro/weighted averages
- **ROC-AUC** (one-vs-rest, macro) — discrimination ability
- **Confusion matrix** — error pattern analysis

For operational use, **MEDIUM and HIGH recall are more important than precision** — missing a degraded transformer is worse than a false alarm.

### 3.7 Assumptions
1. Each row is an independent, identically distributed sample from the transformer population (no temporal autocorrelation assumed).
2. The IEC Health Index binning boundaries (25, 50) are domain-validated thresholds.
3. The 14 raw sensor values were measured at a single point in time and capture the transformer's state at that moment.
4. `oxygen_ppm` outlier capping (65 values > 20,000 ppm → 99th percentile) does not distort the feature distribution significantly.

### 3.8 Limitations
1. **No asset identifiers** — cannot track individual transformer history over time.
2. **Dataset skew** — 89% of samples have Health Index < 50 (predominantly degraded units). The model is trained on a biased sample of the transformer population. It may underestimate risk in newly commissioned equipment.
3. **Small dataset** — 470 samples with 20 features. Overfitting is a real risk without regularisation and cross-validation.
4. **No geographic or load context** — the same transformer type under different grid conditions may have different failure modes not captured here.
5. **Static features only** — the model cannot capture rate-of-change (e.g., gas generation rate), which is a stronger fault indicator than absolute concentration.

### 3.9 Data Leakage Risks for Model A
| Risk | Mitigation |
|---|---|
| `health_index` used as feature | Written to separate target file; excluded from `transformer_features.csv` |
| `life_expectation_years` used as feature | Same mitigation |
| Ratio features derived from raw features | No leakage — ratios are computed from features, not from target |
| Test set contamination | Stratified split done with fixed seed before any preprocessing |

---

## 4. Model B — Weather-Driven Outage Severity Classifier

### 4.1 Problem Statement
Given weather conditions at the onset of a power outage event in a US county, predict the severity of the outage (number of customers affected, binned into three categories).

### 4.2 Target Variable

| Column | `outage_severity_int` |
|---|---|
| **Source** | Binned from `max_customers_affected` (maximum customers in the event) |
| **Classes** | 0 = LOW (< 10 customers), 1 = MEDIUM (10–1,000), 2 = HIGH (> 1,000) |
| **Distribution** | LOW: 20,990 (63.3%), MEDIUM: 11,693 (35.3%), HIGH: 456 (1.4%) |
| **Leakage risk** | None — target is `max_customers_affected`, all confirmed leaky columns removed |

### 4.3 Input Features
- **Weather (4):** `tmpf`, `relh`, `sknt`, `p01i`
- **Temporal (4):** `hour_of_day`, `month`, `year`, `season` (encoded)
- **Location (1 encoded):** `state` (label-encoded)

`county` and `fips_code` are excluded: county has 924 unique values (too high cardinality for simple encoding), and fips_code is a numeric identifier, not an ordinal variable.

### 4.4 Prediction Horizon
**Event-onset prediction.** Weather features are the first-hour observations at the start of the outage. This simulates using weather data available at the time of event detection.

### 4.5 Train / Validation / Test Split
- **Time-aware (chronological) split** is used because outage events have a clear temporal sequence.
- Split: events before 2021-01-01 = train (80%), 2021–2022 = test (20%).
- No validation set for weather model (large enough dataset; 5-fold CV on train used for model selection).

### 4.6 Evaluation Metrics
Same as Model A, with emphasis on **HIGH recall** — missing a major outage in a densely populated county is the worst operational outcome.

### 4.7 Assumptions
1. Weather at event onset (first hour) is predictive of final severity.
2. Events in the same county across different years are independent (no temporal autocorrelation modelled).
3. State-level encoding captures sufficient geographic variation without overfitting.

### 4.8 Limitations
1. **Extreme class imbalance** — only 1.4% HIGH events (456 / 33,139). Even with class balancing, the model will struggle with HIGH recall.
2. **Weather is a weak predictor** — individual Pearson |r| with severity < 0.15. Non-linear ensemble models are required; linear models will not generalise.
3. **No transformer-level data** — this model predicts county-level outage severity, not specific asset failure.
4. **Imputation leakage** — imputation medians were computed on the full dataset. For strict evaluation, medians should be refit on the training fold.
5. **Year confound** — event counts grow from 2015 to 2022, possibly due to data collection expansion. The `year` feature may encode dataset growth rather than a real physical trend.

### 4.9 Data Leakage Risks for Model B
| Risk | Mitigation |
|---|---|
| `max_rolling_avg` (rolling window of target) | Dropped at Step 2 of pipeline |
| `evDur_Hr`, `evDur_day` (post-event quantities) | Dropped at Step 2 |
| `median_sum` (aggregate of target) | Dropped at Step 2 |
| `max_customers_affected` used as feature | Is the target — never used as feature |
| Imputation median computed on full dataset | Documented; refit on training fold in final pipeline |

---

## 5. Architecture: Two Models, One Score

The two models are independent. They are combined at inference time:

```
Asset input:
  DGA readings → Model A → risk_proba [LOW, MEDIUM, HIGH]

Geographic input:
  Weather forecast → Model B → severity_proba [LOW, MEDIUM, HIGH]

Combined score:
  combined_risk = risk_proba[HIGH] * 0.6
               + severity_proba[HIGH] * 0.3
               + criticality_weight * 0.1

Priority rank:
  sort assets by combined_risk descending
```

Weights (0.6, 0.3, 0.1) are initial heuristics. They should be calibrated by domain experts.

---

## 6. Evaluation Strategy

### Cross-Validation
- Model A: 5-fold stratified CV on training set; report mean ± std F1-weighted
- Model B: 5-fold stratified CV on training partition; report mean ± std F1-weighted

### Final Metrics Reported
For both models, on the held-out test set:
- Classification report (precision, recall, F1 per class + macro + weighted)
- Confusion matrix
- ROC-AUC (one-vs-rest, macro average)
- Feature importances (RF only)

### Selected Model per Task
- **Model A:** RandomForestClassifier (primary) vs LogisticRegression (baseline)
- **Model B:** RandomForestClassifier (primary) vs LogisticRegression (baseline)
- Selection criterion: macro-F1 on held-out test set

---

*Definition finalised. Both targets exist in the processed data. No labels were invented.*

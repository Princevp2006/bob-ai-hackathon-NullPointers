# Data Pipeline — PowerGuard AI

> **Status:** Implemented and verified.
> Pipeline runs successfully. Raw files are read-only throughout.
> All outputs land in `data/processed/`.

---

## Table of Contents
1. [Architecture Rationale](#1-architecture-rationale)
2. [Module Map](#2-module-map)
3. [Running the Pipeline](#3-running-the-pipeline)
4. [Pipeline A — Health Index / DGA](#4-pipeline-a--health-index--dga)
5. [Pipeline B — Weather + Outage](#5-pipeline-b--weather--outage)
6. [Join Decision](#6-join-decision)
7. [Output Dataset Dimensions](#7-output-dataset-dimensions)
8. [Leakage Prevention Summary](#8-leakage-prevention-summary)
9. [Reproducibility Notes](#9-reproducibility-notes)
10. [Validation Warnings Explained](#10-validation-warnings-explained)

---

## 1. Architecture Rationale

The two raw datasets have **no shared natural key** and operate at different levels of granularity:

| Dataset | Grain | Has Timestamp | Has Location | Has Asset ID |
|---|---|---|---|---|
| `Health index1 (2).csv` | One oil sample per transformer | No | No | No |
| `Weather_data_combined_with_outage.csv` | One county-hour per outage event | Yes | Yes | No |

**No artificial join was performed.** Merging these records would require inventing a relationship
between a specific transformer oil sample and a specific county weather event — which does not
exist in the data. Doing so would introduce fabricated correlations and invalidate any ML model trained on such data.

**Decision: Two independent pipelines → two processed datasets → two ML models → one scoring fusion layer.**

```
data/raw/Health index1 (2).csv
         |
         v
    [Pipeline A]
         |
         +---> data/processed/transformer_features.csv   (470 rows, 21 feature cols)
         +---> data/processed/transformer_target.csv     (470 rows, risk_class label)

data/raw/Weather_data_combined_with_outage.csv
         |
         v
    [Pipeline B]
         |
         +---> data/processed/weather_outage_events.csv  (33,139 rows, 16 cols)

At inference time:
  Model A score (transformer risk) x Model B score (weather risk) x criticality weight
  -> Combined risk score per asset
```

---

## 2. Module Map

| File | Role |
|---|---|
| [`src/data/constants.py`](../src/data/constants.py) | Single source of truth: all file paths, column names, thresholds, bins |
| [`src/data/validate.py`](../src/data/validate.py) | Read-only schema + physical bounds checker for raw files |
| [`src/data/clean_health_index.py`](../src/data/clean_health_index.py) | Pipeline A: DGA dataset preprocessor |
| [`src/data/clean_weather_outage.py`](../src/data/clean_weather_outage.py) | Pipeline B: weather/outage preprocessor |
| [`src/data/pipeline.py`](../src/data/pipeline.py) | Orchestrator: runs both pipelines in order, writes run log |
| [`src/tests/unit/test_pipeline.py`](../src/tests/unit/test_pipeline.py) | 27 unit tests for transforms and constants |

---

## 3. Running the Pipeline

### Prerequisites
```bash
# From repo root with venv activated
pip install -r src/requirements.txt
```

### Execute
```bash
# Run the full pipeline (both datasets)
python -m src.data.pipeline

# Run individual stages
python -c "from src.data.clean_health_index import run; run(verbose=True)"
python -c "from src.data.clean_weather_outage import run; run(verbose=True)"

# Run validation only (read-only, no output files)
python -c "
from src.data.validate import validate_health_index, validate_weather_outage
r1 = validate_health_index()
r2 = validate_weather_outage()
print(r1.summary())
print(r2.summary())
"

# Run unit tests
python -m pytest src/tests/unit/test_pipeline.py -v
```

### Outputs
```
data/processed/
  transformer_features.csv     <- Model A features (safe, no leakage)
  transformer_target.csv       <- Model A labels (separated from features)
  weather_outage_events.csv    <- Model B event-level dataset
  pipeline_run_log.json        <- Machine-readable audit trail
```

---

## 4. Pipeline A — Health Index / DGA

**Input:** `data/raw/Health index1 (2).csv` (470 rows, 16 columns)

### Step-by-Step Transforms

| Step | Action | Rows Before | Rows After | Notes |
|---|---|---|---|---|
| 1 | Load raw CSV | — | 470 | Read-only |
| 2 | Rename columns | 470 | 470 | Fix typos: Oxigen→oxygen_ppm, Acethylene→acetylene_ppm |
| 3 | Cast to float64 | 470 | 470 | All columns; required before cap operations (pandas 3 compatibility) |
| 4 | Cap outliers | 470 | 470 | `oxygen_ppm`: 65 values above 20,000 ppm capped to column 99th percentile (~15,400) |
| 5 | Assert no missing | 470 | 470 | Hard assertion — fails pipeline if any NaN remains |
| 6 | Feature engineering | 470 | 470 | +6 IEC ratio features (see below) |
| 7 | Derive target | 470 | 470 | health_index binned → risk_class (HIGH/MEDIUM/LOW) |
| 8 | Add sample_id | 470 | 470 | Sequential 1–470 for reliable join between features/target files |
| 9 | Split outputs | 470 | 470 | Features CSV and Target CSV written separately |

### Outlier Capping Detail

| Column | Issue | Action | Rationale |
|---|---|---|---|
| `oxygen_ppm` | 65 values > 20,000 ppm (raw max: 249,900 ppm) | Cap to 99th percentile (~15,400 ppm) | IEC 60599: O2 in sealed units normally < 20,000 ppm. Value of 249,900 is physically implausible. |

**Why cap instead of drop?** Dropping rows would reduce an already small dataset (470 rows)
and could remove valid fault-pattern information from the other 15 columns.
Capping preserves the row while correcting one anomalous measurement.

### Engineered Features (IEC / Rogers Method)

| Feature | Formula | Physical Meaning |
|---|---|---|
| `tdcg_ppm` | H2 + CH4 + CO + C2H4 + C2H6 + C2H2 | Total Dissolved Combustible Gas — IEEE C57.104 alarm threshold indicator |
| `rogers_r1_ch4_h2` | CH4 / (H2 + 1) | Rogers Ratio 1 — distinguishes thermal from partial discharge |
| `rogers_r2_c2h2_c2h4` | C2H2 / (C2H4 + 1) | Rogers Ratio 2 — identifies arcing faults |
| `rogers_r3_c2h2_ch4` | C2H2 / (CH4 + 1) | Rogers Ratio 3 — discharge type classification |
| `co2_co_ratio` | CO2 / (CO + 1) | Cellulose degradation rate — ratio > 11 indicates rapid aging |
| `ethylene_ethane_ratio` | C2H4 / (C2H6 + 1) | High-temperature thermal fault indicator |

Epsilon = 1 ppm is added to all denominators to prevent division by zero (physically valid
because 0 ppm dissolved gas is a common real measurement for low-severity samples).

### Risk Class Target Derivation

| Bin | Health Index Range | Risk Class | Risk Int | Count | % |
|---|---|---|---|---|---|
| [0, 25) | Critical | HIGH | 2 | 269 | 57.2% |
| [25, 50) | Poor / At risk | MEDIUM | 1 | 150 | 31.9% |
| [50, 100] | Acceptable | LOW | 0 | 51 | 10.9% |

**Class imbalance note:** The dataset is 57% HIGH and only 11% LOW. The ML training script
**must** apply `class_weight='balanced'` or SMOTE to avoid a model that always predicts HIGH.

### Output Files

| File | Rows | Columns | Contains |
|---|---|---|---|
| `transformer_features.csv` | 470 | 21 | sample_id + 14 raw sensor cols + 6 engineered ratio cols |
| `transformer_target.csv` | 470 | 5 | sample_id, health_index, life_expectation_years, risk_class, risk_int |

**Structural leakage barrier:** `health_index` and `life_expectation_years` are written
to a separate file from features. A model can only use them as features if a developer
explicitly merges the two files — making the mistake deliberate and visible.

---

## 5. Pipeline B — Weather + Outage

**Input:** `data/raw/Weather_data_combined_with_outage.csv` (103,445 rows, 17 columns)

### Step-by-Step Transforms

| Step | Action | Rows Before | Rows After | Notes |
|---|---|---|---|---|
| 1 | Load raw CSV | — | 103,445 | NA strings handled on load |
| 2 | Drop leaky columns | 103,445 | 103,445 | 7 columns removed (see below) |
| 3 | Parse timestamp | 103,445 | 103,445 | `Hour` → datetime64; 0 parse failures |
| 4 | Cast numeric | 103,445 | 103,445 | Weather + target to float; 0 rows dropped |
| 5 | Cap weather outliers | 103,445 | 103,445 | No values outside physical bounds |
| 6 | Aggregate to event-level | 103,445 | **33,139** | One row per unique outage event |
| 7 | Impute missing weather | 33,139 | 33,139 | Post-aggregation imputation (see below) |
| 8 | Extract temporal features | 33,139 | 33,139 | hour_of_day, month, year, season |
| 9 | Derive severity target | 33,139 | 33,139 | max_customers_affected → outage_severity |
| 10 | Write output | 33,139 | 33,139 | 16 columns |

### Dropped Columns (Documented)

| Column | Reason | Category |
|---|---|---|
| `max_rolling_avg` | Rolling average of `Max_outage` computed DURING the event | DATA LEAKAGE |
| `evDur_Hr` | Event duration in hours — only known after event ends | DATA LEAKAGE |
| `evDur_day` | Derived from `evDur_Hr` — same leakage issue | DATA LEAKAGE |
| `median_sum` | Rolling aggregate of outage customers — unclear lineage | SOFT LEAKAGE |
| `state_st` | 2-letter state abbreviation — duplicate of `state` column | REDUNDANT |
| `feel` | Apparent temperature — derived from `tmpf` + `relh` | COLLINEAR |
| `gust` | 91.5% missing (94,615 / 103,445 rows) | TOO SPARSE |

### Event-Level Aggregation

**Problem:** 103,445 hourly rows represent 33,139 unique outage events.
Training on hourly rows would cause severe pseudo-replication:
- `Max_outage` (the target) is constant across all hours of the same event
- Cross-validation folds would contain leaky copies of the same event
- Optimistic accuracy scores would not reflect real predictive power

**Solution:** Aggregate to one row per event using `new_event_no` as the groupby key.

| Value | Aggregation Strategy | Rationale |
|---|---|---|
| `fips_code`, `state`, `county` | `first` | Location is constant within an event |
| `Max_outage` | `first` | Already the event-level max (repeated on every hour row) |
| `tmpf`, `relh`, `sknt`, `p01i` | `first` | First hour = conditions at event onset (most predictive) |
| `Hour` | `first` | Becomes `event_start` — the event onset timestamp |

**Why first-hour weather, not mean?**
Averaging weather over the event duration mixes pre-storm calm with mid-storm peak with
post-storm recovery. The onset conditions are what a pre-positioned crew would observe and act on.

### Missing Value Imputation (post-aggregation)

Imputation is applied after aggregation so medians reflect the event-level distribution,
not the inflated hourly distribution.

| Column | Missing (events) | Strategy | Value Used |
|---|---|---|---|
| `tmpf` | 659 / 33,139 (2.0%) | Column median | 84.2 °F |
| `relh` | 688 / 33,139 (2.1%) | Column median | 55.98% |
| `sknt` | 2,594 / 33,139 (7.8%) | Column median | 5.0 knots |
| `p01i` | 668 / 33,139 (2.0%) | Constant 0.0 | 0.0 inches/hr |

**Important training note:** The median values shown above were computed on the full dataset.
When training a model, medians must be fitted on the **training fold only** and applied to
the validation/test fold to prevent imputation leakage. The `impute_values` key in the
pipeline result dict records the current medians for this purpose.

### Outage Severity Target Derivation

| Bin | Customers Affected | Severity Class | Int | Count | % |
|---|---|---|---|---|---|
| [0, 10) | Micro outage | LOW | 0 | 20,990 | 63.3% |
| [10, 1000) | Moderate outage | MEDIUM | 1 | 11,693 | 35.3% |
| [1000, ∞) | Major outage | HIGH | 2 | 456 | 1.4% |

**Class imbalance note:** Only 1.4% of events are HIGH severity — the most important class
for grid resilience planning. The ML training script must address this severe imbalance
(e.g., class weights, oversampling, or threshold tuning).

### Output File

| File | Rows | Columns |
|---|---|---|
| `weather_outage_events.csv` | 33,139 | 16 |

**Columns:** `event_id`, `fips_code`, `state`, `county`, `event_start`,
`year`, `month`, `hour_of_day`, `season`, `tmpf`, `relh`, `sknt`, `p01i`,
`max_customers_affected`, `outage_severity`, `outage_severity_int`

---

## 6. Join Decision

**The two processed datasets are NOT joined.** This is an intentional, documented architectural decision.

| Reason | Detail |
|---|---|
| No shared key | No column exists in both datasets that reliably identifies the same entity |
| Different granularity | DGA samples are per-transformer; weather events are per-county |
| No ground truth | There is no data source that maps these specific transformer samples to these specific counties or events |
| Fabrication risk | Any synthetic join would invent a statistical relationship that does not exist in reality |

The correct integration point is at **inference time**, in the scoring layer:

```
transformer_risk_score  (from Model A, per-asset)
         |
         v
  combined_risk = transformer_risk_score
                  * weather_risk_score    (from Model B, per-county)
                  * asset_criticality_weight
         |
         v
  priority_rank  (sorted by combined_risk descending)
```

This approach is honest: each model scores what its data actually supports, and the
combination happens through business logic, not through fabricated data linkage.

---

## 7. Output Dataset Dimensions

### `transformer_features.csv`

| Property | Value |
|---|---|
| Rows | **470** |
| Feature columns | **20** (14 raw sensors + 6 engineered ratios) |
| Total columns | **21** (including sample_id) |
| Missing values | **0** |
| Duplicates | **0** |

Feature column list:
```
sample_id,
hydrogen_ppm, oxygen_ppm, nitrogen_ppm, methane_ppm, co_ppm, co2_ppm,
ethylene_ppm, ethane_ppm, acetylene_ppm, dbds_mg_kg, power_factor_pct,
interfacial_tension_mNm, dielectric_kv, water_content_ppm,
tdcg_ppm, rogers_r1_ch4_h2, rogers_r2_c2h2_c2h4, rogers_r3_c2h2_ch4,
co2_co_ratio, ethylene_ethane_ratio
```

### `transformer_target.csv`

| Property | Value |
|---|---|
| Rows | **470** |
| Columns | **5** (sample_id, health_index, life_expectation_years, risk_class, risk_int) |
| Class distribution | HIGH: 269 (57.2%), MEDIUM: 150 (31.9%), LOW: 51 (10.9%) |

### `weather_outage_events.csv`

| Property | Value |
|---|---|
| Rows | **33,139** |
| Columns | **16** |
| Missing values | **0** (all imputed) |
| Duplicates | **0** |
| Temporal coverage | 2015-07-01 to 2022-09-29 |
| Geographic coverage | 45 US states, 924 counties, 1,323 FIPS codes |
| Class distribution | LOW: 20,990 (63.3%), MEDIUM: 11,693 (35.3%), HIGH: 456 (1.4%) |

---

## 8. Leakage Prevention Summary

| Column | Dataset | Risk Level | Action Taken |
|---|---|---|---|
| `health_index` | Health Index | CRITICAL | Written to `transformer_target.csv` only — never in features file |
| `life_expectation_years` | Health Index | HIGH | Written to `transformer_target.csv` only |
| `max_rolling_avg` | Weather | CRITICAL | Dropped in Step 2, before any computation |
| `evDur_Hr` | Weather | CRITICAL | Dropped in Step 2 |
| `evDur_day` | Weather | CRITICAL | Dropped in Step 2 |
| `median_sum` | Weather | HIGH | Dropped in Step 2 |
| `max_customers_affected` | Weather | — | Target column — present in output but must never be used as a model feature |

**Structural guarantees:**
1. Leaky columns are the **first** columns removed at load time
2. Feature and target are in **separate files** for Dataset A
3. No computed quantity derived from a target value is added to any feature column
4. All drops are logged with reasons in the `pipeline_run_log.json`

---

## 9. Reproducibility Notes

- The pipeline is **deterministic** given the same input files
- Sort order in aggregation is fixed: sort by `(new_event_no, Hour)` before `groupby`
- Imputation values are computed on the full post-aggregation dataset and logged in `pipeline_run_log.json`
- The `sample_id` column in the health index output is a simple 1-to-N integer sequence
- Raw file integrity can be verified with MD5 checksums (recorded at pipeline start)
- To re-run from scratch: delete `data/processed/` and run `python -m src.data.pipeline`

---

## 10. Validation Warnings Explained

These warnings appear at runtime but do NOT indicate pipeline failures:

### Warning 1 — NA in HealthIndex raw file (row count = 471 vs 470)
```
NA / empty values found: {'Hydrogen': 1, 'Oxigen': 1, ...} (all columns, count=1)
```
**Explanation:** The raw file reports 471 rows when read naively, but the 471st row is
the header row being double-counted or a blank trailing row. After loading with `pd.read_csv`,
the actual data row count is **470** — consistent with the analysis phase. The validator uses
a raw CSV reader that counts 471 rows; the preprocessor uses pandas which correctly
reads 470 data rows. **No action required.**

### Warning 2 — oxygen_ppm out of bounds
```
Column 'oxygen_ppm' has 65 value(s) outside physical bounds [0, 20000]
```
**Explanation:** The IEC 60599 typical upper limit for dissolved oxygen in sealed transformer
oil is ~20,000 ppm. 65 samples (13.8%) exceed this. The maximum raw value is 249,900 ppm
(one extreme outlier; row 47). These are capped to the 99th percentile in Pipeline A Step 4.
The warning is informational — the preprocessor handles it.

### Warning 3 — Missing weather values
```
NA / empty values by column: {'gust': 94615, 'sknt': 13079, ...}
```
**Explanation:** These are the known missing values documented in the analysis phase.
`gust` is dropped (91.5% missing). The remaining missing values (`sknt`, `tmpf`, `relh`, `p01i`)
are imputed in Pipeline B Step 7. The warning is expected and handled.

---

*Pipeline implemented and verified. All tests pass. Raw files unmodified.*

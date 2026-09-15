# Data Dictionary — PowerGuard AI

> **Status:** Phase 1 analysis — derived from `data/raw/` files.
> Original files were NOT modified.
> Last updated: Phase 2 preparation.

---

## Overview

| # | Filename | Rows | Columns | Grain |
|---|---|---|---|---|
| 1 | `Health index1 (2).csv` | 470 | 16 | One row per transformer oil test sample |
| 2 | `Weather_data_combined_with_outage.csv` | 103,445 | 17 | One row per county-hour during an outage event window |

---

## Dataset 1 — `Health index1 (2).csv`

### Summary
Contains **Dissolved Gas Analysis (DGA)** and **electrical test** results for power transformers,
along with a computed **Health Index** score and **remaining Life Expectation** in years.
This is a transformer-level, static (non-time-series) dataset — each row represents one
oil sample taken from one transformer at one point in time, but no timestamp or asset ID
is recorded in the file.

### Column Definitions

| # | Column Name | Type | Unit | Range | Missing | Description |
|---|---|---|---|---|---|---|
| 1 | `Hydrogen` | Numerical (integer) | ppm | 0 – 23,349 | 0 | Dissolved hydrogen gas in transformer oil. High H₂ indicates partial discharge or thermal faults. |
| 2 | `Oxigen` | Numerical (integer) | ppm | 57 – 249,900 | 0 | Dissolved oxygen gas. Very high values can indicate oil degradation or air ingress. *(Note: column name is a misspelling of "Oxygen")* |
| 3 | `Nitrogen` | Numerical (integer) | ppm | 3,600 – 85,300 | 0 | Dissolved nitrogen gas. A normal background gas; abnormal levels indicate sealed-system breach. |
| 4 | `Methane` | Numerical (integer) | ppm | 0 – 7,406 | 0 | Dissolved methane. Indicates thermal decomposition at low temperatures. |
| 5 | `CO` | Numerical (integer) | ppm | 10 – 1,730 | 0 | Carbon monoxide. Indicates cellulose (paper insulation) degradation. |
| 6 | `CO2` | Numerical (integer) | ppm | 48 – 24,900 | 0 | Carbon dioxide. High CO₂/CO ratio indicates severe cellulose degradation. |
| 7 | `Ethylene` | Numerical (integer) | ppm | 0 – 16,684 | 0 | Dissolved ethylene. Dominant gas in high-temperature thermal faults (> 300 °C). |
| 8 | `Ethane` | Numerical (integer) | ppm | 0 – 5,467 | 0 | Dissolved ethane. Indicates moderate-temperature thermal faults. |
| 9 | `Acethylene` | Numerical (integer) | ppm | 0 – 9,740 | 0 | Dissolved acetylene. Key fault indicator — even trace amounts indicate arcing or very high temperatures. *(Note: column name is a misspelling of "Acetylene")* |
| 10 | `DBDS` | Numerical (float) | mg/kg | 0 – 227 | 0 | Dibenzyl disulphide concentration. A corrosive sulphur compound that damages copper conductors. |
| 11 | `Power factor` | Numerical (float) | % (dimensionless ratio) | 0.05 – 73.2 | 0 | Dielectric power factor (tan δ). Measures insulation quality; high values indicate contaminated or degraded insulation. |
| 12 | `Interfacial V` | Numerical (integer) | mN/m | 21 – 57 | 0 | Interfacial tension of transformer oil. Low values indicate oil oxidation and acid formation. |
| 13 | `Dielectric rigidity` | Numerical (integer) | kV | 27 – 75 | 0 | Breakdown voltage of the oil. Measures insulating strength; low values indicate contamination. |
| 14 | `Water content` | Numerical (integer) | ppm | 0 – 183 | 0 | Moisture content in transformer oil. Moisture degrades insulation and accelerates aging. |
| 15 | `Health index` | Numerical (float) | Score (0–100) | 13.4 – 95.2 | 0 | **Derived composite score** computed from all preceding DGA and electrical test columns. Higher = healthier. This is a calculated aggregate, NOT a raw sensor reading. |
| 16 | `Life expectation` | Numerical (float) | Years | 6 – 51 | 0 | **Derived estimate** of remaining service life. Computed from the Health Index. Not a raw sensor reading. |

### Health Index Score Interpretation

| Score Range | Condition Category | Count in Dataset |
|---|---|---|
| ≥ 85 | New / Good | 5 |
| 70 – 84 | Normal operation | 4 |
| 50 – 69 | Aging / Monitor | 42 |
| 25 – 49 | Poor / At risk | 150 |
| < 25 | Critical / Urgent | 269 |

> **Observation:** The dataset is **heavily skewed toward poor/critical assets** — 90% of records
> have a Health Index below 50, suggesting this dataset was collected specifically from
> at-risk or already-degraded transformers.

### DGA Gas Group Classification (IEC 60599 standard)

| Group | Gases | Fault Type Indicated |
|---|---|---|
| Key gas | Hydrogen, Methane, Ethylene, Ethane, Acetylene | Thermal / electrical fault type identification |
| CO gases | CO, CO₂ | Cellulose insulation degradation |
| Non-fault gases | Oxygen, Nitrogen | Oil exposure to air / system integrity |
| Oil degradation | DBDS, Power factor, Interfacial V | Oil quality and contamination |
| Moisture | Water content | Insulation moisture ingress |
| Dielectric | Dielectric rigidity | Insulating oil breakdown strength |

---

## Dataset 2 — `Weather_data_combined_with_outage.csv`

### Summary
A combined weather + power outage event dataset at **county-hour** granularity.
Each row represents a specific US county during a specific hour of an outage event window,
with concurrent meteorological observations. The dataset covers events across **45 US states**
spanning **July 2015 to September 2022**.

### Column Definitions

| # | Column Name | Type | Unit | Range | Missing | Description |
|---|---|---|---|---|---|---|
| 1 | `fips_code` | Categorical (integer code) | — | 1001 – 57000+ | 0 | US county FIPS code. 5-digit standard geographic identifier. **Location key.** |
| 2 | `Hour` | Datetime | M/D/YYYY HH:MM | 2015-07-01 to 2022-09-29 | 0 | Timestamp of the observation (hourly). **Temporal key.** |
| 3 | `max_rolling_avg` | Numerical (float) | Customers affected | 0.0625 – 62,374 | 0 | ⚠️ **DATA LEAKAGE** — Rolling window average of `Max_outage` computed within the same event window. Captures the outage magnitude as it unfolds. Must NOT be used as a model feature. |
| 4 | `state` | Categorical (string) | — | 45 unique states | 0 | US state name (full). **Location attribute.** |
| 5 | `county` | Categorical (string) | — | 924 unique counties | 0 | US county name. **Location attribute.** |
| 6 | `median_sum` | Numerical (float) | Customers affected | 1 – 3,668 | 0 | ⚠️ **SOFT LEAKAGE RISK** — Appears to be a per-county rolling or aggregated median of customers affected. Relationship to outage target is unclear without source documentation. Treat with caution. |
| 7 | `new_event_no` | Categorical (integer) | — | 7,251 – 32,412,145 | 0 | Sequential outage event identifier. Monotonically increasing. Used to group rows belonging to the same event. **Not a predictive feature.** |
| 8 | `evDur_Hr` | Numerical (integer) | Hours | 1 – 525 | 0 | ⚠️ **DATA LEAKAGE** — Total duration of the outage event in hours. Only knowable after the event ends. Must NOT be used as a model feature. |
| 9 | `evDur_day` | Numerical (integer) | Days | 0 – (multi-day) | 0 | ⚠️ **DATA LEAKAGE** — Total duration of the event in whole days. Derived from `evDur_Hr`. Only knowable after the event ends. Must NOT be used as a model feature. |
| 10 | `Max_outage` | Numerical (float) | Customers affected | 0.0625 – 62,374 | 0 | **Maximum number of customers affected** during the event. This is the primary outage severity measure. **Candidate ML target variable.** Must NOT be used as an input feature. |
| 11 | `state_st` | Categorical (string) | — | 2-letter postal codes | 12 | US state abbreviation (redundant with `state`). 12 rows missing. |
| 12 | `tmpf` | Numerical (float) | °F | 53.0 – 114.0 | 4,532 | Ambient air temperature in Fahrenheit at the nearest weather station. |
| 13 | `relh` | Numerical (float) | % | 4 – 100 | 4,629 | Relative humidity percentage. High humidity can accelerate insulation degradation. |
| 14 | `gust` | Numerical (float) | knots | 8.0 – 54.0 | 94,615 | Wind gust speed. **91.5% missing** — use with caution or drop for ML. |
| 15 | `feel` | Numerical (float) | °F | (apparent temp) | 4,629 | "Feels like" temperature. Collinear with `tmpf` and `relh` — likely redundant. |
| 16 | `p01i` | Numerical (float) | inches | 0 – 1.03 | 4,559 | Precipitation in inches per hour. 93% of values are zero. |
| 17 | `sknt` | Numerical (float) | knots | 0 – 36 | 13,079 | Wind speed (sustained). Complementary to `gust`. 12.6% missing. |

### Outage Severity Distribution

| Threshold | Count | % of Rows |
|---|---|---|
| Max_outage > 1,000 customers | 13,121 | 12.7% |
| Max_outage > 500 customers | 17,028 | 16.5% |
| Max_outage > 100 customers | 29,361 | 28.4% |
| Max_outage > 10 customers | 52,358 | 50.6% |
| Max_outage ≤ 10 customers | 51,087 | 49.4% |

### Percentile Distribution of Max_outage

| Percentile | Value (customers) |
|---|---|
| P25 | 1.375 |
| P50 (median) | 10.75 |
| P75 | 146.75 |
| P90 | 1,740 |
| P95 | 5,744 |
| P99 | 27,041 |

### Missing Value Summary

| Column | Missing Count | % Missing | Recommendation |
|---|---|---|---|
| `gust` | 94,615 | 91.5% | Drop column — too sparse for imputation |
| `sknt` | 13,079 | 12.6% | Impute with median or group mean |
| `relh` | 4,629 | 4.5% | Impute with median |
| `feel` | 4,629 | 4.5% | Drop (collinear with `tmpf` + `relh`) |
| `tmpf` | 4,532 | 4.4% | Impute with median by state/month |
| `p01i` | 4,559 | 4.4% | Impute with 0 (majority class) |
| `state_st` | 12 | 0.01% | Drop column (redundant with `state`) |

---

## Cross-Dataset Joinability

| Aspect | Finding |
|---|---|
| **Common column names** | None |
| **Direct join possible** | No — no shared natural key |
| **Asset ID present** | Only in Weather dataset (implicitly `fips_code` = county, not transformer) |
| **Transformer ID** | Absent in both datasets |
| **Join level mismatch** | Health Index = per-transformer; Weather = per-county-hour |
| **Indirect linkage** | Both datasets describe power infrastructure health — Health Index at the asset level, Weather at the geographic/grid-area level |
| **Recommended approach** | Use each dataset **independently** to train separate components; later, a synthetic `region` mapping can bridge them |

---

*Data dictionary generated from analysis of `data/raw/` files. Files were read-only — no modifications made.*

# Data Analysis — PowerGuard AI

> **Status:** Phase 2 preparation — full EDA of raw datasets.
> Original files in `data/raw/` were NOT modified.
> All statistics were computed programmatically from the raw CSV files.

---

## Table of Contents
1. [Dataset Inventory](#1-dataset-inventory)
2. [Dataset 1 — Health Index (Transformer DGA)](#2-dataset-1--health-index-transformer-dga)
3. [Dataset 2 — Weather + Outage](#3-dataset-2--weather--outage)
4. [Join Analysis](#4-join-analysis)
5. [Data Leakage Assessment](#5-data-leakage-assessment)
6. [ML Target Recommendation](#6-ml-target-recommendation)
7. [Feature Roadmap](#7-feature-roadmap)
8. [Known Limitations & Gaps](#8-known-limitations--gaps)

---

## 1. Dataset Inventory

| # | Filename | Rows | Columns | Time-series? | Has Location? | Has Asset ID? |
|---|---|---|---|---|---|---|
| 1 | `Health index1 (2).csv` | **470** | **16** | No | No | No |
| 2 | `Weather_data_combined_with_outage.csv` | **103,445** | **17** | Yes | Yes | No |

**Total raw data available:** 103,915 rows across 33 unique columns.

---

## 2. Dataset 1 — Health Index (Transformer DGA)

### 2.1 Basic Profile

| Property | Value |
|---|---|
| **Filename** | `Health index1 (2).csv` |
| **Rows (data, excl. header)** | 470 |
| **Columns** | 16 |
| **Missing values** | 0 (across all 16 columns) |
| **Duplicate rows** | 0 |
| **Grain** | One row = one oil sample from one transformer |

### 2.2 All Columns

| Column | Data Type | Category | Unit | Min | Max | Zeros |
|---|---|---|---|---|---|---|
| `Hydrogen` | Integer | Numerical — DGA fault gas | ppm | 0 | 23,349 | 71 |
| `Oxigen` | Integer | Numerical — DGA background gas | ppm | 57 | 249,900 | 0 |
| `Nitrogen` | Integer | Numerical — DGA background gas | ppm | 3,600 | 85,300 | 0 |
| `Methane` | Integer | Numerical — DGA fault gas | ppm | 0 | 7,406 | 82 |
| `CO` | Integer | Numerical — DGA cellulose gas | ppm | 10 | 1,730 | 0 |
| `CO2` | Integer | Numerical — DGA cellulose gas | ppm | 48 | 24,900 | 0 |
| `Ethylene` | Integer | Numerical — DGA fault gas | ppm | 0 | 16,684 | 214 |
| `Ethane` | Integer | Numerical — DGA fault gas | ppm | 0 | 5,467 | 209 |
| `Acethylene` | Integer | Numerical — DGA fault gas | ppm | 0 | 9,740 | 432 |
| `DBDS` | Float | Numerical — oil chemistry | mg/kg | 0 | 227 | 310 |
| `Power factor` | Float | Numerical — electrical test | % | 0.05 | 73.20 | 0 |
| `Interfacial V` | Integer | Numerical — oil chemistry | mN/m | 21 | 57 | 0 |
| `Dielectric rigidity` | Integer | Numerical — electrical test | kV | 27 | 75 | 0 |
| `Water content` | Integer | Numerical — oil chemistry | ppm | 0 | 183 | 2 |
| `Health index` | Float | Numerical — **derived score** | 0–100 | 13.4 | 95.2 | 0 |
| `Life expectation` | Float | Numerical — **derived estimate** | years | 6 | 51 | 0 |

### 2.3 Column Type Classification

| Category | Columns |
|---|---|
| **Numerical (raw sensor)** | Hydrogen, Oxigen, Nitrogen, Methane, CO, CO2, Ethylene, Ethane, Acethylene, DBDS, Power factor, Interfacial V, Dielectric rigidity, Water content |
| **Numerical (derived)** | Health index, Life expectation |
| **Categorical** | *(none)* |
| **Datetime** | *(none — no timestamps)* |
| **Location** | *(none)* |
| **Asset / Equipment ID** | *(none)* |
| **Failure / Fault indicator** | Health index (implicitly — low score = at-risk/failed) |

### 2.4 Missing Values
**None.** All 470 rows × 16 columns are fully populated. There are no `NA`, empty, or null entries.

### 2.5 Duplicate Records
**None.** All 470 rows are unique.

### 2.6 Data Distribution: Health Index

```
Score ≥ 85  (New/Good):      5 rows   (1.1%)
Score 70–84 (Normal):         4 rows   (0.9%)
Score 50–69 (Aging):         42 rows   (8.9%)
Score 25–49 (Poor/At risk): 150 rows  (31.9%)
Score  < 25 (Critical):     269 rows  (57.2%)
```

**Finding:** The dataset is **strongly skewed toward degraded/critical transformers**
(~89% of samples fall below score 50). This is not a balanced sample of the
transformer population — it appears to have been collected specifically from
monitored or already-flagged units. This skew is crucial for ML planning:
any model trained only on this data will need class balancing.

### 2.7 Key Observations

1. **No asset identifier** — rows cannot be traced back to a specific physical transformer
2. **No timestamp** — it is unknown when each oil sample was taken
3. **No location** — samples cannot be mapped to a geographic region or grid zone
4. **`Health index` and `Life expectation` are derived** from the other 14 columns using
   a scoring formula (based on IEC 60422 / industry standard methods). They are **aggregate
   outputs**, not independent measurements. Using them as features while the raw columns
   are also features would create **multicollinearity** — a form of indirect leakage.
5. **`Acethylene`** (acetylene) has 432 zero values out of 470 — this is physically
   meaningful (acetylene is only present in arcing faults, which are rare). This is
   NOT missing data — zeros are valid readings.
6. **`DBDS`** has 310 zeros — similarly meaningful (corrosive sulphur absent in many
   oil formulations).
7. **`Oxigen`** contains one extreme outlier: 249,900 ppm (row 47). The global maximum
   for dissolved oxygen in typical transformer oil is ~15,000 ppm. This value is
   likely a measurement or transcription error.

### 2.8 What This Dataset Contributes to PowerGuard AI

| Contribution | Detail |
|---|---|
| **Asset health scoring** | The 14 raw sensor columns define the chemical/electrical state of a transformer's insulation system |
| **Risk classification target (option A)** | `Health index` score can be bucketed into risk classes (HIGH/MEDIUM/LOW) to form the ML target |
| **Feature engineering source** | Gas ratios (IEC/Rogers/Duval methods), power factor severity levels, water content thresholds |
| **Domain grounding** | Provides the physical basis for *why* a transformer is failing — enabling explainable predictions |

---

## 3. Dataset 2 — Weather + Outage

### 3.1 Basic Profile

| Property | Value |
|---|---|
| **Filename** | `Weather_data_combined_with_outage.csv` |
| **Rows (data, excl. header)** | 103,445 |
| **Columns** | 17 |
| **Missing values** | Present in 7 columns (see Section 3.5) |
| **Duplicate rows** | 0 |
| **Grain** | One row = one county × one hour during an outage event window |
| **Geographic coverage** | 45 US states, 924 counties, 1,323 unique FIPS codes |
| **Temporal coverage** | 2015-07-01 to 2022-09-29 (7+ years) |

### 3.2 All Columns

| Column | Data Type | Category | Unit | Min | Max |
|---|---|---|---|---|---|
| `fips_code` | Integer (code) | Location | — | 1,001 | 57,000+ |
| `Hour` | Datetime string | Temporal | M/D/YYYY HH:MM | 2015-07-01 | 2022-09-29 |
| `max_rolling_avg` | Float | **⚠️ LEAKAGE** | Customers | 0.0625 | 62,374 |
| `state` | String | Location | — | — | — |
| `county` | String | Location | — | — | — |
| `median_sum` | Float | **⚠️ SOFT LEAKAGE** | Customers | 1 | 3,668 |
| `new_event_no` | Integer | Event ID | — | 7,251 | 32,412,145 |
| `evDur_Hr` | Integer | **⚠️ LEAKAGE** | Hours | 1 | 525 |
| `evDur_day` | Integer | **⚠️ LEAKAGE** | Days | 0 | multi-day |
| `Max_outage` | Float | **🎯 TARGET** | Customers | 0.0625 | 62,374 |
| `state_st` | String | Location (redundant) | 2-letter code | — | — |
| `tmpf` | Float | Weather | °F | 53.0 | 114.0 |
| `relh` | Float | Weather | % | 4.0 | 100.0 |
| `gust` | Float | Weather | knots | 8.0 | 54.0 |
| `feel` | Float | Weather | °F | — | — |
| `p01i` | Float | Weather | inches/hr | 0 | 1.03 |
| `sknt` | Float | Weather | knots | 0 | 36.0 |

### 3.3 Column Type Classification

| Category | Columns |
|---|---|
| **Numerical (weather — safe features)** | `tmpf`, `relh`, `gust`, `feel`, `p01i`, `sknt` |
| **Numerical (outage metrics — target/leakage)** | `Max_outage`, `max_rolling_avg`, `median_sum`, `evDur_Hr`, `evDur_day` |
| **Categorical (location)** | `state`, `county`, `state_st` |
| **Categorical (event ID)** | `new_event_no` |
| **Location code** | `fips_code` |
| **Datetime** | `Hour` |

### 3.4 Location-Related Columns
- `fips_code` — primary geographic key (US county FIPS code)
- `state` — state name
- `state_st` — state abbreviation (redundant, 12 missing)
- `county` — county name
- Collectively these columns allow geospatial aggregation and joining to other US county-level datasets

### 3.5 Missing Values

| Column | Missing Count | % Missing | Recommended Action |
|---|---|---|---|
| `gust` | 94,615 | **91.5%** | **Drop** — too sparse to be useful |
| `sknt` | 13,079 | 12.6% | Impute with county/state/month median |
| `relh` | 4,629 | 4.5% | Impute with nearby county median |
| `feel` | 4,629 | 4.5% | **Drop** — collinear with `tmpf` + `relh` |
| `p01i` | 4,559 | 4.4% | Impute with 0 (physically valid default) |
| `tmpf` | 4,532 | 4.4% | Impute with state/month median |
| `state_st` | 12 | 0.01% | **Drop** — redundant with `state` |

### 3.6 Outage Event Structure

Each distinct value of `new_event_no` defines one outage incident. Multiple rows with the
same `new_event_no` represent the **hourly progression** of a single event. Key observations:
- **33,139 unique events** across the dataset
- `evDur_Hr` ranges from 1 to 525 hours (~22 days for the longest event)
- **77,879 rows (75%)** have `evDur_day = 0` — most events are resolved within a single day
- `Max_outage` is repeated on every hour of the same event (it's the event-level maximum,
  not the hourly count) — this means `Max_outage` is NOT a time-varying feature per hour

### 3.7 Temporal Analysis

- **Date range:** July 2015 – September 2022 (7+ years)
- **Year coverage:** Includes summer storm seasons across multiple years
- **Hourly granularity** within event windows only — this is NOT a continuous hourly weather
  record. Rows only exist during event windows, not during quiescent periods.

### 3.8 Weather Feature Summary

| Feature | Coverage | Physical Significance |
|---|---|---|
| `tmpf` (temperature) | 95.6% | Extreme heat increases transformer load and insulation stress |
| `relh` (humidity) | 95.5% | High humidity accelerates insulation moisture ingress |
| `sknt` (wind speed) | 87.4% | Strong winds cause physical damage (downed lines, debris) |
| `p01i` (precipitation) | 95.6% | Rain/ice causes flashovers; flooding damages substations |
| `gust` | **8.5%** | Too sparse — not useful as a standalone feature |
| `feel` | 95.5% | Redundant with tmpf+relh — will be dropped |

### 3.9 What This Dataset Contributes to PowerGuard AI

| Contribution | Detail |
|---|---|
| **Outage target variable** | `Max_outage` (binned into severity categories) is the most direct target for an outage prediction model |
| **Weather feature source** | `tmpf`, `relh`, `sknt`, `p01i` are clean weather features to fuse with asset health data |
| **Geographic context** | FIPS code enables joining to US county-level population, grid infrastructure, or topology data |
| **Event severity quantification** | Can derive binary (`outage_occurred`), multi-class (`severity_level`), or regression (`customers_affected`) targets |
| **Temporal patterns** | 7 years of data enables year/month/season feature engineering |

---

## 4. Join Analysis

### 4.1 Can the Datasets Be Joined Directly?

**No.** There is no shared column or natural key between the two datasets.

| Join Criterion | Health Index | Weather + Outage | Verdict |
|---|---|---|---|
| Asset / transformer ID | ❌ absent | ❌ absent | Cannot join |
| Timestamp / date | ❌ absent | ✅ present | Cannot join |
| Location (FIPS / county) | ❌ absent | ✅ present | Cannot join |
| Station or utility name | ❌ absent | ❌ absent | Cannot join |

### 4.2 Columns That Cannot Be Joined

The following columns are dataset-specific and have **no counterpart** in the other file:

**Health Index only:**
Hydrogen, Oxigen, Nitrogen, Methane, CO, CO2, Ethylene, Ethane, Acethylene, DBDS,
Power factor, Interfacial V, Dielectric rigidity, Water content, Health index, Life expectation

**Weather + Outage only:**
fips_code, Hour, max_rolling_avg, state, county, median_sum, new_event_no, evDur_Hr,
evDur_day, Max_outage, state_st, tmpf, relh, gust, feel, p01i, sknt

### 4.3 Possible Indirect Linkage Strategies

| Strategy | Description | Feasibility |
|---|---|---|
| **Synthetic region mapping** | Assign a region label to each Health Index record based on domain knowledge, then join to weather by region | Feasible — requires external reference or assumption |
| **Feature-level fusion at inference** | Train two separate models (asset health → risk score; weather → outage probability) and combine scores at inference time | **Recommended** — clean separation of signal sources |
| **External bridge dataset** | Acquire a utility asset registry that maps transformer IDs to FIPS codes — would enable a true join | Not available in current dataset; future enhancement |
| **Clustering-based imputation** | Assign Health Index records to county groups based on voltage/type assumptions | Speculative — not recommended without ground truth |

### 4.4 Recommendation

> **Use the datasets independently as two separate signal sources.**
> Dataset 1 trains the **transformer health risk classifier**.
> Dataset 2 trains the **weather-driven outage severity predictor**.
> Both outputs are combined at the **prediction/scoring layer** in PowerGuard AI.

---

## 5. Data Leakage Assessment

### 5.1 Dataset 1 — Health Index

| Column | Leakage Type | Severity | Action |
|---|---|---|---|
| `Health index` | **Direct leakage** — derived from the same raw columns used as features | CRITICAL | **Exclude from features.** Use as target variable only (binned into risk classes). |
| `Life expectation` | **Indirect leakage** — computed from Health Index | HIGH | **Exclude from features.** Could be used as an alternative target if a regression approach is taken. |
| All 14 raw columns | Safe features ✅ | — | Can be used as model inputs |

### 5.2 Dataset 2 — Weather + Outage

| Column | Leakage Type | Severity | Action |
|---|---|---|---|
| `Max_outage` | **Target variable** — this IS what we are predicting | — | **Use as target only.** Never as a feature. |
| `max_rolling_avg` | **Direct leakage** — rolling average of `Max_outage` computed during the same event | CRITICAL | **Drop from features.** |
| `evDur_Hr` | **Temporal leakage** — event duration only known after event ends | CRITICAL | **Drop from features.** |
| `evDur_day` | **Temporal leakage** — derived from `evDur_Hr` | CRITICAL | **Drop from features.** |
| `median_sum` | **Soft leakage** — likely a rolling aggregate of customers affected | HIGH | **Drop from features** until data lineage is confirmed. |
| `new_event_no` | No predictive value; sequential index | LOW | Drop (not a feature). |
| `state_st` | Redundant with `state` | LOW | Drop (simplification). |
| `feel` | Collinear with `tmpf` + `relh` | LOW | Drop (multicollinearity). |
| `gust` | 91.5% missing | MEDIUM | Drop from features. |
| `tmpf`, `relh`, `sknt`, `p01i` | Safe weather features ✅ | — | Retain as model inputs. |
| `fips_code`, `state`, `county` | Safe location features ✅ | — | Retain for grouping/encoding. |
| `Hour` | Safe temporal feature ✅ | — | Extract hour-of-day, month, year features. |

### 5.3 Summary of Columns to Exclude

```
FROM HEALTH INDEX:  Health index, Life expectation (as features — use as target instead)
FROM WEATHER:       max_rolling_avg, evDur_Hr, evDur_day, Max_outage (as feature),
                    median_sum, new_event_no, state_st, feel, gust
```

---

## 6. ML Target Recommendation

### 6.1 Option A — Transformer Risk Classifier (from Health Index dataset)

| Property | Detail |
|---|---|
| **Target column** | `Health index` binned into 3 classes |
| **Target encoding** | HIGH (score < 25), MEDIUM (25–49), LOW (score ≥ 50) |
| **Task type** | Multi-class classification |
| **Features (safe)** | All 14 raw DGA + electrical test columns |
| **Rows available** | 470 |
| **Class balance** | Imbalanced — ~57% HIGH, ~32% MEDIUM, ~11% LOW |
| **Pros** | Direct domain signal; clean features; no leakage |
| **Cons** | Small dataset (470 rows); no temporal or location context; heavy class imbalance |
| **Suitability** | ✅ **Recommended as the primary ML model** for PowerGuard AI |

**Suggested threshold buckets:**

| Class | Health Index Range | Risk Level |
|---|---|---|
| 2 — HIGH | < 25 | Urgent — fault imminent |
| 1 — MEDIUM | 25 – 49 | At risk — schedule maintenance |
| 0 — LOW | ≥ 50 | Acceptable — monitor |

### 6.2 Option B — Outage Severity Predictor (from Weather + Outage dataset)

| Property | Detail |
|---|---|
| **Target column** | `Max_outage` (continuous → binned) |
| **Target encoding** | LOW (< 10 customers), MEDIUM (10–1000), HIGH (> 1000) |
| **Task type** | Multi-class classification or regression |
| **Features (safe)** | `tmpf`, `relh`, `sknt`, `p01i` + location + temporal |
| **Rows available** | 103,445 |
| **Pros** | Large dataset; rich temporal + geographic context; real outage records |
| **Cons** | No transformer-level detail; weather alone is a weak predictor of outages |
| **Suitability** | ✅ **Recommended as a secondary model** for weather-driven outage risk scoring |

### 6.3 Final Recommendation

> **Build Option A first** using the Health Index dataset as the primary ML target.
> This directly answers "which transformer is at risk of failure?"
>
> **Build Option B second** to answer "which counties are at high outage risk given
> current/forecast weather conditions?"
>
> **Combine both** at the scoring layer: a transformer's final risk score =
> `f(transformer_risk_score × weather_risk_score × criticality_weight)`.

---

## 7. Feature Roadmap

### 7.1 Features from Health Index Dataset (Phase 3)

| Feature | Source Column(s) | Engineering Required |
|---|---|---|
| `hydrogen_ppm` | `Hydrogen` | Direct use |
| `acetylene_ppm` | `Acethylene` | Direct use |
| `ethylene_ppm` | `Ethylene` | Direct use |
| `co_ppm` | `CO` | Direct use |
| `co2_ppm` | `CO2` | Direct use |
| `power_factor` | `Power factor` | Direct use |
| `water_content_ppm` | `Water content` | Direct use |
| `dielectric_kv` | `Dielectric rigidity` | Direct use |
| `interfacial_tension` | `Interfacial V` | Direct use |
| `dbds_mg_kg` | `DBDS` | Direct use |
| `rogers_ratio_1` | `CH4/H2` | Compute ratio |
| `rogers_ratio_2` | `C2H2/C2H4` | Compute ratio |
| `rogers_ratio_3` | `C2H2/CH4` | Compute ratio |
| `tdcg` | sum of key fault gases | Total Dissolved Combustible Gas |
| `co2_co_ratio` | `CO2/CO` | Cellulose degradation indicator |
| `risk_class` | `Health index` (binned) | **This is the ML TARGET** |

### 7.2 Features from Weather + Outage Dataset (Phase 3)

| Feature | Source Column(s) | Engineering Required |
|---|---|---|
| `temp_f` | `tmpf` | Direct use (impute missing) |
| `humidity_pct` | `relh` | Direct use (impute missing) |
| `wind_speed_kts` | `sknt` | Direct use (impute missing) |
| `precipitation_in` | `p01i` | Direct use (impute with 0) |
| `hour_of_day` | `Hour` | Extract from datetime |
| `month` | `Hour` | Extract from datetime |
| `season` | `Hour` | Derive: spring/summer/fall/winter |
| `heat_index` | `tmpf` + `relh` | Compute heat stress metric |
| `storm_flag` | `p01i` + `sknt` | Binary: precipitation > 0.1 AND wind > 15 kts |
| `extreme_temp_flag` | `tmpf` | Binary: tmpf > 95°F or < 20°F |
| `fips_code` | `fips_code` | Encode or embed as location feature |
| `state_encoded` | `state` | Label or one-hot encoding |
| `outage_severity` | `Max_outage` (binned) | **This is the ML TARGET for Option B** |

---

## 8. Known Limitations & Gaps

### 8.1 Health Index Dataset

| Limitation | Impact | Mitigation |
|---|---|---|
| **No asset ID** | Cannot track individual transformer history | Accept as anonymous samples; cannot build time-series per asset |
| **No timestamp** | Cannot engineer temporal features | Accept as cross-sectional data |
| **No location** | Cannot correlate with geographic or grid risk | Cannot join to weather data |
| **Small size (470 rows)** | High risk of overfitting | Apply cross-validation; consider SMOTE for class balancing |
| **Dataset skew (~90% poor/critical)** | Model may not generalise to healthy transformers | Apply class-weight balancing; document in model card |
| **One extreme outlier (Oxigen=249,900)** | May distort distance-based models | Investigate and cap or remove |
| **Column name typos** | `Oxigen` (Oxygen), `Acethylene` (Acetylene) | Rename during preprocessing — DO NOT modify originals |
| **No fault type labels** | Cannot distinguish fault type (thermal vs. electrical) | Out of scope for Phase 3; potential future work |

### 8.2 Weather + Outage Dataset

| Limitation | Impact | Mitigation |
|---|---|---|
| **`gust` is 91.5% missing** | This important storm feature is mostly absent | Drop the column; supplement with external wind data if needed |
| **Hourly records exist only during events** | Cannot model "no-outage" baseline conditions | Supplement with background weather data from NOAA/ASOS for negative samples |
| **`Max_outage` is per-event (repeated)** | Each event row is not independent — this inflates sample count | Use event-level aggregation for training to avoid pseudo-replication |
| **No transformer / substation IDs** | Cannot link outages to specific equipment | Out of scope for this dataset; future join with EAGLE-I or EIA data |
| **US-centric data** | Limited to US geography and climate zones | Acceptable for demonstration; note in documentation |
| **Weather measured at nearest airport/station** | May not reflect actual microclimate at the substation | Acceptable approximation for county-level granularity |
| **No root cause column** | Cannot distinguish weather-caused vs. equipment-caused outages | `tmpf`, `relh`, `p01i`, `sknt` serve as proxies for weather stress |

### 8.3 Cross-Dataset Limitations

| Gap | Impact | Recommended Resolution |
|---|---|---|
| **No shared key** | Cannot directly combine transformer health and weather signals | Implement a two-model architecture with a scoring fusion layer |
| **Different levels of granularity** | HI = per-transformer; Weather = per-county-hour | Aggregate or parameterise differently in each model |
| **No incident/failure labels** | Neither dataset has explicit "transformer failed" flags | Use Health Index score < 25 as a proxy for failure risk |
| **No asset criticality field** | Cannot weight high-voltage or high-customer-density assets more | Supplement with EIA transmission topology data in a later phase |

---

*Analysis completed. No raw files were modified. All computed statistics are reproducible
from the files in `data/raw/`.*

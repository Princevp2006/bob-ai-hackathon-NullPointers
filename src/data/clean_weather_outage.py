"""
PowerGuard AI — Weather + Outage Preprocessor
================================================
Transforms the raw county-hour weather/outage dataset into a clean,
leakage-free event-level dataset ready for ML training.

Key design decisions (all documented in docs/data_pipeline.md)
---------------------------------------------------------------
1.  LEAKY COLUMNS ARE DROPPED FIRST, before any other operation,
    so they can never accidentally contaminate downstream steps.

2.  The raw data is hourly rows during event windows.  This causes
    pseudo-replication: Max_outage (the target) is constant across
    all hours of the same event.  Training a model on hourly rows
    would give artificially optimistic cross-validation scores.
    Solution: aggregate to ONE ROW PER EVENT using first-observation
    weather features for the predictive features and event-peak
    Max_outage for the target.

3.  Missing weather values are imputed with the column median computed
    ONLY on the training partition (fit-on-train strategy).  Because
    we are not splitting here, we use the full-dataset median and note
    this in the pipeline log — the model training step must re-apply
    the same medians from the training fold only.

4.  `p01i` (precipitation) missing values are imputed to 0.0 because
    absence of a precipitation record at an ASOS station almost always
    means trace/zero precipitation, not a sensor gap.

5.  `gust` (91.5% missing) is dropped entirely.
    `feel` (collinear) and `state_st` (redundant) are dropped.

6.  Temporal features (hour_of_day, month, year, season) are extracted
    from the event start timestamp.

7.  The outage severity target is derived by binning Max_outage:
    LOW < 10, MEDIUM 10–1000, HIGH > 1000 customers affected.

Output
------
  data/processed/weather_outage_events.csv
    One row per outage event (unique new_event_no).
    Columns: event_id, fips_code, state, county,
             event_start, hour_of_day, month, year, season,
             tmpf, relh, sknt, p01i,
             max_customers_affected (target — continuous),
             outage_severity (target — categorical),
             outage_severity_int (target — integer encoded)

Usage
-----
    from src.data.clean_weather_outage import run
    result = run(verbose=True)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.constants import (
    RAW_WEATHER,
    PROCESSED_DIR,
    PROC_WEATHER,
    WO_DROP_COLS,
    WO_SAFE_FEATURE_COLS,
    WO_LOCATION_COLS,
    WO_TIMESTAMP_COL,
    WO_TIMESTAMP_FMT,
    WO_TARGET_COL,
    WO_IMPUTE_DEFAULTS,
    WO_PHYSICAL_BOUNDS,
    WO_SEVERITY_BINS,
    WO_SEVERITY_LABELS,
    WO_SEVERITY_INT,
    WO_SEASON_MAP,
)

log = logging.getLogger(__name__)

_NA_STRINGS = {"NA", "N/A", "NULL", "NONE", ""}


@dataclass
class TransformLog:
    steps: list[dict[str, Any]] = field(default_factory=list)

    def record(self, step: str, detail: str, rows_before: int, rows_after: int) -> None:
        entry = {
            "step": step,
            "detail": detail,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_before - rows_after,
        }
        self.steps.append(entry)
        log.info("[WeatherOutage] %s | %s | rows %d → %d", step, detail, rows_before, rows_after)


def run(verbose: bool = False) -> dict[str, Any]:
    """
    Execute the full Weather + Outage preprocessing pipeline.

    Steps
    -----
    1.  Load raw CSV (read-only; NA strings handled on load).
    2.  Drop leaky and redundant columns (documented in WO_DROP_COLS).
    3.  Parse timestamp column to datetime.
    4.  Cast weather and target columns to float.
    5.  Cap physically impossible weather values.
    6.  Aggregate hourly rows → one row per outage event.
    7.  Impute missing weather values with column medians.
    8.  Extract temporal features from event_start timestamp.
    9.  Derive outage severity target (binned).
    10. Write output to data/processed/.

    Returns
    -------
    dict with keys: rows_in, rows_out, columns, class_distribution,
                    output_path, transform_log
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    tlog = TransformLog()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    log.info("[WeatherOutage] Loading raw file: %s", RAW_WEATHER)
    # Replace all NA sentinel strings with proper NaN on load
    df = pd.read_csv(RAW_WEATHER, na_values=list(_NA_STRINGS), keep_default_na=True)
    rows_in = len(df)
    tlog.record("load", f"Read {rows_in} rows, {len(df.columns)} columns", rows_in, rows_in)

    # ── Step 2: Drop leaky and redundant columns ──────────────────────────────
    # Drop order matters: we remove leaky columns BEFORE any computation.
    cols_to_drop = [c for c in WO_DROP_COLS.keys() if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    drop_reasons = {c: WO_DROP_COLS[c] for c in cols_to_drop}
    tlog.record(
        "drop_leaky_cols",
        f"Dropped {len(cols_to_drop)} columns: {drop_reasons}",
        len(df), len(df),
    )

    # ── Step 3: Parse timestamp ───────────────────────────────────────────────
    df[WO_TIMESTAMP_COL] = pd.to_datetime(df[WO_TIMESTAMP_COL], format=WO_TIMESTAMP_FMT)
    ts_nulls = df[WO_TIMESTAMP_COL].isnull().sum()
    if ts_nulls:
        log.warning("[WeatherOutage] %d timestamp parse failures — dropping those rows", ts_nulls)
        df = df.dropna(subset=[WO_TIMESTAMP_COL])
    tlog.record("parse_timestamp", f"Parsed '{WO_TIMESTAMP_COL}'; {ts_nulls} failures dropped",
                rows_in, len(df))

    # ── Step 4: Cast numeric columns ──────────────────────────────────────────
    numeric_cols = WO_SAFE_FEATURE_COLS + [WO_TARGET_COL, "fips_code"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where the target (Max_outage) is missing — can't use them
    n_before = len(df)
    df = df.dropna(subset=[WO_TARGET_COL])
    n_dropped_target = n_before - len(df)
    tlog.record(
        "cast_numeric",
        f"Numeric cast applied. Dropped {n_dropped_target} rows with missing target.",
        n_before, len(df),
    )

    # ── Step 5: Cap physically impossible weather values ──────────────────────
    cap_log: list[str] = []
    for col, (lo, hi, _desc) in WO_PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        n_hi = (df[col] > hi).sum()
        n_lo = (df[col] < lo).sum()
        if n_hi or n_lo:
            p99 = df[col].quantile(0.99)
            df.loc[df[col] > hi, col] = p99
            df.loc[df[col] < lo, col] = lo
            cap_log.append(f"{col}: {n_hi} above max, {n_lo} below min → capped")
    tlog.record(
        "cap_weather_outliers",
        "; ".join(cap_log) if cap_log else "No weather values outside physical bounds",
        len(df), len(df),
    )

    # ── Step 6: Aggregate hourly rows → one row per event ────────────────────
    # The key design decision: each unique new_event_no defines one outage.
    # Max_outage is ALREADY the event-level maximum (repeated on each hour row).
    # Weather features: we take the FIRST hour's reading for each event.
    # This approximates "conditions at the onset of the outage" which is the
    # closest we can get to pre-event weather without a continuous baseline.
    #
    # Why first hour, not mean?
    #   - Mean would mix pre-storm and post-storm weather within one event.
    #   - First hour is consistent and corresponds to event_start timestamp.
    #   - Location and target are constant within an event — no information loss.

    log.info("[WeatherOutage] Aggregating %d hourly rows to event-level ...", len(df))

    # Sort by event and time so "first" is deterministic
    df = df.sort_values(["new_event_no", WO_TIMESTAMP_COL])

    # Columns that are constant within an event (take first value)
    constant_cols = WO_LOCATION_COLS + [WO_TARGET_COL]

    # Weather features — take first observation of the event
    weather_cols = [c for c in WO_SAFE_FEATURE_COLS if c in df.columns]

    agg_spec: dict[str, Any] = {}
    for col in constant_cols:
        if col in df.columns:
            agg_spec[col] = "first"
    for col in weather_cols:
        agg_spec[col] = "first"
    agg_spec[WO_TIMESTAMP_COL] = "first"   # event start time

    df_events = (
        df.groupby("new_event_no", sort=False)
          .agg(agg_spec)
          .reset_index()
    )
    df_events = df_events.rename(columns={
        "new_event_no":   "event_id",
        WO_TIMESTAMP_COL: "event_start",
        WO_TARGET_COL:    "max_customers_affected",
    })

    rows_events = len(df_events)
    tlog.record(
        "aggregate_to_event_level",
        (
            f"Aggregated {len(df)} hourly rows → {rows_events} unique events. "
            "Weather features: first-hour observation. "
            "Target: max_customers_affected (event-level max)."
        ),
        len(df), rows_events,
    )

    # ── Step 7: Impute missing weather values ─────────────────────────────────
    # IMPORTANT: in a real train/test split, medians must be computed on the
    # training fold only and applied to the test fold.  We record the medians
    # here so the training script can replicate or verify this step.
    impute_log: list[str] = []
    impute_values: dict[str, float] = {}

    for col in weather_cols:
        n_missing = df_events[col].isnull().sum()
        if n_missing == 0:
            continue

        default = WO_IMPUTE_DEFAULTS.get(col)
        if default is not None:
            fill_val = default
            strategy = f"constant({default})"
        else:
            fill_val = df_events[col].median()
            strategy = f"column_median({fill_val:.4f})"

        df_events[col] = df_events[col].fillna(fill_val)
        impute_values[col] = fill_val
        impute_log.append(f"{col}: {n_missing} missing → {strategy}")

    tlog.record(
        "impute_missing_weather",
        "; ".join(impute_log) if impute_log else "No missing values to impute",
        len(df_events), len(df_events),
    )

    # ── Step 8: Extract temporal features ────────────────────────────────────
    df_events["hour_of_day"] = df_events["event_start"].dt.hour
    df_events["month"]       = df_events["event_start"].dt.month
    df_events["year"]        = df_events["event_start"].dt.year
    df_events["season"]      = df_events["month"].map(WO_SEASON_MAP)

    tlog.record(
        "extract_temporal_features",
        "Added hour_of_day, month, year, season from event_start timestamp",
        len(df_events), len(df_events),
    )

    # ── Step 9: Derive outage severity target ─────────────────────────────────
    # Bins: LOW [0, 10), MEDIUM [10, 1000), HIGH [1000, ∞)
    # pd.cut: include_lowest=True makes the first bin left-closed
    df_events["outage_severity"] = pd.cut(
        df_events["max_customers_affected"],
        bins=WO_SEVERITY_BINS,
        labels=WO_SEVERITY_LABELS,
        right=False,
        include_lowest=True,
    )
    df_events["outage_severity_int"] = df_events["outage_severity"].map(WO_SEVERITY_INT)

    class_dist = df_events["outage_severity"].value_counts().to_dict()
    tlog.record(
        "derive_severity_target",
        f"Binned max_customers_affected into outage_severity. Distribution: {class_dist}",
        len(df_events), len(df_events),
    )

    # ── Step 10: Final column ordering and write ──────────────────────────────
    output_col_order = (
        ["event_id", "fips_code", "state", "county", "event_start",
         "year", "month", "hour_of_day", "season"]
        + weather_cols
        + ["max_customers_affected", "outage_severity", "outage_severity_int"]
    )
    output_col_order = [c for c in output_col_order if c in df_events.columns]
    df_events = df_events[output_col_order]

    df_events.to_csv(PROC_WEATHER, index=False)

    rows_out = len(df_events)
    tlog.record(
        "write_output",
        f"Written {rows_out} event rows, {len(df_events.columns)} columns → {PROC_WEATHER.name}",
        rows_out, rows_out,
    )

    log.info(
        "[WeatherOutage] Pipeline complete. "
        "Rows in: %d  |  Events out: %d  |  Columns: %d",
        rows_in, rows_out, len(df_events.columns),
    )

    return {
        "dataset": "weather_outage",
        "rows_in": rows_in,
        "rows_out": rows_out,
        "columns": list(df_events.columns),
        "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
        "impute_values": impute_values,
        "output_path": str(PROC_WEATHER),
        "transform_log": tlog.steps,
    }

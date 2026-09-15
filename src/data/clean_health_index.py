"""
PowerGuard AI — Health Index / DGA Preprocessor
=================================================
Transforms the raw transformer DGA dataset into two clean outputs:

  data/processed/transformer_features.csv  — 14 feature columns + sample_id
  data/processed/transformer_target.csv    — sample_id + risk_class (label) + risk_int

Design principles applied
--------------------------
* Read-only on data/raw/ — this module NEVER writes to the raw directory.
* Target variable (health_index, life_expectation_years) is separated from
  features at write time to make accidental leakage structurally impossible.
* Column names are corrected (Oxigen → oxygen_ppm, Acethylene → acetylene_ppm)
  only in the processed output; the raw file is untouched.
* One outlier row is capped rather than dropped (oxygen_ppm = 249,900 ppm is
  physically implausible — capped to the column's 99th percentile).
* Engineered IEC ratio features are added but clearly labelled _ratio so
  they are distinguishable from raw measurements.
* All transformations are recorded in the returned TransformLog.

Usage
-----
    from src.data.clean_health_index import run
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
    RAW_HEALTH_INDEX,
    PROCESSED_DIR,
    PROC_HEALTH_FEATURES,
    PROC_HEALTH_TARGET,
    HI_RENAME_MAP,
    HI_FEATURE_COLS,
    HI_EXCLUDE_FROM_FEATURES,
    HI_PHYSICAL_BOUNDS,
    HI_RISK_BINS,
    HI_RISK_LABELS,
    HI_RISK_INT,
)

log = logging.getLogger(__name__)


# ── Transform audit log ───────────────────────────────────────────────────────

@dataclass
class TransformLog:
    """Records every mutation applied to the dataset."""

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
        log.info("[HealthIndex] %s | %s | rows %d → %d", step, detail, rows_before, rows_after)


# ── Main preprocessing function ───────────────────────────────────────────────

def run(verbose: bool = False) -> dict[str, Any]:
    """
    Execute the full Health Index preprocessing pipeline.

    Steps
    -----
    1.  Load raw CSV (read-only).
    2.  Rename columns to clean snake_case names.
    3.  Cast all columns to float64.
    4.  Detect and cap physically impossible values (oxygen outlier).
    5.  Assert no missing values remain.
    6.  Engineer IEC gas-ratio features (Rogers ratios, TDCG, CO₂/CO).
    7.  Derive risk_class target from health_index (binning).
    8.  Split: features CSV vs target CSV (structural leakage barrier).
    9.  Write outputs to data/processed/.
    10. Return summary dict.

    Returns
    -------
    dict with keys: rows_in, rows_out, features_path, target_path, log_steps
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    tlog = TransformLog()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    log.info("[HealthIndex] Loading raw file: %s", RAW_HEALTH_INDEX)
    df = pd.read_csv(RAW_HEALTH_INDEX)
    rows_in = len(df)
    tlog.record("load", f"Read {rows_in} rows, {len(df.columns)} columns", rows_in, rows_in)

    # ── Step 2: Rename columns ────────────────────────────────────────────────
    df = df.rename(columns=HI_RENAME_MAP)
    tlog.record("rename_columns", f"Applied rename map ({len(HI_RENAME_MAP)} cols)", rows_in, len(df))

    # ── Step 3: Cast to float64 ───────────────────────────────────────────────
    # Explicit float64 cast is required before any cap operations so that
    # pandas 3 doesn't raise TypeError when assigning a float cap to an int col.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    n_na_after_cast = df.isnull().sum().sum()
    if n_na_after_cast > 0:
        log.warning("[HealthIndex] %d cells became NaN after numeric cast", n_na_after_cast)
    tlog.record(
        "cast_numeric",
        f"All columns cast to float64; {n_na_after_cast} NaN produced",
        len(df), len(df),
    )

    # ── Step 4: Cap / fix physically impossible values ────────────────────────
    # oxygen_ppm: one row has 249,900 ppm which exceeds physically plausible
    # max for dissolved oxygen in sealed transformer oil (~15,000–20,000 ppm).
    # Action: cap to the 99th percentile of the column (not the outlier row).
    cap_log: list[str] = []
    for clean_name, (lo, hi, desc) in HI_PHYSICAL_BOUNDS.items():
        if clean_name not in df.columns:
            continue
        oob_mask = (df[clean_name] < lo) | (df[clean_name] > hi)
        n_oob = oob_mask.sum()
        if n_oob > 0:
            # Cap strategy: values above hi → 99th percentile (preserves distribution shape)
            #               values below lo → lo (floor, usually 0)
            p99 = df[clean_name].quantile(0.99)
            df.loc[df[clean_name] > hi, clean_name] = p99
            df.loc[df[clean_name] < lo, clean_name] = lo
            cap_log.append(f"{clean_name}: {n_oob} values capped (bounds [{lo}, {hi}])")
    if cap_log:
        tlog.record("cap_outliers", "; ".join(cap_log), len(df), len(df))
    else:
        tlog.record("cap_outliers", "No values outside physical bounds", len(df), len(df))

    # ── Step 5: Assert no missing values ─────────────────────────────────────
    missing_total = df.isnull().sum().sum()
    assert missing_total == 0, (
        f"Unexpected NaN values after preprocessing: {df.isnull().sum().to_dict()}"
    )
    tlog.record("assert_no_missing", "Zero missing values confirmed", len(df), len(df))

    # ── Step 6: Engineer IEC ratio features ──────────────────────────────────
    # These ratios are standard IEC 60599 / Rogers method diagnostic tools.
    # They are computed from the raw feature columns — NOT from the target —
    # so they introduce no leakage.
    #
    # All divisions use a small epsilon (1 ppm = 1e-9 in context) to avoid /0.
    eps = 1.0

    # Total Dissolved Combustible Gas (TDCG) — IEEE C57.104
    df["tdcg_ppm"] = (
        df["hydrogen_ppm"] + df["methane_ppm"] + df["co_ppm"]
        + df["ethylene_ppm"] + df["ethane_ppm"] + df["acetylene_ppm"]
    )

    # Rogers Ratio 1: CH4 / H2 — distinguishes thermal from electrical faults
    df["rogers_r1_ch4_h2"] = df["methane_ppm"] / (df["hydrogen_ppm"] + eps)

    # Rogers Ratio 2: C2H2 / C2H4 — identifies arcing faults
    df["rogers_r2_c2h2_c2h4"] = df["acetylene_ppm"] / (df["ethylene_ppm"] + eps)

    # Rogers Ratio 3: C2H2 / CH4 — discharge type identification
    df["rogers_r3_c2h2_ch4"] = df["acetylene_ppm"] / (df["methane_ppm"] + eps)

    # CO₂ / CO ratio — cellulose insulation degradation rate
    # High ratio (> 11) indicates rapid paper aging
    df["co2_co_ratio"] = df["co2_ppm"] / (df["co_ppm"] + eps)

    # Doernenburg-style C2H4 / C2H6 — identifies high-temperature thermal faults
    df["ethylene_ethane_ratio"] = df["ethylene_ppm"] / (df["ethane_ppm"] + eps)

    engineered = ["tdcg_ppm", "rogers_r1_ch4_h2", "rogers_r2_c2h2_c2h4",
                  "rogers_r3_c2h2_ch4", "co2_co_ratio", "ethylene_ethane_ratio"]
    tlog.record(
        "feature_engineering",
        f"Added {len(engineered)} IEC ratio features: {engineered}",
        len(df), len(df),
    )

    # ── Step 7: Derive risk_class target ─────────────────────────────────────
    # health_index is binned into 3 ordered risk classes.
    # Bins: [0, 25) → HIGH, [25, 50) → MEDIUM, [50, 100] → LOW
    # pd.cut uses left-closed, right-open for all but the last bin.
    df["risk_class"] = pd.cut(
        df["health_index"],
        bins=HI_RISK_BINS,
        labels=HI_RISK_LABELS,
        right=False,      # [left, right) — left-inclusive
        include_lowest=True,
    )
    df["risk_int"] = df["risk_class"].map(HI_RISK_INT)

    class_dist = df["risk_class"].value_counts().to_dict()
    tlog.record(
        "derive_target",
        f"Derived risk_class from health_index. Distribution: {class_dist}",
        len(df), len(df),
    )

    # ── Step 8: Add sample_id ─────────────────────────────────────────────────
    # A simple sequential ID so features and target CSVs can always be re-joined.
    df.insert(0, "sample_id", range(1, len(df) + 1))

    # ── Step 9: Split features from target ────────────────────────────────────
    # Feature columns = raw sensor columns + engineered ratios
    # Explicitly exclude health_index and life_expectation_years (leakage columns)
    feature_cols_all = (
        ["sample_id"]
        + HI_FEATURE_COLS
        + engineered
    )
    # Verify all expected feature columns exist
    missing_feat_cols = [c for c in feature_cols_all if c not in df.columns]
    if missing_feat_cols:
        log.warning("[HealthIndex] Missing feature columns after engineering: %s", missing_feat_cols)

    df_features = df[feature_cols_all].copy()
    df_target = df[["sample_id", "health_index", "life_expectation_years",
                     "risk_class", "risk_int"]].copy()

    tlog.record(
        "split_features_target",
        (
            f"Features: {len(df_features.columns)} cols. "
            f"Target: {len(df_target.columns)} cols. "
            "health_index and life_expectation_years removed from features."
        ),
        len(df), len(df),
    )

    # ── Step 10: Write outputs ────────────────────────────────────────────────
    df_features.to_csv(PROC_HEALTH_FEATURES, index=False)
    df_target.to_csv(PROC_HEALTH_TARGET, index=False)

    rows_out = len(df_features)
    tlog.record(
        "write_outputs",
        f"Features → {PROC_HEALTH_FEATURES.name}  |  Target → {PROC_HEALTH_TARGET.name}",
        rows_out, rows_out,
    )

    log.info(
        "[HealthIndex] Pipeline complete. "
        "Rows in: %d, Rows out: %d, Feature cols: %d",
        rows_in, rows_out, len(df_features.columns),
    )

    return {
        "dataset": "health_index",
        "rows_in": rows_in,
        "rows_out": rows_out,
        "feature_columns": list(df_features.columns),
        "target_columns": list(df_target.columns),
        "class_distribution": class_dist,
        "features_path": str(PROC_HEALTH_FEATURES),
        "target_path": str(PROC_HEALTH_TARGET),
        "transform_log": tlog.steps,
    }

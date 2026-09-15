"""
PowerGuard AI — Feature Builder for Model A (Transformer Risk)
================================================================
Loads the two processed CSVs, merges them on sample_id, and returns
(X, y) arrays ready for scikit-learn.

No feature engineering is done here — the pipeline in data/clean_health_index.py
already computed the IEC ratio features.  This module is a clean I/O layer.

Usage
-----
    from src.ml.features.build_features import load_transformer_dataset
    X, y, feature_names = load_transformer_dataset()
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.ml_constants import (
    TRANSFORMER_FEATURES,
    TRANSFORMER_TARGET,
    MODEL_A_FEATURE_COLS,
    MODEL_A_TARGET_COL,
)

log = logging.getLogger(__name__)


def load_transformer_dataset(
    features_path: Path = TRANSFORMER_FEATURES,
    target_path:   Path = TRANSFORMER_TARGET,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Load and merge the transformer feature and target CSVs.

    Parameters
    ----------
    features_path : Path
        Path to transformer_features.csv
    target_path : Path
        Path to transformer_target.csv

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (470 rows × 20 feature columns).
    y : pd.Series
        Integer target labels (0=LOW, 1=MEDIUM, 2=HIGH).
    feature_names : list[str]
        Ordered list of feature column names (same order as X columns).

    Raises
    ------
    FileNotFoundError
        If either CSV is missing.
    ValueError
        If the merge results in row count mismatch.
    """
    log.info("Loading transformer features from %s", features_path)
    log.info("Loading transformer targets from  %s", target_path)

    feat_df   = pd.read_csv(features_path)
    target_df = pd.read_csv(target_path)

    df = feat_df.merge(target_df[["sample_id", MODEL_A_TARGET_COL]], on="sample_id", how="inner")

    if len(df) != len(feat_df):
        raise ValueError(
            f"Merge row count mismatch: features={len(feat_df)}, merged={len(df)}. "
            "Check that sample_id values match between the two files."
        )

    # Verify all expected feature columns are present
    missing = [c for c in MODEL_A_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Expected feature columns missing from data: {missing}")

    X = df[MODEL_A_FEATURE_COLS].copy()
    y = df[MODEL_A_TARGET_COL].copy()

    log.info("Loaded transformer dataset: X=%s  y=%s  classes=%s",
             X.shape, y.shape, dict(y.value_counts().sort_index()))

    return X, y, MODEL_A_FEATURE_COLS


def load_weather_dataset(
    events_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Load the weather outage events dataset and return (X, y, feature_names).

    Label-encodes `season` and `state` columns.
    Drops identifier columns (event_id, fips_code, county, event_start).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (33,139 rows × ~9 features).
    y : pd.Series
        Integer target labels (0=LOW, 1=MEDIUM, 2=HIGH).
    feature_names : list[str]
        Column names in X.
    """
    from src.ml.ml_constants import (
        WEATHER_EVENTS,
        MODEL_B_NUMERIC_COLS,
        MODEL_B_CATEGORY_COLS,
        MODEL_B_TARGET_COL,
    )
    from sklearn.preprocessing import LabelEncoder

    path = events_path or WEATHER_EVENTS
    log.info("Loading weather events from %s", path)

    df = pd.read_csv(path, parse_dates=["event_start"])

    # Encode categoricals
    encoders: dict = {}
    for col in MODEL_B_CATEGORY_COLS:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Build feature list: numeric + encoded categoricals
    feature_cols = MODEL_B_NUMERIC_COLS + [f"{c}_enc" for c in MODEL_B_CATEGORY_COLS]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Expected weather feature columns missing: {missing}")

    X = df[feature_cols].copy()
    y = df[MODEL_B_TARGET_COL].copy()

    log.info("Loaded weather dataset: X=%s  y=%s  classes=%s",
             X.shape, y.shape, dict(y.value_counts().sort_index()))

    return X, y, feature_cols, df["event_start"]  # return timestamps for chronological split

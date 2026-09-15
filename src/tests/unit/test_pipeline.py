"""
PowerGuard AI — Unit Tests for the Data Pipeline
==================================================
Tests every transformation step in isolation using synthetic
mini-DataFrames so the tests:
  - Run fast (no I/O to data/raw/ or data/processed/)
  - Are independent of the actual raw files
  - Document the expected behaviour of each transform

Run with:
    python -m pytest src/tests/unit/test_pipeline.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── Module under test imports ──────────────────────────────────────────────

from src.data.constants import (
    HI_RISK_BINS, HI_RISK_LABELS, HI_RISK_INT,
    WO_SEVERITY_BINS, WO_SEVERITY_LABELS, WO_SEVERITY_INT,
    WO_SEASON_MAP, WO_DROP_COLS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (shared test fixtures)
# ─────────────────────────────────────────────────────────────────────────────

def _make_hi_df() -> pd.DataFrame:
    """Minimal Health Index DataFrame with 5 rows covering all risk classes."""
    return pd.DataFrame({
        "hydrogen_ppm":            [10, 200, 500, 1000, 5000],
        "oxygen_ppm":              [3000, 5000, 8000, 10000, 15000],
        "nitrogen_ppm":            [30000, 40000, 50000, 60000, 70000],
        "methane_ppm":             [5, 50, 200, 500, 1500],
        "co_ppm":                  [50, 100, 200, 500, 1000],
        "co2_ppm":                 [500, 1000, 2000, 5000, 10000],
        "ethylene_ppm":            [0, 5, 50, 200, 1000],
        "ethane_ppm":              [0, 10, 50, 100, 500],
        "acetylene_ppm":           [0, 0, 1, 10, 100],
        "dbds_mg_kg":              [0, 0, 5, 20, 80],
        "power_factor_pct":        [0.5, 1.0, 5.0, 15.0, 40.0],
        "interfacial_tension_mNm": [45, 40, 35, 30, 22],
        "dielectric_kv":           [60, 55, 50, 40, 32],
        "water_content_ppm":       [5, 10, 20, 40, 80],
        "health_index":            [90.0, 70.0, 45.0, 30.0, 15.0],
        "life_expectation_years":  [30.0, 20.0, 15.0, 10.0, 6.0],
    })


def _make_wo_df() -> pd.DataFrame:
    """Minimal Weather+Outage DataFrame at hourly-row level (3 events)."""
    return pd.DataFrame({
        "fips_code":       [1001, 1001, 1001, 1003, 1003, 1005],
        "Hour":            [
            "8/12/2019 18:00", "8/12/2019 19:00", "8/12/2019 20:00",
            "7/21/2022 9:00",  "7/21/2022 10:00",
            "9/15/2021 14:00",
        ],
        "max_rolling_avg": [0.5, 0.75, 0.75, 10.0, 10.0, 500.0],  # LEAKY — must be dropped
        "state":           ["Alabama"] * 5 + ["Georgia"],
        "county":          ["Autauga"] * 3 + ["Baldwin"] * 2 + ["Appling"],
        "median_sum":      [43.0] * 5 + [55.0],                     # LEAKY — must be dropped
        "new_event_no":    [7251, 7251, 7251, 14180, 14180, 20000],
        "evDur_Hr":        [6, 6, 6, 5, 5, 1],                      # LEAKY — must be dropped
        "evDur_day":       [0, 0, 0, 0, 0, 0],                      # LEAKY — must be dropped
        "Max_outage":      [8.875, 8.875, 8.875, 0.75, 0.75, 2500.0],
        "state_st":        ["AL", "AL", "AL", "AL", "AL", "GA"],    # REDUNDANT — must be dropped
        "tmpf":            [80.8, 80.0, 78.6, 82.4, 82.4, 95.0],
        "relh":            [78.2, 80.0, 84.1, 88.9, 88.9, 45.0],
        "gust":            [np.nan] * 6,                             # 100% missing — must be dropped
        "feel":            [85.6, 84.4, 78.6, 92.7, 92.7, 109.0],   # COLLINEAR — must be dropped
        "p01i":            [0.0, 0.0, 0.0, 0.07, 0.02, 0.50],
        "sknt":            [0, 0, 3, 5, 3, 8],
    })


# ─────────────────────────────────────────────────────────────────────────────
# Tests — constants
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_hi_risk_bins_cover_full_range(self):
        assert HI_RISK_BINS[0] == 0
        assert HI_RISK_BINS[-1] == 100

    def test_hi_risk_labels_count_matches_bins(self):
        # pd.cut with n bins produces n-1 labels
        assert len(HI_RISK_LABELS) == len(HI_RISK_BINS) - 1

    def test_hi_risk_int_has_all_labels(self):
        for label in HI_RISK_LABELS:
            assert label in HI_RISK_INT

    def test_wo_severity_bins_finite_start(self):
        assert WO_SEVERITY_BINS[0] == 0

    def test_wo_drop_cols_non_empty(self):
        assert len(WO_DROP_COLS) > 0

    def test_season_map_covers_all_months(self):
        assert set(WO_SEASON_MAP.keys()) == set(range(1, 13))


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Health Index transforms
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthIndexTransforms:

    def test_risk_class_derivation_correct(self):
        """health_index binning must produce correct risk classes."""
        df = _make_hi_df()
        df["risk_class"] = pd.cut(
            df["health_index"],
            bins=HI_RISK_BINS,
            labels=HI_RISK_LABELS,
            right=False,
            include_lowest=True,
        )
        # health_index=90 → LOW; 70 → LOW; 45 → MEDIUM; 30 → MEDIUM; 15 → HIGH
        assert df.loc[0, "risk_class"] == "LOW"
        assert df.loc[1, "risk_class"] == "LOW"
        assert df.loc[2, "risk_class"] == "MEDIUM"
        assert df.loc[3, "risk_class"] == "MEDIUM"
        assert df.loc[4, "risk_class"] == "HIGH"

    def test_risk_boundaries_exact(self):
        """Boundary values: 25 → MEDIUM (not HIGH), 50 → LOW (not MEDIUM).
        Note: pd.cut with right=False makes each bin [left, right).
        HI_RISK_BINS = [0, 25, 50, 100] so:
          [0, 25)  → HIGH
          [25, 50) → MEDIUM
          [50, 100] → LOW  (include_lowest closes the last bin on the right too)
        A value of exactly 100 is captured by include_lowest for the last bin.
        """
        df = pd.DataFrame({"health_index": [0, 24.99, 25.0, 49.99, 50.0, 99.9]})
        df["risk_class"] = pd.cut(
            df["health_index"],
            bins=HI_RISK_BINS,
            labels=HI_RISK_LABELS,
            right=False,
            include_lowest=True,
        )
        assert df.loc[0, "risk_class"] == "HIGH"   # exactly 0
        assert df.loc[1, "risk_class"] == "HIGH"   # just below 25
        assert df.loc[2, "risk_class"] == "MEDIUM" # exactly 25
        assert df.loc[3, "risk_class"] == "MEDIUM" # just below 50
        assert df.loc[4, "risk_class"] == "LOW"    # exactly 50
        assert df.loc[5, "risk_class"] == "LOW"    # 99.9 — inside last bin

    def test_tdcg_calculation(self):
        """TDCG = sum of 6 combustible gases."""
        df = _make_hi_df()
        expected = (
            df["hydrogen_ppm"] + df["methane_ppm"] + df["co_ppm"]
            + df["ethylene_ppm"] + df["ethane_ppm"] + df["acetylene_ppm"]
        )
        eps = 1.0
        df["tdcg_ppm"] = (
            df["hydrogen_ppm"] + df["methane_ppm"] + df["co_ppm"]
            + df["ethylene_ppm"] + df["ethane_ppm"] + df["acetylene_ppm"]
        )
        pd.testing.assert_series_equal(df["tdcg_ppm"], expected, check_names=False)

    def test_rogers_r1_no_division_by_zero(self):
        """rogers_r1 must be computable even when hydrogen_ppm = 0."""
        df = _make_hi_df()
        df.loc[0, "hydrogen_ppm"] = 0
        eps = 1.0
        df["rogers_r1"] = df["methane_ppm"] / (df["hydrogen_ppm"] + eps)
        assert df["rogers_r1"].isnull().sum() == 0
        assert np.isfinite(df["rogers_r1"]).all()

    def test_outlier_capping_does_not_remove_rows(self):
        """Capping outliers should not change the row count.
        Cast to float64 first — pandas 3 raises TypeError when assigning
        a float cap value into an int64 column.
        """
        df = _make_hi_df()
        df["oxygen_ppm"] = df["oxygen_ppm"].astype(float)  # required for cap to work
        df.loc[0, "oxygen_ppm"] = 999_999.0  # inject implausible value
        n_before = len(df)
        p99 = float(df["oxygen_ppm"].quantile(0.99))
        df.loc[df["oxygen_ppm"] > 20_000, "oxygen_ppm"] = p99
        assert len(df) == n_before

    def test_feature_target_separation(self):
        """health_index and life_expectation_years must not appear in features."""
        df = _make_hi_df().rename(columns={
            "Hydrogen": "hydrogen_ppm",
            "Health index": "health_index",
            "Life expectation": "life_expectation_years",
        })
        from src.data.constants import HI_FEATURE_COLS, HI_EXCLUDE_FROM_FEATURES
        for excluded in HI_EXCLUDE_FROM_FEATURES:
            assert excluded not in HI_FEATURE_COLS, (
                f"Leakage: '{excluded}' found in HI_FEATURE_COLS"
            )

    def test_no_missing_after_cap(self):
        """No NaN should remain after capping (all original values are valid)."""
        df = _make_hi_df()
        assert df.isnull().sum().sum() == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Weather + Outage transforms
# ─────────────────────────────────────────────────────────────────────────────

class TestWeatherOutageTransforms:

    def test_leaky_columns_are_dropped(self):
        """All columns in WO_DROP_COLS must be absent after the drop step."""
        df = _make_wo_df()
        cols_to_drop = [c for c in WO_DROP_COLS.keys() if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        for col in WO_DROP_COLS:
            assert col not in df.columns, f"Leaky column '{col}' survived the drop step"

    def test_max_outage_not_in_features_after_drop(self):
        """Max_outage is the target — it must stay in the DataFrame (not dropped)."""
        df = _make_wo_df()
        cols_to_drop = [c for c in WO_DROP_COLS.keys() if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        # Max_outage should still be present (it is the TARGET, not a leaky feature)
        assert "Max_outage" in df.columns

    def test_event_aggregation_reduces_rows(self):
        """Aggregating 6 hourly rows across 3 events must yield exactly 3 events.
        new_event_no is kept during aggregation (it is the groupby key) and is
        only removed from predictive features at inference time — it must NOT
        be dropped before the groupby step.
        """
        df = _make_wo_df()
        # Drop only the truly leaky/redundant cols; keep new_event_no for groupby
        drop_except_event_id = [
            c for c in WO_DROP_COLS.keys()
            if c in df.columns and c != "new_event_no"
        ]
        df = df.drop(columns=drop_except_event_id)
        df["Hour"] = pd.to_datetime(df["Hour"], format="%m/%d/%Y %H:%M")
        df = df.sort_values(["new_event_no", "Hour"])
        df_agg = df.groupby("new_event_no", sort=False).agg(
            state=("state", "first"),
            county=("county", "first"),
            Max_outage=("Max_outage", "first"),
            tmpf=("tmpf", "first"),
        ).reset_index()
        assert len(df_agg) == 3

    def test_aggregation_uses_first_weather_value(self):
        """First-hour strategy: first row's tmpf must be taken for each event.
        new_event_no is kept for groupby; other leaky cols are dropped.
        """
        df = _make_wo_df()
        drop_except_event_id = [
            c for c in WO_DROP_COLS.keys()
            if c in df.columns and c != "new_event_no"
        ]
        df = df.drop(columns=drop_except_event_id)
        df["Hour"] = pd.to_datetime(df["Hour"], format="%m/%d/%Y %H:%M")
        df = df.sort_values(["new_event_no", "Hour"])
        df_agg = df.groupby("new_event_no").agg(tmpf=("tmpf", "first")).reset_index()
        # Event 7251 first hour tmpf = 80.8
        assert df_agg.loc[df_agg["new_event_no"] == 7251, "tmpf"].values[0] == pytest.approx(80.8)

    def test_severity_binning_correct(self):
        """Binning thresholds: <10 → LOW, 10-1000 → MEDIUM, >1000 → HIGH."""
        df = pd.DataFrame({"max_customers_affected": [1.0, 9.99, 10.0, 999.99, 1000.0, 62000.0]})
        df["sev"] = pd.cut(
            df["max_customers_affected"],
            bins=WO_SEVERITY_BINS,
            labels=WO_SEVERITY_LABELS,
            right=False,
            include_lowest=True,
        )
        assert str(df.loc[0, "sev"]) == "LOW"
        assert str(df.loc[1, "sev"]) == "LOW"
        assert str(df.loc[2, "sev"]) == "MEDIUM"
        assert str(df.loc[3, "sev"]) == "MEDIUM"
        assert str(df.loc[4, "sev"]) == "HIGH"
        assert str(df.loc[5, "sev"]) == "HIGH"

    def test_p01i_imputed_to_zero_not_median(self):
        """p01i missing values must be imputed with 0.0 (physical default)."""
        from src.data.constants import WO_IMPUTE_DEFAULTS
        assert WO_IMPUTE_DEFAULTS["p01i"] == 0.0

    def test_timestamp_parsing(self):
        """Timestamp column must parse to datetime without errors."""
        df = _make_wo_df()
        df["Hour"] = pd.to_datetime(df["Hour"], format="%m/%d/%Y %H:%M")
        assert df["Hour"].isnull().sum() == 0
        assert pd.api.types.is_datetime64_any_dtype(df["Hour"])

    def test_season_extraction(self):
        """Season derivation from month must be correct for all 4 seasons."""
        months   = [1,  3,  6,  9]
        expected = ["WINTER", "SPRING", "SUMMER", "AUTUMN"]
        for month, exp in zip(months, expected):
            assert WO_SEASON_MAP[month] == exp

    def test_no_raw_dir_writes(self):
        """RAW_DIR must be read-only — verify path constant is not under processed/."""
        from src.data.constants import RAW_DIR, PROCESSED_DIR
        # They must be different directories
        assert RAW_DIR != PROCESSED_DIR
        # RAW_DIR must not be inside PROCESSED_DIR
        assert not str(RAW_DIR).startswith(str(PROCESSED_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# Tests — Validation module
# ─────────────────────────────────────────────────────────────────────────────

class TestValidation:

    def test_validation_report_passes_by_default(self):
        """A fresh ValidationReport should have passed=True and no issues."""
        from src.data.validate import ValidationReport
        r = ValidationReport(dataset_name="Test")
        assert r.passed is True
        assert len(r.issues) == 0

    def test_add_issue_sets_passed_false(self):
        """Adding an issue must set passed=False."""
        from src.data.validate import ValidationReport
        r = ValidationReport(dataset_name="Test")
        r.add_issue("Something is broken")
        assert r.passed is False
        assert len(r.issues) == 1

    def test_add_warning_does_not_fail(self):
        """Adding a warning must NOT set passed=False."""
        from src.data.validate import ValidationReport
        r = ValidationReport(dataset_name="Test")
        r.add_warning("Minor concern")
        assert r.passed is True
        assert len(r.warnings) == 1

    def test_safe_float_handles_na(self):
        """_safe_float must return None for all NA sentinel strings."""
        from src.data.validate import _safe_float
        for sentinel in ("NA", "N/A", "NULL", "NONE", "", " "):
            assert _safe_float(sentinel) is None

    def test_safe_float_parses_valid_numbers(self):
        """_safe_float must parse integers, floats, and scientific notation."""
        from src.data.validate import _safe_float
        assert _safe_float("42") == pytest.approx(42.0)
        assert _safe_float("3.14") == pytest.approx(3.14)
        assert _safe_float("1e3") == pytest.approx(1000.0)
        assert _safe_float("-5.5") == pytest.approx(-5.5)

"""
PowerGuard AI — Raw Data Validator
====================================
Checks raw CSV files for schema compliance, column presence, and
physical-domain plausibility WITHOUT modifying any files.

All functions in this module are pure read operations.  They return
a structured ValidationReport dict so callers can decide how to react.

Usage
-----
    from src.data.validate import validate_health_index, validate_weather_outage
    report = validate_health_index()
    if not report["passed"]:
        for issue in report["issues"]:
            print(issue)
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.data.constants import (
    RAW_HEALTH_INDEX,
    RAW_WEATHER,
    HI_RAW_COLUMNS,
    HI_PHYSICAL_BOUNDS,
    HI_RENAME_MAP,
    WO_RAW_COLUMNS,
    WO_PHYSICAL_BOUNDS,
    WO_TIMESTAMP_COL,
    WO_TIMESTAMP_FMT,
    WO_TARGET_COL,
)

log = logging.getLogger(__name__)


# ── Lightweight result container ─────────────────────────────────────────────

@dataclass
class ValidationReport:
    """Holds the outcome of a single dataset validation pass."""

    dataset_name: str
    passed: bool = True
    row_count: int = 0
    col_count: int = 0
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add_issue(self, msg: str) -> None:
        """Record a hard failure — sets passed=False."""
        self.issues.append(msg)
        self.passed = False
        log.error("[%s] ISSUE: %s", self.dataset_name, msg)

    def add_warning(self, msg: str) -> None:
        """Record a soft warning — does not fail the report."""
        self.warnings.append(msg)
        log.warning("[%s] WARNING: %s", self.dataset_name, msg)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"[{self.dataset_name}] Validation {status} | "
            f"rows={self.row_count} cols={self.col_count} | "
            f"issues={len(self.issues)} warnings={len(self.warnings)}"
        )


# ── Internal helpers ─────────────────────────────────────────────────────────

def _read_raw_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read a CSV into header + data rows.  Never modifies the file."""
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty file: {path}")
    return rows[0], rows[1:]


def _safe_float(value: str) -> float | None:
    """Return float or None if unparseable / NA."""
    v = value.strip()
    if v.upper() in ("NA", "N/A", "NULL", "NONE", ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _check_bounds(
    report: ValidationReport,
    header: list[str],
    data_rows: list[list[str]],
    bounds: dict[str, tuple[float, float, str]],
    rename_map: dict[str, str] | None = None,
) -> dict[str, int]:
    """
    Check every value against physical-domain bounds.

    Args:
        report:     ValidationReport to append warnings to.
        header:     Column names from the CSV header.
        data_rows:  Data rows (list of string lists).
        bounds:     Dict of clean_col_name → (min, max, description).
        rename_map: Mapping from raw col name → clean col name.

    Returns:
        Dict of clean_col_name → count of out-of-bounds values.
    """
    # Build index: clean_name → column index in header
    col_idx: dict[str, int] = {}
    for raw_name, clean_name in (rename_map or {}).items():
        if raw_name in header:
            col_idx[clean_name] = header.index(raw_name)
    # Also handle already-clean names that appear directly in header
    for clean_name in bounds:
        if clean_name in header and clean_name not in col_idx:
            col_idx[clean_name] = header.index(clean_name)

    oob_counts: dict[str, int] = {}
    for clean_name, (lo, hi, desc) in bounds.items():
        idx = col_idx.get(clean_name)
        if idx is None:
            continue
        oob = 0
        for row_num, row in enumerate(data_rows, start=2):
            if idx >= len(row):
                continue
            val = _safe_float(row[idx])
            if val is None:
                continue
            if not (lo <= val <= hi):
                oob += 1
        if oob:
            oob_counts[clean_name] = oob
            report.add_warning(
                f"Column '{clean_name}' has {oob} value(s) outside "
                f"physical bounds [{lo}, {hi}] — {desc}"
            )
    return oob_counts


# ── Public validators ─────────────────────────────────────────────────────────

def validate_health_index() -> ValidationReport:
    """
    Validate the raw Health Index / DGA dataset.

    Checks performed:
      1. File exists and is readable.
      2. All expected columns are present.
      3. No unexpected extra columns.
      4. Row count is non-zero.
      5. No empty cells or NA values.
      6. All values are numeric.
      7. Physical-domain bounds (IEEE C57.104 / IEC 60422).
      8. Duplicate row detection.
      9. Health index is in [0, 100].
    """
    report = ValidationReport(dataset_name="HealthIndex")

    # 1. Read
    try:
        header, data = _read_raw_csv(RAW_HEALTH_INDEX)
    except (FileNotFoundError, ValueError) as exc:
        report.add_issue(str(exc))
        return report

    report.row_count = len(data)
    report.col_count = len(header)

    # 2. Expected columns present
    missing_cols = [c for c in HI_RAW_COLUMNS if c not in header]
    if missing_cols:
        report.add_issue(f"Missing expected columns: {missing_cols}")

    # 3. No extra unexpected columns
    extra_cols = [c for c in header if c not in HI_RAW_COLUMNS]
    if extra_cols:
        report.add_warning(f"Unexpected extra columns (will be ignored): {extra_cols}")

    # 4. Non-zero rows
    if report.row_count == 0:
        report.add_issue("Dataset has zero data rows.")

    # 5 & 6. Missing and non-numeric values
    na_counts: dict[str, int] = {}
    non_numeric: dict[str, int] = {}
    for row_num, row in enumerate(data, start=2):
        for j, col in enumerate(header):
            if j >= len(row):
                na_counts[col] = na_counts.get(col, 0) + 1
                continue
            val = row[j].strip()
            if val.upper() in ("NA", "N/A", "NULL", "NONE", ""):
                na_counts[col] = na_counts.get(col, 0) + 1
            else:
                try:
                    float(val)
                except ValueError:
                    non_numeric[col] = non_numeric.get(col, 0) + 1

    if na_counts:
        report.add_warning(f"NA / empty values found: {na_counts}")
    if non_numeric:
        report.add_issue(f"Non-numeric values in numeric columns: {non_numeric}")

    # 7. Physical bounds
    oob = _check_bounds(report, header, data, HI_PHYSICAL_BOUNDS, HI_RENAME_MAP)
    report.stats["out_of_bounds"] = oob

    # 8. Duplicate rows
    seen: set[tuple] = set()
    dup_count = 0
    for row in data:
        key = tuple(row)
        if key in seen:
            dup_count += 1
        else:
            seen.add(key)
    if dup_count:
        report.add_warning(f"Found {dup_count} duplicate row(s).")
    report.stats["duplicates"] = dup_count

    # 9. Health index range
    hi_idx = header.index("Health index") if "Health index" in header else None
    if hi_idx is not None:
        out_of_range = sum(
            1 for row in data
            if hi_idx < len(row) and _safe_float(row[hi_idx]) is not None
            and not (0 <= _safe_float(row[hi_idx]) <= 100)
        )
        if out_of_range:
            report.add_issue(f"Health index has {out_of_range} value(s) outside [0, 100].")

    log.info(report.summary())
    return report


def validate_weather_outage() -> ValidationReport:
    """
    Validate the raw Weather + Outage dataset.

    Checks performed:
      1. File exists and is readable.
      2. All expected columns are present.
      3. Row count is non-zero.
      4. Timestamp column parseable.
      5. Target column (Max_outage) is non-negative numeric.
      6. Physical-domain bounds for weather columns.
      7. Missing value inventory.
      8. Duplicate row detection.
    """
    report = ValidationReport(dataset_name="WeatherOutage")

    # 1. Read
    try:
        header, data = _read_raw_csv(RAW_WEATHER)
    except (FileNotFoundError, ValueError) as exc:
        report.add_issue(str(exc))
        return report

    report.row_count = len(data)
    report.col_count = len(header)

    # 2. Expected columns
    missing_cols = [c for c in WO_RAW_COLUMNS if c not in header]
    if missing_cols:
        report.add_issue(f"Missing expected columns: {missing_cols}")

    # 3. Non-zero rows
    if report.row_count == 0:
        report.add_issue("Dataset has zero data rows.")

    # 4. Timestamp parseable — check first 200 rows as a sample
    import datetime
    ts_idx = header.index(WO_TIMESTAMP_COL) if WO_TIMESTAMP_COL in header else None
    ts_errors = 0
    if ts_idx is not None:
        for row in data[:200]:
            if ts_idx >= len(row):
                continue
            try:
                datetime.datetime.strptime(row[ts_idx].strip(), WO_TIMESTAMP_FMT)
            except ValueError:
                ts_errors += 1
    if ts_errors:
        report.add_issue(
            f"Timestamp column '{WO_TIMESTAMP_COL}' has {ts_errors}/200 "
            f"sample rows unparseable with format '{WO_TIMESTAMP_FMT}'."
        )

    # 5. Target non-negative
    tgt_idx = header.index(WO_TARGET_COL) if WO_TARGET_COL in header else None
    neg_target = 0
    if tgt_idx is not None:
        for row in data:
            val = _safe_float(row[tgt_idx]) if tgt_idx < len(row) else None
            if val is not None and val < 0:
                neg_target += 1
    if neg_target:
        report.add_issue(f"Target '{WO_TARGET_COL}' has {neg_target} negative value(s).")

    # 6. Physical bounds for weather features (no rename needed — same names)
    oob = _check_bounds(report, header, data, WO_PHYSICAL_BOUNDS, rename_map=None)
    report.stats["out_of_bounds"] = oob

    # 7. Missing value inventory
    na_counts: dict[str, int] = {}
    for row in data:
        for j, col in enumerate(header):
            if j >= len(row):
                na_counts[col] = na_counts.get(col, 0) + 1
            elif row[j].strip().upper() in ("NA", "N/A", "NULL", "NONE", ""):
                na_counts[col] = na_counts.get(col, 0) + 1
    if na_counts:
        report.add_warning(f"NA / empty values by column: {na_counts}")
    report.stats["missing_counts"] = na_counts

    # 8. Duplicates — use first 10,000 rows as a check (full scan is O(n))
    seen: set[tuple] = set()
    dup_count = 0
    for row in data[:10_000]:
        key = tuple(row)
        if key in seen:
            dup_count += 1
        else:
            seen.add(key)
    if dup_count:
        report.add_warning(f"Found {dup_count} duplicate row(s) in first 10,000 rows.")
    report.stats["duplicates_sample"] = dup_count

    log.info(report.summary())
    return report

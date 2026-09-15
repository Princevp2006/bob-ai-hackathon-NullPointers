"""
PowerGuard AI -- Data Pipeline Orchestrator
============================================
Runs the complete preprocessing pipeline in the correct order:

  1. Validate raw files (schema + physical bounds).
  2. Preprocess Health Index / DGA dataset.
  3. Preprocess Weather + Outage dataset.
  4. Write a machine-readable pipeline run log to data/processed/.
  5. Print a human-readable summary report.

No raw files in data/raw/ are modified at any point.
No artificial joins are performed between the two datasets.

Usage
-----
Run from the repository root:

    python -m src.data.pipeline

Or import programmatically:

    from src.data.pipeline import run_pipeline
    results = run_pipeline(verbose=True)
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src.data.constants import PROCESSED_DIR, PROC_PIPELINE_LOG
from src.data.validate import validate_health_index, validate_weather_outage
from src.data.clean_health_index import run as run_health_index
from src.data.clean_weather_outage import run as run_weather_outage

log = logging.getLogger(__name__)


def run_pipeline(verbose: bool = True, fail_on_validation_error: bool = True) -> dict:
    """
    Execute the full data preprocessing pipeline.

    Parameters
    ----------
    verbose:
        If True, configure root logger to INFO and print a summary to stdout.
    fail_on_validation_error:
        If True (default), abort if a raw file fails hard validation.
        If False, log the error and continue (useful for partial runs / CI).

    Returns
    -------
    dict with keys: validation_reports, health_index_result,
                    weather_result, pipeline_status, run_at
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    run_at = datetime.now(tz=timezone.utc).isoformat()
    pipeline_log: dict = {"run_at": run_at, "stages": []}
    pipeline_status = "SUCCESS"

    # --------------------------------------------------------------------------
    # Stage 1 -- Validation
    # --------------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("  PowerGuard AI -- Data Pipeline")
    print("=" * 64)
    print(f"\n[1/3] Validating raw files ...\n")

    val_hi = validate_health_index()
    val_wo = validate_weather_outage()

    for report in (val_hi, val_wo):
        status_icon = "[OK]" if report.passed else "[X]"
        print(f"  {status_icon} {report.dataset_name}: {report.summary()}")
        if report.warnings:
            for w in report.warnings:
                print(f"       !  {w}")
        if report.issues:
            for e in report.issues:
                print(f"       X  {e}")

    pipeline_log["stages"].append({
        "stage": "validation",
        "health_index": {
            "passed": val_hi.passed,
            "rows": val_hi.row_count,
            "issues": val_hi.issues,
            "warnings": val_hi.warnings,
            "stats": val_hi.stats,
        },
        "weather_outage": {
            "passed": val_wo.passed,
            "rows": val_wo.row_count,
            "issues": val_wo.issues,
            "warnings": val_wo.warnings,
            "stats": {k: str(v) for k, v in val_wo.stats.items()},
        },
    })

    if fail_on_validation_error and (not val_hi.passed or not val_wo.passed):
        pipeline_status = "ABORTED_VALIDATION_FAILURE"
        pipeline_log["pipeline_status"] = pipeline_status
        _write_log(pipeline_log)
        print(f"\n  Pipeline aborted due to validation failures.\n")
        return {
            "validation_reports": {"health_index": val_hi, "weather_outage": val_wo},
            "health_index_result": None,
            "weather_result": None,
            "pipeline_status": pipeline_status,
            "run_at": run_at,
        }

    # --------------------------------------------------------------------------
    # Stage 2 -- Health Index preprocessing
    # --------------------------------------------------------------------------
    print(f"\n[2/3] Preprocessing Health Index / DGA dataset ...\n")
    hi_result = None
    try:
        hi_result = run_health_index(verbose=verbose)
        print(f"  [OK] Health Index processed.")
        print(f"     Rows in : {hi_result['rows_in']}")
        print(f"     Rows out: {hi_result['rows_out']}")
        print(f"     Feature columns ({len(hi_result['feature_columns'])}): "
              f"{hi_result['feature_columns']}")
        print(f"     Target columns : {hi_result['target_columns']}")
        print(f"     Class distribution: {hi_result['class_distribution']}")
        print(f"     Features -> {Path(hi_result['features_path']).name}")
        print(f"     Target   -> {Path(hi_result['target_path']).name}")
        pipeline_log["stages"].append({
            "stage": "health_index_preprocessing",
            "status": "SUCCESS",
            "result": {k: v for k, v in hi_result.items() if k != "transform_log"},
        })
    except Exception as exc:
        pipeline_status = "PARTIAL_FAILURE"
        tb = traceback.format_exc()
        print(f"  [X] Health Index preprocessing FAILED: {exc}")
        log.error("Health Index pipeline error:\n%s", tb)
        pipeline_log["stages"].append({
            "stage": "health_index_preprocessing",
            "status": "FAILED",
            "error": str(exc),
        })

    # --------------------------------------------------------------------------
    # Stage 3 -- Weather + Outage preprocessing
    # --------------------------------------------------------------------------
    print(f"\n[3/3] Preprocessing Weather + Outage dataset ...\n")
    wo_result = None
    try:
        wo_result = run_weather_outage(verbose=verbose)
        print(f"  [OK] Weather + Outage processed.")
        print(f"     Rows in (hourly): {wo_result['rows_in']}")
        print(f"     Events out      : {wo_result['rows_out']}")
        print(f"     Columns ({len(wo_result['columns'])}): {wo_result['columns']}")
        print(f"     Class distribution: {wo_result['class_distribution']}")
        if wo_result.get("impute_values"):
            print(f"     Imputed medians: {wo_result['impute_values']}")
        print(f"     Output -> {Path(wo_result['output_path']).name}")
        pipeline_log["stages"].append({
            "stage": "weather_outage_preprocessing",
            "status": "SUCCESS",
            "result": {k: v for k, v in wo_result.items() if k != "transform_log"},
        })
    except Exception as exc:
        pipeline_status = "PARTIAL_FAILURE"
        tb = traceback.format_exc()
        print(f"  [X] Weather + Outage preprocessing FAILED: {exc}")
        log.error("Weather+Outage pipeline error:\n%s", tb)
        pipeline_log["stages"].append({
            "stage": "weather_outage_preprocessing",
            "status": "FAILED",
            "error": str(exc),
        })

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------
    pipeline_log["pipeline_status"] = pipeline_status
    _write_log(pipeline_log)

    print("\n" + "-" * 64)
    print("  PIPELINE SUMMARY")
    print("-" * 64)
    print(f"  Status   : {pipeline_status}")
    print(f"  Run at   : {run_at}")
    print(f"  Log      : {PROC_PIPELINE_LOG}")
    print(f"\n  Architecture note:")
    print(f"  +-----------------------------------------------------+")
    print(f"  |  Dataset 1 (Health Index) and Dataset 2 (Weather)   |")
    print(f"  |  have NO common key and are kept SEPARATE.           |")
    print(f"  |                                                       |")
    print(f"  |  Model A: transformer_features.csv  ->  risk_class   |")
    print(f"  |  Model B: weather_outage_events.csv ->  outage_sev.  |")
    print(f"  |                                                       |")
    print(f"  |  Scores are fused at inference time -- not by join.   |")
    print(f"  +-----------------------------------------------------+")
    print()

    return {
        "validation_reports": {"health_index": val_hi, "weather_outage": val_wo},
        "health_index_result": hi_result,
        "weather_result": wo_result,
        "pipeline_status": pipeline_status,
        "run_at": run_at,
    }


def _write_log(log_data: dict) -> None:
    """Write the pipeline run log as JSON (for machine consumption)."""
    try:
        with open(PROC_PIPELINE_LOG, "w", encoding="utf-8") as fh:
            json.dump(log_data, fh, indent=2, default=str)
    except Exception as exc:
        log.error("Could not write pipeline log: %s", exc)


if __name__ == "__main__":
    results = run_pipeline(verbose=True)
    sys.exit(0 if results["pipeline_status"] == "SUCCESS" else 1)

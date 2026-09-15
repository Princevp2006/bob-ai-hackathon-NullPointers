"""
PowerGuard AI — Model Evaluator
================================
Computes and formats all evaluation metrics for a fitted classifier.

Produces:
  - classification_report dict
  - confusion_matrix array
  - ROC-AUC (macro OVR)
  - feature importances (if RF)
  - saves a JSON results file

Usage
-----
    from src.ml.evaluation.evaluator import evaluate_classifier, save_results
    results = evaluate_classifier(clf, X_test, y_test, class_names=['LOW','MED','HIGH'])
    save_results(results, Path('artifacts/model_a_results.json'))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

log = logging.getLogger(__name__)


def evaluate_classifier(
    model,
    X_test:      pd.DataFrame | np.ndarray,
    y_test:      pd.Series    | np.ndarray,
    class_names: list[str],
    model_name:  str = "model",
) -> dict[str, Any]:
    """
    Compute a full set of evaluation metrics for a fitted classifier.

    Parameters
    ----------
    model       : fitted classifier with .predict() and .predict_proba()
    X_test      : feature matrix for the test set
    y_test      : true integer labels for the test set
    class_names : ordered list of class name strings
    model_name  : label used in log messages and result dict

    Returns
    -------
    dict with keys:
        model_name, accuracy, classification_report, confusion_matrix,
        roc_auc_macro, n_test_samples, class_counts
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)

    # Classification report as a dict (each class + macro + weighted averages)
    cr_dict = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cr_str = classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred).tolist()

    # ROC-AUC (one-vs-rest, macro average) — requires at least 2 classes in test
    n_classes = len(class_names)
    classes   = list(range(n_classes))
    y_bin = label_binarize(y_test, classes=classes)
    try:
        roc_auc = roc_auc_score(
            y_bin, y_proba,
            average="macro",
            multi_class="ovr",
        )
    except ValueError as exc:
        # Happens when a class is absent from y_test
        log.warning("ROC-AUC could not be computed: %s", exc)
        roc_auc = float("nan")

    # Class counts in the test set
    unique, counts = np.unique(np.asarray(y_test), return_counts=True)
    class_counts = {class_names[int(u)]: int(c) for u, c in zip(unique, counts)}

    results = {
        "model_name":              model_name,
        "accuracy":                round(float(acc), 4),
        "classification_report":   cr_dict,
        "classification_report_str": cr_str,
        "confusion_matrix":        cm,
        "confusion_matrix_labels": class_names,
        "roc_auc_macro":           round(float(roc_auc), 4) if not np.isnan(roc_auc) else None,
        "n_test_samples":          int(len(y_test)),
        "class_counts":            class_counts,
    }

    log.info(
        "[%s] accuracy=%.4f  roc_auc=%.4f  F1_macro=%.4f",
        model_name, acc, roc_auc if not np.isnan(roc_auc) else -1,
        cr_dict.get("macro avg", {}).get("f1-score", -1),
    )

    return results


def save_results(results: dict[str, Any], path: Path | str) -> None:
    """Write evaluation results dict to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    log.info("Evaluation results saved to %s", path)


def print_results(results: dict[str, Any]) -> None:
    """Print a human-readable summary of evaluation results."""
    print(f"\n{'='*60}")
    print(f"  {results['model_name']} — Evaluation Results")
    print(f"{'='*60}")
    print(f"  Test samples : {results['n_test_samples']}")
    print(f"  Class counts : {results['class_counts']}")
    print(f"  Accuracy     : {results['accuracy']:.4f}")
    print(f"  ROC-AUC (macro OVR): {results.get('roc_auc_macro', 'N/A')}")
    print()
    print(results["classification_report_str"])
    print("  Confusion matrix (rows=true, cols=pred):")
    labels = results["confusion_matrix_labels"]
    print(f"  Labels: {labels}")
    for row in results["confusion_matrix"]:
        print(f"    {row}")
    print()

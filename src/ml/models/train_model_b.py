"""
PowerGuard AI — Model B Training Script
=========================================
Trains and evaluates the Weather Outage Severity Classifier.

Steps
-----
1.  Load weather_outage_events.csv via load_weather_dataset()
2.  Chronological split: train on events before MODEL_B_TRAIN_CUTOFF,
    test on events on/after MODEL_B_TRAIN_CUTOFF
3.  Fit LogisticRegression baseline
4.  Fit RandomForestClassifier main model
5.  5-fold stratified CV on training set for both (on random sub-sample
    for speed: 10,000 samples max)
6.  Evaluate both on held-out test set
7.  Save RF model artifact and metadata to src/ml/models/artifacts/
8.  Save evaluation results to artifacts/model_b_results.json
9.  Print summary report

Notes
-----
- Dataset B has no shared key with Dataset A — these are independent models.
- Chronological split is mandatory for time-series event data to prevent
  look-ahead leakage.
- Classes are severely imbalanced (LOW 63%, MEDIUM 35%, HIGH 1.4%).
  class_weight='balanced' is set in MODEL_B_RF_PARAMS.

Run from the repository root:

    python -m src.ml.models.train_model_b

Returns
-------
dict with all results (also returned from train_model_b() for programmatic use)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.ml.features.build_features import load_weather_dataset
from src.ml.models.weather_classifier import WeatherClassifier
from src.ml.evaluation.evaluator import evaluate_classifier, print_results, save_results
from src.ml.ml_constants import (
    MODEL_B_CLASS_NAMES,
    MODEL_B_TRAIN_CUTOFF,
    MODEL_B_RANDOM_STATE,
    MODEL_B_PATH,
    MODEL_B_META_PATH,
    ARTIFACTS_DIR,
)

log = logging.getLogger(__name__)

# Maximum samples used for CV (full 26k-row train set is slow for CV)
_CV_MAX_SAMPLES = 10_000


def train_model_b(verbose: bool = True) -> dict:
    """
    Full training pipeline for Model B (Weather Outage Severity Classifier).

    Returns a dict containing all training and evaluation results.
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        )

    print("\n" + "=" * 62)
    print("  PowerGuard AI — Model B: Weather Outage Severity")
    print("=" * 62)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X, y, feature_names, timestamps = load_weather_dataset()
    print(f"\n  Dataset : {X.shape[0]} samples x {X.shape[1]} features")
    print(f"  Classes : {dict(zip(MODEL_B_CLASS_NAMES, np.bincount(y)))}")
    pct = np.bincount(y) / len(y) * 100
    print(f"  Imbalance (%) : LOW={pct[0]:.1f}%  MED={pct[1]:.1f}%  HIGH={pct[2]:.1f}%")

    # ── 2. Chronological split ────────────────────────────────────────────────
    # All events strictly before CUTOFF → train; on/after CUTOFF → test
    cutoff = pd.Timestamp(MODEL_B_TRAIN_CUTOFF)
    train_mask = timestamps < cutoff
    test_mask  = timestamps >= cutoff

    X_train = X.loc[train_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    X_test  = X.loc[test_mask].reset_index(drop=True)
    y_test  = y.loc[test_mask].reset_index(drop=True)

    print(f"\n  Cutoff  : {MODEL_B_TRAIN_CUTOFF}")
    print(f"  Train   : {len(X_train)} samples  {np.bincount(y_train)}")
    print(f"  Test    : {len(X_test)} samples  {np.bincount(y_test)}")

    if len(X_test) == 0:
        raise RuntimeError(
            f"No test samples after cutoff {MODEL_B_TRAIN_CUTOFF}. "
            "Check MODEL_B_TRAIN_CUTOFF in ml_constants.py."
        )

    # ── 3. Stratified CV on a sub-sample of train set ─────────────────────────
    # Sub-sample to keep CV fast; preserve class proportions via stratification
    rng = np.random.default_rng(MODEL_B_RANDOM_STATE)
    cv_size = min(_CV_MAX_SAMPLES, len(X_train))
    cv_idx  = rng.choice(len(X_train), size=cv_size, replace=False)
    X_cv = X_train.iloc[cv_idx]
    y_cv = y_train.iloc[cv_idx]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=MODEL_B_RANDOM_STATE)

    print(f"\n  Running 5-fold CV on {cv_size}-sample sub-set of training data ...")

    lr_clf = WeatherClassifier(model_type="lr")
    rf_clf = WeatherClassifier(model_type="rf")

    lr_pipe = lr_clf._build_pipeline()
    rf_pipe = rf_clf._build_pipeline()

    lr_cv_f1  = cross_val_score(lr_pipe, X_cv, y_cv, cv=cv, scoring="f1_weighted")
    rf_cv_f1  = cross_val_score(rf_pipe, X_cv, y_cv, cv=cv, scoring="f1_weighted")
    lr_cv_f1m = cross_val_score(lr_pipe, X_cv, y_cv, cv=cv, scoring="f1_macro")
    rf_cv_f1m = cross_val_score(rf_pipe, X_cv, y_cv, cv=cv, scoring="f1_macro")

    cv_results = {
        "cv_sample_size":          cv_size,
        "lr_cv_f1_weighted_mean":  float(lr_cv_f1.mean()),
        "lr_cv_f1_weighted_std":   float(lr_cv_f1.std()),
        "rf_cv_f1_weighted_mean":  float(rf_cv_f1.mean()),
        "rf_cv_f1_weighted_std":   float(rf_cv_f1.std()),
        "lr_cv_f1_macro_mean":     float(lr_cv_f1m.mean()),
        "rf_cv_f1_macro_mean":     float(rf_cv_f1m.mean()),
    }
    print(f"  LR  5-fold CV  F1-weighted: {lr_cv_f1.mean():.4f} +/- {lr_cv_f1.std():.4f}")
    print(f"  RF  5-fold CV  F1-weighted: {rf_cv_f1.mean():.4f} +/- {rf_cv_f1.std():.4f}")
    print(f"  LR  5-fold CV  F1-macro   : {lr_cv_f1m.mean():.4f}")
    print(f"  RF  5-fold CV  F1-macro   : {rf_cv_f1m.mean():.4f}")

    # ── 4. Fit both models on full training set ───────────────────────────────
    print("\n  Fitting Logistic Regression baseline on full train set ...")
    lr_clf.fit(X_train, y_train, feature_names=feature_names)

    print("  Fitting Random Forest main model on full train set ...")
    rf_clf.fit(X_train, y_train, feature_names=feature_names)

    # ── 5. Evaluate on held-out test set ─────────────────────────────────────
    print("\n  Evaluating on HELD-OUT TEST SET ...")
    rf_test_results = evaluate_classifier(rf_clf, X_test, y_test, MODEL_B_CLASS_NAMES,
                                          model_name="RF_Main_test_FINAL")
    lr_test_results = evaluate_classifier(lr_clf, X_test, y_test, MODEL_B_CLASS_NAMES,
                                          model_name="LR_Baseline_test_FINAL")
    print_results(rf_test_results)
    print_results(lr_test_results)

    # ── 6. Feature importances ────────────────────────────────────────────────
    fi = rf_clf.feature_importances()
    print("  Top feature importances (RF):")
    for feat, imp in fi.items():
        print(f"    {feat:<30} {imp:.4f}")

    # ── 7. Save model artifacts ───────────────────────────────────────────────
    rf_clf.save(MODEL_B_PATH)
    rf_clf.save_metadata(
        MODEL_B_META_PATH,
        extra={
            "train_cutoff":   MODEL_B_TRAIN_CUTOFF,
            "cv_f1_weighted": f"{rf_cv_f1.mean():.4f} +/- {rf_cv_f1.std():.4f}",
            "test_accuracy":  rf_test_results["accuracy"],
            "test_roc_auc":   rf_test_results["roc_auc_macro"],
            "test_f1_macro":  rf_test_results["classification_report"]["macro avg"]["f1-score"],
        },
    )

    # Save all results to JSON
    all_results = {
        "model_b_weather_severity": {
            "cv":         cv_results,
            "test_rf":    rf_test_results,
            "test_lr":    lr_test_results,
            "feature_importances": fi.to_dict(),
            "split_info": {
                "train_cutoff":       MODEL_B_TRAIN_CUTOFF,
                "n_train":            len(X_train),
                "n_test":             len(X_test),
                "train_class_dist":   np.bincount(y_train).tolist(),
                "test_class_dist":    np.bincount(y_test).tolist(),
            },
        }
    }
    results_path = ARTIFACTS_DIR / "model_b_results.json"
    save_results(all_results, results_path)

    print(f"\n  Model saved    : {MODEL_B_PATH}")
    print(f"  Metadata saved : {MODEL_B_META_PATH}")
    print(f"  Results saved  : {results_path}")
    print("\n  Selected model: Random Forest")
    f1_macro   = rf_test_results["classification_report"]["macro avg"]["f1-score"]
    f1_weighted = rf_test_results["classification_report"]["weighted avg"]["f1-score"]
    print(f"  Test F1 (macro)   : {f1_macro:.4f}")
    print(f"  Test F1 (weighted): {f1_weighted:.4f}")
    print(f"  Test ROC-AUC      : {rf_test_results['roc_auc_macro']}")

    return all_results


if __name__ == "__main__":
    train_model_b(verbose=True)
    sys.exit(0)

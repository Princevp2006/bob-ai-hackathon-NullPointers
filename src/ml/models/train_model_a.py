"""
PowerGuard AI — Model A Training Script
=========================================
Trains and evaluates the Transformer Risk Classifier.

Steps
-----
1.  Load transformer_features.csv + transformer_target.csv
2.  Stratified split: 70% train / 15% val / 15% test  (random_state=42)
3.  Fit LogisticRegression baseline
4.  Fit RandomForestClassifier main model
5.  5-fold stratified CV on training set for both
6.  Evaluate both on held-out validation set
7.  Evaluate selected model on held-out test set
8.  Save RF model artifact and metadata to src/ml/models/artifacts/
9.  Save evaluation results to artifacts/model_a_results.json
10. Print summary report

Run from the repository root:

    python -m src.ml.models.train_model_a

Returns
-------
dict with all results (also returned from train_model_a() for programmatic use)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_score

from src.ml.features.build_features import load_transformer_dataset
from src.ml.models.risk_classifier import RiskClassifier
from src.ml.evaluation.evaluator import evaluate_classifier, print_results, save_results
from src.ml.ml_constants import (
    MODEL_A_CLASS_NAMES,
    MODEL_A_TEST_SIZE,
    MODEL_A_VAL_SIZE,
    MODEL_A_RANDOM_STATE,
    MODEL_A_PATH,
    MODEL_A_META_PATH,
    ARTIFACTS_DIR,
)

log = logging.getLogger(__name__)


def train_model_a(verbose: bool = True) -> dict:
    """
    Full training pipeline for Model A (Transformer Risk Classifier).

    Returns a dict containing all training and evaluation results.
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        )

    print("\n" + "=" * 62)
    print("  PowerGuard AI — Model A: Transformer Risk Classifier")
    print("=" * 62)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X, y, feature_names = load_transformer_dataset()
    print(f"\n  Dataset : {X.shape[0]} samples x {X.shape[1]} features")
    print(f"  Classes : {dict(zip(MODEL_A_CLASS_NAMES, np.bincount(y)))}")
    print(f"  Imbalance: {np.bincount(y) / len(y) * 100}")

    # ── 2. Stratified train / val / test split ────────────────────────────────
    # First cut: 70% train, 30% temp
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(MODEL_A_TEST_SIZE + MODEL_A_VAL_SIZE),
        random_state=MODEL_A_RANDOM_STATE,
    )
    tr_idx, tv_idx = next(sss1.split(X, y))
    X_train, X_tv = X.iloc[tr_idx], X.iloc[tv_idx]
    y_train, y_tv = y.iloc[tr_idx], y.iloc[tv_idx]

    # Second cut: split the 30% evenly into val + test
    val_fraction = MODEL_A_VAL_SIZE / (MODEL_A_TEST_SIZE + MODEL_A_VAL_SIZE)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=(1 - val_fraction),
        random_state=MODEL_A_RANDOM_STATE,
    )
    val_idx, test_idx = next(sss2.split(X_tv, y_tv))
    X_val,  X_test  = X_tv.iloc[val_idx],  X_tv.iloc[test_idx]
    y_val,  y_test  = y_tv.iloc[val_idx],  y_tv.iloc[test_idx]

    print(f"\n  Train : {len(X_train)} samples  {np.bincount(y_train)}")
    print(f"  Val   : {len(X_val)} samples  {np.bincount(y_val)}")
    print(f"  Test  : {len(X_test)} samples  {np.bincount(y_test)}")

    # ── 3. 5-fold CV on training set ─────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=MODEL_A_RANDOM_STATE)

    print("\n  Running 5-fold CV on training set ...")

    lr_clf = RiskClassifier(model_type="lr")
    rf_clf = RiskClassifier(model_type="rf")

    # Build unfitted pipelines for CV
    lr_pipe = lr_clf._build_pipeline()
    rf_pipe = rf_clf._build_pipeline()

    lr_cv_f1  = cross_val_score(lr_pipe, X_train, y_train, cv=cv, scoring="f1_weighted")
    rf_cv_f1  = cross_val_score(rf_pipe, X_train, y_train, cv=cv, scoring="f1_weighted")
    lr_cv_acc = cross_val_score(lr_pipe, X_train, y_train, cv=cv, scoring="accuracy")
    rf_cv_acc = cross_val_score(rf_pipe, X_train, y_train, cv=cv, scoring="accuracy")

    cv_results = {
        "lr_cv_f1_weighted_mean":  float(lr_cv_f1.mean()),
        "lr_cv_f1_weighted_std":   float(lr_cv_f1.std()),
        "rf_cv_f1_weighted_mean":  float(rf_cv_f1.mean()),
        "rf_cv_f1_weighted_std":   float(rf_cv_f1.std()),
        "lr_cv_accuracy_mean":     float(lr_cv_acc.mean()),
        "rf_cv_accuracy_mean":     float(rf_cv_acc.mean()),
    }
    print(f"\n  LR  5-fold CV  F1-weighted: {lr_cv_f1.mean():.4f} +/- {lr_cv_f1.std():.4f}")
    print(f"  RF  5-fold CV  F1-weighted: {rf_cv_f1.mean():.4f} +/- {rf_cv_f1.std():.4f}")

    # ── 4. Fit both models on full training set ───────────────────────────────
    print("\n  Fitting Logistic Regression baseline ...")
    lr_clf.fit(X_train, y_train, feature_names=feature_names)

    print("  Fitting Random Forest main model ...")
    rf_clf.fit(X_train, y_train, feature_names=feature_names)

    # ── 5. Validate on val set ────────────────────────────────────────────────
    print("\n  Evaluating on validation set ...")
    lr_val_results = evaluate_classifier(lr_clf, X_val, y_val, MODEL_A_CLASS_NAMES,
                                          model_name="LR_Baseline_val")
    rf_val_results = evaluate_classifier(rf_clf, X_val, y_val, MODEL_A_CLASS_NAMES,
                                          model_name="RF_Main_val")
    print_results(lr_val_results)
    print_results(rf_val_results)

    # ── 6. Final test evaluation (RF is the selected model) ───────────────────
    print("\n  Evaluating on HELD-OUT TEST SET ...")
    rf_test_results = evaluate_classifier(rf_clf, X_test, y_test, MODEL_A_CLASS_NAMES,
                                           model_name="RF_Main_test_FINAL")
    lr_test_results = evaluate_classifier(lr_clf, X_test, y_test, MODEL_A_CLASS_NAMES,
                                           model_name="LR_Baseline_test_FINAL")
    print_results(rf_test_results)
    print_results(lr_test_results)

    # ── 7. Feature importances ────────────────────────────────────────────────
    fi = rf_clf.feature_importances()
    print("  Top 10 feature importances (RF):")
    for feat, imp in fi.head(10).items():
        print(f"    {feat:<35} {imp:.4f}")

    # ── 8. Save model artifacts ───────────────────────────────────────────────
    rf_clf.save(MODEL_A_PATH)
    rf_clf.save_metadata(
        MODEL_A_META_PATH,
        extra={
            "cv_f1_weighted": f"{rf_cv_f1.mean():.4f} +/- {rf_cv_f1.std():.4f}",
            "test_accuracy":  rf_test_results["accuracy"],
            "test_roc_auc":   rf_test_results["roc_auc_macro"],
            "test_f1_macro":  rf_test_results["classification_report"]["macro avg"]["f1-score"],
        },
    )

    # Save evaluation results JSON
    all_results = {
        "model_a_transformer_risk": {
            "cv":          cv_results,
            "val_rf":      rf_val_results,
            "val_lr":      lr_val_results,
            "test_rf":     rf_test_results,
            "test_lr":     lr_test_results,
            "feature_importances": fi.to_dict(),
            "split_info": {
                "n_train": len(X_train),
                "n_val":   len(X_val),
                "n_test":  len(X_test),
                "train_class_dist": np.bincount(y_train).tolist(),
                "val_class_dist":   np.bincount(y_val).tolist(),
                "test_class_dist":  np.bincount(y_test).tolist(),
            },
        }
    }
    results_path = ARTIFACTS_DIR / "model_a_results.json"
    save_results(all_results, results_path)

    print(f"\n  Model saved    : {MODEL_A_PATH}")
    print(f"  Metadata saved : {MODEL_A_META_PATH}")
    print(f"  Results saved  : {results_path}")
    print("\n  Selected model: Random Forest")
    print(f"  Test F1 (macro)   : {rf_test_results['classification_report']['macro avg']['f1-score']:.4f}")
    print(f"  Test F1 (weighted): {rf_test_results['classification_report']['weighted avg']['f1-score']:.4f}")
    print(f"  Test ROC-AUC      : {rf_test_results['roc_auc_macro']}")

    return all_results


if __name__ == "__main__":
    train_model_a(verbose=True)
    sys.exit(0)

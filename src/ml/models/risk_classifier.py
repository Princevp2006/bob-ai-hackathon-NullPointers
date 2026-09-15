"""
PowerGuard AI — Risk Classifier (Model A)
==========================================
Encapsulates the full scikit-learn pipeline for transformer risk prediction:

    StandardScaler → RandomForestClassifier

The scaler is included in the pipeline so that the model artifact stores the
fitted scaler alongside the classifier — no separate scaler file needed.

Both a Random Forest (main model) and a Logistic Regression (baseline) are
provided. The training script in train_model_a.py fits both and compares them.

Usage
-----
    from src.ml.models.risk_classifier import RiskClassifier
    clf = RiskClassifier(model_type='rf')
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    clf.save('artifacts/transformer_risk_model.joblib')
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.ml_constants import (
    MODEL_A_RF_PARAMS,
    MODEL_A_LR_PARAMS,
    MODEL_A_CLASS_NAMES,
)

log = logging.getLogger(__name__)


class RiskClassifier:
    """
    Wrapper around a scikit-learn Pipeline for transformer risk classification.

    Parameters
    ----------
    model_type : {'rf', 'lr'}
        'rf' = RandomForestClassifier (main model)
        'lr' = LogisticRegression (baseline)
    """

    CLASS_NAMES = MODEL_A_CLASS_NAMES  # ["LOW", "MEDIUM", "HIGH"]

    def __init__(self, model_type: Literal["rf", "lr"] = "rf") -> None:
        self.model_type = model_type
        self._pipeline: Pipeline | None = None
        self._feature_names: list[str] = []
        self._is_fitted: bool = False

    def _build_pipeline(self) -> Pipeline:
        """Construct a fresh unfitted pipeline."""
        if self.model_type == "rf":
            estimator = RandomForestClassifier(**MODEL_A_RF_PARAMS)
        elif self.model_type == "lr":
            estimator = LogisticRegression(**MODEL_A_LR_PARAMS)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type!r}. Use 'rf' or 'lr'.")

        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    estimator),
        ])

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series  | np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "RiskClassifier":
        """
        Fit the pipeline on training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
        y_train : array-like of shape (n_samples,) — integer labels 0/1/2
        feature_names : list of str, optional
        """
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X_train, y_train)
        self._feature_names = (
            list(feature_names)
            if feature_names is not None
            else [f"feature_{i}" for i in range(
                X_train.shape[1] if hasattr(X_train, "shape") else len(X_train[0])
            )]
        )
        self._is_fitted = True
        log.info("RiskClassifier(%s) fitted on %d samples.", self.model_type,
                 len(y_train) if hasattr(y_train, "__len__") else "?")
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return integer class predictions."""
        self._assert_fitted()
        return self._pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Return probability matrix of shape (n_samples, 3).
        Columns: [P(LOW), P(MEDIUM), P(HIGH)]
        """
        self._assert_fitted()
        return self._pipeline.predict_proba(X)

    def feature_importances(self) -> pd.Series:
        """
        Return feature importances as a named Series (RF only).
        Raises AttributeError for Logistic Regression.
        """
        self._assert_fitted()
        clf = self._pipeline.named_steps["clf"]
        if not hasattr(clf, "feature_importances_"):
            raise AttributeError(
                f"Model type '{self.model_type}' does not support feature importances. "
                "Use model_type='rf'."
            )
        return pd.Series(
            clf.feature_importances_,
            index=self._feature_names,
        ).sort_values(ascending=False)

    def coef_magnitudes(self) -> pd.Series:
        """
        Return mean absolute coefficient magnitudes (Logistic Regression only).
        """
        self._assert_fitted()
        clf = self._pipeline.named_steps["clf"]
        if not hasattr(clf, "coef_"):
            raise AttributeError("coef_ only available for LogisticRegression.")
        return pd.Series(
            np.abs(clf.coef_).mean(axis=0),
            index=self._feature_names,
        ).sort_values(ascending=False)

    def save(self, path: Path | str) -> None:
        """Serialise the fitted pipeline with joblib."""
        self._assert_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)
        log.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: Path | str, model_type: str = "rf") -> "RiskClassifier":
        """Load a previously saved pipeline."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        obj = cls(model_type=model_type)
        obj._pipeline   = joblib.load(path)
        obj._is_fitted  = True
        log.info("Model loaded from %s", path)
        return obj

    def save_metadata(self, path: Path | str, extra: dict | None = None) -> None:
        """Save a JSON metadata file alongside the model artifact."""
        path = Path(path)
        meta = {
            "model_type":    self.model_type,
            "feature_names": self._feature_names,
            "class_names":   self.CLASS_NAMES,
            "n_features":    len(self._feature_names),
        }
        if extra:
            meta.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)
        log.info("Metadata saved to %s", path)

    def _assert_fitted(self) -> None:
        if not self._is_fitted or self._pipeline is None:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

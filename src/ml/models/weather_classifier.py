"""
PowerGuard AI — Weather Outage Classifier (Model B)
=====================================================
Exactly mirrors the RiskClassifier API but serves the weather-severity task.
Shared logic lives in the parent; this class only overrides defaults.

Usage
-----
    from src.ml.models.weather_classifier import WeatherClassifier
    clf = WeatherClassifier(model_type='rf')
    clf.fit(X_train, y_train, feature_names=feat_names)
    proba = clf.predict_proba(X_test)
    clf.save('artifacts/weather_outage_model.joblib')
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
    MODEL_B_RF_PARAMS,
    MODEL_B_LR_PARAMS,
    MODEL_B_CLASS_NAMES,
)

log = logging.getLogger(__name__)


class WeatherClassifier:
    """
    Wrapper around a scikit-learn Pipeline for weather-driven outage severity
    classification.

    Parameters
    ----------
    model_type : {'rf', 'lr'}
    """

    CLASS_NAMES = MODEL_B_CLASS_NAMES  # ["LOW", "MEDIUM", "HIGH"]

    def __init__(self, model_type: Literal["rf", "lr"] = "rf") -> None:
        self.model_type = model_type
        self._pipeline: Pipeline | None = None
        self._feature_names: list[str] = []
        self._is_fitted: bool = False

    def _build_pipeline(self) -> Pipeline:
        if self.model_type == "rf":
            estimator = RandomForestClassifier(**MODEL_B_RF_PARAMS)
        elif self.model_type == "lr":
            estimator = LogisticRegression(**MODEL_B_LR_PARAMS)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type!r}. Use 'rf' or 'lr'.")
        return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])

    def fit(self, X_train, y_train, feature_names=None) -> "WeatherClassifier":
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X_train, y_train)
        self._feature_names = list(feature_names) if feature_names is not None else []
        self._is_fitted = True
        log.info("WeatherClassifier(%s) fitted on %d samples.", self.model_type, len(y_train))
        return self

    def predict(self, X) -> np.ndarray:
        self._assert_fitted()
        return self._pipeline.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        self._assert_fitted()
        return self._pipeline.predict_proba(X)

    def feature_importances(self) -> pd.Series:
        self._assert_fitted()
        clf = self._pipeline.named_steps["clf"]
        if not hasattr(clf, "feature_importances_"):
            raise AttributeError("feature_importances_ only available for RF model_type.")
        return pd.Series(clf.feature_importances_, index=self._feature_names).sort_values(ascending=False)

    def save(self, path: Path | str) -> None:
        self._assert_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)
        log.info("WeatherClassifier saved to %s", path)

    @classmethod
    def load(cls, path: Path | str, model_type: str = "rf") -> "WeatherClassifier":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        obj = cls(model_type=model_type)
        obj._pipeline  = joblib.load(path)
        obj._is_fitted = True
        log.info("WeatherClassifier loaded from %s", path)
        return obj

    def save_metadata(self, path: Path | str, extra: dict | None = None) -> None:
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

    def _assert_fitted(self) -> None:
        if not self._is_fitted or self._pipeline is None:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

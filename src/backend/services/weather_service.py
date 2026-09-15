"""
PowerGuard AI — Weather Service
=================================
Handles weather data input and regional severity queries.

Provides:
  - WeatherService.query(state, weather_dict) → severity prediction dict
  - WeatherService.save_record(state, region, weather_dict) → WeatherRecord
  - WeatherService.get_latest(state) → most recent WeatherRecord for a state
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.backend.services.prediction_service import PredictionService

log = logging.getLogger(__name__)


class WeatherService:
    """
    Wraps Model B (WeatherClassifier) for standalone weather-severity queries.

    A PredictionService instance is accepted at construction so that
    the same loaded model is reused across both services.
    """

    def __init__(self, prediction_service: PredictionService) -> None:
        self._svc = prediction_service

    def query(self, state: str, weather_dict: dict) -> dict[str, Any]:
        """
        Run Model B for a given state and weather snapshot.

        Parameters
        ----------
        state       : 2-letter US state abbreviation
        weather_dict: dict with keys tmpf, relh, sknt, p01i

        Returns a dict with severity_class and probabilities.
        """
        import numpy as np
        from src.ml.ml_constants import MODEL_B_CLASS_NAMES

        X_b = PredictionService._build_weather_features(state, None, weather_dict=weather_dict)
        proba_b = self._svc._model_b.predict_proba(X_b)[0]
        class_b = MODEL_B_CLASS_NAMES[int(proba_b.argmax())]

        return {
            "state":             state.upper(),
            "weather_input":     weather_dict,
            "severity_class":    class_b,
            "proba_low":         round(float(proba_b[0]), 4),
            "proba_medium":      round(float(proba_b[1]), 4),
            "proba_high":        round(float(proba_b[2]), 4),
            "predicted_at":      datetime.now(timezone.utc).isoformat(),
        }

    def save_record(self, state: str, region: str, weather_dict: dict) -> dict:
        """
        Persist a WeatherRecord to the database and return its dict.

        Parameters
        ----------
        state       : 2-letter US state abbreviation
        region      : free-text region name
        weather_dict: dict with keys tmpf, relh, sknt, p01i
        """
        from src.backend.db.models import db, WeatherRecord

        record = WeatherRecord(
            state       = state.upper(),
            region      = region,
            recorded_at = datetime.now(timezone.utc),
            tmpf        = float(weather_dict.get("tmpf", 84.2)),
            relh        = float(weather_dict.get("relh", 55.98)),
            sknt        = float(weather_dict.get("sknt", 5.0)),
            p01i        = float(weather_dict.get("p01i", 0.0)),
        )
        db.session.add(record)
        db.session.commit()
        log.info("WeatherRecord saved: state=%s  region=%s", state, region)
        return record.to_dict()

    def get_latest(self, state: str) -> dict | None:
        """Return the most recent WeatherRecord for a state, or None."""
        from src.backend.db.models import WeatherRecord

        record = (
            WeatherRecord.query
            .filter_by(state=state.upper())
            .order_by(WeatherRecord.recorded_at.desc())
            .first()
        )
        return record.to_dict() if record else None

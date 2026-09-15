"""
PowerGuard AI — Prediction Service
====================================
Loads the two trained joblib models and runs the full inference pipeline
for one asset (or all assets).

Inference steps
---------------
1. Fetch the most recent SensorReading for the asset from the DB
2. Build the 20-feature vector for Model A (RiskClassifier)
3. Run Model A → (risk_class, probabilities)
4. Fetch or synthesise a WeatherRecord for the asset's region
5. Build the 9-feature vector for Model B (WeatherClassifier)
6. Run Model B → (severity_class, probabilities)
7. Compute fusion score:
       combined = 0.60 × P_A(HIGH) + 0.30 × P_B(HIGH) + 0.10 × criticality
8. Bucket the combined score → risk_level (LOW/MEDIUM/HIGH/CRITICAL)
9. Persist a RiskScore row and return it as a dict
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.ml.ml_constants import (
    MODEL_A_PATH, MODEL_B_PATH,
    MODEL_A_FEATURE_COLS, MODEL_A_CLASS_NAMES,
    MODEL_B_CLASS_NAMES,
    FUSION_WEIGHT_A, FUSION_WEIGHT_B, FUSION_WEIGHT_C,
)

log = logging.getLogger(__name__)

# ── Fusion bucketing thresholds ───────────────────────────────────────────────
_CRITICAL_THRESHOLD = 0.65
_HIGH_THRESHOLD     = 0.40
_MEDIUM_THRESHOLD   = 0.20

# ── State → integer encoding (matches the LabelEncoder fitted on training data)
# Must be consistent with what the WeatherClassifier was trained on.
# We rebuild a mapping from a sorted list of the 50 US state abbreviations.
_US_STATES_SORTED = sorted([
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY",
])
_STATE_ENC = {s: i for i, s in enumerate(_US_STATES_SORTED)}

_SEASON_ENC = {"winter": 0, "spring": 1, "summer": 2, "fall": 3}


class PredictionService:
    """
    Wraps the trained ML models and handles the full inference + persistence
    pipeline.

    Thread-safety: models are loaded once at construction and are read-only
    during predict calls.  SQLAlchemy session management is per-call.
    """

    def __init__(self, model_a_path: Path = MODEL_A_PATH,
                 model_b_path: Path = MODEL_B_PATH) -> None:
        self._model_a = self._load_model(model_a_path, "Model A")
        self._model_b = self._load_model(model_b_path, "Model B")

    # ── Public interface ──────────────────────────────────────────────────────

    def predict_asset(self, asset_id: int) -> dict[str, Any]:
        """
        Run the full inference pipeline for a single asset.

        Returns a dict representation of the saved RiskScore row.
        Raises ValueError if the asset has no sensor readings.
        """
        from src.backend.db.models import db, Asset, SensorReading, WeatherRecord, RiskScore

        asset = Asset.query.get_or_404(asset_id)

        # ── Step 1: most recent sensor reading ────────────────────────────────
        reading = (SensorReading.query
                   .filter_by(asset_id=asset_id)
                   .order_by(SensorReading.recorded_at.desc())
                   .first())
        if reading is None:
            raise ValueError(f"Asset {asset_id} has no sensor readings.")

        # ── Step 2 & 3: Model A ───────────────────────────────────────────────
        feat_dict = reading.to_feature_dict()
        X_a = pd.DataFrame([feat_dict])[MODEL_A_FEATURE_COLS]
        proba_a = self._model_a.predict_proba(X_a)[0]   # shape (3,) LOW/MED/HIGH
        class_a = MODEL_A_CLASS_NAMES[int(np.argmax(proba_a))]

        # ── Step 4 & 5 & 6: Model B ──────────────────────────────────────────
        weather = (WeatherRecord.query
                   .filter_by(state=asset.state)
                   .order_by(WeatherRecord.recorded_at.desc())
                   .first())
        weather_record_id = weather.id if weather else None
        X_b = self._build_weather_features(asset.state, weather)
        proba_b = self._model_b.predict_proba(X_b)[0]   # shape (3,) LOW/MED/HIGH
        class_b = MODEL_B_CLASS_NAMES[int(np.argmax(proba_b))]

        # ── Step 7: Fusion score ──────────────────────────────────────────────
        combined = (
            FUSION_WEIGHT_A * float(proba_a[2]) +   # P_A(HIGH)
            FUSION_WEIGHT_B * float(proba_b[2]) +   # P_B(HIGH)
            FUSION_WEIGHT_C * float(asset.criticality_score)
        )
        combined = float(np.clip(combined, 0.0, 1.0))

        # ── Step 8: Bucket ────────────────────────────────────────────────────
        if combined >= _CRITICAL_THRESHOLD:
            risk_level = "CRITICAL"
        elif combined >= _HIGH_THRESHOLD:
            risk_level = "HIGH"
        elif combined >= _MEDIUM_THRESHOLD:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # ── Step 9: Persist ───────────────────────────────────────────────────
        score = RiskScore(
            asset_id          = asset_id,
            sensor_reading_id = reading.id,
            weather_record_id = weather_record_id,
            predicted_at      = datetime.now(timezone.utc),
            model_a_risk_class = class_a,
            model_a_proba_low  = float(proba_a[0]),
            model_a_proba_med  = float(proba_a[1]),
            model_a_proba_high = float(proba_a[2]),
            model_b_risk_class = class_b,
            model_b_proba_low  = float(proba_b[0]),
            model_b_proba_med  = float(proba_b[1]),
            model_b_proba_high = float(proba_b[2]),
            combined_risk_score = combined,
            risk_level          = risk_level,
        )
        db.session.add(score)
        db.session.commit()

        log.info("Predicted asset_id=%d  risk=%s  combined=%.4f", asset_id, risk_level, combined)
        return {**score.to_dict(), "asset": asset.to_dict()}

    def predict_all(self) -> list[dict[str, Any]]:
        """Run predict_asset() for every asset in the database."""
        from src.backend.db.models import Asset
        results = []
        for asset in Asset.query.all():
            try:
                results.append(self.predict_asset(asset.id))
            except ValueError as exc:
                log.warning("Skipping asset %d: %s", asset.id, exc)
        return results

    def predict_from_payload(self, payload: dict, state: str = "TX",
                              weather: dict | None = None) -> dict[str, Any]:
        """
        Run inference from a raw feature dict without touching the database.
        Useful for ad-hoc API calls where the caller supplies the sensor data.

        Parameters
        ----------
        payload : dict
            Must contain the 14 raw DGA keys.  IEC ratios are computed here.
        state   : str  Two-letter US state for weather context.
        weather : dict | None  Optional weather dict (tmpf/relh/sknt/p01i).

        Returns a dict with all prediction outputs (no DB write).
        """
        # Compute IEC ratios from raw gases
        from src.backend.db.database import _compute_iec_ratios
        payload = dict(payload)
        ratios = _compute_iec_ratios(payload)
        payload.update(ratios)

        X_a = pd.DataFrame([payload])[MODEL_A_FEATURE_COLS]
        proba_a = self._model_a.predict_proba(X_a)[0]
        class_a = MODEL_A_CLASS_NAMES[int(np.argmax(proba_a))]

        X_b = self._build_weather_features(state, None, weather_dict=weather)
        proba_b = self._model_b.predict_proba(X_b)[0]
        class_b = MODEL_B_CLASS_NAMES[int(np.argmax(proba_b))]

        # Use 0.5 criticality when not asset-backed
        criticality = float((weather or {}).get("criticality", 0.5))
        combined = float(np.clip(
            FUSION_WEIGHT_A * float(proba_a[2]) +
            FUSION_WEIGHT_B * float(proba_b[2]) +
            FUSION_WEIGHT_C * criticality,
            0.0, 1.0,
        ))
        if combined >= _CRITICAL_THRESHOLD:
            risk_level = "CRITICAL"
        elif combined >= _HIGH_THRESHOLD:
            risk_level = "HIGH"
        elif combined >= _MEDIUM_THRESHOLD:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "model_a_risk_class":  class_a,
            "model_a_proba_low":   round(float(proba_a[0]), 4),
            "model_a_proba_med":   round(float(proba_a[1]), 4),
            "model_a_proba_high":  round(float(proba_a[2]), 4),
            "model_b_risk_class":  class_b,
            "model_b_proba_low":   round(float(proba_b[0]), 4),
            "model_b_proba_med":   round(float(proba_b[1]), 4),
            "model_b_proba_high":  round(float(proba_b[2]), 4),
            "combined_risk_score": round(combined, 4),
            "risk_level":          risk_level,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _load_model(path: Path, name: str):
        """Load a joblib pipeline from disk.  Raises FileNotFoundError if missing."""
        if not path.exists():
            raise FileNotFoundError(
                f"{name} artifact not found at {path}. "
                "Run the training script first:\n"
                "  python -m src.ml.models.train_model_a\n"
                "  python -m src.ml.models.train_model_b"
            )
        model = joblib.load(path)
        log.info("%s loaded from %s", name, path)
        return model

    @staticmethod
    def _build_weather_features(state: str, weather_record,
                                 weather_dict: dict | None = None) -> pd.DataFrame:
        """
        Build the 9-feature DataFrame expected by WeatherClassifier.

        Priority: weather_dict > weather_record > dataset medians.
        """
        now = datetime.now(timezone.utc)
        hour  = now.hour
        month = now.month
        year  = now.year

        # Season encoding
        if month in (12, 1, 2):
            season = "winter"
        elif month in (3, 4, 5):
            season = "spring"
        elif month in (6, 7, 8):
            season = "summer"
        else:
            season = "fall"

        # Default to dataset medians (imputed values from clean_weather_outage.py)
        tmpf, relh, sknt, p01i = 84.2, 55.98, 5.0, 0.0

        if weather_dict:
            tmpf = float(weather_dict.get("tmpf", tmpf))
            relh = float(weather_dict.get("relh", relh))
            sknt = float(weather_dict.get("sknt", sknt))
            p01i = float(weather_dict.get("p01i", p01i))
        elif weather_record is not None:
            tmpf = float(weather_record.tmpf or tmpf)
            relh = float(weather_record.relh or relh)
            sknt = float(weather_record.sknt or sknt)
            p01i = float(weather_record.p01i or p01i)

        state_enc  = _STATE_ENC.get(state.upper(), 0)
        season_enc = _SEASON_ENC.get(season, 0)

        return pd.DataFrame([{
            "tmpf":        tmpf,
            "relh":        relh,
            "sknt":        sknt,
            "p01i":        p01i,
            "hour_of_day": float(hour),
            "month":       float(month),
            "year":        float(year),
            "season_enc":  float(season_enc),
            "state_enc":   float(state_enc),
        }])

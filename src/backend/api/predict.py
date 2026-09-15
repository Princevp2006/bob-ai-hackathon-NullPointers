"""
PowerGuard AI — Prediction API Blueprint
==========================================
Endpoints:

    POST /api/predict/<asset_id>   — Run prediction for one asset, store result
    POST /api/predict/all          — Run prediction for all assets
    POST /api/predict/payload      — Ad-hoc prediction from raw DGA JSON payload
    GET  /api/predict/<asset_id>/latest — Fetch most recent stored risk score
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request, abort, current_app

predict_bp = Blueprint("predict", __name__)


def _get_svc():
    """Lazy-load PredictionService once per app context (stored in app extensions)."""
    from src.backend.services.prediction_service import PredictionService
    if "prediction_service" not in current_app.extensions:
        current_app.extensions["prediction_service"] = PredictionService()
    return current_app.extensions["prediction_service"]


@predict_bp.route("/<int:asset_id>", methods=["POST"])
def predict_asset(asset_id: int):
    """
    Trigger a fresh ML prediction for one asset.

    Fetches the most recent SensorReading, runs both models,
    fuses the scores, persists a RiskScore row, and returns it.
    """
    svc = _get_svc()
    try:
        result = svc.predict_asset(asset_id)
    except ValueError as exc:
        abort(422, description=str(exc))
    return jsonify(result), 200


@predict_bp.route("/all", methods=["POST"])
def predict_all():
    """
    Trigger predictions for every asset in the database.

    Returns a summary with counts per risk level plus the full list.
    """
    svc = _get_svc()
    results = svc.predict_all()

    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in results:
        level = r.get("risk_level", "LOW")
        counts[level] = counts.get(level, 0) + 1

    return jsonify({
        "predictions":  results,
        "count":        len(results),
        "risk_summary": counts,
    }), 200


@predict_bp.route("/payload", methods=["POST"])
def predict_payload():
    """
    Run an ad-hoc prediction from a raw DGA JSON payload.
    Does NOT require an Asset in the database; does NOT write to the DB.

    Required JSON body keys (14 raw DGA features):
        hydrogen_ppm, oxygen_ppm, nitrogen_ppm, methane_ppm,
        co_ppm, co2_ppm, ethylene_ppm, ethane_ppm, acetylene_ppm,
        dbds_mg_kg, power_factor_pct, interfacial_tension_mNm,
        dielectric_kv, water_content_ppm

    Optional:
        state   : 2-letter US state (default "TX")
        tmpf    : temperature °F
        relh    : relative humidity %
        sknt    : wind speed knots
        p01i    : 1-hour precipitation inches
    """
    data = request.get_json(force=True)
    if not data:
        abort(400, description="Request body must be JSON.")

    raw_dga_keys = [
        "hydrogen_ppm", "oxygen_ppm", "nitrogen_ppm", "methane_ppm",
        "co_ppm", "co2_ppm", "ethylene_ppm", "ethane_ppm", "acetylene_ppm",
        "dbds_mg_kg", "power_factor_pct", "interfacial_tension_mNm",
        "dielectric_kv", "water_content_ppm",
    ]
    missing = [k for k in raw_dga_keys if k not in data]
    if missing:
        abort(400, description=f"Missing required DGA feature keys: {missing}")

    state = data.pop("state", "TX")
    weather = {k: data.pop(k) for k in ("tmpf", "relh", "sknt", "p01i") if k in data}

    svc = _get_svc()
    result = svc.predict_from_payload(data, state=state, weather=weather or None)
    return jsonify(result), 200


@predict_bp.route("/<int:asset_id>/latest", methods=["GET"])
def get_latest_prediction(asset_id: int):
    """Return the most recently stored RiskScore for an asset."""
    from src.backend.db.models import Asset, RiskScore

    Asset.query.get_or_404(asset_id)
    score = (
        RiskScore.query
        .filter_by(asset_id=asset_id)
        .order_by(RiskScore.predicted_at.desc())
        .first()
    )
    if score is None:
        return jsonify({
            "asset_id": asset_id,
            "message": "No predictions yet. POST /api/predict/<id> to trigger one."
        }), 404

    return jsonify(score.to_dict()), 200

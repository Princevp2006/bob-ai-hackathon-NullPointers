"""
PowerGuard AI — Weather API Blueprint
=======================================
Endpoints:

    POST /api/weather/query        — Predict outage severity for a state + weather snapshot
    POST /api/weather/record       — Save a weather observation to the DB
    GET  /api/weather/latest/<state> — Fetch most recent weather record for a state
    GET  /api/weather/maintenance  — Return the current maintenance plan (with re-gen option)
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request, abort, current_app

weather_bp = Blueprint("weather", __name__)


def _get_svc():
    """Lazy-load services once per app context."""
    from src.backend.services.prediction_service import PredictionService
    from src.backend.services.weather_service import WeatherService
    from src.backend.services.maintenance_service import MaintenanceService

    ext = current_app.extensions

    if "prediction_service" not in ext:
        ext["prediction_service"] = PredictionService()
    if "weather_service" not in ext:
        ext["weather_service"] = WeatherService(ext["prediction_service"])
    if "maintenance_service" not in ext:
        ext["maintenance_service"] = MaintenanceService()

    return ext["weather_service"], ext["maintenance_service"]


@weather_bp.route("/query", methods=["POST"])
def weather_query():
    """
    Predict outage severity for a weather snapshot.

    Required JSON body:
        state  : 2-letter US state
        tmpf   : temperature °F
        relh   : relative humidity %
        sknt   : wind speed knots
        p01i   : 1-hour precipitation inches
    """
    data = request.get_json(force=True)
    if not data:
        abort(400, description="Request body must be JSON.")

    required = ["state", "tmpf", "relh", "sknt"]
    missing = [k for k in required if k not in data]
    if missing:
        abort(400, description=f"Missing required fields: {missing}")

    weather_svc, _ = _get_svc()
    state = data.pop("state")
    data.setdefault("p01i", 0.0)
    result = weather_svc.query(state, data)
    return jsonify(result), 200


@weather_bp.route("/record", methods=["POST"])
def save_weather_record():
    """
    Persist a weather observation.

    Required JSON body:
        state  : 2-letter US state
        region : free-text region name
        tmpf   : temperature °F
        relh   : relative humidity %
        sknt   : wind speed knots
        p01i   : 1-hour precipitation inches (optional, default 0)
    """
    data = request.get_json(force=True)
    if not data:
        abort(400, description="Request body must be JSON.")

    required = ["state", "region", "tmpf", "relh", "sknt"]
    missing = [k for k in required if k not in data]
    if missing:
        abort(400, description=f"Missing required fields: {missing}")

    weather_svc, _ = _get_svc()
    state  = data.pop("state")
    region = data.pop("region")
    data.setdefault("p01i", 0.0)
    record = weather_svc.save_record(state, region, data)
    return jsonify(record), 201


@weather_bp.route("/latest/<state>", methods=["GET"])
def get_latest_weather(state: str):
    """Return the most recent WeatherRecord for a state."""
    weather_svc, _ = _get_svc()
    record = weather_svc.get_latest(state)
    if record is None:
        return jsonify({"state": state.upper(), "message": "No weather records found."}), 404
    return jsonify(record), 200


@weather_bp.route("/maintenance", methods=["GET"])
def get_maintenance_plan():
    """
    Return the current pending maintenance plan.

    Query param: regenerate=true  — re-run MaintenanceService.generate_plan()
    """
    _, maintenance_svc = _get_svc()

    regenerate = request.args.get("regenerate", "false").lower() == "true"
    if regenerate:
        plan = maintenance_svc.generate_plan()
    else:
        plan = maintenance_svc.get_current_plan()
        if not plan:
            # Auto-generate if no plan exists yet
            plan = maintenance_svc.generate_plan()

    return jsonify({"plan": plan, "count": len(plan)}), 200

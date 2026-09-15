"""
PowerGuard AI — Assets API Blueprint
======================================
Endpoints:

    GET  /api/assets/           — List all assets with latest risk score
    GET  /api/assets/<id>       — Single asset + sensor history + risk history
    POST /api/assets/           — Register a new asset
    GET  /api/assets/<id>/readings — All sensor readings for an asset
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request, abort, current_app

assets_bp = Blueprint("assets", __name__)


@assets_bp.route("/", methods=["GET"])
def list_assets():
    """
    Return all assets, each enriched with its most recent risk score.

    Query parameters
    ----------------
    region : str   Filter by region name (case-insensitive, partial match)
    state  : str   Filter by 2-letter state code
    level  : str   Filter by latest risk_level (LOW/MEDIUM/HIGH/CRITICAL)
    """
    from src.backend.db.models import Asset, RiskScore

    query = Asset.query

    region = request.args.get("region")
    state  = request.args.get("state")
    level  = request.args.get("level")

    if region:
        query = query.filter(Asset.region.ilike(f"%{region}%"))
    if state:
        query = query.filter(Asset.state == state.upper())

    assets = query.order_by(Asset.name).all()

    result = []
    for asset in assets:
        d = asset.to_dict()
        latest_score = (
            RiskScore.query
            .filter_by(asset_id=asset.id)
            .order_by(RiskScore.predicted_at.desc())
            .first()
        )
        d["latest_risk"] = latest_score.to_dict() if latest_score else None

        # Apply level filter after joining (SQLite doesn't have a direct FK join here)
        if level and (d["latest_risk"] is None or
                      d["latest_risk"]["risk_level"] != level.upper()):
            continue

        result.append(d)

    return jsonify({"assets": result, "count": len(result)}), 200


@assets_bp.route("/<int:asset_id>", methods=["GET"])
def get_asset(asset_id: int):
    """
    Return full details for one asset: metadata + last 10 sensor readings +
    last 10 risk scores.
    """
    from src.backend.db.models import Asset, SensorReading, RiskScore

    asset = Asset.query.get_or_404(asset_id)
    d = asset.to_dict()

    readings = (
        SensorReading.query
        .filter_by(asset_id=asset_id)
        .order_by(SensorReading.recorded_at.desc())
        .limit(10)
        .all()
    )
    d["sensor_readings"] = [r.to_dict() for r in readings]

    scores = (
        RiskScore.query
        .filter_by(asset_id=asset_id)
        .order_by(RiskScore.predicted_at.desc())
        .limit(10)
        .all()
    )
    d["risk_scores"] = [s.to_dict() for s in scores]

    return jsonify(d), 200


@assets_bp.route("/", methods=["POST"])
def create_asset():
    """
    Register a new asset.

    Required JSON fields: name, region, state, latitude, longitude,
                          voltage_kv, install_year, customers_served
    """
    from src.backend.db.models import db, Asset
    from src.backend.db.database import _compute_criticality

    data = request.get_json(force=True)
    if not data:
        abort(400, description="Request body must be JSON.")

    required = ["name", "region", "state", "latitude", "longitude",
                "voltage_kv", "install_year", "customers_served"]
    missing = [f for f in required if f not in data]
    if missing:
        abort(400, description=f"Missing required fields: {missing}")

    # Compute criticality relative to existing assets
    existing = Asset.query.all()
    all_v = [a.voltage_kv       for a in existing] + [float(data["voltage_kv"])]
    all_c = [a.customers_served for a in existing] + [int(data["customers_served"])]
    crit = _compute_criticality(
        float(data["voltage_kv"]), int(data["customers_served"]), all_v, all_c
    )

    asset = Asset(
        name             = data["name"],
        asset_type       = data.get("asset_type", "transformer"),
        region           = data["region"],
        state            = data["state"].upper(),
        latitude         = float(data["latitude"]),
        longitude        = float(data["longitude"]),
        voltage_kv       = float(data["voltage_kv"]),
        install_year     = int(data["install_year"]),
        customers_served = int(data["customers_served"]),
        criticality_score = crit,
    )
    db.session.add(asset)
    db.session.commit()

    return jsonify(asset.to_dict()), 201


@assets_bp.route("/<int:asset_id>/readings", methods=["GET"])
def get_readings(asset_id: int):
    """Return the last N sensor readings for an asset.  Default N=20."""
    from src.backend.db.models import Asset, SensorReading

    Asset.query.get_or_404(asset_id)
    limit = min(int(request.args.get("limit", 20)), 100)

    readings = (
        SensorReading.query
        .filter_by(asset_id=asset_id)
        .order_by(SensorReading.recorded_at.desc())
        .limit(limit)
        .all()
    )
    return jsonify({
        "asset_id": asset_id,
        "readings": [r.to_dict() for r in readings],
        "count":    len(readings),
    }), 200


@assets_bp.route("/<int:asset_id>/readings", methods=["POST"])
def add_reading(asset_id: int):
    """
    Add a new sensor reading for an asset.

    Accepts the 14 raw DGA keys.  IEC ratio features are auto-computed.
    """
    from src.backend.db.models import db, Asset, SensorReading
    from src.backend.db.database import _compute_iec_ratios

    Asset.query.get_or_404(asset_id)
    data = request.get_json(force=True)
    if not data:
        abort(400, description="Request body must be JSON.")

    ratios = _compute_iec_ratios(data)
    reading = SensorReading(asset_id=asset_id, **data, **ratios)
    db.session.add(reading)
    db.session.commit()

    return jsonify(reading.to_dict()), 201

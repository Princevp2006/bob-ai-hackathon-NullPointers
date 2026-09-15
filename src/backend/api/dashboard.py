"""
PowerGuard AI — Dashboard API Blueprint
=========================================
Provides aggregated data for Chart.js widgets on the frontend.

Endpoints:

    GET /api/dashboard/summary        — Risk counts, asset stats, top risks
    GET /api/dashboard/risk-trend     — Risk score history for sparklines
    GET /api/dashboard/map-data       — Asset locations + risk levels for map
    GET /api/dashboard/feature-importance — Model A feature importances
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, jsonify, current_app

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/summary", methods=["GET"])
def summary():
    """
    Return the high-level dashboard summary:

    - total_assets, assets_by_risk_level
    - top 5 critical assets (sorted by combined_risk_score desc)
    - total_customers_at_risk (sum of customers_served for CRITICAL+HIGH assets)
    - pending maintenance orders count
    """
    from src.backend.db.models import Asset, RiskScore, MaintenanceOrder

    assets = Asset.query.all()
    total_assets = len(assets)

    risk_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNSCORED": 0}
    top_risks = []
    total_at_risk_customers = 0

    for asset in assets:
        score = (
            RiskScore.query
            .filter_by(asset_id=asset.id)
            .order_by(RiskScore.predicted_at.desc())
            .first()
        )
        if score is None:
            risk_counts["UNSCORED"] += 1
            continue

        level = score.risk_level
        risk_counts[level] = risk_counts.get(level, 0) + 1

        if level in ("CRITICAL", "HIGH"):
            total_at_risk_customers += asset.customers_served
            top_risks.append({
                "asset_id":           asset.id,
                "asset_name":         asset.name,
                "region":             asset.region,
                "state":              asset.state,
                "voltage_kv":         asset.voltage_kv,
                "customers_served":   asset.customers_served,
                "risk_level":         level,
                "combined_risk_score": round(score.combined_risk_score, 4),
                "model_a_risk_class": score.model_a_risk_class,
                "model_b_risk_class": score.model_b_risk_class,
                "predicted_at":       score.predicted_at.isoformat(),
            })

    top_risks.sort(key=lambda x: x["combined_risk_score"], reverse=True)
    top_risks = top_risks[:5]

    pending_orders = MaintenanceOrder.query.filter_by(status="pending").count()

    return jsonify({
        "total_assets":              total_assets,
        "risk_counts":               risk_counts,
        "top_risks":                 top_risks,
        "total_at_risk_customers":   total_at_risk_customers,
        "pending_maintenance_orders": pending_orders,
    }), 200


@dashboard_bp.route("/risk-trend", methods=["GET"])
def risk_trend():
    """
    Return the last N risk scores for each asset, suitable for sparkline charts.

    Query params: limit (default 6)
    """
    from src.backend.db.models import Asset, RiskScore

    from flask import request as _req
    limit = min(int(_req.args.get("limit", 6)), 20)

    assets = Asset.query.order_by(Asset.name).all()
    trends = []
    for asset in assets:
        scores = (
            RiskScore.query
            .filter_by(asset_id=asset.id)
            .order_by(RiskScore.predicted_at.asc())
            .all()
        )
        trends.append({
            "asset_id":   asset.id,
            "asset_name": asset.name,
            "trend": [
                {
                    "predicted_at":      s.predicted_at.isoformat(),
                    "combined_risk_score": round(s.combined_risk_score, 4),
                    "risk_level":        s.risk_level,
                }
                for s in scores
            ],
        })

    return jsonify({"trends": trends}), 200


@dashboard_bp.route("/map-data", methods=["GET"])
def map_data():
    """
    Return asset coordinates + latest risk level for the geographic map widget.
    """
    from src.backend.db.models import Asset, RiskScore

    assets = Asset.query.all()
    features = []
    for asset in assets:
        score = (
            RiskScore.query
            .filter_by(asset_id=asset.id)
            .order_by(RiskScore.predicted_at.desc())
            .first()
        )
        features.append({
            "asset_id":           asset.id,
            "name":               asset.name,
            "latitude":           asset.latitude,
            "longitude":          asset.longitude,
            "region":             asset.region,
            "state":              asset.state,
            "voltage_kv":         asset.voltage_kv,
            "customers_served":   asset.customers_served,
            "risk_level":         score.risk_level if score else "UNSCORED",
            "combined_risk_score": round(score.combined_risk_score, 4) if score else None,
        })

    return jsonify({"assets": features, "count": len(features)}), 200


@dashboard_bp.route("/feature-importance", methods=["GET"])
def feature_importance():
    """
    Return Model A feature importances from the saved JSON artifact.
    Used to render the feature importance bar chart on the dashboard.
    """
    results_path = current_app.config["MODEL_DIR"] / "model_a_results.json"
    if not results_path.exists():
        return jsonify({"error": "Model artifacts not found. Run training first."}), 503

    with open(results_path) as f:
        data = json.load(f)

    fi = data["model_a_transformer_risk"]["feature_importances"]
    # Sort descending by importance value
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)

    return jsonify({
        "feature_importances": [
            {"feature": k, "importance": round(v, 6)}
            for k, v in sorted_fi
        ]
    }), 200

"""
PowerGuard AI — Maintenance Service
=====================================
Generates a prioritised maintenance and crew pre-positioning plan from the
most recent RiskScore for every asset.

Ranking formula
---------------
    priority_score = combined_risk_score × 0.70
                   + criticality_score   × 0.20
                   + age_factor          × 0.10

    age_factor = min(asset_age_years / 40, 1.0)   # normalised to 40-year horizon

Action type mapping
-------------------
    CRITICAL  → "Emergency Inspection & Crew Pre-positioning"
    HIGH      → "Priority Inspection (48 h)"
    MEDIUM    → "Scheduled Maintenance (next cycle)"
    LOW       → "Routine Monitoring"

Crew sizing
-----------
    CRITICAL  → 4 crew, 8 h estimated
    HIGH      → 3 crew, 6 h estimated
    MEDIUM    → 2 crew, 4 h estimated
    LOW       → 1 crew, 2 h estimated
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

log = logging.getLogger(__name__)

# ── Action / crew tables ──────────────────────────────────────────────────────

_ACTION = {
    "CRITICAL": "Emergency Inspection & Crew Pre-positioning",
    "HIGH":     "Priority Inspection (48 h)",
    "MEDIUM":   "Scheduled Maintenance (next cycle)",
    "LOW":      "Routine Monitoring",
}
_CREW = {
    "CRITICAL": (4, 8.0),
    "HIGH":     (3, 6.0),
    "MEDIUM":   (2, 4.0),
    "LOW":      (1, 2.0),
}
_DAYS_OUT = {
    "CRITICAL": 0,   # immediate
    "HIGH":     2,
    "MEDIUM":   14,
    "LOW":      90,
}


class MaintenanceService:
    """
    Builds the maintenance plan from stored RiskScore rows.

    Usage (inside a Flask app context):
        svc = MaintenanceService()
        plan = svc.generate_plan()   # returns list[dict], sorted by priority
    """

    def generate_plan(self) -> list[dict[str, Any]]:
        """
        Fetch the latest RiskScore per asset, rank by priority_score,
        create / update MaintenanceOrder rows, and return the ranked plan.

        Returns
        -------
        list of dicts, each a MaintenanceOrder.to_dict() enriched with
        asset details and priority_score.  Sorted ascending by priority_rank.
        """
        from src.backend.db.models import db, Asset, RiskScore, MaintenanceOrder
        from datetime import datetime, timezone

        assets = Asset.query.all()
        if not assets:
            return []

        scored: list[tuple[float, Asset, RiskScore]] = []
        for asset in assets:
            latest_score = (
                RiskScore.query
                .filter_by(asset_id=asset.id)
                .order_by(RiskScore.predicted_at.desc())
                .first()
            )
            if latest_score is None:
                continue

            age_years = datetime.now(timezone.utc).year - asset.install_year
            age_factor = min(age_years / 40.0, 1.0)

            priority_score = (
                latest_score.combined_risk_score * 0.70 +
                asset.criticality_score          * 0.20 +
                age_factor                       * 0.10
            )
            scored.append((priority_score, asset, latest_score))

        # Sort descending → highest priority first
        scored.sort(key=lambda t: t[0], reverse=True)

        # Delete stale pending orders, then re-create from scratch
        MaintenanceOrder.query.filter_by(status="pending").delete()
        db.session.flush()

        plan = []
        for rank, (priority_score, asset, risk_score) in enumerate(scored, start=1):
            level  = risk_score.risk_level
            action = _ACTION[level]
            crew_size, est_hours = _CREW[level]
            rec_date = (date.today() + timedelta(days=_DAYS_OUT[level])).isoformat()

            age_years = datetime.now(timezone.utc).year - asset.install_year
            notes = (
                f"Combined risk score: {risk_score.combined_risk_score:.3f}  |  "
                f"Model A: {risk_score.model_a_risk_class} (p={risk_score.model_a_proba_high:.2f})  |  "
                f"Model B: {risk_score.model_b_risk_class} (p={risk_score.model_b_proba_high:.2f})  |  "
                f"Asset age: {age_years} yr  |  "
                f"Criticality: {asset.criticality_score:.3f}"
            )

            order = MaintenanceOrder(
                asset_id         = asset.id,
                risk_score_id    = risk_score.id,
                priority_rank    = rank,
                action_type      = action,
                recommended_date = rec_date,
                crew_size        = crew_size,
                estimated_hours  = est_hours,
                notes            = notes,
                status           = "pending",
            )
            db.session.add(order)
            db.session.flush()   # get order.id for to_dict()

            entry = order.to_dict()
            entry["priority_score"] = round(priority_score, 4)
            entry["risk_level"]     = level
            entry["region"]         = asset.region
            entry["voltage_kv"]     = asset.voltage_kv
            entry["customers_served"] = asset.customers_served
            plan.append(entry)

        db.session.commit()
        log.info("Maintenance plan generated: %d orders.", len(plan))
        return plan

    def get_current_plan(self) -> list[dict[str, Any]]:
        """
        Return the current pending maintenance orders without re-generating.
        Used by the dashboard to fetch the existing plan.
        """
        from src.backend.db.models import MaintenanceOrder, RiskScore, Asset

        orders = (
            MaintenanceOrder.query
            .filter_by(status="pending")
            .order_by(MaintenanceOrder.priority_rank.asc())
            .all()
        )
        results = []
        for order in orders:
            entry = order.to_dict()
            # Enrich with current risk level from linked RiskScore
            if order.risk_score_id:
                score = RiskScore.query.get(order.risk_score_id)
                entry["risk_level"] = score.risk_level if score else "UNKNOWN"
                entry["combined_risk_score"] = score.combined_risk_score if score else None
            if order.asset:
                entry["region"]           = order.asset.region
                entry["voltage_kv"]       = order.asset.voltage_kv
                entry["customers_served"] = order.asset.customers_served
            results.append(entry)
        return results

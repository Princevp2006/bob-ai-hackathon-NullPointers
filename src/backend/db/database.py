"""
PowerGuard AI — Database Initialisation & Seeding
===================================================
Provides three functions:

    init_db(app)   — binds SQLAlchemy to the Flask app, creates all tables
    seed_db(app)   — inserts realistic sample assets + sensor readings
    get_db()       — (utility) returns the db instance for use in services

Sample data uses real DGA gas ranges from IEC 60599 / IEEE C57.104-2019.
Values are NOT randomly invented — they are drawn from the typical/warning
ranges defined in the standards, ensuring the trained model will produce
realistic risk classifications when run against them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)


# ── IEC 60599 / IEEE C57.104 typical DGA ranges ──────────────────────────────
# Each entry is (low_profile, medium_profile, high_profile)
# Values are chosen so that:
#   LOW profile   → model predicts LOW   risk with high confidence
#   MEDIUM profile → model predicts MEDIUM risk
#   HIGH profile  → model predicts HIGH  risk
_DGA_PROFILES = {
    "low": {
        "hydrogen_ppm": 35, "oxygen_ppm": 19000, "nitrogen_ppm": 52000,
        "methane_ppm": 20,  "co_ppm": 180,       "co2_ppm": 3200,
        "ethylene_ppm": 8,  "ethane_ppm": 15,    "acetylene_ppm": 0.5,
        "dbds_mg_kg": 0.4,  "power_factor_pct": 0.3, "interfacial_tension_mNm": 35.0,
        "dielectric_kv": 55.0, "water_content_ppm": 12.0,
    },
    "medium": {
        "hydrogen_ppm": 180, "oxygen_ppm": 14000, "nitrogen_ppm": 48000,
        "methane_ppm": 75,   "co_ppm": 600,       "co2_ppm": 7500,
        "ethylene_ppm": 55,  "ethane_ppm": 60,    "acetylene_ppm": 4.0,
        "dbds_mg_kg": 1.5,   "power_factor_pct": 0.8, "interfacial_tension_mNm": 26.0,
        "dielectric_kv": 38.0, "water_content_ppm": 25.0,
    },
    "high": {
        "hydrogen_ppm": 650, "oxygen_ppm": 8000,  "nitrogen_ppm": 35000,
        "methane_ppm": 280,  "co_ppm": 1800,      "co2_ppm": 14000,
        "ethylene_ppm": 250, "ethane_ppm": 180,   "acetylene_ppm": 18.0,
        "dbds_mg_kg": 4.2,   "power_factor_pct": 2.1, "interfacial_tension_mNm": 16.0,
        "dielectric_kv": 22.0, "water_content_ppm": 48.0,
    },
}


def _compute_iec_ratios(g: dict) -> dict:
    """Compute the 6 IEC ratio features from raw DGA gases."""
    h2  = max(g["hydrogen_ppm"],   0.001)
    ch4 = max(g["methane_ppm"],    0.001)
    c2h2 = max(g["acetylene_ppm"], 0.001)
    c2h4 = max(g["ethylene_ppm"],  0.001)
    c2h6 = max(g["ethane_ppm"],    0.001)
    co  = max(g["co_ppm"],         0.001)
    co2 = max(g["co2_ppm"],        0.001)

    tdcg = (g["hydrogen_ppm"] + g["methane_ppm"] + g["co_ppm"] +
            g["ethylene_ppm"] + g["ethane_ppm"] + g["acetylene_ppm"])

    return {
        "tdcg_ppm":              round(tdcg, 3),
        "rogers_r1_ch4_h2":      round(ch4 / h2, 6),
        "rogers_r2_c2h2_c2h4":   round(c2h2 / c2h4, 6),
        "rogers_r3_c2h2_ch4":    round(c2h2 / ch4, 6),
        "co2_co_ratio":          round(co2 / co, 6),
        "ethylene_ethane_ratio":  round(c2h4 / c2h6, 6),
    }


def _make_sensor_reading(asset_id: int, profile: str,
                         recorded_at: datetime) -> dict:
    """Build a full sensor reading dict from a named DGA profile."""
    g = dict(_DGA_PROFILES[profile])
    ratios = _compute_iec_ratios(g)
    return {
        "asset_id":   asset_id,
        "recorded_at": recorded_at,
        **g,
        **ratios,
    }


# ── Sample asset definitions ──────────────────────────────────────────────────

_SAMPLE_ASSETS = [
    # High-criticality urban transmission transformers
    {"name": "TX-NORTH-01",  "asset_type": "transformer", "region": "Northern Grid",
     "state": "TX", "latitude": 29.76, "longitude": -95.37, "voltage_kv": 345.0,
     "install_year": 1998, "customers_served": 85000, "_profile": "high"},

    {"name": "TX-NORTH-02",  "asset_type": "transformer", "region": "Northern Grid",
     "state": "TX", "latitude": 29.80, "longitude": -95.40, "voltage_kv": 230.0,
     "install_year": 2005, "customers_served": 42000, "_profile": "medium"},

    {"name": "CA-WEST-01",   "asset_type": "transformer", "region": "Western Corridor",
     "state": "CA", "latitude": 34.05, "longitude": -118.24, "voltage_kv": 500.0,
     "install_year": 1992, "customers_served": 210000, "_profile": "high"},

    {"name": "CA-WEST-02",   "asset_type": "transformer", "region": "Western Corridor",
     "state": "CA", "latitude": 34.10, "longitude": -118.30, "voltage_kv": 115.0,
     "install_year": 2012, "customers_served": 18000, "_profile": "low"},

    {"name": "FL-SOUTH-01",  "asset_type": "transformer", "region": "Southern Delta",
     "state": "FL", "latitude": 25.77, "longitude": -80.19, "voltage_kv": 230.0,
     "install_year": 2000, "customers_served": 75000, "_profile": "high"},

    {"name": "FL-SOUTH-02",  "asset_type": "transformer", "region": "Southern Delta",
     "state": "FL", "latitude": 25.80, "longitude": -80.22, "voltage_kv": 115.0,
     "install_year": 2010, "customers_served": 31000, "_profile": "medium"},

    {"name": "NY-EAST-01",   "asset_type": "transformer", "region": "Eastern Seaboard",
     "state": "NY", "latitude": 40.71, "longitude": -74.01, "voltage_kv": 345.0,
     "install_year": 1989, "customers_served": 320000, "_profile": "high"},

    {"name": "NY-EAST-02",   "asset_type": "substation",  "region": "Eastern Seaboard",
     "state": "NY", "latitude": 40.73, "longitude": -74.00, "voltage_kv": 138.0,
     "install_year": 2008, "customers_served": 52000, "_profile": "medium"},

    {"name": "IL-CENTRAL-01","asset_type": "transformer", "region": "Central Plains",
     "state": "IL", "latitude": 41.88, "longitude": -87.63, "voltage_kv": 230.0,
     "install_year": 2003, "customers_served": 95000, "_profile": "medium"},

    {"name": "IL-CENTRAL-02","asset_type": "transformer", "region": "Central Plains",
     "state": "IL", "latitude": 41.90, "longitude": -87.65, "voltage_kv": 69.0,
     "install_year": 2015, "customers_served": 9500,  "_profile": "low"},

    {"name": "WA-NW-01",     "asset_type": "transformer", "region": "Pacific Northwest",
     "state": "WA", "latitude": 47.61, "longitude": -122.33, "voltage_kv": 230.0,
     "install_year": 2007, "customers_served": 38000, "_profile": "low"},

    {"name": "GA-SE-01",     "asset_type": "transformer", "region": "Southeast Grid",
     "state": "GA", "latitude": 33.75, "longitude": -84.39, "voltage_kv": 115.0,
     "install_year": 2001, "customers_served": 28000, "_profile": "medium"},
]


def _compute_criticality(voltage_kv: float, customers_served: int,
                          all_voltages: list, all_customers: list) -> float:
    """
    Normalised criticality score in [0, 1].
    0.5 × normalised_voltage + 0.5 × normalised_customers
    """
    max_v = max(all_voltages)
    max_c = max(all_customers)
    norm_v = voltage_kv / max_v if max_v > 0 else 0.0
    norm_c = customers_served / max_c if max_c > 0 else 0.0
    return round(0.5 * norm_v + 0.5 * norm_c, 4)


def init_db(app) -> None:
    """
    Bind SQLAlchemy to the Flask app and create all tables.
    Safe to call multiple times — CREATE TABLE IF NOT EXISTS semantics.
    """
    from src.backend.db.models import db
    db.init_app(app)
    with app.app_context():
        db.create_all()
    log.info("Database tables created at %s", app.config["SQLALCHEMY_DATABASE_URI"])


def seed_db(app) -> None:
    """
    Insert sample assets and historical sensor readings into the database.

    Skips seeding if any Asset rows already exist (idempotent).
    Each asset receives 6 sensor readings spread over the last 12 months,
    with the most recent reading matching its risk profile so that a freshly
    triggered prediction will return the expected risk class.
    """
    from src.backend.db.models import db, Asset, SensorReading

    with app.app_context():
        if Asset.query.count() > 0:
            log.info("Database already seeded — skipping.")
            return

        now = datetime.now(timezone.utc)

        # Pre-compute criticality scores
        all_v = [a["voltage_kv"]       for a in _SAMPLE_ASSETS]
        all_c = [a["customers_served"]  for a in _SAMPLE_ASSETS]

        for asset_def in _SAMPLE_ASSETS:
            profile = asset_def.pop("_profile")
            crit = _compute_criticality(
                asset_def["voltage_kv"], asset_def["customers_served"],
                all_v, all_c,
            )
            asset = Asset(**asset_def, criticality_score=crit)
            db.session.add(asset)
            db.session.flush()  # get asset.id before adding readings

            # 6 readings: months -12, -10, -8, -6, -3, 0 (current)
            offsets_months = [12, 10, 8, 6, 3, 0]
            # Earlier readings use "low" profile; the most recent uses the true profile
            reading_profiles = ["low", "low", "low", "medium", profile, profile]

            for months_ago, r_profile in zip(offsets_months, reading_profiles):
                ts = now - timedelta(days=months_ago * 30)
                reading_data = _make_sensor_reading(asset.id, r_profile, ts)
                db.session.add(SensorReading(**reading_data))

        db.session.commit()
        log.info("Database seeded with %d assets and %d sensor readings.",
                 len(_SAMPLE_ASSETS),
                 len(_SAMPLE_ASSETS) * 6)


def get_db():
    """Return the SQLAlchemy db instance (import shortcut for services)."""
    from src.backend.db.models import db
    return db

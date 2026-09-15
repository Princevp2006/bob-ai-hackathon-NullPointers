"""
PowerGuard AI — SQLAlchemy Database Models
===========================================
Five tables:

    Asset            — Transformer / substation registry
    SensorReading    — DGA oil-analysis readings per asset
    WeatherRecord    — Regional weather snapshots
    RiskScore        — Stored ML inference results
    MaintenanceOrder — Generated work orders / crew pre-positioning

All timestamps are UTC.  Float scores are stored as REAL (SQLite).
"""

from __future__ import annotations

from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Asset ──────────────────────────────────────────────────────────────────────

class Asset(db.Model):
    """A power transformer or substation in the grid."""

    __tablename__ = "asset"

    id               = db.Column(db.Integer, primary_key=True)
    name             = db.Column(db.String(120), nullable=False, unique=True)
    asset_type       = db.Column(db.String(40),  nullable=False, default="transformer")
    region           = db.Column(db.String(80),  nullable=False)
    state            = db.Column(db.String(2),   nullable=False)   # 2-letter US state
    latitude         = db.Column(db.Float,       nullable=False)
    longitude        = db.Column(db.Float,       nullable=False)
    voltage_kv       = db.Column(db.Float,       nullable=False)
    install_year     = db.Column(db.Integer,     nullable=False)
    customers_served = db.Column(db.Integer,     nullable=False, default=0)
    # 0.0–1.0 composite score: normalised(voltage_kv) * 0.5 + normalised(customers) * 0.5
    criticality_score = db.Column(db.Float,      nullable=False, default=0.5)
    created_at       = db.Column(db.DateTime,    default=_utcnow)

    # Relationships
    sensor_readings  = db.relationship("SensorReading",  back_populates="asset",
                                       cascade="all, delete-orphan", lazy="dynamic")
    risk_scores      = db.relationship("RiskScore",       back_populates="asset",
                                       cascade="all, delete-orphan", lazy="dynamic")
    maintenance_orders = db.relationship("MaintenanceOrder", back_populates="asset",
                                         cascade="all, delete-orphan", lazy="dynamic")

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "name":             self.name,
            "asset_type":       self.asset_type,
            "region":           self.region,
            "state":            self.state,
            "latitude":         self.latitude,
            "longitude":        self.longitude,
            "voltage_kv":       self.voltage_kv,
            "install_year":     self.install_year,
            "customers_served": self.customers_served,
            "criticality_score": round(self.criticality_score, 4),
            "age_years":        datetime.now(timezone.utc).year - self.install_year,
        }


# ── SensorReading ──────────────────────────────────────────────────────────────

class SensorReading(db.Model):
    """
    A DGA (Dissolved Gas Analysis) oil-test reading for a transformer.

    Column names mirror the features in transformer_features.csv so they can
    be passed directly to the RiskClassifier feature vector.
    """

    __tablename__ = "sensor_reading"

    id              = db.Column(db.Integer, primary_key=True)
    asset_id        = db.Column(db.Integer, db.ForeignKey("asset.id"), nullable=False)
    recorded_at     = db.Column(db.DateTime, nullable=False, default=_utcnow, index=True)

    # Raw DGA gas concentrations (ppm / mg/kg)
    hydrogen_ppm             = db.Column(db.Float)
    oxygen_ppm               = db.Column(db.Float)
    nitrogen_ppm             = db.Column(db.Float)
    methane_ppm              = db.Column(db.Float)
    co_ppm                   = db.Column(db.Float)
    co2_ppm                  = db.Column(db.Float)
    ethylene_ppm             = db.Column(db.Float)
    ethane_ppm               = db.Column(db.Float)
    acetylene_ppm            = db.Column(db.Float)
    dbds_mg_kg               = db.Column(db.Float)

    # Electrical / physical tests
    power_factor_pct         = db.Column(db.Float)
    interfacial_tension_mNm  = db.Column(db.Float)
    dielectric_kv            = db.Column(db.Float)
    water_content_ppm        = db.Column(db.Float)

    # Computed IEC ratio features (stored for auditability; can be re-derived)
    tdcg_ppm                 = db.Column(db.Float)
    rogers_r1_ch4_h2         = db.Column(db.Float)
    rogers_r2_c2h2_c2h4      = db.Column(db.Float)
    rogers_r3_c2h2_ch4       = db.Column(db.Float)
    co2_co_ratio             = db.Column(db.Float)
    ethylene_ethane_ratio    = db.Column(db.Float)

    # Relationship
    asset = db.relationship("Asset", back_populates="sensor_readings")

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "asset_id":    self.asset_id,
            "recorded_at": self.recorded_at.isoformat(),
            "hydrogen_ppm":             self.hydrogen_ppm,
            "oxygen_ppm":               self.oxygen_ppm,
            "nitrogen_ppm":             self.nitrogen_ppm,
            "methane_ppm":              self.methane_ppm,
            "co_ppm":                   self.co_ppm,
            "co2_ppm":                  self.co2_ppm,
            "ethylene_ppm":             self.ethylene_ppm,
            "ethane_ppm":               self.ethane_ppm,
            "acetylene_ppm":            self.acetylene_ppm,
            "dbds_mg_kg":               self.dbds_mg_kg,
            "power_factor_pct":         self.power_factor_pct,
            "interfacial_tension_mNm":  self.interfacial_tension_mNm,
            "dielectric_kv":            self.dielectric_kv,
            "water_content_ppm":        self.water_content_ppm,
            "tdcg_ppm":                 self.tdcg_ppm,
            "rogers_r1_ch4_h2":         self.rogers_r1_ch4_h2,
            "rogers_r2_c2h2_c2h4":      self.rogers_r2_c2h2_c2h4,
            "rogers_r3_c2h2_ch4":       self.rogers_r3_c2h2_ch4,
            "co2_co_ratio":             self.co2_co_ratio,
            "ethylene_ethane_ratio":    self.ethylene_ethane_ratio,
        }

    def to_feature_dict(self) -> dict:
        """Return only the 20 features expected by RiskClassifier."""
        return {
            "hydrogen_ppm":             self.hydrogen_ppm,
            "oxygen_ppm":               self.oxygen_ppm,
            "nitrogen_ppm":             self.nitrogen_ppm,
            "methane_ppm":              self.methane_ppm,
            "co_ppm":                   self.co_ppm,
            "co2_ppm":                  self.co2_ppm,
            "ethylene_ppm":             self.ethylene_ppm,
            "ethane_ppm":               self.ethane_ppm,
            "acetylene_ppm":            self.acetylene_ppm,
            "dbds_mg_kg":               self.dbds_mg_kg,
            "power_factor_pct":         self.power_factor_pct,
            "interfacial_tension_mNm":  self.interfacial_tension_mNm,
            "dielectric_kv":            self.dielectric_kv,
            "water_content_ppm":        self.water_content_ppm,
            "tdcg_ppm":                 self.tdcg_ppm,
            "rogers_r1_ch4_h2":         self.rogers_r1_ch4_h2,
            "rogers_r2_c2h2_c2h4":      self.rogers_r2_c2h2_c2h4,
            "rogers_r3_c2h2_ch4":       self.rogers_r3_c2h2_ch4,
            "co2_co_ratio":             self.co2_co_ratio,
            "ethylene_ethane_ratio":    self.ethylene_ethane_ratio,
        }


# ── WeatherRecord ──────────────────────────────────────────────────────────────

class WeatherRecord(db.Model):
    """Regional weather snapshot — used by WeatherService to build Model B features."""

    __tablename__ = "weather_record"

    id           = db.Column(db.Integer, primary_key=True)
    region       = db.Column(db.String(80), nullable=False, index=True)
    state        = db.Column(db.String(2),  nullable=False)
    recorded_at  = db.Column(db.DateTime,  nullable=False, default=_utcnow, index=True)

    # Weather features matching Model B columns
    tmpf         = db.Column(db.Float)   # temperature °F
    relh         = db.Column(db.Float)   # relative humidity %
    sknt         = db.Column(db.Float)   # wind speed knots
    p01i         = db.Column(db.Float)   # 1-hour precip inches

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "region":      self.region,
            "state":       self.state,
            "recorded_at": self.recorded_at.isoformat(),
            "tmpf":        self.tmpf,
            "relh":        self.relh,
            "sknt":        self.sknt,
            "p01i":        self.p01i,
        }


# ── RiskScore ──────────────────────────────────────────────────────────────────

class RiskScore(db.Model):
    """
    Stored ML inference result for one asset at one point in time.

    Keeps both individual model outputs and the fused combined_risk_score
    so the dashboard can show trends over time.
    """

    __tablename__ = "risk_score"

    id                = db.Column(db.Integer,  primary_key=True)
    asset_id          = db.Column(db.Integer,  db.ForeignKey("asset.id"), nullable=False)
    sensor_reading_id = db.Column(db.Integer,  db.ForeignKey("sensor_reading.id"))
    weather_record_id = db.Column(db.Integer,  db.ForeignKey("weather_record.id"))
    predicted_at      = db.Column(db.DateTime, nullable=False, default=_utcnow, index=True)

    # Model A outputs (transformer health)
    model_a_risk_class  = db.Column(db.String(10))    # LOW / MEDIUM / HIGH
    model_a_proba_low   = db.Column(db.Float)
    model_a_proba_med   = db.Column(db.Float)
    model_a_proba_high  = db.Column(db.Float)

    # Model B outputs (weather severity)
    model_b_risk_class  = db.Column(db.String(10))
    model_b_proba_low   = db.Column(db.Float)
    model_b_proba_med   = db.Column(db.Float)
    model_b_proba_high  = db.Column(db.Float)

    # Fused combined score (0.0–1.0)
    combined_risk_score = db.Column(db.Float,  nullable=False)
    # Final bucketed label after fusion
    risk_level          = db.Column(db.String(10), nullable=False)  # LOW/MEDIUM/HIGH/CRITICAL

    # Relationship
    asset = db.relationship("Asset", back_populates="risk_scores")

    def to_dict(self) -> dict:
        return {
            "id":                 self.id,
            "asset_id":           self.asset_id,
            "predicted_at":       self.predicted_at.isoformat(),
            "model_a_risk_class": self.model_a_risk_class,
            "model_a_proba_low":  round(self.model_a_proba_low  or 0, 4),
            "model_a_proba_med":  round(self.model_a_proba_med  or 0, 4),
            "model_a_proba_high": round(self.model_a_proba_high or 0, 4),
            "model_b_risk_class": self.model_b_risk_class,
            "model_b_proba_low":  round(self.model_b_proba_low  or 0, 4),
            "model_b_proba_med":  round(self.model_b_proba_med  or 0, 4),
            "model_b_proba_high": round(self.model_b_proba_high or 0, 4),
            "combined_risk_score": round(self.combined_risk_score, 4),
            "risk_level":          self.risk_level,
        }


# ── MaintenanceOrder ──────────────────────────────────────────────────────────

class MaintenanceOrder(db.Model):
    """
    A generated work order for an at-risk asset.

    Created by MaintenanceService after each prediction run.
    """

    __tablename__ = "maintenance_order"

    id              = db.Column(db.Integer,  primary_key=True)
    asset_id        = db.Column(db.Integer,  db.ForeignKey("asset.id"), nullable=False)
    risk_score_id   = db.Column(db.Integer,  db.ForeignKey("risk_score.id"))
    generated_at    = db.Column(db.DateTime, nullable=False, default=_utcnow)
    priority_rank   = db.Column(db.Integer,  nullable=False)   # 1 = highest priority
    action_type     = db.Column(db.String(60), nullable=False)  # e.g. "Emergency Inspection"
    recommended_date = db.Column(db.String(20))                 # ISO date string
    crew_size       = db.Column(db.Integer,  default=2)
    estimated_hours = db.Column(db.Float,    default=4.0)
    notes           = db.Column(db.Text,     default="")
    status          = db.Column(db.String(20), default="pending")  # pending/in_progress/done

    # Relationship
    asset = db.relationship("Asset", back_populates="maintenance_orders")

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "asset_id":         self.asset_id,
            "asset_name":       self.asset.name if self.asset else None,
            "risk_score_id":    self.risk_score_id,
            "generated_at":     self.generated_at.isoformat(),
            "priority_rank":    self.priority_rank,
            "action_type":      self.action_type,
            "recommended_date": self.recommended_date,
            "crew_size":        self.crew_size,
            "estimated_hours":  self.estimated_hours,
            "notes":            self.notes,
            "status":           self.status,
        }

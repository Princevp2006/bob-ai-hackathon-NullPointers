"""
PowerGuard AI — Application Configuration

Centralised configuration using environment variables with safe defaults.
All environment-specific values (secrets, paths, flags) live here.
"""

import os
from pathlib import Path

# Resolve the repository root regardless of where the process starts
_REPO_ROOT = Path(__file__).resolve().parents[3]  # src/backend/core/config.py → repo root


class Config:
    """Base configuration — suitable for development."""

    # ------------------------------------------------------------------
    # Flask core
    # ------------------------------------------------------------------
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG: bool = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    TESTING: bool = False

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    DATABASE_PATH: Path = Path(
        os.environ.get("DATABASE_PATH", str(_REPO_ROOT / "data" / "powerguard.db"))
    )
    SQLALCHEMY_DATABASE_URI: str = f"sqlite:///{DATABASE_PATH}"
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False

    # ------------------------------------------------------------------
    # ML model artifacts
    # ------------------------------------------------------------------
    MODEL_DIR: Path = Path(
        os.environ.get("MODEL_DIR", str(_REPO_ROOT / "src" / "ml" / "models" / "artifacts"))
    )

    # ------------------------------------------------------------------
    # Data directories
    # ------------------------------------------------------------------
    RAW_DATA_DIR: Path = _REPO_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR: Path = _REPO_ROOT / "data" / "processed"
    SAMPLE_DATA_DIR: Path = _REPO_ROOT / "data" / "sample"

    # ------------------------------------------------------------------
    # External APIs (optional — for real weather data)
    # ------------------------------------------------------------------
    OPENWEATHER_API_KEY: str = os.environ.get("OPENWEATHER_API_KEY", "")
    OPENWEATHER_BASE_URL: str = "https://api.openweathermap.org/data/2.5"

    # ------------------------------------------------------------------
    # Prediction thresholds
    # ------------------------------------------------------------------
    # Probability above which an asset is flagged as HIGH risk
    HIGH_RISK_THRESHOLD: float = float(os.environ.get("HIGH_RISK_THRESHOLD", "0.7"))
    # Probability above which an asset is flagged as MEDIUM risk
    MEDIUM_RISK_THRESHOLD: float = float(os.environ.get("MEDIUM_RISK_THRESHOLD", "0.4"))


class TestingConfig(Config):
    """Configuration used during automated tests."""

    TESTING: bool = True
    DEBUG: bool = True
    DATABASE_PATH: Path = Path(":memory:")
    SQLALCHEMY_DATABASE_URI: str = "sqlite:///:memory:"


class ProductionConfig(Config):
    """Production configuration — override DEBUG and enforce secrets."""

    DEBUG: bool = False

    def __init__(self) -> None:
        # Fail fast at instantiation time (not at import time) if the secret is missing.
        if not os.environ.get("SECRET_KEY"):
            raise OSError(
                "SECRET_KEY environment variable must be set in production. "
                "Set it before starting the application."
            )
        self.SECRET_KEY = os.environ["SECRET_KEY"]

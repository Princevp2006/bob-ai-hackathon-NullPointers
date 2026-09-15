"""
PowerGuard AI — Database Initialisation Script
================================================
Creates all SQLite tables and seeds the database with sample assets
and sensor readings.

Run from the repository root:

    python scripts/init_db.py
    python scripts/init_db.py --reset   # drop + recreate all tables

The database is created at data/powerguard.db (or DATABASE_PATH env var).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main(reset: bool = False) -> None:
    from src.backend.app import create_app
    from src.backend.db.models import db

    if reset:
        # Create a minimal app without seeding to drop tables first
        app = create_app()
        with app.app_context():
            log.warning("Dropping all tables ...")
            db.drop_all()
            log.info("All tables dropped.")
        # Re-create fresh (create_app will call init_db + seed_db)
        app = create_app()
    else:
        app = create_app()

    with app.app_context():
        from src.backend.db.models import Asset, SensorReading
        n_assets   = Asset.query.count()
        n_readings = SensorReading.query.count()

    log.info("Database ready:  %d assets,  %d sensor readings.", n_assets, n_readings)
    log.info("DB path: %s", app.config["SQLALCHEMY_DATABASE_URI"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialise the PowerGuard AI database.")
    parser.add_argument("--reset", action="store_true",
                        help="Drop all tables before re-creating (data will be lost).")
    args = parser.parse_args()
    main(reset=args.reset)

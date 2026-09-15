"""
PowerGuard AI — Flask Application Factory
==========================================
Creates and configures the Flask app instance.
All blueprints are registered here.

Usage
-----
Development server (from repo root):

    python run.py                  # uses Config() defaults
    flask --app run:app run        # same

Production (gunicorn):

    gunicorn "run:create_app()" -w 4 -b 0.0.0.0:8000
"""

from __future__ import annotations

import logging

from flask import Flask, jsonify

from src.backend.core.config import Config

log = logging.getLogger(__name__)


def create_app(config_object: Config = None) -> Flask:
    """
    Application factory.

    Parameters
    ----------
    config_object : Config or subclass instance.
                    Defaults to Config() (development settings).
    """
    from pathlib import Path
    _repo_root = Path(__file__).resolve().parents[2]
    app = Flask(
        __name__,
        template_folder=str(_repo_root / "src" / "frontend" / "templates"),
        static_folder=str(_repo_root / "src" / "frontend" / "static"),
    )

    cfg = config_object or Config()
    app.config.from_object(cfg)

    # ── Logging ────────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO if not app.config.get("DEBUG") else logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Database ───────────────────────────────────────────────────────────────
    from src.backend.db.database import init_db, seed_db
    init_db(app)
    seed_db(app)

    # ── Blueprints ─────────────────────────────────────────────────────────────
    from src.backend.api.assets    import assets_bp
    from src.backend.api.predict   import predict_bp
    from src.backend.api.dashboard import dashboard_bp
    from src.backend.api.weather   import weather_bp

    app.register_blueprint(assets_bp,    url_prefix="/api/assets")
    app.register_blueprint(predict_bp,   url_prefix="/api/predict")
    app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")
    app.register_blueprint(weather_bp,   url_prefix="/api/weather")

    # ── Page routes ───────────────────────────────────────────────────────────
    from flask import render_template

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/assets")
    def assets():
        return render_template("assets.html")

    @app.route("/maintenance")
    def maintenance():
        return render_template("maintenance.html")

    # ── Health-check ───────────────────────────────────────────────────────────
    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "PowerGuard AI"}), 200

    # ── Generic error handlers ─────────────────────────────────────────────────
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad Request", "detail": str(e.description)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not Found", "detail": str(e.description)}), 404

    @app.errorhandler(422)
    def unprocessable(e):
        return jsonify({"error": "Unprocessable", "detail": str(e.description)}), 422

    @app.errorhandler(500)
    def server_error(e):
        log.exception("Unhandled server error")
        return jsonify({"error": "Internal Server Error"}), 500

    log.info("PowerGuard AI app created. DB=%s", app.config["SQLALCHEMY_DATABASE_URI"])
    return app


if __name__ == "__main__":
    application = create_app()
    application.run(debug=True, host="0.0.0.0", port=5000)

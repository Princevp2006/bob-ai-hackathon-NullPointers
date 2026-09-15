"""
PowerGuard AI — Development Server Entry Point
================================================
Run from the repository root:

    python run.py

Or with Flask CLI:

    flask --app run:app run --debug
"""

from src.backend.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

"""
PowerGuard AI — Sample Data Generator Script

Usage:
    python src/scripts/generate_sample_data.py

Generates realistic synthetic CSV files for:
  - data/sample/assets.csv
  - data/sample/sensor_readings.csv
  - data/sample/weather_records.csv
  - data/sample/incidents.csv

Uses only the Python standard library + NumPy + pandas.
No real asset data is used — all values are synthetic.

To be implemented in Phase 2.
"""

# TODO (Phase 2):
#   import numpy as np
#   import pandas as pd
#   from pathlib import Path
#
#   OUTPUT_DIR = Path(__file__).parents[2] / "data" / "sample"
#
#   def generate_assets(n=50) -> pd.DataFrame: ...
#   def generate_sensor_readings(assets_df, days=365) -> pd.DataFrame: ...
#   def generate_weather_records(regions, days=365) -> pd.DataFrame: ...
#   def generate_incidents(assets_df, n=200) -> pd.DataFrame: ...
#
#   if __name__ == "__main__":
#       assets = generate_assets()
#       ...

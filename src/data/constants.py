"""
PowerGuard AI — Pipeline Constants
===================================
Single source of truth for every file path, column name, threshold,
and schema definition used across the preprocessing pipeline.

Rules enforced here:
  - RAW_DIR is treated as read-only.  No script may write to it.
  - Column names are always referenced via these constants, never
    as bare strings scattered through other modules.
  - Physical-domain bounds are documented with IEEE / IEC references
    so future engineers understand where the numbers come from.
"""

from pathlib import Path

# ── Repository root (3 levels up from src/data/constants.py) ──────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Directory layout ──────────────────────────────────────────────────────
RAW_DIR       = _REPO_ROOT / "data" / "raw"
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"

# ── Raw filenames (never modified) ───────────────────────────────────────
RAW_HEALTH_INDEX = RAW_DIR / "Health index1 (2).csv"
RAW_WEATHER      = RAW_DIR / "Weather_data_combined_with_outage.csv"

# ── Processed output filenames ────────────────────────────────────────────
PROC_HEALTH_FEATURES = PROCESSED_DIR / "transformer_features.csv"
PROC_HEALTH_TARGET   = PROCESSED_DIR / "transformer_target.csv"   # separate to avoid accidental leakage
PROC_WEATHER         = PROCESSED_DIR / "weather_outage_events.csv"
PROC_PIPELINE_LOG    = PROCESSED_DIR / "pipeline_run_log.json"

# =============================================================================
# Dataset 1 — Health Index / DGA
# =============================================================================

# Original column names exactly as they appear in the raw file
HI_RAW_COLUMNS = [
    "Hydrogen", "Oxigen", "Nitrogen", "Methane", "CO", "CO2",
    "Ethylene", "Ethane", "Acethylene", "DBDS", "Power factor",
    "Interfacial V", "Dielectric rigidity", "Water content",
    "Health index", "Life expectation",
]

# Columns renamed during preprocessing (original → clean)
# Rationale: fix typos without altering raw files
HI_RENAME_MAP = {
    "Oxigen":      "oxygen_ppm",
    "Acethylene":  "acetylene_ppm",
    "Hydrogen":    "hydrogen_ppm",
    "Nitrogen":    "nitrogen_ppm",
    "Methane":     "methane_ppm",
    "CO":          "co_ppm",
    "CO2":         "co2_ppm",
    "Ethylene":    "ethylene_ppm",
    "Ethane":      "ethane_ppm",
    "DBDS":        "dbds_mg_kg",
    "Power factor":       "power_factor_pct",
    "Interfacial V":      "interfacial_tension_mNm",
    "Dielectric rigidity":"dielectric_kv",
    "Water content":      "water_content_ppm",
    "Health index":       "health_index",
    "Life expectation":   "life_expectation_years",
}

# Feature columns (raw sensor readings — safe to use as ML inputs)
HI_FEATURE_COLS = [
    "hydrogen_ppm", "oxygen_ppm", "nitrogen_ppm", "methane_ppm",
    "co_ppm", "co2_ppm", "ethylene_ppm", "ethane_ppm", "acetylene_ppm",
    "dbds_mg_kg", "power_factor_pct", "interfacial_tension_mNm",
    "dielectric_kv", "water_content_ppm",
]

# Columns that must NEVER appear in the feature matrix (leakage / target)
HI_EXCLUDE_FROM_FEATURES = ["health_index", "life_expectation_years"]

# Physical-domain bounds for anomaly detection
# Sources: IEEE C57.104-2019 (DGA limits), IEC 60422 (oil quality limits)
HI_PHYSICAL_BOUNDS = {
    # (min_valid, max_valid, description)
    "hydrogen_ppm":            (0,      50_000, "IEEE C57.104 — H2 alarm at 1800 ppm"),
    "oxygen_ppm":              (0,      20_000, "IEC 60599 — O2 normal < 20,000 ppm in sealed units"),
    "nitrogen_ppm":            (0,     100_000, "IEC 60599 — N2 background gas"),
    "methane_ppm":             (0,      20_000, "IEEE C57.104 — CH4 alarm at 1000 ppm"),
    "co_ppm":                  (0,       5_000, "IEEE C57.104 — CO alarm at 1000 ppm"),
    "co2_ppm":                 (0,      50_000, "IEEE C57.104 — CO2 alarm at 10,000 ppm"),
    "ethylene_ppm":            (0,      20_000, "IEEE C57.104 — C2H4 alarm at 200 ppm"),
    "ethane_ppm":              (0,      10_000, "IEEE C57.104 — C2H6 alarm at 200 ppm"),
    "acetylene_ppm":           (0,      10_000, "IEEE C57.104 — C2H2 alarm at 35 ppm"),
    "dbds_mg_kg":              (0,        500,  "IEC 62697 — DBDS threshold at 50 mg/kg"),
    "power_factor_pct":        (0,        100,  "IEC 60247 — tan δ > 10% is severe degradation"),
    "interfacial_tension_mNm": (0,         80,  "ASTM D971 — IFT range 20–50 mN/m typical"),
    "dielectric_kv":           (0,        100,  "IEC 60156 — BDV > 30 kV required"),
    "water_content_ppm":       (0,        200,  "IEC 60422 — moisture alarm at 35 ppm for ≥400 kV"),
    "health_index":            (0,        100,  "Score 0–100"),
    "life_expectation_years":  (0,        100,  "Remaining life in years"),
}

# Health Index → risk class mapping (used to derive the ML target)
# Boundaries informed by IEC TC10 and industry practice
HI_RISK_BINS   = [0, 25, 50, 100]          # left-closed, right-open except last
HI_RISK_LABELS = ["HIGH", "MEDIUM", "LOW"] # HIGH = score < 25
HI_RISK_INT    = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}  # integer encoding for sklearn

# =============================================================================
# Dataset 2 — Weather + Outage
# =============================================================================

WO_RAW_COLUMNS = [
    "fips_code", "Hour", "max_rolling_avg", "state", "county",
    "median_sum", "new_event_no", "evDur_Hr", "evDur_day",
    "Max_outage", "state_st", "tmpf", "relh", "gust", "feel",
    "p01i", "sknt",
]

# Columns to DROP before any further processing
# Documented reason for each drop.
# NOTE: new_event_no is NOT here — it is the groupby key for event aggregation.
#       It is renamed to event_id in the output and retained as an identifier.
WO_DROP_COLS = {
    "max_rolling_avg": "DATA LEAKAGE -- rolling window of Max_outage computed during event",
    "evDur_Hr":        "DATA LEAKAGE -- event duration only known after event ends",
    "evDur_day":       "DATA LEAKAGE -- derived from evDur_Hr; same leakage issue",
    "median_sum":      "SOFT LEAKAGE -- rolling aggregate of outage customers; unclear lineage",
    "state_st":        "REDUNDANT -- duplicate of state column (2-letter abbreviation)",
    "feel":            "COLLINEAR -- apparent temperature derived from tmpf + relh",
    "gust":            "SPARSE -- 91.5% missing; not usable as a feature",
}

# Columns retained as features after dropping leaky ones
WO_SAFE_FEATURE_COLS = ["tmpf", "relh", "sknt", "p01i"]

# Location and grouping keys
WO_LOCATION_COLS = ["fips_code", "state", "county"]

# Temporal key
WO_TIMESTAMP_COL = "Hour"
WO_TIMESTAMP_FMT = "%m/%d/%Y %H:%M"

# Target column
WO_TARGET_COL = "Max_outage"

# Imputation values for weather features (used when not enough context for median)
WO_IMPUTE_DEFAULTS = {
    "tmpf": None,   # use column median
    "relh": None,   # use column median
    "sknt": None,   # use column median
    "p01i": 0.0,    # 0 precipitation is a valid physical default
}

# Physical bounds for weather features
# Sources: ASOS/METAR observation ranges, NOAA documentation
WO_PHYSICAL_BOUNDS = {
    "tmpf": (-60.0, 150.0, "Air temperature in Fahrenheit — CONUS plausible range"),
    "relh": (  0.0, 100.0, "Relative humidity — 0 to 100%"),
    "sknt": (  0.0, 200.0, "Wind speed in knots — 200 kts would be hurricane-level extreme"),
    "p01i": (  0.0,  10.0, "Precipitation in inches/hr — 10 in/hr is extreme"),
}

# Outage severity bins → risk label for the county-level model
# Thresholds reflect NERC major event declaration guidance
WO_SEVERITY_BINS   = [0, 10, 1_000, float("inf")]
WO_SEVERITY_LABELS = ["LOW", "MEDIUM", "HIGH"]
WO_SEVERITY_INT    = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

# Derived temporal feature columns (added during processing)
WO_TEMPORAL_FEATURES = ["hour_of_day", "month", "year", "season"]

# Season mapping (month → season name)
WO_SEASON_MAP = {
    12: "WINTER", 1: "WINTER", 2: "WINTER",
    3:  "SPRING", 4: "SPRING", 5: "SPRING",
    6:  "SUMMER", 7: "SUMMER", 8: "SUMMER",
    9:  "AUTUMN", 10: "AUTUMN", 11: "AUTUMN",
}

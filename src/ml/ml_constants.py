"""
PowerGuard AI — ML Constants
==============================
Central registry for every path, column list, and hyperparameter used
across the ML layer.  Import from here; never hard-code strings elsewhere.
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Artifact storage ───────────────────────────────────────────────────────────
ARTIFACTS_DIR = _REPO_ROOT / "src" / "ml" / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_A_PATH      = ARTIFACTS_DIR / "transformer_risk_model.joblib"
MODEL_A_META_PATH = ARTIFACTS_DIR / "transformer_risk_model_meta.json"
MODEL_B_PATH      = ARTIFACTS_DIR / "weather_outage_model.joblib"
MODEL_B_META_PATH = ARTIFACTS_DIR / "weather_outage_model_meta.json"

# ── Processed data paths ───────────────────────────────────────────────────────
PROC_DIR             = _REPO_ROOT / "data" / "processed"
TRANSFORMER_FEATURES = PROC_DIR / "transformer_features.csv"
TRANSFORMER_TARGET   = PROC_DIR / "transformer_target.csv"
WEATHER_EVENTS       = PROC_DIR / "weather_outage_events.csv"

# ── Model A — Transformer Risk ─────────────────────────────────────────────────
# Feature columns used for training (excludes sample_id, health_index, life_expectation_years)
MODEL_A_FEATURE_COLS = [
    "hydrogen_ppm", "oxygen_ppm", "nitrogen_ppm", "methane_ppm",
    "co_ppm", "co2_ppm", "ethylene_ppm", "ethane_ppm", "acetylene_ppm",
    "dbds_mg_kg", "power_factor_pct", "interfacial_tension_mNm",
    "dielectric_kv", "water_content_ppm",
    "tdcg_ppm", "rogers_r1_ch4_h2", "rogers_r2_c2h2_c2h4",
    "rogers_r3_c2h2_ch4", "co2_co_ratio", "ethylene_ethane_ratio",
]
MODEL_A_TARGET_COL  = "risk_int"
MODEL_A_CLASS_NAMES = ["LOW", "MEDIUM", "HIGH"]   # index = integer label

# Stratified split proportions (no time-axis in this dataset)
MODEL_A_TEST_SIZE = 0.15   # 15 % held-out test
MODEL_A_VAL_SIZE  = 0.15   # 15 % validation (from remainder)
MODEL_A_RANDOM_STATE = 42

# Random Forest hyperparameters for Model A
MODEL_A_RF_PARAMS = {
    "n_estimators":    200,
    "max_depth":       None,
    "min_samples_leaf": 2,
    "class_weight":    "balanced",
    "random_state":    42,
    "n_jobs":          -1,
}

# Logistic Regression baseline hyperparameters
MODEL_A_LR_PARAMS = {
    "C":             0.1,
    "class_weight":  "balanced",
    "max_iter":      1000,
    "solver":        "lbfgs",
    "random_state":  42,
}

# ── Model B — Weather Outage Severity ─────────────────────────────────────────
MODEL_B_NUMERIC_COLS  = ["tmpf", "relh", "sknt", "p01i", "hour_of_day", "month", "year"]
MODEL_B_CATEGORY_COLS = ["season", "state"]   # label-encoded
MODEL_B_TARGET_COL    = "outage_severity_int"
MODEL_B_CLASS_NAMES   = ["LOW", "MEDIUM", "HIGH"]

# Chronological split: train on events before cutoff, test on events after
MODEL_B_TRAIN_CUTOFF  = "2021-01-01"   # ~80/20 split by event count
MODEL_B_RANDOM_STATE  = 42

MODEL_B_RF_PARAMS = {
    "n_estimators":    200,
    "max_depth":       None,
    "min_samples_leaf": 5,
    "class_weight":    "balanced",
    "random_state":    42,
    "n_jobs":          -1,
}

MODEL_B_LR_PARAMS = {
    "C":             0.1,
    "class_weight":  "balanced",
    "max_iter":      1000,
    "solver":        "lbfgs",
    "random_state":  42,
}

# ── Inference fusion weights ───────────────────────────────────────────────────
# combined_risk = model_a_proba_high * W_A + model_b_proba_high * W_B + criticality * W_C
FUSION_WEIGHT_A = 0.6   # transformer health dominates
FUSION_WEIGHT_B = 0.3   # weather modifies
FUSION_WEIGHT_C = 0.1   # criticality (from asset registry)

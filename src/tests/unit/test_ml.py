"""
PowerGuard AI — ML Layer Unit Tests
=====================================
Tests the ML constants, feature loaders, classifier classes,
and evaluator WITHOUT training full models on the real dataset.

All tests use small synthetic data so they run in milliseconds
and do not depend on trained artifact files being present.

Run from the repository root:

    python -m pytest src/tests/unit/test_ml.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_clf_data():
    """
    Synthetic 3-class balanced dataset (150 samples × 5 features).
    Returns (X_df, y_series, feature_names).
    """
    X_arr, y_arr = make_classification(
        n_samples=150,
        n_features=5,
        n_informative=4,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    feature_names = [f"feat_{i}" for i in range(5)]
    X = pd.DataFrame(X_arr, columns=feature_names)
    y = pd.Series(y_arr, name="label")
    return X, y, feature_names


@pytest.fixture
def tiny_weather_data():
    """
    Synthetic weather-like 3-class dataset (300 samples × 9 features).
    Returns (X_df, y_series, feature_names).
    """
    rng = np.random.default_rng(42)
    n = 300
    data = {
        "tmpf":       rng.normal(70, 15, n),
        "relh":       rng.uniform(20, 100, n),
        "sknt":       rng.exponential(5, n),
        "p01i":       rng.exponential(0.05, n),
        "hour_of_day": rng.integers(0, 24, n).astype(float),
        "month":      rng.integers(1, 13, n).astype(float),
        "year":       rng.choice([2019, 2020, 2021], size=n).astype(float),
        "season_enc": rng.integers(0, 4, n).astype(float),
        "state_enc":  rng.integers(0, 50, n).astype(float),
    }
    X = pd.DataFrame(data)
    # Imbalanced target like the real dataset: 0=LOW(63%), 1=MEDIUM(35%), 2=HIGH(2%)
    y = pd.Series(rng.choice([0, 1, 2], size=n, p=[0.63, 0.35, 0.02]), name="outage_severity_int")
    return X, y, list(X.columns)


# ── ML Constants Tests ────────────────────────────────────────────────────────

class TestMLConstants:
    def test_artifacts_dir_exists(self):
        from src.ml.ml_constants import ARTIFACTS_DIR
        assert ARTIFACTS_DIR.exists(), "ARTIFACTS_DIR should be created on import"

    def test_model_a_feature_count(self):
        from src.ml.ml_constants import MODEL_A_FEATURE_COLS
        # 14 raw DGA cols + 6 engineered IEC ratio cols = 20
        assert len(MODEL_A_FEATURE_COLS) == 20

    def test_model_a_class_names(self):
        from src.ml.ml_constants import MODEL_A_CLASS_NAMES
        assert MODEL_A_CLASS_NAMES == ["LOW", "MEDIUM", "HIGH"]

    def test_model_b_class_names(self):
        from src.ml.ml_constants import MODEL_B_CLASS_NAMES
        assert MODEL_B_CLASS_NAMES == ["LOW", "MEDIUM", "HIGH"]

    def test_fusion_weights_sum_to_one(self):
        from src.ml.ml_constants import FUSION_WEIGHT_A, FUSION_WEIGHT_B, FUSION_WEIGHT_C
        total = FUSION_WEIGHT_A + FUSION_WEIGHT_B + FUSION_WEIGHT_C
        assert abs(total - 1.0) < 1e-9, f"Fusion weights must sum to 1.0, got {total}"

    def test_split_sizes_sum_to_at_most_one(self):
        from src.ml.ml_constants import MODEL_A_TEST_SIZE, MODEL_A_VAL_SIZE
        assert MODEL_A_TEST_SIZE + MODEL_A_VAL_SIZE < 1.0

    def test_model_a_rf_has_class_weight(self):
        from src.ml.ml_constants import MODEL_A_RF_PARAMS
        assert MODEL_A_RF_PARAMS["class_weight"] == "balanced"

    def test_model_b_rf_has_class_weight(self):
        from src.ml.ml_constants import MODEL_B_RF_PARAMS
        assert MODEL_B_RF_PARAMS["class_weight"] == "balanced"

    def test_processed_paths_are_path_objects(self):
        from src.ml.ml_constants import TRANSFORMER_FEATURES, TRANSFORMER_TARGET, WEATHER_EVENTS
        assert isinstance(TRANSFORMER_FEATURES, Path)
        assert isinstance(TRANSFORMER_TARGET, Path)
        assert isinstance(WEATHER_EVENTS, Path)

    def test_model_b_numeric_cols_count(self):
        from src.ml.ml_constants import MODEL_B_NUMERIC_COLS
        # tmpf, relh, sknt, p01i, hour_of_day, month, year = 7
        assert len(MODEL_B_NUMERIC_COLS) == 7

    def test_model_b_category_cols_count(self):
        from src.ml.ml_constants import MODEL_B_CATEGORY_COLS
        # season, state = 2
        assert len(MODEL_B_CATEGORY_COLS) == 2


# ── RiskClassifier Tests (Model A) ────────────────────────────────────────────

class TestRiskClassifier:
    def test_rf_fit_and_predict(self, tiny_clf_data):
        from src.ml.models.risk_classifier import RiskClassifier
        X, y, feat_names = tiny_clf_data
        clf = RiskClassifier(model_type="rf")
        clf.fit(X, y, feature_names=feat_names)
        preds = clf.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1, 2})

    def test_lr_fit_and_predict(self, tiny_clf_data):
        from src.ml.models.risk_classifier import RiskClassifier
        X, y, feat_names = tiny_clf_data
        clf = RiskClassifier(model_type="lr")
        clf.fit(X, y, feature_names=feat_names)
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_predict_proba_shape(self, tiny_clf_data):
        from src.ml.models.risk_classifier import RiskClassifier
        X, y, feat_names = tiny_clf_data
        clf = RiskClassifier(model_type="rf")
        clf.fit(X, y, feature_names=feat_names)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 3)

    def test_predict_proba_rows_sum_to_one(self, tiny_clf_data):
        from src.ml.models.risk_classifier import RiskClassifier
        X, y, feat_names = tiny_clf_data
        clf = RiskClassifier(model_type="rf")
        clf.fit(X, y, feature_names=feat_names)
        proba = clf.predict_proba(X)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_feature_importances_rf(self, tiny_clf_data):
        from src.ml.models.risk_classifier import RiskClassifier
        X, y, feat_names = tiny_clf_data
        clf = RiskClassifier(model_type="rf")
        clf.fit(X, y, feature_names=feat_names)
        fi = clf.feature_importances()
        assert isinstance(fi, pd.Series)
        assert len(fi) == len(feat_names)
        assert abs(fi.sum() - 1.0) < 1e-6

    def test_feature_importances_lr_raises(self, tiny_clf_data):
        from src.ml.models.risk_classifier import RiskClassifier
        X, y, feat_names = tiny_clf_data
        clf = RiskClassifier(model_type="lr")
        clf.fit(X, y, feature_names=feat_names)
        with pytest.raises(AttributeError):
            clf.feature_importances()

    def test_unfitted_predict_raises(self, tiny_clf_data):
        from src.ml.models.risk_classifier import RiskClassifier
        X, _, _ = tiny_clf_data
        clf = RiskClassifier(model_type="rf")
        with pytest.raises(RuntimeError):
            clf.predict(X)

    def test_invalid_model_type_raises(self):
        from src.ml.models.risk_classifier import RiskClassifier
        with pytest.raises(ValueError):
            RiskClassifier(model_type="xgb")._build_pipeline()

    def test_save_and_load(self, tiny_clf_data):
        from src.ml.models.risk_classifier import RiskClassifier
        X, y, feat_names = tiny_clf_data
        clf = RiskClassifier(model_type="rf")
        clf.fit(X, y, feature_names=feat_names)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.joblib"
            clf.save(path)
            assert path.exists()
            loaded = RiskClassifier.load(path, model_type="rf")
            preds_orig   = clf.predict(X)
            preds_loaded = loaded.predict(X)
            assert np.array_equal(preds_orig, preds_loaded)

    def test_save_metadata(self, tiny_clf_data):
        from src.ml.models.risk_classifier import RiskClassifier
        X, y, feat_names = tiny_clf_data
        clf = RiskClassifier(model_type="rf")
        clf.fit(X, y, feature_names=feat_names)
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "meta.json"
            clf.save_metadata(meta_path, extra={"test_accuracy": 0.85})
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["model_type"] == "rf"
            assert meta["test_accuracy"] == 0.85
            assert "feature_names" in meta

    def test_build_pipeline_returns_pipeline(self):
        from src.ml.models.risk_classifier import RiskClassifier
        from sklearn.pipeline import Pipeline
        clf = RiskClassifier(model_type="rf")
        pipe = clf._build_pipeline()
        assert isinstance(pipe, Pipeline)


# ── WeatherClassifier Tests (Model B) ────────────────────────────────────────

class TestWeatherClassifier:
    def test_rf_fit_and_predict(self, tiny_weather_data):
        from src.ml.models.weather_classifier import WeatherClassifier
        X, y, feat_names = tiny_weather_data
        clf = WeatherClassifier(model_type="rf")
        clf.fit(X, y, feature_names=feat_names)
        preds = clf.predict(X)
        assert len(preds) == len(y)

    def test_predict_proba_shape(self, tiny_weather_data):
        from src.ml.models.weather_classifier import WeatherClassifier
        X, y, feat_names = tiny_weather_data
        clf = WeatherClassifier(model_type="rf")
        clf.fit(X, y, feature_names=feat_names)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 3)

    def test_save_and_load(self, tiny_weather_data):
        from src.ml.models.weather_classifier import WeatherClassifier
        X, y, feat_names = tiny_weather_data
        clf = WeatherClassifier(model_type="rf")
        clf.fit(X, y, feature_names=feat_names)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weather_model.joblib"
            clf.save(path)
            loaded = WeatherClassifier.load(path, model_type="rf")
            assert np.array_equal(clf.predict(X), loaded.predict(X))

    def test_unfitted_raises(self, tiny_weather_data):
        from src.ml.models.weather_classifier import WeatherClassifier
        X, _, _ = tiny_weather_data
        with pytest.raises(RuntimeError):
            WeatherClassifier(model_type="rf").predict(X)

    def test_feature_importances_sums_to_one(self, tiny_weather_data):
        from src.ml.models.weather_classifier import WeatherClassifier
        X, y, feat_names = tiny_weather_data
        clf = WeatherClassifier(model_type="rf")
        clf.fit(X, y, feature_names=feat_names)
        fi = clf.feature_importances()
        assert abs(fi.sum() - 1.0) < 1e-6


# ── Evaluator Tests ───────────────────────────────────────────────────────────

class TestEvaluator:
    def test_evaluate_returns_expected_keys(self, tiny_clf_data):
        from src.ml.models.risk_classifier import RiskClassifier
        from src.ml.evaluation.evaluator import evaluate_classifier
        X, y, feat_names = tiny_clf_data
        clf = RiskClassifier(model_type="rf")
        clf.fit(X, y, feature_names=feat_names)
        results = evaluate_classifier(clf, X, y, ["LOW", "MEDIUM", "HIGH"])
        expected_keys = {
            "model_name", "accuracy", "classification_report",
            "confusion_matrix", "roc_auc_macro", "n_test_samples", "class_counts",
        }
        assert expected_keys.issubset(results.keys())

    def test_accuracy_in_range(self, tiny_clf_data):
        from src.ml.models.risk_classifier import RiskClassifier
        from src.ml.evaluation.evaluator import evaluate_classifier
        X, y, feat_names = tiny_clf_data
        clf = RiskClassifier(model_type="rf")
        clf.fit(X, y, feature_names=feat_names)
        results = evaluate_classifier(clf, X, y, ["LOW", "MEDIUM", "HIGH"])
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_confusion_matrix_shape(self, tiny_clf_data):
        from src.ml.models.risk_classifier import RiskClassifier
        from src.ml.evaluation.evaluator import evaluate_classifier
        X, y, feat_names = tiny_clf_data
        clf = RiskClassifier(model_type="rf")
        clf.fit(X, y, feature_names=feat_names)
        results = evaluate_classifier(clf, X, y, ["LOW", "MEDIUM", "HIGH"])
        cm = results["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_save_results_writes_json(self, tiny_clf_data):
        from src.ml.models.risk_classifier import RiskClassifier
        from src.ml.evaluation.evaluator import evaluate_classifier, save_results
        X, y, feat_names = tiny_clf_data
        clf = RiskClassifier(model_type="rf")
        clf.fit(X, y, feature_names=feat_names)
        results = evaluate_classifier(clf, X, y, ["LOW", "MEDIUM", "HIGH"])
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "results.json"
            save_results({"model_a": results}, out)
            assert out.exists()
            with open(out) as f:
                loaded = json.load(f)
            assert "model_a" in loaded

    def test_n_test_samples_correct(self, tiny_clf_data):
        from src.ml.models.risk_classifier import RiskClassifier
        from src.ml.evaluation.evaluator import evaluate_classifier
        X, y, feat_names = tiny_clf_data
        clf = RiskClassifier(model_type="rf")
        clf.fit(X, y, feature_names=feat_names)
        results = evaluate_classifier(clf, X, y, ["LOW", "MEDIUM", "HIGH"])
        assert results["n_test_samples"] == len(y)

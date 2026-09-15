"""
PowerGuard AI — Unit Tests for Configuration

To be expanded in Phase 2.
"""


def test_config_defaults():
    """Verify that Config loads without error and has sane defaults."""
    from src.backend.core.config import Config

    cfg = Config()
    assert cfg.HIGH_RISK_THRESHOLD == 0.7
    assert cfg.MEDIUM_RISK_THRESHOLD == 0.4
    assert cfg.SQLALCHEMY_TRACK_MODIFICATIONS is False


def test_testing_config_uses_in_memory_db():
    """TestingConfig must point to an in-memory SQLite database."""
    from src.backend.core.config import TestingConfig

    cfg = TestingConfig()
    assert cfg.TESTING is True
    assert ":memory:" in cfg.SQLALCHEMY_DATABASE_URI

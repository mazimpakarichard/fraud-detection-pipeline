"""Tests for fraud detection models."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.models.isolation_forest import IsolationForestModel
from fraud_detection.models.rules import RuleBasedModel


@pytest.fixture
def sample_features():
    """Create sample feature data."""
    n = 100
    return pd.DataFrame(
        {
            "amount": np.random.lognormal(4, 1, n).round(2),
            "amount_log": np.log1p(np.random.lognormal(4, 1, n)),
            "hour_sin": np.sin(2 * np.pi * np.random.randint(0, 24, n) / 24),
            "hour_cos": np.cos(2 * np.pi * np.random.randint(0, 24, n) / 24),
            "is_night": np.random.choice([0, 1], n, p=[0.8, 0.2]),
            "is_weekend": np.random.choice([0, 1], n, p=[0.7, 0.3]),
            "suspicious_email": np.random.choice([0, 1], n, p=[0.95, 0.05]),
            "country_mismatch": np.random.choice([0, 1], n, p=[0.9, 0.1]),
            "high_risk_category": np.random.choice([0, 1], n, p=[0.85, 0.15]),
            "is_online": np.random.choice([0, 1], n, p=[0.3, 0.7]),
            "card_txn_count_1h": np.random.poisson(2, n),
            "card_txn_count_24h": np.random.poisson(10, n),
        }
    )


class TestRuleBasedModel:
    """Tests for RuleBasedModel."""

    def test_score_basic(self, sample_features):
        """Test basic scoring."""
        model = RuleBasedModel()
        scores, reasons = model.score(sample_features)

        assert len(scores) == len(sample_features)
        assert all(0 <= s <= 1 for s in scores)
        assert len(reasons) == len(sample_features)

    def test_high_amount_detection(self):
        """Test high amount rule."""
        model = RuleBasedModel(amount_threshold=1000)
        df = pd.DataFrame(
            {
                "amount": [500, 5000, 10000],
            }
        )

        scores, _reasons = model.score(df)

        # Higher amounts should have higher scores
        assert scores[0] < scores[1] < scores[2]

    def test_velocity_detection(self):
        """Test velocity rule."""
        model = RuleBasedModel(velocity_threshold_1h=3)
        df = pd.DataFrame(
            {
                "card_txn_count_1h": [1, 5, 10],
            }
        )

        scores, _reasons = model.score(df)

        # Higher velocity should have higher scores
        assert scores[0] < scores[1] < scores[2]

    def test_reason_codes_generated(self, sample_features):
        """Test that reason codes are generated for flagged transactions."""
        model = RuleBasedModel()
        _scores, reasons = model.score(sample_features)

        # At least some transactions should have reason codes
        has_reasons = sum(1 for r in reasons if len(r) > 0)
        assert has_reasons > 0


class TestIsolationForestModel:
    """Tests for IsolationForestModel."""

    def test_fit_and_score(self, sample_features):
        """Test fitting and scoring."""
        model = IsolationForestModel(n_estimators=50, random_state=42)
        model.fit(sample_features)

        scores = model.score(sample_features)

        assert len(scores) == len(sample_features)
        assert all(0 <= s <= 1 for s in scores)

    def test_predict(self, sample_features):
        """Test prediction."""
        model = IsolationForestModel(n_estimators=50, random_state=42)
        model.fit(sample_features)

        predictions = model.predict(sample_features)

        assert len(predictions) == len(sample_features)
        assert all(p in [0, 1] for p in predictions)

    def test_feature_importances(self, sample_features):
        """Test feature importances are computed."""
        model = IsolationForestModel(n_estimators=50, random_state=42)
        model.fit(sample_features)

        assert model.feature_importances_ is not None
        assert len(model.feature_importances_) == len(sample_features.columns)

    def test_top_anomaly_features(self, sample_features):
        """Test getting top anomaly features."""
        model = IsolationForestModel(n_estimators=50, random_state=42)
        model.fit(sample_features)

        top_features = model.get_top_anomaly_features(sample_features, n_top=3)

        assert len(top_features) == len(sample_features)
        assert all(len(f) == 3 for f in top_features)

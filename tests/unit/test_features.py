"""Tests for feature engineering."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from fraud_detection.features.engineering import FastFeatureEngineer, FeatureEngineer


@pytest.fixture
def sample_transactions():
    """Create sample transaction data."""
    n = 100
    base_date = datetime(2024, 1, 15, 12, 0, 0)

    return pd.DataFrame(
        {
            "transaction_id": [f"txn_{i:04d}" for i in range(n)],
            "timestamp": [base_date + timedelta(hours=i) for i in range(n)],
            "amount": np.random.lognormal(4, 1, n).round(2),
            "merchant_id": [f"merchant_{i % 10}" for i in range(n)],
            "card_id": [f"card_{i % 20}" for i in range(n)],
            "customer_id": [f"cust_{i % 15}" for i in range(n)],
            "email_domain": ["gmail.com"] * 90 + ["tempmail.com"] * 10,
            "billing_country": ["USA"] * 95 + ["GBR"] * 5,
            "shipping_country": ["USA"] * 90 + ["CAN"] * 10,
            "is_online": [True] * 80 + [False] * 20,
            "merchant_category": ["grocery"] * 50 + ["electronics"] * 30 + ["jewelry"] * 20,
        }
    )


class TestFeatureEngineer:
    """Tests for FeatureEngineer."""

    def test_fit_transform(self, sample_transactions):
        """Test basic feature transformation."""
        engineer = FeatureEngineer()
        features = engineer.fit_transform(sample_transactions)

        assert len(features) == len(sample_transactions)
        assert len(features.columns) > 0

    def test_temporal_features(self, sample_transactions):
        """Test temporal feature generation."""
        engineer = FeatureEngineer(
            compute_velocity=False,
            compute_aggregations=False,
            compute_risk_indicators=False,
        )
        features = engineer.fit_transform(sample_transactions)

        expected_features = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_night"]
        for feat in expected_features:
            assert feat in features.columns, f"Missing temporal feature: {feat}"

    def test_risk_indicators(self, sample_transactions):
        """Test risk indicator features."""
        engineer = FeatureEngineer(
            compute_velocity=False,
            compute_aggregations=False,
            compute_temporal=False,
        )
        features = engineer.fit_transform(sample_transactions)

        assert "suspicious_email" in features.columns
        assert "country_mismatch" in features.columns

        # Check suspicious email detection
        suspicious_count = features["suspicious_email"].sum()
        assert suspicious_count == 10  # We set 10 tempmail.com emails

    def test_no_nan_values(self, sample_transactions):
        """Test that output has no NaN values."""
        engineer = FeatureEngineer()
        features = engineer.fit_transform(sample_transactions)

        assert not features.isna().any().any(), "Features contain NaN values"

    def test_feature_names(self, sample_transactions):
        """Test that feature names are tracked."""
        engineer = FeatureEngineer()
        features = engineer.fit_transform(sample_transactions)

        assert len(engineer.feature_names) == len(features.columns)
        assert engineer.feature_names == features.columns.tolist()


class TestFastFeatureEngineer:
    """Tests for FastFeatureEngineer."""

    def test_transform(self, sample_transactions):
        """Test fast transformation."""
        engineer = FastFeatureEngineer()
        features = engineer.transform(sample_transactions)

        assert len(features) == len(sample_transactions)
        assert "amount" in features.columns
        assert "hour_sin" in features.columns

    def test_no_nan_values(self, sample_transactions):
        """Test that output has no NaN values."""
        engineer = FastFeatureEngineer()
        features = engineer.transform(sample_transactions)

        assert not features.isna().any().any()

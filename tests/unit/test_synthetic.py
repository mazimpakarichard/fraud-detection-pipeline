"""Tests for synthetic data generator."""

import pytest

from fraud_detection.data.synthetic import AnomalyConfig, SyntheticTransactionGenerator


class TestSyntheticGenerator:
    """Tests for SyntheticTransactionGenerator."""

    def test_generate_basic(self):
        """Test basic data generation."""
        generator = SyntheticTransactionGenerator(seed=42)
        df = generator.generate(n_transactions=1000)

        # Allow small variance due to anomaly injection/deduplication
        assert len(df) >= 950  # At least 95% of requested transactions
        assert "transaction_id" in df.columns
        assert "amount" in df.columns
        assert "is_fraud" in df.columns
        assert "timestamp" in df.columns

    def test_anomaly_rate(self):
        """Test that anomaly rate is approximately correct."""
        config = AnomalyConfig(total_anomaly_rate=0.05)
        generator = SyntheticTransactionGenerator(anomaly_config=config, seed=42)
        df = generator.generate(n_transactions=10000)

        fraud_rate = df["is_fraud"].mean()
        # Allow some variance
        assert 0.03 < fraud_rate < 0.10

    def test_required_columns(self):
        """Test that all required columns are present."""
        generator = SyntheticTransactionGenerator(seed=42)
        df = generator.generate(n_transactions=100)

        required_cols = [
            "transaction_id",
            "timestamp",
            "amount",
            "merchant_id",
            "card_id",
            "is_fraud",
            "data_source",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_data_source_label(self):
        """Test that data_source is correctly set."""
        generator = SyntheticTransactionGenerator(seed=42)
        df = generator.generate(n_transactions=100)

        assert (df["data_source"] == "SYNTHETIC").all()

    def test_amount_positive(self):
        """Test that all amounts are positive."""
        generator = SyntheticTransactionGenerator(seed=42)
        df = generator.generate(n_transactions=1000)

        assert (df["amount"] > 0).all()

    def test_unique_transaction_ids(self):
        """Test that transaction IDs are unique."""
        generator = SyntheticTransactionGenerator(seed=42)
        df = generator.generate(n_transactions=1000)

        assert df["transaction_id"].nunique() == len(df)

    def test_reproducibility(self):
        """Test that same seed produces same data."""
        gen1 = SyntheticTransactionGenerator(seed=123)
        gen2 = SyntheticTransactionGenerator(seed=123)

        df1 = gen1.generate(n_transactions=100)
        df2 = gen2.generate(n_transactions=100)

        assert df1["amount"].sum() == df2["amount"].sum()
        assert df1["is_fraud"].sum() == df2["is_fraud"].sum()


class TestAnomalyConfig:
    """Tests for AnomalyConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AnomalyConfig()
        config.validate()  # Should not raise

    def test_invalid_ratios(self):
        """Test that invalid ratios raise error."""
        config = AnomalyConfig(
            amount_outlier_ratio=0.5,
            velocity_anomaly_ratio=0.5,
            geographic_anomaly_ratio=0.5,  # This makes sum > 1
        )
        with pytest.raises(ValueError):
            config.validate()

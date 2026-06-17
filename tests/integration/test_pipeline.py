"""Integration tests for the fraud detection pipeline."""

import pytest


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    def test_synthetic_data_generation(self):
        """Test that synthetic data can be generated."""
        from fraud_detection.data.synthetic import SyntheticTransactionGenerator

        generator = SyntheticTransactionGenerator(seed=42)
        df = generator.generate(n_transactions=100)

        assert len(df) >= 90  # Allow some variance
        assert "transaction_id" in df.columns
        assert "amount" in df.columns
        assert "is_fraud" in df.columns

    def test_feature_engineering(self):
        """Test feature engineering on synthetic data."""
        from fraud_detection.data.synthetic import SyntheticTransactionGenerator
        from fraud_detection.features.engineering import FeatureEngineer

        generator = SyntheticTransactionGenerator(seed=42)
        df = generator.generate(n_transactions=100)

        engineer = FeatureEngineer()
        features = engineer.fit_transform(df)

        assert len(features) == len(df)
        assert not features.isna().any().any()

    def test_model_training_and_prediction(self):
        """Test model training and prediction."""
        from fraud_detection.data.synthetic import SyntheticTransactionGenerator
        from fraud_detection.features.engineering import FeatureEngineer
        from fraud_detection.models.isolation_forest import IsolationForestModel

        # Generate data
        generator = SyntheticTransactionGenerator(seed=42)
        df = generator.generate(n_transactions=500)

        # Engineer features
        engineer = FeatureEngineer()
        features = engineer.fit_transform(df)

        # Train model
        model = IsolationForestModel(n_estimators=50, random_state=42)
        model.fit(features)

        # Make predictions
        scores = model.score(features)

        assert len(scores) == len(features)
        assert all(0 <= s <= 1 for s in scores)

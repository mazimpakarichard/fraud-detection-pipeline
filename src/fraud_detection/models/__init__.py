"""Fraud detection models."""

from fraud_detection.models.ensemble import EnsembleScorer
from fraud_detection.models.isolation_forest import IsolationForestModel
from fraud_detection.models.rules import RuleBasedModel


# Lazy import for PyTorch model (optional dependency)
def get_autoencoder_model():
    from fraud_detection.models.autoencoder import AutoencoderModel

    return AutoencoderModel


__all__ = [
    "EnsembleScorer",
    "IsolationForestModel",
    "RuleBasedModel",
    "get_autoencoder_model",
]

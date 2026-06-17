"""Ensemble scoring with explainability.

Combines multiple anomaly detection models:
- Rule-based (interpretable domain rules)
- Isolation Forest (unsupervised anomaly detection)
- Autoencoder (deep learning reconstruction)

Provides:
- Weighted ensemble score
- Per-transaction reason codes
- Feature importance explanations
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from fraud_detection.models.rules import RuleBasedModel
from fraud_detection.models.isolation_forest import IsolationForestModel
from fraud_detection.utils.config import Settings, get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScoringResult:
    """Result for a single transaction."""
    
    transaction_id: str
    ensemble_score: float
    is_flagged: bool
    rule_score: float
    isolation_forest_score: float
    autoencoder_score: float | None
    reason_codes: list[dict[str, Any]]
    model_versions: dict[str, str]


@dataclass
class BatchScoringResult:
    """Results for a batch of transactions."""
    
    batch_id: str
    scores: np.ndarray
    is_flagged: np.ndarray
    rule_scores: np.ndarray
    isolation_forest_scores: np.ndarray
    autoencoder_scores: np.ndarray | None
    reason_codes: list[list[dict[str, Any]]]
    model_versions: dict[str, str]
    stats: dict[str, Any] = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for database insertion."""
        df = pd.DataFrame({
            "ensemble_score": self.scores,
            "is_flagged": self.is_flagged,
            "rule_score": self.rule_scores,
            "isolation_forest_score": self.isolation_forest_scores,
            "autoencoder_score": self.autoencoder_scores if self.autoencoder_scores is not None else np.nan,
            "reason_codes": [json.dumps(r) for r in self.reason_codes],
            "model_versions": json.dumps(self.model_versions),
        })
        return df


class EnsembleScorer:
    """
    Ensemble fraud scoring combining multiple models.
    
    Combines scores from:
    - Rule-based model (30% default weight)
    - Isolation Forest (35% default weight)
    - Autoencoder (35% default weight, if available)
    
    Generates per-transaction reason codes for explainability.
    """
    
    VERSION = "1.0.0"
    
    def __init__(
        self,
        weights: dict[str, float] | None = None,
        flag_threshold: float = 0.5,
        use_autoencoder: bool = True,
        settings: Settings | None = None,
    ) -> None:
        """
        Initialize ensemble scorer.
        
        Args:
            weights: Model weights (rules, isolation_forest, autoencoder).
            flag_threshold: Score threshold for flagging transactions.
            use_autoencoder: Whether to use autoencoder model.
            settings: Application settings.
        """
        self.settings = settings or get_settings()
        
        # Default weights from settings
        if weights is None:
            weights = self.settings.ensemble_weights
        
        self.weights = weights
        self.flag_threshold = flag_threshold
        self.use_autoencoder = use_autoencoder
        
        # Initialize models
        self.rule_model = RuleBasedModel()
        self.isolation_forest: IsolationForestModel | None = None
        self.autoencoder: Any = None  # Lazy load
        
        self._is_fitted = False
    
    @property
    def model_versions(self) -> dict[str, str]:
        """Get versions of all models."""
        versions = {
            "ensemble": self.VERSION,
            "rules": "1.0.0",
        }
        if self.isolation_forest is not None:
            versions["isolation_forest"] = "1.0.0"
        if self.autoencoder is not None:
            versions["autoencoder"] = "1.0.0"
        return versions
    
    def fit(
        self,
        X: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> "EnsembleScorer":
        """
        Fit ensemble models on training data.
        
        Args:
            X: Training features.
            feature_names: Feature names.
            
        Returns:
            Self for chaining.
        """
        logger.info("Fitting ensemble models", n_samples=len(X))
        
        # Fit Isolation Forest
        self.isolation_forest = IsolationForestModel(
            contamination=self.settings.isolation_forest_contamination,
        )
        self.isolation_forest.fit(X, feature_names)
        
        # Fit Autoencoder if requested
        if self.use_autoencoder:
            try:
                from fraud_detection.models.autoencoder import AutoencoderModel
                self.autoencoder = AutoencoderModel(
                    threshold_percentile=self.settings.autoencoder_threshold_percentile,
                )
                self.autoencoder.fit(X, feature_names)
            except ImportError:
                logger.warning("PyTorch not available, skipping autoencoder")
                self.use_autoencoder = False
        
        # Adjust weights if autoencoder not used
        if not self.use_autoencoder or self.autoencoder is None:
            # Redistribute autoencoder weight
            ae_weight = self.weights.get("autoencoder", 0)
            rule_weight = self.weights.get("rules", 0.3)
            if_weight = self.weights.get("isolation_forest", 0.35)
            total = rule_weight + if_weight
            self.weights = {
                "rules": rule_weight + (ae_weight * rule_weight / total),
                "isolation_forest": if_weight + (ae_weight * if_weight / total),
                "autoencoder": 0,
            }
        
        self._is_fitted = True
        logger.info("Ensemble fitting complete", weights=self.weights)
        
        return self
    
    def score_batch(
        self,
        X: pd.DataFrame,
        transaction_ids: list[str] | None = None,
        batch_id: str = "default",
    ) -> BatchScoringResult:
        """
        Score a batch of transactions.
        
        Args:
            X: Feature DataFrame.
            transaction_ids: Transaction IDs.
            batch_id: Batch identifier.
            
        Returns:
            BatchScoringResult with all scores and reason codes.
        """
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        n_samples = len(X)
        logger.info("Scoring batch", batch_id=batch_id, n_samples=n_samples)
        
        # Get rule-based scores and reasons
        rule_scores, rule_reasons = self.rule_model.score(X)
        
        # Get Isolation Forest scores
        if_scores = self.isolation_forest.score(X)
        if_contributions = self.isolation_forest.get_top_anomaly_features(X, n_top=3)
        
        # Get Autoencoder scores if available
        ae_scores = None
        ae_contributions = None
        if self.autoencoder is not None:
            ae_scores = self.autoencoder.score(X)
            ae_contributions = self.autoencoder.get_reconstruction_contributions(X, n_top=3)
        
        # Compute ensemble scores
        ensemble_scores = (
            self.weights.get("rules", 0) * rule_scores +
            self.weights.get("isolation_forest", 0) * if_scores
        )
        
        if ae_scores is not None:
            ensemble_scores += self.weights.get("autoencoder", 0) * ae_scores
        
        # Clip to 0-1
        ensemble_scores = np.clip(ensemble_scores, 0, 1)
        
        # Flag transactions
        is_flagged = ensemble_scores >= self.flag_threshold
        
        # Compile reason codes
        reason_codes: list[list[dict[str, Any]]] = []
        for i in range(n_samples):
            reasons = []
            
            # Add rule-based reasons
            for reason in rule_reasons[i]:
                reasons.append({
                    "model": "rules",
                    **reason,
                })
            
            # Add Isolation Forest feature contributions
            for feature, contrib in if_contributions[i]:
                if contrib > 0.1:  # Only include significant contributions
                    reasons.append({
                        "model": "isolation_forest",
                        "feature": feature,
                        "contribution": round(contrib, 4),
                        "reason": f"Feature '{feature}' contributed to anomaly score",
                    })
            
            # Add Autoencoder contributions
            if ae_contributions is not None:
                for feature, error in ae_contributions[i]:
                    if error > 0.1:
                        reasons.append({
                            "model": "autoencoder",
                            "feature": feature,
                            "reconstruction_error": round(error, 4),
                            "reason": f"High reconstruction error for '{feature}'",
                        })
            
            # Sort by contribution/score
            reasons.sort(key=lambda x: x.get("score", x.get("contribution", 0)), reverse=True)
            reason_codes.append(reasons[:5])  # Keep top 5 reasons
        
        # Compute stats
        stats = {
            "total": n_samples,
            "flagged": int(is_flagged.sum()),
            "flag_rate": float(is_flagged.mean()),
            "mean_score": float(ensemble_scores.mean()),
            "max_score": float(ensemble_scores.max()),
            "p95_score": float(np.percentile(ensemble_scores, 95)),
        }
        
        logger.info(
            "Batch scoring complete",
            batch_id=batch_id,
            **stats,
        )
        
        return BatchScoringResult(
            batch_id=batch_id,
            scores=ensemble_scores,
            is_flagged=is_flagged,
            rule_scores=rule_scores,
            isolation_forest_scores=if_scores,
            autoencoder_scores=ae_scores,
            reason_codes=reason_codes,
            model_versions=self.model_versions,
            stats=stats,
        )
    
    def score_single(
        self,
        X: pd.DataFrame,
        transaction_id: str,
    ) -> ScoringResult:
        """Score a single transaction."""
        batch_result = self.score_batch(X, batch_id="single")
        
        return ScoringResult(
            transaction_id=transaction_id,
            ensemble_score=float(batch_result.scores[0]),
            is_flagged=bool(batch_result.is_flagged[0]),
            rule_score=float(batch_result.rule_scores[0]),
            isolation_forest_score=float(batch_result.isolation_forest_scores[0]),
            autoencoder_score=float(batch_result.autoencoder_scores[0]) if batch_result.autoencoder_scores is not None else None,
            reason_codes=batch_result.reason_codes[0],
            model_versions=batch_result.model_versions,
        )
    
    def save(self, model_dir: str | Path) -> None:
        """Save all models to directory."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Isolation Forest
        if self.isolation_forest is not None:
            self.isolation_forest.save(model_dir / "isolation_forest.pkl")
        
        # Save Autoencoder
        if self.autoencoder is not None:
            self.autoencoder.save(model_dir / "autoencoder.pt")
        
        # Save ensemble config
        config = {
            "weights": self.weights,
            "flag_threshold": self.flag_threshold,
            "use_autoencoder": self.use_autoencoder,
            "version": self.VERSION,
        }
        with open(model_dir / "ensemble_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info("Ensemble models saved", path=str(model_dir))
    
    @classmethod
    def load(cls, model_dir: str | Path) -> "EnsembleScorer":
        """Load ensemble from directory."""
        model_dir = Path(model_dir)
        
        # Load config
        with open(model_dir / "ensemble_config.json") as f:
            config = json.load(f)
        
        instance = cls(
            weights=config["weights"],
            flag_threshold=config["flag_threshold"],
            use_autoencoder=config["use_autoencoder"],
        )
        
        # Load Isolation Forest
        if_path = model_dir / "isolation_forest.pkl"
        if if_path.exists():
            instance.isolation_forest = IsolationForestModel.load(if_path)
        
        # Load Autoencoder
        ae_path = model_dir / "autoencoder.pt"
        if ae_path.exists() and config["use_autoencoder"]:
            try:
                from fraud_detection.models.autoencoder import AutoencoderModel
                instance.autoencoder = AutoencoderModel.load(ae_path)
            except ImportError:
                logger.warning("PyTorch not available, autoencoder not loaded")
        
        instance._is_fitted = True
        logger.info("Ensemble models loaded", path=str(model_dir))
        
        return instance

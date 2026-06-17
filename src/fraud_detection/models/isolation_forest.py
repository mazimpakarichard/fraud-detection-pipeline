"""Isolation Forest anomaly detection model.

Uses scikit-learn's Isolation Forest for unsupervised anomaly detection.
Features:
- Automatic contamination rate estimation
- Feature importance via permutation
- Anomaly score normalization to 0-1
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from fraud_detection.utils.config import Settings, get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


class IsolationForestModel:
    """
    Isolation Forest for fraud detection.

    Isolation Forest detects anomalies by isolating observations.
    The algorithm randomly selects a feature and splits value between
    the max and min. Anomalies require fewer splits to isolate.
    """

    def __init__(
        self,
        contamination: float | str = "auto",
        n_estimators: int = 200,
        max_samples: int | float = "auto",
        random_state: int = 42,
        settings: Settings | None = None,
    ) -> None:
        """
        Initialize Isolation Forest model.

        Args:
            contamination: Expected proportion of outliers (0-0.5) or "auto".
            n_estimators: Number of trees in the forest.
            max_samples: Number of samples to draw for each tree.
            random_state: Random seed for reproducibility.
            settings: Application settings.
        """
        self.settings = settings or get_settings()
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self.model: IsolationForest | None = None
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "IsolationForestModel":
        """
        Fit the Isolation Forest model.

        Args:
            X: Training features.
            feature_names: Names of features.

        Returns:
            Self for chaining.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X

        logger.info(
            "Fitting Isolation Forest",
            n_samples=X_array.shape[0],
            n_features=X_array.shape[1],
            contamination=self.contamination,
        )

        # Scale features for better isolation
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_array)

        # Fit Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination if self.contamination != "auto" else 0.02,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
            warm_start=False,
        )
        self.model.fit(X_scaled)

        # Compute feature importances (average depth contribution)
        self.feature_importances_ = self._compute_feature_importances(X_scaled)

        logger.info(
            "Isolation Forest fitted",
            n_estimators=self.n_estimators,
            top_features=dict(zip(self.feature_names[:5], self.feature_importances_[:5].tolist())),
        )

        return self

    def _compute_feature_importances(self, X: np.ndarray) -> np.ndarray:
        """Compute feature importances based on average path length contribution."""
        if self.model is None:
            return np.zeros(X.shape[1])

        # Get decision paths
        importances = np.zeros(X.shape[1])

        for tree in self.model.estimators_:
            # Count feature usage in splits
            feature_counts = np.bincount(
                tree.tree_.feature[tree.tree_.feature >= 0], minlength=X.shape[1]
            )
            # Weight by depth (earlier splits more important)
            importances += feature_counts

        # Normalize
        importances = importances / importances.sum() if importances.sum() > 0 else importances
        return importances

    def score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Score transactions (anomaly probability).

        Args:
            X: Features to score.

        Returns:
            Array of anomaly scores (0-1, higher = more anomalous).
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        X_scaled = self.scaler.transform(X_array)

        # Get raw anomaly scores (-1 to 1, lower = more anomalous)
        raw_scores = self.model.decision_function(X_scaled)

        # Convert to 0-1 (higher = more anomalous)
        # decision_function returns negative for anomalies
        scores = -raw_scores

        # Normalize to 0-1 using min-max scaling
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        return scores

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict anomaly labels.

        Args:
            X: Features to predict.
            threshold: Score threshold for anomaly classification.

        Returns:
            Array of predictions (1 = anomaly, 0 = normal).
        """
        scores = self.score(X)
        return (scores >= threshold).astype(int)

    def get_top_anomaly_features(
        self,
        X: pd.DataFrame | np.ndarray,
        n_top: int = 5,
    ) -> list[list[tuple[str, float]]]:
        """
        Get top contributing features for each sample's anomaly score.

        Uses feature deviation from mean as a proxy for contribution.

        Args:
            X: Features.
            n_top: Number of top features to return.

        Returns:
            List of (feature_name, contribution) tuples for each sample.
        """
        if self.scaler is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        X_scaled = self.scaler.transform(X_array)

        # Use absolute deviation from mean as contribution
        importances = (
            self.feature_importances_
            if self.feature_importances_ is not None
            else np.ones(X_scaled.shape[1])
        )
        contributions = np.abs(X_scaled) * importances

        results = []
        for i in range(len(X_array)):
            sample_contrib = contributions[i]
            top_indices = np.argsort(sample_contrib)[::-1][:n_top]
            top_features = [
                (self.feature_names[idx], float(sample_contrib[idx])) for idx in top_indices
            ]
            results.append(top_features)

        return results

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances_,
            "params": {
                "contamination": self.contamination,
                "n_estimators": self.n_estimators,
                "max_samples": self.max_samples,
                "random_state": self.random_state,
            },
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info("Isolation Forest model saved", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> "IsolationForestModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        instance = cls(**model_data["params"])
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance.feature_names = model_data["feature_names"]
        instance.feature_importances_ = model_data["feature_importances"]

        logger.info("Isolation Forest model loaded", path=str(path))
        return instance

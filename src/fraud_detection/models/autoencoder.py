"""PyTorch Autoencoder for anomaly detection.

Uses reconstruction error as anomaly score:
- Normal transactions are well-reconstructed (low error)
- Anomalous transactions have high reconstruction error

Architecture:
- Symmetric encoder-decoder
- Batch normalization for stable training
- Bottleneck layer forces learning of normal patterns
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from fraud_detection.utils.config import Settings, get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


def check_torch_available() -> None:
    """Check if PyTorch is available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch not installed. Install with: pip install torch\n"
            "Or use CPU-only version: pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )


class Autoencoder(nn.Module):
    """
    Symmetric autoencoder for anomaly detection.

    Architecture:
        Encoder: input -> hidden1 -> hidden2 -> bottleneck
        Decoder: bottleneck -> hidden2 -> hidden1 -> output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [64, 32],
        bottleneck_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize autoencoder.

        Args:
            input_dim: Number of input features.
            hidden_dims: Dimensions of hidden layers.
            bottleneck_dim: Dimension of bottleneck layer.
            dropout: Dropout rate.
        """
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (reverse of encoder)
        decoder_layers = []
        prev_dim = bottleneck_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (reconstruction, latent representation).
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation."""
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent."""
        return self.decoder(latent)


class AutoencoderModel:
    """
    Wrapper for training and scoring with autoencoder.

    Uses reconstruction error as anomaly score:
    - Low error = normal transaction
    - High error = potential fraud
    """

    def __init__(
        self,
        hidden_dims: list[int] = [64, 32],
        bottleneck_dim: int = 16,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 50,
        threshold_percentile: float = 98.0,
        device: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        """
        Initialize autoencoder model.

        Args:
            hidden_dims: Hidden layer dimensions.
            bottleneck_dim: Bottleneck dimension.
            dropout: Dropout rate.
            learning_rate: Optimizer learning rate.
            batch_size: Training batch size.
            epochs: Training epochs.
            threshold_percentile: Percentile for anomaly threshold.
            device: Device to use ('cuda' or 'cpu').
            settings: Application settings.
        """
        check_torch_available()

        self.settings = settings or get_settings()
        self.hidden_dims = hidden_dims
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.threshold_percentile = threshold_percentile

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Autoencoder | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.feature_names: list[str] = []
        self.threshold: float = 0.0
        self.training_losses: list[float] = []

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        feature_names: list[str] | None = None,
        validation_split: float = 0.1,
    ) -> "AutoencoderModel":
        """
        Train autoencoder on normal transactions.

        Args:
            X: Training features (should be mostly normal transactions).
            feature_names: Names of features.
            validation_split: Fraction for validation.

        Returns:
            Self for chaining.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values.astype(np.float32)
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X.astype(np.float32)

        logger.info(
            "Training autoencoder",
            n_samples=X_array.shape[0],
            n_features=X_array.shape[1],
            device=str(self.device),
        )

        # Normalize features
        self.feature_mean = X_array.mean(axis=0)
        self.feature_std = X_array.std(axis=0) + 1e-8
        X_normalized = (X_array - self.feature_mean) / self.feature_std

        # Create data loaders
        n_val = int(len(X_normalized) * validation_split)
        X_train = X_normalized[n_val:]
        X_val = X_normalized[:n_val]

        train_dataset = TensorDataset(torch.FloatTensor(X_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Initialize model
        input_dim = X_array.shape[1]
        self.model = Autoencoder(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            bottleneck_dim=self.bottleneck_dim,
            dropout=self.dropout,
        ).to(self.device)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Training loop
        best_val_loss = float("inf")
        self.training_losses = []

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                x = batch[0].to(self.device)

                optimizer.zero_grad()
                reconstruction, _ = self.model(x)
                loss = criterion(reconstruction, x)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(x)

            train_loss /= len(X_train)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(self.device)
                    reconstruction, _ = self.model(x)
                    loss = criterion(reconstruction, x)
                    val_loss += loss.item() * len(x)

            val_loss /= len(X_val)
            scheduler.step(val_loss)

            self.training_losses.append(val_loss)

            best_val_loss = min(best_val_loss, val_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs}",
                    train_loss=f"{train_loss:.6f}",
                    val_loss=f"{val_loss:.6f}",
                )

        # Compute threshold from training data reconstruction errors
        train_errors = self._compute_reconstruction_error(X_train)
        self.threshold = np.percentile(train_errors, self.threshold_percentile)

        logger.info(
            "Autoencoder training complete",
            final_val_loss=f"{best_val_loss:.6f}",
            threshold=f"{self.threshold:.6f}",
        )

        return self

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize features."""
        return (X - self.feature_mean) / self.feature_std

    def _compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute reconstruction errors."""
        if self.model is None:
            raise ValueError("Model not fitted")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            reconstruction, _ = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstruction) ** 2, dim=1)

        return errors.cpu().numpy()

    def score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Score transactions (anomaly probability).

        Args:
            X: Features to score.

        Returns:
            Array of anomaly scores (0-1, higher = more anomalous).
        """
        if self.model is None or self.feature_mean is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            X_array = X.values.astype(np.float32)
        else:
            X_array = X.astype(np.float32)

        X_normalized = self._normalize(X_array)
        errors = self._compute_reconstruction_error(X_normalized)

        # Normalize scores to 0-1 using threshold
        scores = errors / (self.threshold * 2)  # Scale so threshold ~= 0.5
        scores = np.clip(scores, 0, 1)

        return scores

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float | None = None) -> np.ndarray:
        """
        Predict anomaly labels.

        Args:
            X: Features to predict.
            threshold: Score threshold (uses learned threshold if None).

        Returns:
            Array of predictions (1 = anomaly, 0 = normal).
        """
        scores = self.score(X)
        threshold = threshold or 0.5  # Threshold corresponds to learned percentile
        return (scores >= threshold).astype(int)

    def get_reconstruction_contributions(
        self,
        X: pd.DataFrame | np.ndarray,
        n_top: int = 5,
    ) -> list[list[tuple[str, float]]]:
        """
        Get top contributing features to reconstruction error.

        Args:
            X: Features.
            n_top: Number of top features.

        Returns:
            List of (feature_name, contribution) tuples for each sample.
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        if isinstance(X, pd.DataFrame):
            X_array = X.values.astype(np.float32)
        else:
            X_array = X.astype(np.float32)

        X_normalized = self._normalize(X_array)
        X_tensor = torch.FloatTensor(X_normalized).to(self.device)

        self.model.eval()
        with torch.no_grad():
            reconstruction, _ = self.model(X_tensor)
            feature_errors = (X_tensor - reconstruction) ** 2

        feature_errors = feature_errors.cpu().numpy()

        results = []
        for i in range(len(X_array)):
            sample_errors = feature_errors[i]
            top_indices = np.argsort(sample_errors)[::-1][:n_top]
            top_features = [
                (self.feature_names[idx], float(sample_errors[idx])) for idx in top_indices
            ]
            results.append(top_features)

        return results

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not fitted")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model_state": self.model.state_dict(),
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "training_losses": self.training_losses,
            "params": {
                "hidden_dims": self.hidden_dims,
                "bottleneck_dim": self.bottleneck_dim,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "threshold_percentile": self.threshold_percentile,
            },
            "input_dim": self.model.encoder[0].in_features,
        }

        torch.save(model_data, path)
        logger.info("Autoencoder model saved", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> "AutoencoderModel":
        """Load model from disk."""
        check_torch_available()

        model_data = torch.load(path, map_location="cpu")

        instance = cls(**model_data["params"])
        instance.feature_mean = model_data["feature_mean"]
        instance.feature_std = model_data["feature_std"]
        instance.feature_names = model_data["feature_names"]
        instance.threshold = model_data["threshold"]
        instance.training_losses = model_data["training_losses"]

        # Rebuild model architecture
        instance.model = Autoencoder(
            input_dim=model_data["input_dim"],
            hidden_dims=model_data["params"]["hidden_dims"],
            bottleneck_dim=model_data["params"]["bottleneck_dim"],
            dropout=model_data["params"]["dropout"],
        ).to(instance.device)

        instance.model.load_state_dict(model_data["model_state"])

        logger.info("Autoencoder model loaded", path=str(path))
        return instance

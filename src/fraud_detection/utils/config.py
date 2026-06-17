"""Configuration management with secure secrets handling."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Security: All sensitive values use SecretStr and are loaded from
    environment variables. Never hardcode credentials in code.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FRAUD_",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Fraud Detection Pipeline"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Database - SECURE: credentials from environment only
    db_host: str = Field(default="localhost", description="PostgreSQL host")
    db_port: int = Field(default=5432, description="PostgreSQL port")
    db_name: str = Field(default="fraud_detection", description="Database name")
    db_user: str = Field(default="postgres", description="Database user")
    db_password: SecretStr = Field(default=SecretStr("postgres"), description="Database password")
    db_schema: str = Field(default="fraud", description="Database schema")

    # Connection pool settings
    db_pool_size: int = Field(default=5, ge=1, le=20)
    db_max_overflow: int = Field(default=10, ge=0, le=50)

    # Paths
    data_dir: Path = Field(default=Path("data"))
    models_dir: Path = Field(default=Path("models"))
    results_dir: Path = Field(default=Path("results"))
    reports_dir: Path = Field(default=Path("reports"))

    # Data source
    use_synthetic_data: bool = Field(
        default=True,
        description="Use synthetic data instead of IEEE-CIS dataset",
    )
    synthetic_rows: int = Field(default=1_000_000, ge=1000)
    anomaly_rate: float = Field(default=0.02, ge=0.001, le=0.1)

    # Model settings
    isolation_forest_contamination: float = Field(default=0.02, ge=0.001, le=0.1)
    autoencoder_threshold_percentile: float = Field(default=98.0, ge=90.0, le=99.9)
    ensemble_weights: dict[str, float] = Field(
        default={"rules": 0.3, "isolation_forest": 0.35, "autoencoder": 0.35}
    )

    # Airflow
    airflow_dag_id: str = "fraud_detection_pipeline"

    @property
    def database_url(self) -> str:
        """Construct database URL (without exposing password in logs)."""
        return (
            f"postgresql://{self.db_user}:{self.db_password.get_secret_value()}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def database_url_masked(self) -> str:
        """Database URL with masked password for logging."""
        return (
            f"postgresql://{self.db_user}:****"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for dir_path in [self.data_dir, self.models_dir, self.results_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()

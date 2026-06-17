"""Command-line interface for fraud detection pipeline."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer

from fraud_detection.utils.config import get_settings
from fraud_detection.utils.logging import configure_logging, get_logger

app = typer.Typer(
    name="fraud-cli",
    help="Fraud Detection Pipeline CLI",
    add_completion=False,
)

logger = get_logger(__name__)


@app.command()
def generate(
    rows: int = typer.Option(100_000, help="Number of transactions to generate"),
    output: Optional[Path] = typer.Option(None, help="Output path (parquet or csv)"),
    seed: int = typer.Option(42, help="Random seed"),
    anomaly_rate: float = typer.Option(0.02, help="Anomaly injection rate"),
) -> None:
    """Generate synthetic transaction data."""
    configure_logging()

    from fraud_detection.data.synthetic import SyntheticTransactionGenerator, AnomalyConfig

    typer.echo(f"Generating {rows:,} synthetic transactions...")

    config = AnomalyConfig(total_anomaly_rate=anomaly_rate)
    generator = SyntheticTransactionGenerator(anomaly_config=config, seed=seed)
    df = generator.generate(n_transactions=rows)

    if output:
        if output.suffix == ".csv":
            generator.save_to_csv(df, str(output))
        else:
            generator.save_to_parquet(df, str(output))
        typer.echo(f"Saved to {output}")
    else:
        settings = get_settings()
        output_path = settings.data_dir / f"synthetic_{datetime.now():%Y%m%d_%H%M%S}.parquet"
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        generator.save_to_parquet(df, str(output_path))
        typer.echo(f"Saved to {output_path}")

    typer.echo(f"Generated {len(df):,} transactions")
    typer.echo(f"Fraud rate: {df['is_fraud'].mean():.2%}")


@app.command()
def train(
    data_path: Optional[Path] = typer.Option(None, help="Path to training data"),
    use_synthetic: bool = typer.Option(True, help="Generate synthetic data if no path"),
    rows: int = typer.Option(100_000, help="Synthetic data rows"),
    output_dir: Optional[Path] = typer.Option(None, help="Model output directory"),
) -> None:
    """Train ensemble fraud detection model."""
    configure_logging()

    import pandas as pd
    from fraud_detection.features.engineering import FeatureEngineer
    from fraud_detection.models.ensemble import EnsembleScorer

    settings = get_settings()
    output_dir = output_dir or settings.models_dir / "ensemble"

    # Load or generate data
    if data_path:
        typer.echo(f"Loading data from {data_path}...")
        df = pd.read_parquet(data_path)
    elif use_synthetic:
        typer.echo(f"Generating {rows:,} synthetic transactions...")
        from fraud_detection.data.synthetic import SyntheticTransactionGenerator
        generator = SyntheticTransactionGenerator(seed=42)
        df = generator.generate(n_transactions=rows)
    else:
        typer.echo("No data source specified", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loaded {len(df):,} transactions")

    # Feature engineering
    typer.echo("Engineering features...")
    engineer = FeatureEngineer()
    features = engineer.fit_transform(df)

    typer.echo(f"Created {len(features.columns)} features")

    # Train ensemble
    typer.echo("Training ensemble model...")
    scorer = EnsembleScorer(settings=settings)
    scorer.fit(features)

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    scorer.save(output_dir)

    typer.echo(f"Model saved to {output_dir}")

    # Quick evaluation
    typer.echo("\nQuick evaluation on training data:")
    results = scorer.score_batch(features, batch_id="train_eval")

    from sklearn.metrics import precision_score, recall_score, roc_auc_score

    y_true = df["is_fraud"].astype(int).values
    y_pred = results.is_flagged.astype(int)
    y_score = results.scores

    typer.echo(f"Precision: {precision_score(y_true, y_pred):.3f}")
    typer.echo(f"Recall: {recall_score(y_true, y_pred):.3f}")
    typer.echo(f"ROC-AUC: {roc_auc_score(y_true, y_score):.3f}")


@app.command()
def score(
    data_path: Optional[Path] = typer.Option(None, help="Path to data to score"),
    model_dir: Optional[Path] = typer.Option(None, help="Model directory"),
    output: Optional[Path] = typer.Option(None, help="Output path for scores"),
    batch_size: int = typer.Option(10_000, help="Batch size for scoring"),
) -> None:
    """Score transactions with trained ensemble model."""
    configure_logging()

    import pandas as pd
    from fraud_detection.features.engineering import FastFeatureEngineer
    from fraud_detection.models.ensemble import EnsembleScorer

    settings = get_settings()
    model_dir = model_dir or settings.models_dir / "ensemble"

    if not (model_dir / "ensemble_config.json").exists():
        typer.echo(f"Model not found at {model_dir}. Run 'train' first.", err=True)
        raise typer.Exit(1)

    # Load model
    typer.echo(f"Loading model from {model_dir}...")
    scorer = EnsembleScorer.load(model_dir)

    # Load data
    if data_path:
        typer.echo(f"Loading data from {data_path}...")
        df = pd.read_parquet(data_path)
    else:
        # Generate some test data
        typer.echo("No data path provided, generating test data...")
        from fraud_detection.data.synthetic import SyntheticTransactionGenerator
        generator = SyntheticTransactionGenerator(seed=123)
        df = generator.generate(n_transactions=batch_size)

    typer.echo(f"Loaded {len(df):,} transactions")

    # Fast feature engineering
    engineer = FastFeatureEngineer()
    features = engineer.transform(df)

    # Score
    typer.echo("Scoring transactions...")
    results = scorer.score_batch(features, batch_id=f"cli_{datetime.now():%Y%m%d_%H%M%S}")

    # Output results
    scores_df = results.to_dataframe()
    scores_df["transaction_id"] = df["transaction_id"].values

    if output:
        scores_df.to_parquet(output)
        typer.echo(f"Scores saved to {output}")

    # Summary
    typer.echo(f"\nScoring complete:")
    typer.echo(f"  Total: {len(scores_df):,}")
    typer.echo(f"  Flagged: {results.is_flagged.sum():,} ({results.is_flagged.mean():.1%})")
    typer.echo(f"  Mean score: {results.scores.mean():.3f}")
    typer.echo(f"  Max score: {results.scores.max():.3f}")

    # Show top flagged
    if results.is_flagged.sum() > 0:
        typer.echo(f"\nTop 5 flagged transactions:")
        top_indices = results.scores.argsort()[::-1][:5]
        for i, idx in enumerate(top_indices):
            typer.echo(f"  {i+1}. {df.iloc[idx]['transaction_id']}: "
                      f"score={results.scores[idx]:.3f}, "
                      f"amount=${df.iloc[idx]['amount']:.2f}")


@app.command()
def validate(
    data_path: Path = typer.Argument(..., help="Path to data to validate"),
    strict: bool = typer.Option(False, help="Raise error on validation failure"),
) -> None:
    """Validate transaction data quality."""
    configure_logging()

    import pandas as pd
    from fraud_detection.validation.expectations import TransactionValidator

    typer.echo(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    typer.echo(f"Validating {len(df):,} transactions...")
    validator = TransactionValidator()
    results = validator.validate(df, batch_id=f"cli_{datetime.now():%Y%m%d_%H%M%S}", strict=strict)

    if results["success"]:
        typer.echo("Validation PASSED")
    else:
        typer.echo("Validation FAILED", err=True)
        for failure in results["failed_expectations"]:
            typer.echo(f"  - {failure}", err=True)
        if strict:
            raise typer.Exit(1)


@app.command()
def init_db(
    schema_file: Path = typer.Option(Path("sql/001_schema.sql"), help="Schema SQL file"),
    seed_file: Optional[Path] = typer.Option(None, help="Seed SQL file"),
) -> None:
    """Initialize database schema."""
    configure_logging()

    from fraud_detection.utils.database import DatabaseManager

    db = DatabaseManager()

    typer.echo("Initializing database schema...")
    db.execute_sql_file(str(schema_file))
    typer.echo("Schema created successfully")

    if seed_file:
        typer.echo("Running seed script...")
        db.execute_sql_file(str(seed_file))
        typer.echo("Seed data loaded")


@app.command()
def run_pipeline(
    rows: int = typer.Option(10_000, help="Number of transactions"),
    save_results: bool = typer.Option(True, help="Save results to database"),
) -> None:
    """Run complete pipeline: generate -> validate -> train -> score."""
    configure_logging()

    import pandas as pd
    from fraud_detection.data.synthetic import SyntheticTransactionGenerator
    from fraud_detection.features.engineering import FeatureEngineer
    from fraud_detection.models.ensemble import EnsembleScorer
    from fraud_detection.validation.expectations import TransactionValidator

    settings = get_settings()

    typer.echo("=" * 60)
    typer.echo("FRAUD DETECTION PIPELINE")
    typer.echo("=" * 60)

    # 1. Generate data
    typer.echo("\n[1/5] Generating synthetic data...")
    generator = SyntheticTransactionGenerator(seed=42)
    df = generator.generate(n_transactions=rows)
    typer.echo(f"Generated {len(df):,} transactions ({df['is_fraud'].mean():.1%} fraud)")

    # 2. Validate
    typer.echo("\n[2/5] Validating data quality...")
    validator = TransactionValidator()
    validation_results = validator.validate(df, batch_id="pipeline_run")
    typer.echo(f"Validation: {'PASSED' if validation_results['success'] else 'FAILED'}")

    # 3. Feature engineering
    typer.echo("\n[3/5] Engineering features...")
    engineer = FeatureEngineer()
    features = engineer.fit_transform(df)
    typer.echo(f"Created {len(features.columns)} features")

    # 4. Train/load model and score
    typer.echo("\n[4/5] Training and scoring...")
    scorer = EnsembleScorer(settings=settings)
    scorer.fit(features)

    results = scorer.score_batch(features, batch_id="pipeline_run")
    typer.echo(f"Flagged {results.is_flagged.sum():,} transactions ({results.is_flagged.mean():.1%})")

    # 5. Evaluate
    typer.echo("\n[5/5] Evaluating performance...")
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    y_true = df["is_fraud"].astype(int).values
    y_pred = results.is_flagged.astype(int)
    y_score = results.scores

    metrics = {
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_score),
    }

    typer.echo("\nPerformance Metrics:")
    for name, value in metrics.items():
        typer.echo(f"  {name}: {value:.3f}")

    # Show example flagged transactions
    typer.echo("\nExample Flagged Transactions:")
    flagged_indices = results.scores.argsort()[::-1][:3]
    for idx in flagged_indices:
        txn = df.iloc[idx]
        typer.echo(f"  - {txn['transaction_id']}: ${txn['amount']:.2f}, "
                  f"score={results.scores[idx]:.3f}, "
                  f"actual_fraud={txn['is_fraud']}")
        if results.reason_codes[idx]:
            for reason in results.reason_codes[idx][:2]:
                typer.echo(f"      {reason.get('reason', 'N/A')}")

    typer.echo("\n" + "=" * 60)
    typer.echo("Pipeline complete!")


if __name__ == "__main__":
    app()

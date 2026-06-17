"""Fraud Detection Pipeline DAG.

Orchestrates the end-to-end fraud detection workflow:
1. Extract: Load unscored transactions from database or generate synthetic data
2. Validate: Run Great Expectations data quality checks
3. Transform: Engineer features from raw transactions
4. Score: Apply ensemble fraud detection models
5. Persist: Store results back to database
6. Alert: Notify on high-risk transactions
7. Monitor: Generate drift and performance reports
"""

from datetime import datetime, timedelta
from typing import Any
import json
import uuid

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule


# Default DAG arguments
default_args = {
    "owner": "fraud_detection",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


def generate_batch_id(**context) -> str:
    """Generate unique batch ID for this run."""
    execution_date = context["execution_date"]
    batch_id = f"batch_{execution_date.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    context["ti"].xcom_push(key="batch_id", value=batch_id)
    return batch_id


def extract_transactions(**context) -> dict[str, Any]:
    """Extract transactions for scoring."""
    from fraud_detection.utils.config import get_settings
    from fraud_detection.utils.database import DatabaseManager
    from fraud_detection.data.synthetic import SyntheticTransactionGenerator
    from fraud_detection.monitoring.audit import AuditLogger
    
    batch_id = context["ti"].xcom_pull(key="batch_id")
    settings = get_settings()
    
    db = DatabaseManager(settings)
    audit = AuditLogger(db)
    start_time = audit.log_start(batch_id, "EXTRACT")
    
    try:
        if settings.use_synthetic_data:
            # Generate synthetic data
            generator = SyntheticTransactionGenerator(settings, seed=42)
            df = generator.generate(n_transactions=settings.synthetic_rows)
            
            # Store in database
            db.write_dataframe(df, "transactions", schema="fraud", if_exists="append")
        else:
            # Extract unscored transactions from database
            query = """
                SELECT * FROM fraud.extract_unscored_batch(:batch_size)
            """
            df = db.read_sql(query, {"batch_size": 10000})
        
        # Save to temp location for next task
        temp_path = f"/tmp/fraud_batch_{batch_id}.parquet"
        df.to_parquet(temp_path)
        
        audit.log_complete(
            batch_id, "EXTRACT", start_time,
            records_processed=len(df),
            metadata={"source": "synthetic" if settings.use_synthetic_data else "database"}
        )
        
        return {"rows": len(df), "path": temp_path}
        
    except Exception as e:
        audit.log_failure(batch_id, "EXTRACT", start_time, e)
        raise


def validate_transactions(**context) -> dict[str, Any]:
    """Validate transaction data quality."""
    from fraud_detection.validation.expectations import TransactionValidator
    from fraud_detection.monitoring.audit import AuditLogger
    from fraud_detection.utils.database import DatabaseManager
    import pandas as pd
    
    batch_id = context["ti"].xcom_pull(key="batch_id")
    extract_result = context["ti"].xcom_pull(task_ids="extract_transactions")
    
    db = DatabaseManager()
    audit = AuditLogger(db)
    start_time = audit.log_start(batch_id, "VALIDATE")
    
    try:
        # Load data
        df = pd.read_parquet(extract_result["path"])
        
        # Run validation
        validator = TransactionValidator()
        results = validator.validate(df, batch_id, strict=False)
        
        audit.log_complete(
            batch_id, "VALIDATE", start_time,
            records_processed=len(df),
            metadata={"success": results["success"], "failed": len(results["failed_expectations"])}
        )
        
        return results
        
    except Exception as e:
        audit.log_failure(batch_id, "VALIDATE", start_time, e)
        raise


def engineer_features(**context) -> dict[str, Any]:
    """Engineer features from raw transactions."""
    from fraud_detection.features.engineering import FeatureEngineer
    from fraud_detection.monitoring.audit import AuditLogger
    from fraud_detection.utils.database import DatabaseManager
    import pandas as pd
    
    batch_id = context["ti"].xcom_pull(key="batch_id")
    extract_result = context["ti"].xcom_pull(task_ids="extract_transactions")
    
    db = DatabaseManager()
    audit = AuditLogger(db)
    start_time = audit.log_start(batch_id, "TRANSFORM")
    
    try:
        # Load data
        df = pd.read_parquet(extract_result["path"])
        
        # Engineer features
        engineer = FeatureEngineer()
        features = engineer.fit_transform(df)
        
        # Save features with original data
        result_df = pd.concat([
            df[["transaction_id", "timestamp", "amount", "is_fraud"]],
            features
        ], axis=1)
        
        features_path = f"/tmp/fraud_features_{batch_id}.parquet"
        result_df.to_parquet(features_path)
        
        audit.log_complete(
            batch_id, "TRANSFORM", start_time,
            records_processed=len(df),
            metadata={"n_features": len(features.columns)}
        )
        
        return {"rows": len(df), "n_features": len(features.columns), "path": features_path}
        
    except Exception as e:
        audit.log_failure(batch_id, "TRANSFORM", start_time, e)
        raise


def score_transactions(**context) -> dict[str, Any]:
    """Score transactions with ensemble models."""
    from fraud_detection.models.ensemble import EnsembleScorer
    from fraud_detection.monitoring.audit import AuditLogger
    from fraud_detection.utils.database import DatabaseManager
    from fraud_detection.utils.config import get_settings
    import pandas as pd
    
    batch_id = context["ti"].xcom_pull(key="batch_id")
    features_result = context["ti"].xcom_pull(task_ids="engineer_features")
    
    settings = get_settings()
    db = DatabaseManager()
    audit = AuditLogger(db)
    start_time = audit.log_start(batch_id, "SCORE")
    
    try:
        # Load features
        df = pd.read_parquet(features_result["path"])
        
        # Get feature columns (exclude metadata)
        feature_cols = [c for c in df.columns if c not in ["transaction_id", "timestamp", "is_fraud"]]
        X = df[feature_cols]
        
        # Load or fit ensemble
        model_path = settings.models_dir / "ensemble"
        if (model_path / "ensemble_config.json").exists():
            scorer = EnsembleScorer.load(model_path)
        else:
            # Fit on this batch (for initial run)
            scorer = EnsembleScorer(settings=settings)
            scorer.fit(X)
            model_path.mkdir(parents=True, exist_ok=True)
            scorer.save(model_path)
        
        # Score batch
        results = scorer.score_batch(X, batch_id=batch_id)
        
        # Combine with transaction IDs
        scores_df = results.to_dataframe()
        scores_df["transaction_id"] = df["transaction_id"].values
        scores_df["batch_id"] = batch_id
        
        scores_path = f"/tmp/fraud_scores_{batch_id}.parquet"
        scores_df.to_parquet(scores_path)
        
        audit.log_complete(
            batch_id, "SCORE", start_time,
            records_processed=len(df),
            records_flagged=int(results.is_flagged.sum()),
            metadata=results.stats
        )
        
        return {
            "rows": len(df),
            "flagged": int(results.is_flagged.sum()),
            "path": scores_path,
            "stats": results.stats
        }
        
    except Exception as e:
        audit.log_failure(batch_id, "SCORE", start_time, e)
        raise


def persist_results(**context) -> dict[str, Any]:
    """Persist scoring results to database."""
    from fraud_detection.utils.database import DatabaseManager
    from fraud_detection.monitoring.audit import AuditLogger
    import pandas as pd
    
    batch_id = context["ti"].xcom_pull(key="batch_id")
    score_result = context["ti"].xcom_pull(task_ids="score_transactions")
    
    db = DatabaseManager()
    audit = AuditLogger(db)
    start_time = audit.log_start(batch_id, "PERSIST")
    
    try:
        # Load scores
        df = pd.read_parquet(score_result["path"])
        
        # Write to database
        db.write_dataframe(
            df,
            "scoring_results",
            schema="fraud",
            if_exists="append"
        )
        
        audit.log_complete(
            batch_id, "PERSIST", start_time,
            records_processed=len(df),
            metadata={"table": "fraud.scoring_results"}
        )
        
        return {"rows_written": len(df)}
        
    except Exception as e:
        audit.log_failure(batch_id, "PERSIST", start_time, e)
        raise


def generate_alerts(**context) -> dict[str, Any]:
    """Generate alerts for high-risk transactions."""
    from fraud_detection.monitoring.audit import AuditLogger
    from fraud_detection.utils.database import DatabaseManager
    from fraud_detection.utils.logging import get_logger
    import pandas as pd
    
    logger = get_logger(__name__)
    
    batch_id = context["ti"].xcom_pull(key="batch_id")
    score_result = context["ti"].xcom_pull(task_ids="score_transactions")
    
    db = DatabaseManager()
    audit = AuditLogger(db)
    start_time = audit.log_start(batch_id, "ALERT")
    
    try:
        # Load scores
        df = pd.read_parquet(score_result["path"])
        
        # Filter high-risk (score > 0.8)
        high_risk = df[df["ensemble_score"] > 0.8]
        
        alerts_generated = 0
        for _, row in high_risk.iterrows():
            # Log alert (in production, would send to alert system)
            logger.warning(
                "HIGH RISK TRANSACTION",
                transaction_id=row["transaction_id"],
                score=row["ensemble_score"],
                reasons=row.get("reason_codes", []),
            )
            alerts_generated += 1
        
        audit.log_complete(
            batch_id, "ALERT", start_time,
            records_processed=len(high_risk),
            metadata={"alerts_generated": alerts_generated}
        )
        
        return {"alerts_generated": alerts_generated}
        
    except Exception as e:
        audit.log_failure(batch_id, "ALERT", start_time, e)
        raise


def generate_drift_report(**context) -> dict[str, Any]:
    """Generate drift monitoring report."""
    from fraud_detection.monitoring.drift import DriftMonitor
    from fraud_detection.utils.config import get_settings
    import pandas as pd
    
    batch_id = context["ti"].xcom_pull(key="batch_id")
    features_result = context["ti"].xcom_pull(task_ids="engineer_features")
    
    settings = get_settings()
    
    # Load current features
    df = pd.read_parquet(features_result["path"])
    feature_cols = [c for c in df.columns if c not in ["transaction_id", "timestamp", "is_fraud"]]
    
    # Initialize monitor (would load reference from previous batch in production)
    monitor = DriftMonitor(settings=settings)
    
    # Detect drift
    drift_results = monitor.detect_drift(
        df[feature_cols],
        batch_id=batch_id,
        generate_report=True
    )
    
    return drift_results


def cleanup_temp_files(**context) -> None:
    """Clean up temporary files."""
    import os
    
    batch_id = context["ti"].xcom_pull(key="batch_id")
    
    temp_files = [
        f"/tmp/fraud_batch_{batch_id}.parquet",
        f"/tmp/fraud_features_{batch_id}.parquet",
        f"/tmp/fraud_scores_{batch_id}.parquet",
    ]
    
    for filepath in temp_files:
        if os.path.exists(filepath):
            os.remove(filepath)


# Create DAG
with DAG(
    dag_id="fraud_detection_pipeline",
    default_args=default_args,
    description="End-to-end fraud detection pipeline",
    schedule_interval="@hourly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["fraud", "ml", "production"],
    max_active_runs=1,
) as dag:
    
    # Task: Generate batch ID
    start = PythonOperator(
        task_id="generate_batch_id",
        python_callable=generate_batch_id,
    )
    
    # Task: Extract transactions
    extract = PythonOperator(
        task_id="extract_transactions",
        python_callable=extract_transactions,
    )
    
    # Task: Validate data
    validate = PythonOperator(
        task_id="validate_transactions",
        python_callable=validate_transactions,
    )
    
    # Task: Engineer features
    transform = PythonOperator(
        task_id="engineer_features",
        python_callable=engineer_features,
    )
    
    # Task: Score transactions
    score = PythonOperator(
        task_id="score_transactions",
        python_callable=score_transactions,
    )
    
    # Task: Persist results
    persist = PythonOperator(
        task_id="persist_results",
        python_callable=persist_results,
    )
    
    # Task: Generate alerts
    alert = PythonOperator(
        task_id="generate_alerts",
        python_callable=generate_alerts,
    )
    
    # Task: Generate drift report
    drift = PythonOperator(
        task_id="generate_drift_report",
        python_callable=generate_drift_report,
    )
    
    # Task: Cleanup
    cleanup = PythonOperator(
        task_id="cleanup_temp_files",
        python_callable=cleanup_temp_files,
        trigger_rule=TriggerRule.ALL_DONE,
    )
    
    # Task: End
    end = DummyOperator(
        task_id="end",
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )
    
    # Define task dependencies
    start >> extract >> validate >> transform >> score
    score >> [persist, alert, drift]
    [persist, alert, drift] >> cleanup >> end

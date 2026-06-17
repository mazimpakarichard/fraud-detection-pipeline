"""Data drift and model performance monitoring.

Uses Evidently AI for:
- Data drift detection
- Target drift (fraud rate changes)
- Model performance monitoring
- Feature distribution analysis

Generates HTML reports and JSON metrics.
"""

from datetime import datetime
from typing import Any

import pandas as pd

from fraud_detection.utils.config import Settings, get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy import for Evidently
try:
    from evidently import ColumnMapping
    from evidently.metrics import DataDriftTable, DatasetDriftMetric
    from evidently.report import Report
    from evidently.test_suite import TestSuite
    from evidently.tests import TestColumnDrift, TestShareOfDriftedColumns

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False


class DriftMonitor:
    """
    Monitor data drift and model performance.

    Compares current batch against reference distribution:
    - Feature drift (input distribution changes)
    - Target drift (fraud rate changes)
    - Model performance degradation
    """

    def __init__(
        self,
        reference_data: pd.DataFrame | None = None,
        settings: Settings | None = None,
    ) -> None:
        """
        Initialize drift monitor.

        Args:
            reference_data: Reference dataset for comparison.
            settings: Application settings.
        """
        self.settings = settings or get_settings()
        self.reference_data = reference_data
        self.reports_dir = self.settings.reports_dir

        # Feature column mapping
        self.numerical_features = [
            "amount",
            "amount_log",
            "hour_sin",
            "hour_cos",
            "card_hours_since_last",
            "card_txn_count_1h",
            "card_txn_count_24h",
            "card_amount_zscore_24h",
        ]
        self.categorical_features = [
            "is_weekend",
            "is_night",
            "suspicious_email",
            "country_mismatch",
            "high_risk_category",
            "is_online",
        ]

    def set_reference(self, reference_data: pd.DataFrame) -> None:
        """Set reference dataset for drift comparison."""
        self.reference_data = reference_data
        logger.info("Reference data set", n_samples=len(reference_data))

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        batch_id: str,
        generate_report: bool = True,
    ) -> dict[str, Any]:
        """
        Detect data drift between reference and current data.

        Args:
            current_data: Current batch to analyze.
            batch_id: Batch identifier.
            generate_report: Generate HTML report.

        Returns:
            Dictionary with drift metrics.
        """
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available, skipping drift detection")
            return {"drift_detected": False, "error": "evidently not installed"}

        if self.reference_data is None:
            logger.warning("No reference data set, using current data as reference")
            self.reference_data = current_data
            return {"drift_detected": False, "note": "reference data initialized"}

        logger.info(
            "Detecting drift",
            batch_id=batch_id,
            reference_size=len(self.reference_data),
            current_size=len(current_data),
        )

        # Filter to available features
        available_numerical = [f for f in self.numerical_features if f in current_data.columns]
        available_categorical = [f for f in self.categorical_features if f in current_data.columns]

        all_features = available_numerical + available_categorical

        # Subset data to common features
        ref_subset = self.reference_data[all_features].copy()
        cur_subset = current_data[all_features].copy()

        # Create column mapping
        column_mapping = ColumnMapping(
            numerical_features=available_numerical,
            categorical_features=available_categorical,
        )

        # Create drift report
        report = Report(
            metrics=[
                DatasetDriftMetric(),
                DataDriftTable(),
            ]
        )

        report.run(
            reference_data=ref_subset,
            current_data=cur_subset,
            column_mapping=column_mapping,
        )

        # Extract results
        results = report.as_dict()

        # Parse drift metrics
        dataset_drift = results["metrics"][0]["result"]
        drift_detected = dataset_drift.get("dataset_drift", False)
        drift_share = dataset_drift.get("share_of_drifted_columns", 0)

        drift_result = {
            "batch_id": batch_id,
            "timestamp": datetime.now().isoformat(),
            "drift_detected": drift_detected,
            "drift_share": drift_share,
            "n_drifted_columns": dataset_drift.get("number_of_drifted_columns", 0),
            "n_columns": dataset_drift.get("number_of_columns", len(all_features)),
            "drift_by_column": {},
        }

        # Extract per-column drift
        if len(results["metrics"]) > 1:
            drift_table = results["metrics"][1]["result"]
            if "drift_by_columns" in drift_table:
                for col_name, col_data in drift_table["drift_by_columns"].items():
                    drift_result["drift_by_column"][col_name] = {
                        "drift_detected": col_data.get("drift_detected", False),
                        "drift_score": col_data.get("drift_score", 0),
                        "stattest": col_data.get("stattest_name", "unknown"),
                    }

        # Generate report if requested
        if generate_report:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = self.reports_dir / f"drift_report_{batch_id}.html"
            report.save_html(str(report_path))
            drift_result["report_path"] = str(report_path)
            logger.info("Drift report saved", path=str(report_path))

        logger.info(
            "Drift detection complete",
            drift_detected=drift_detected,
            drift_share=f"{drift_share:.1%}",
        )

        return drift_result

    def run_drift_tests(
        self,
        current_data: pd.DataFrame,
        batch_id: str,
        drift_threshold: float = 0.3,
    ) -> dict[str, Any]:
        """
        Run automated drift tests.

        Args:
            current_data: Current batch.
            batch_id: Batch identifier.
            drift_threshold: Threshold for share of drifted columns.

        Returns:
            Test results dictionary.
        """
        if not EVIDENTLY_AVAILABLE:
            return {"success": True, "note": "evidently not installed, tests skipped"}

        if self.reference_data is None:
            return {"success": True, "note": "no reference data"}

        # Filter to available features
        available_numerical = [f for f in self.numerical_features if f in current_data.columns]
        available_categorical = [f for f in self.categorical_features if f in current_data.columns]
        all_features = available_numerical + available_categorical

        ref_subset = self.reference_data[all_features].copy()
        cur_subset = current_data[all_features].copy()

        column_mapping = ColumnMapping(
            numerical_features=available_numerical,
            categorical_features=available_categorical,
        )

        # Create test suite
        test_suite = TestSuite(
            tests=[
                TestShareOfDriftedColumns(lt=drift_threshold),
                *[
                    TestColumnDrift(column_name=col) for col in available_numerical[:5]
                ],  # Test top 5 numerical
            ]
        )

        test_suite.run(
            reference_data=ref_subset,
            current_data=cur_subset,
            column_mapping=column_mapping,
        )

        results = test_suite.as_dict()

        test_results = {
            "batch_id": batch_id,
            "timestamp": datetime.now().isoformat(),
            "success": results.get("summary", {}).get("all_passed", False),
            "tests": [],
        }

        for test in results.get("tests", []):
            test_results["tests"].append(
                {
                    "name": test.get("name", "unknown"),
                    "status": test.get("status", "unknown"),
                    "description": test.get("description", ""),
                }
            )

        return test_results

    def generate_performance_report(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame,
        batch_id: str,
    ) -> dict[str, Any]:
        """
        Generate model performance report.

        Args:
            predictions: Model predictions with scores.
            actuals: Actual fraud labels.
            batch_id: Batch identifier.

        Returns:
            Performance metrics.
        """
        # Merge predictions and actuals
        if "is_fraud" not in predictions.columns and "is_fraud" in actuals.columns:
            data = predictions.merge(
                actuals[["transaction_id", "is_fraud"]], on="transaction_id", how="left"
            )
        else:
            data = predictions.copy()

        if "is_fraud" not in data.columns:
            logger.warning("No ground truth labels available")
            return {"error": "no ground truth labels"}

        # Calculate metrics
        from sklearn.metrics import (
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_true = data["is_fraud"].astype(int)
        y_pred = data["is_flagged"].astype(int)
        y_score = data["ensemble_score"]

        # Handle edge cases
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            logger.warning("No class variation in ground truth")
            return {
                "batch_id": batch_id,
                "error": "no class variation",
                "fraud_count": int(y_true.sum()),
            }

        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "batch_id": batch_id,
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(data),
            "fraud_rate": float(y_true.mean()),
            "flag_rate": float(y_pred.mean()),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_score)),
            "pr_auc": float(average_precision_score(y_true, y_score)),
            "confusion_matrix": {
                "true_negatives": int(cm[0, 0]),
                "false_positives": int(cm[0, 1]),
                "false_negatives": int(cm[1, 0]),
                "true_positives": int(cm[1, 1]),
            },
        }

        logger.info(
            "Performance metrics calculated",
            batch_id=batch_id,
            precision=f"{metrics['precision']:.3f}",
            recall=f"{metrics['recall']:.3f}",
            roc_auc=f"{metrics['roc_auc']:.3f}",
        )

        return metrics

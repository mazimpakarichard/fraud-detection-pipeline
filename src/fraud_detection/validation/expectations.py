"""Data validation with Great Expectations.

Validates transaction data before processing:
- Required columns present
- Data types correct
- Values in expected ranges
- No critical nulls

Generates validation reports for audit.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_detection.utils.config import Settings, get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)

# Lazy import Great Expectations
try:
    import great_expectations as gx
    from great_expectations.core.expectation_configuration import ExpectationConfiguration
    GX_AVAILABLE = True
except ImportError:
    GX_AVAILABLE = False


class TransactionValidator:
    """
    Validate transaction data quality.
    
    Uses Great Expectations for:
    - Schema validation
    - Data quality checks
    - Business rule validation
    """
    
    REQUIRED_COLUMNS = [
        "transaction_id",
        "timestamp",
        "amount",
        "merchant_id",
        "card_id",
    ]
    
    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize validator."""
        self.settings = settings or get_settings()
        self.results_dir = self.settings.results_dir / "validation"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def validate(
        self,
        df: pd.DataFrame,
        batch_id: str,
        strict: bool = False,
    ) -> dict[str, Any]:
        """
        Validate transaction DataFrame.
        
        Args:
            df: Transaction data to validate.
            batch_id: Batch identifier.
            strict: Raise exception on validation failure.
            
        Returns:
            Validation results dictionary.
        """
        logger.info("Validating transactions", batch_id=batch_id, rows=len(df))
        
        results: dict[str, Any] = {
            "batch_id": batch_id,
            "timestamp": datetime.now().isoformat(),
            "n_rows": len(df),
            "success": True,
            "expectations": [],
            "failed_expectations": [],
        }
        
        # Check required columns
        missing_columns = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_columns:
            results["success"] = False
            results["failed_expectations"].append({
                "expectation": "columns_present",
                "missing": list(missing_columns),
            })
        
        # Use Great Expectations if available
        if GX_AVAILABLE:
            gx_results = self._validate_with_great_expectations(df, batch_id)
            results["expectations"].extend(gx_results.get("expectations", []))
            results["failed_expectations"].extend(gx_results.get("failed_expectations", []))
            results["success"] = results["success"] and gx_results.get("success", True)
        else:
            # Fallback to basic validation
            basic_results = self._basic_validation(df)
            results["expectations"].extend(basic_results.get("expectations", []))
            results["failed_expectations"].extend(basic_results.get("failed_expectations", []))
            results["success"] = results["success"] and basic_results.get("success", True)
        
        # Save results
        self._save_results(results)
        
        if strict and not results["success"]:
            raise ValueError(f"Validation failed: {results['failed_expectations']}")
        
        logger.info(
            "Validation complete",
            batch_id=batch_id,
            success=results["success"],
            n_failed=len(results["failed_expectations"]),
        )
        
        return results
    
    def _validate_with_great_expectations(
        self,
        df: pd.DataFrame,
        batch_id: str,
    ) -> dict[str, Any]:
        """Run Great Expectations validation."""
        context = gx.get_context()
        
        # Create expectations suite
        expectations = [
            # Transaction ID expectations
            ("expect_column_to_exist", {"column": "transaction_id"}),
            ("expect_column_values_to_not_be_null", {"column": "transaction_id"}),
            ("expect_column_values_to_be_unique", {"column": "transaction_id"}),
            
            # Amount expectations
            ("expect_column_to_exist", {"column": "amount"}),
            ("expect_column_values_to_not_be_null", {"column": "amount"}),
            ("expect_column_values_to_be_between", {
                "column": "amount", "min_value": 0, "max_value": 1000000
            }),
            
            # Timestamp expectations
            ("expect_column_to_exist", {"column": "timestamp"}),
            ("expect_column_values_to_not_be_null", {"column": "timestamp"}),
            
            # Card ID expectations
            ("expect_column_to_exist", {"column": "card_id"}),
            ("expect_column_values_to_not_be_null", {"column": "card_id"}),
            
            # Merchant ID expectations
            ("expect_column_to_exist", {"column": "merchant_id"}),
            ("expect_column_values_to_not_be_null", {"column": "merchant_id"}),
        ]
        
        # Create validator
        validator = context.sources.pandas_default.read_dataframe(df)
        
        results = {
            "success": True,
            "expectations": [],
            "failed_expectations": [],
        }
        
        for expectation_type, kwargs in expectations:
            try:
                method = getattr(validator, expectation_type)
                result = method(**kwargs)
                
                expectation_result = {
                    "expectation": expectation_type,
                    "kwargs": kwargs,
                    "success": result.success,
                }
                results["expectations"].append(expectation_result)
                
                if not result.success:
                    results["success"] = False
                    results["failed_expectations"].append(expectation_result)
                    
            except Exception as e:
                logger.warning(f"Expectation {expectation_type} failed: {e}")
        
        return results
    
    def _basic_validation(self, df: pd.DataFrame) -> dict[str, Any]:
        """Basic validation without Great Expectations."""
        results = {
            "success": True,
            "expectations": [],
            "failed_expectations": [],
        }
        
        # Check for required columns
        for col in self.REQUIRED_COLUMNS:
            expectation = {"expectation": "column_exists", "column": col}
            if col in df.columns:
                expectation["success"] = True
            else:
                expectation["success"] = False
                results["success"] = False
                results["failed_expectations"].append(expectation)
            results["expectations"].append(expectation)
        
        # Check for null transaction_ids
        if "transaction_id" in df.columns:
            null_count = df["transaction_id"].isna().sum()
            expectation = {
                "expectation": "no_null_transaction_ids",
                "null_count": int(null_count),
                "success": null_count == 0,
            }
            results["expectations"].append(expectation)
            if not expectation["success"]:
                results["success"] = False
                results["failed_expectations"].append(expectation)
        
        # Check for duplicate transaction_ids
        if "transaction_id" in df.columns:
            duplicate_count = df["transaction_id"].duplicated().sum()
            expectation = {
                "expectation": "unique_transaction_ids",
                "duplicate_count": int(duplicate_count),
                "success": duplicate_count == 0,
            }
            results["expectations"].append(expectation)
            if not expectation["success"]:
                results["success"] = False
                results["failed_expectations"].append(expectation)
        
        # Check amount range
        if "amount" in df.columns:
            invalid_amounts = ((df["amount"] < 0) | (df["amount"] > 1000000)).sum()
            expectation = {
                "expectation": "valid_amount_range",
                "invalid_count": int(invalid_amounts),
                "success": invalid_amounts == 0,
            }
            results["expectations"].append(expectation)
            if not expectation["success"]:
                results["success"] = False
                results["failed_expectations"].append(expectation)
        
        # Check null amounts
        if "amount" in df.columns:
            null_amounts = df["amount"].isna().sum()
            expectation = {
                "expectation": "no_null_amounts",
                "null_count": int(null_amounts),
                "success": null_amounts == 0,
            }
            results["expectations"].append(expectation)
            if not expectation["success"]:
                results["success"] = False
                results["failed_expectations"].append(expectation)
        
        return results
    
    def _save_results(self, results: dict[str, Any]) -> None:
        """Save validation results to file."""
        import json
        
        filepath = self.results_dir / f"validation_{results['batch_id']}.json"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.debug("Validation results saved", path=str(filepath))
    
    def get_validation_summary(self, batch_ids: list[str] | None = None) -> pd.DataFrame:
        """Get summary of validation results."""
        import json
        
        records = []
        for filepath in self.results_dir.glob("validation_*.json"):
            with open(filepath) as f:
                result = json.load(f)
            
            if batch_ids is None or result["batch_id"] in batch_ids:
                records.append({
                    "batch_id": result["batch_id"],
                    "timestamp": result["timestamp"],
                    "n_rows": result["n_rows"],
                    "success": result["success"],
                    "n_expectations": len(result["expectations"]),
                    "n_failed": len(result["failed_expectations"]),
                })
        
        return pd.DataFrame(records)

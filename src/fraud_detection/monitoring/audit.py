"""Audit logging for pipeline operations.

Provides immutable audit trail for:
- Data extraction
- Validation results
- Scoring operations
- Alert generation

All operations are logged to database for compliance.
"""

from datetime import datetime
from typing import Any

from fraud_detection.utils.database import DatabaseManager
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


class AuditLogger:
    """
    Immutable audit logging for pipeline operations.
    
    Logs all operations to fraud.audit_log table for:
    - Regulatory compliance
    - Debugging and troubleshooting
    - Performance monitoring
    """
    
    OPERATIONS = {
        "EXTRACT": "Data extraction from source",
        "VALIDATE": "Data validation with Great Expectations",
        "TRANSFORM": "Feature engineering",
        "SCORE": "Model scoring",
        "PERSIST": "Results persistence to database",
        "ALERT": "Alert generation",
    }
    
    STATUSES = {"STARTED", "COMPLETED", "FAILED"}
    
    def __init__(self, db_manager: DatabaseManager | None = None) -> None:
        """Initialize audit logger."""
        self.db_manager = db_manager or DatabaseManager()
    
    def log(
        self,
        batch_id: str,
        operation: str,
        status: str,
        records_processed: int | None = None,
        records_flagged: int | None = None,
        duration_seconds: float | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
        triggered_by: str = "pipeline",
        source_system: str = "fraud_detection",
    ) -> None:
        """
        Log an operation to audit table.
        
        Args:
            batch_id: Batch identifier.
            operation: Operation type (EXTRACT, VALIDATE, etc.).
            status: Operation status (STARTED, COMPLETED, FAILED).
            records_processed: Number of records processed.
            records_flagged: Number of records flagged.
            duration_seconds: Operation duration.
            error_message: Error details if failed.
            metadata: Additional metadata.
            triggered_by: User or system that triggered.
            source_system: Source system name.
        """
        if operation not in self.OPERATIONS:
            logger.warning(f"Unknown operation type: {operation}")
        
        if status not in self.STATUSES:
            logger.warning(f"Unknown status: {status}")
        
        import json
        metadata_json = json.dumps(metadata) if metadata else None
        
        sql = """
            INSERT INTO fraud.audit_log (
                batch_id, operation, status, records_processed, records_flagged,
                duration_seconds, error_message, metadata, triggered_by, source_system
            ) VALUES (
                :batch_id, :operation, :status, :records_processed, :records_flagged,
                :duration_seconds, :error_message, :metadata::jsonb, :triggered_by, :source_system
            )
        """
        
        params = {
            "batch_id": batch_id,
            "operation": operation,
            "status": status,
            "records_processed": records_processed,
            "records_flagged": records_flagged,
            "duration_seconds": duration_seconds,
            "error_message": error_message,
            "metadata": metadata_json,
            "triggered_by": triggered_by,
            "source_system": source_system,
        }
        
        try:
            self.db_manager.execute_sql(sql, params)
            logger.info(
                "Audit log entry created",
                batch_id=batch_id,
                operation=operation,
                status=status,
            )
        except Exception as e:
            # Log to file if database unavailable
            logger.error(
                "Failed to write audit log to database",
                error=str(e),
                batch_id=batch_id,
                operation=operation,
            )
    
    def log_start(
        self,
        batch_id: str,
        operation: str,
        metadata: dict[str, Any] | None = None,
    ) -> datetime:
        """Log operation start and return timestamp."""
        start_time = datetime.now()
        self.log(
            batch_id=batch_id,
            operation=operation,
            status="STARTED",
            metadata=metadata,
        )
        return start_time
    
    def log_complete(
        self,
        batch_id: str,
        operation: str,
        start_time: datetime,
        records_processed: int | None = None,
        records_flagged: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log operation completion with duration."""
        duration = (datetime.now() - start_time).total_seconds()
        self.log(
            batch_id=batch_id,
            operation=operation,
            status="COMPLETED",
            records_processed=records_processed,
            records_flagged=records_flagged,
            duration_seconds=duration,
            metadata=metadata,
        )
    
    def log_failure(
        self,
        batch_id: str,
        operation: str,
        start_time: datetime,
        error: Exception,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log operation failure."""
        duration = (datetime.now() - start_time).total_seconds()
        self.log(
            batch_id=batch_id,
            operation=operation,
            status="FAILED",
            duration_seconds=duration,
            error_message=str(error),
            metadata=metadata,
        )
    
    def get_batch_history(self, batch_id: str) -> list[dict[str, Any]]:
        """Get all audit entries for a batch."""
        sql = """
            SELECT *
            FROM fraud.audit_log
            WHERE batch_id = :batch_id
            ORDER BY timestamp
        """
        
        df = self.db_manager.read_sql(sql, {"batch_id": batch_id})
        return df.to_dict("records")
    
    def get_recent_failures(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get recent failed operations."""
        sql = """
            SELECT *
            FROM fraud.audit_log
            WHERE status = 'FAILED'
              AND timestamp >= NOW() - INTERVAL ':hours hours'
            ORDER BY timestamp DESC
        """
        
        df = self.db_manager.read_sql(sql, {"hours": hours})
        return df.to_dict("records")

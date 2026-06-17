"""Database connection management with connection pooling."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from fraud_detection.utils.config import Settings, get_settings
from fraud_detection.utils.logging import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """
    PostgreSQL database connection manager.

    Provides:
    - Connection pooling
    - Context managers for sessions
    - Efficient bulk operations
    - Schema management
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize database manager."""
        self.settings = settings or get_settings()
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None

    @property
    def engine(self) -> Engine:
        """Get or create SQLAlchemy engine with connection pooling."""
        if self._engine is None:
            self._engine = create_engine(
                self.settings.database_url,
                pool_size=self.settings.db_pool_size,
                max_overflow=self.settings.db_max_overflow,
                pool_pre_ping=True,  # Verify connections before use
                echo=self.settings.debug,
            )
            logger.info(
                "Database engine created",
                url=self.settings.database_url_masked,
                pool_size=self.settings.db_pool_size,
            )
        return self._engine

    @property
    def session_factory(self) -> sessionmaker[Session]:
        """Get session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions with auto-commit/rollback."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def connection(self) -> Generator[Any, None, None]:
        """Context manager for raw database connections."""
        conn = self.engine.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute_sql(self, sql: str, params: dict[str, Any] | None = None) -> None:
        """Execute a SQL statement."""
        with self.connection() as conn:
            conn.execute(text(sql), params or {})

    def execute_sql_file(self, file_path: str) -> None:
        """Execute SQL from a file."""
        with open(file_path) as f:
            sql = f.read()

        # Split on semicolons for multiple statements
        statements = [s.strip() for s in sql.split(";") if s.strip()]

        with self.connection() as conn:
            for statement in statements:
                if statement:
                    conn.execute(text(statement))

        logger.info("Executed SQL file", path=file_path, statements=len(statements))

    def read_sql(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        chunksize: int | None = None,
    ) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
        """
        Read SQL query into DataFrame.

        Args:
            query: SQL query string.
            params: Query parameters.
            chunksize: If provided, return iterator of DataFrames.

        Returns:
            DataFrame or iterator of DataFrames.
        """
        return pd.read_sql(query, self.engine, params=params, chunksize=chunksize)

    def write_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: str | None = None,
        if_exists: str = "append",
        index: bool = False,
        chunksize: int = 10000,
    ) -> int:
        """
        Write DataFrame to database table efficiently.

        Args:
            df: DataFrame to write.
            table_name: Target table name.
            schema: Database schema.
            if_exists: How to handle existing table ('fail', 'replace', 'append').
            index: Write DataFrame index.
            chunksize: Rows per batch for bulk insert.

        Returns:
            Number of rows written.
        """
        schema = schema or self.settings.db_schema

        df.to_sql(
            table_name,
            self.engine,
            schema=schema,
            if_exists=if_exists,
            index=index,
            chunksize=chunksize,
            method="multi",
        )

        logger.info(
            "DataFrame written to database",
            table=f"{schema}.{table_name}",
            rows=len(df),
        )
        return len(df)

    def table_exists(self, table_name: str, schema: str | None = None) -> bool:
        """Check if a table exists."""
        schema = schema or self.settings.db_schema
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = :schema AND table_name = :table
            )
        """
        with self.connection() as conn:
            result = conn.execute(text(query), {"schema": schema, "table": table_name})
            return bool(result.scalar())

    def get_row_count(self, table_name: str, schema: str | None = None) -> int:
        """Get approximate row count for a table."""
        schema = schema or self.settings.db_schema
        query = f"SELECT COUNT(*) FROM {schema}.{table_name}"
        with self.connection() as conn:
            result = conn.execute(text(query))
            return int(result.scalar() or 0)

    def close(self) -> None:
        """Close database connections."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database connections closed")

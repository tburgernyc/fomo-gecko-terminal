from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Iterator

import duckdb

from src.config import Settings, get_settings


class Database:
    """Owns the DuckDB connection and serializes transaction scopes."""

    def __init__(self, path: str, schema_path: str):
        self._lock = RLock()
        self._connection = duckdb.connect(path)
        schema_file = Path(schema_path)
        if not schema_file.is_file():
            self._connection.close()
            raise FileNotFoundError(f"DuckDB schema not found: {schema_file}")
        try:
            with self.transaction() as cursor:
                cursor.execute(schema_file.read_text(encoding="utf-8"))
        except Exception:
            self._connection.close()
            raise

    @contextmanager
    def cursor(self) -> Iterator[duckdb.DuckDBPyConnection]:
        with self._lock:
            cursor = self._connection.cursor()
            try:
                yield cursor
            finally:
                cursor.close()

    @contextmanager
    def transaction(self) -> Iterator[duckdb.DuckDBPyConnection]:
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("BEGIN TRANSACTION")
            try:
                yield cursor
            except Exception:
                cursor.execute("ROLLBACK")
                raise
            else:
                cursor.execute("COMMIT")
            finally:
                cursor.close()

    def close(self) -> None:
        with self._lock:
            self._connection.close()


@contextmanager
def cursor_scope(connection: Database | duckdb.DuckDBPyConnection) -> Iterator[duckdb.DuckDBPyConnection]:
    if isinstance(connection, Database):
        with connection.cursor() as cursor:
            yield cursor
        return
    cursor = connection.cursor()
    try:
        yield cursor
    finally:
        cursor.close()


@contextmanager
def transaction_scope(
    connection: Database | duckdb.DuckDBPyConnection,
) -> Iterator[duckdb.DuckDBPyConnection]:
    if isinstance(connection, Database):
        with connection.transaction() as cursor:
            yield cursor
        return
    cursor = connection.cursor()
    cursor.execute("BEGIN TRANSACTION")
    try:
        yield cursor
    except Exception:
        cursor.execute("ROLLBACK")
        raise
    else:
        cursor.execute("COMMIT")
    finally:
        cursor.close()


_database: Database | None = None
_database_lock = RLock()


def initialize_database(settings: Settings | None = None) -> Database:
    global _database
    configured = settings or get_settings()
    with _database_lock:
        if _database is None:
            _database = Database(configured.duckdb_path, configured.schema_path)
        return _database


def close_database() -> None:
    global _database
    with _database_lock:
        if _database is not None:
            _database.close()
            _database = None

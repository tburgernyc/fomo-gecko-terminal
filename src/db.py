from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
from threading import RLock
from typing import Iterator

import duckdb

from src.config import Settings, get_settings


class Database:
    """Owns the DuckDB connection and serializes transaction scopes."""

    def __init__(self, path: str, schema_path: str):
        self._lock = RLock()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(str(self._path))
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
    def get_readonly_snapshot_conn(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Yield an isolated read-only connection to an atomic database snapshot."""
        snapshot_path = self._path.with_suffix(".snapshot.duckdb")
        snapshot_temp: Path | None = None
        reader_path: Path | None = None
        connection: duckdb.DuckDBPyConnection | None = None
        try:
            with self._lock:
                self._connection.execute("CHECKPOINT")
                with tempfile.NamedTemporaryFile(
                    prefix=f".{snapshot_path.name}.",
                    suffix=".tmp",
                    dir=snapshot_path.parent,
                    delete=False,
                ) as temporary:
                    snapshot_temp = Path(temporary.name)
                shutil.copyfile(self._path, snapshot_temp)
                snapshot_temp.replace(snapshot_path)
                snapshot_temp = None

            with tempfile.NamedTemporaryFile(
                prefix=".fomo-reader.",
                suffix=".duckdb",
                dir=snapshot_path.parent,
                delete=False,
            ) as temporary:
                reader_path = Path(temporary.name)
            with snapshot_path.open("rb") as source, reader_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            connection = duckdb.connect(str(reader_path), read_only=True)
            yield connection
        finally:
            if connection is not None:
                connection.close()
            if snapshot_temp is not None:
                snapshot_temp.unlink(missing_ok=True)
            if reader_path is not None:
                reader_path.unlink(missing_ok=True)

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

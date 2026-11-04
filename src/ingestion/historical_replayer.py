"""Event-by-event historical market data replay backed by DuckDB and Parquet."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncIterator

import duckdb
import numpy as np

from src.config import Settings, get_settings
from src.quant.indicators import calculate_indicators

logger = logging.getLogger(__name__)
REQUIRED_COLUMNS = {
    "timestamp",
    "slot",
    "token_address",
    "signal_type",
    "price_usd",
    "reserve_in_usd",
    "volume_usd_h24",
    "fdv_usd",
}


class HistoricalReplayer:
    """Stream ordered events and expose only causal state to market evaluation."""

    def __init__(
        self,
        parquet_path: str | Path,
        connection: duckdb.DuckDBPyConnection | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.is_file():
            raise FileNotFoundError(f"historical Parquet dataset does not exist: {self.parquet_path}")
        self.connection = connection or duckdb.connect(":memory:")
        self._owns_connection = connection is None
        self.settings = settings or get_settings()
        self.current_event_id = 0
        self.current_slot = 0
        self.current_timestamp: Any = None

        self.connection.execute(
            """
            CREATE OR REPLACE TABLE historical_events AS
            SELECT *, row_number() OVER (
                ORDER BY timestamp, slot, token_address
            ) AS _event_id
            FROM read_parquet(?)
            """,
            [str(self.parquet_path.resolve())],
        )
        columns = {
            str(row[0]).lower()
            for row in self.connection.execute("DESCRIBE historical_events").fetchall()
        }
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"historical Parquet dataset is missing columns: {sorted(missing)}")
        count = int(self.connection.execute("SELECT COUNT(*) FROM historical_events").fetchone()[0])
        if count == 0:
            raise ValueError("historical Parquet dataset contains no events")
        logger.info("Loaded %d historical events from %s", count, self.parquet_path)

    async def stream_events(self) -> AsyncIterator[dict[str, Any]]:
        """Yield one chronologically ordered event at a time without materializing the file."""
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                SELECT _event_id, timestamp, slot, token_address, signal_type,
                       price_usd, reserve_in_usd, volume_usd_h24, fdv_usd
                FROM historical_events
                ORDER BY _event_id
                """
            )
            while row := cursor.fetchone():
                (
                    event_id,
                    timestamp,
                    slot,
                    token_address,
                    signal_type,
                    price,
                    reserve,
                    volume,
                    fdv,
                ) = row
                self.current_event_id = int(event_id)
                self.current_slot = int(slot)
                self.current_timestamp = timestamp
                normalized_type = str(signal_type).upper()
                event: dict[str, Any] = {
                    "timestamp": timestamp,
                    "slot": self.current_slot,
                    "token_address": str(token_address),
                    "signal_type": normalized_type,
                    "price_usd": float(price),
                    "reserve_in_usd": float(reserve),
                    "volume_usd_h24": float(volume),
                    "fdv_usd": float(fdv),
                }
                if normalized_type in {"FOMO_BUY", "FOMO_SELL"}:
                    event.update(
                        {
                            "userHandle": self._trader_for_token(str(token_address)),
                            "network": "solana",
                            "tokenAddress": str(token_address),
                            "side": "BUY" if normalized_type == "FOMO_BUY" else "SELL",
                            "sleeve_weight": 0.10,
                        }
                    )
                yield event
                await asyncio.sleep(0)
        finally:
            cursor.close()

    @staticmethod
    def _trader_for_token(token_address: str) -> str:
        """Assign a stable synthetic trader to each token so sells close its buys."""
        return f"historical_{token_address[-8:].lower()}"

    def get_pool_state_at_slot(
        self, token_address: str, target_slot: int
    ) -> dict[str, Any] | None:
        """Return the first observed pool state at or after a requested execution slot."""
        row = self.connection.execute(
            """
            SELECT timestamp, slot, price_usd, reserve_in_usd, volume_usd_h24, fdv_usd
            FROM historical_events
            WHERE token_address = ? AND slot >= ?
            ORDER BY slot, timestamp, _event_id
            LIMIT 1
            """,
            [token_address, int(target_slot)],
        ).fetchone()
        if row is None:
            return None
        timestamp, slot, price, reserve, volume, fdv = row
        return {
            "timestamp": timestamp,
            "slot": int(slot),
            "price_usd": float(price),
            "reserve_in_usd": float(reserve),
            "volume_usd_h24": float(volume),
            "fdv_usd": float(fdv),
        }

    def current_pool_states(self) -> dict[str, dict[str, Any]]:
        """Return the latest observed state for each token up to the replay cursor."""
        rows = self.connection.execute(
            """
            SELECT token_address, timestamp, slot, price_usd, reserve_in_usd,
                   volume_usd_h24, fdv_usd
            FROM historical_events
            WHERE _event_id <= ?
            QUALIFY row_number() OVER (
                PARTITION BY token_address ORDER BY _event_id DESC
            ) = 1
            """,
            [self.current_event_id],
        ).fetchall()
        return {
            str(token): {
                "timestamp": timestamp,
                "slot": int(slot),
                "price_usd": float(price),
                "reserve_in_usd": float(reserve),
                "volume_usd_h24": float(volume),
                "fdv_usd": float(fdv),
            }
            for token, timestamp, slot, price, reserve, volume, fdv in rows
        }

    async def evaluate_token(
        self, client: Any, network: str, token_address: str
    ) -> dict[str, Any]:
        """Provide gatekeeper market fields from history available at the current event."""
        if str(network).lower() not in {"solana", "sol", "1399811149"}:
            return {"passed": False, "reason": f"unsupported_network:{network}"}
        row = self.connection.execute(
            """
            SELECT timestamp, slot, price_usd, reserve_in_usd, volume_usd_h24, fdv_usd
            FROM historical_events
            WHERE token_address = ? AND _event_id <= ?
            ORDER BY _event_id DESC
            LIMIT 1
            """,
            [token_address, self.current_event_id],
        ).fetchone()
        if row is None:
            return {"passed": False, "reason": "no_historical_pool_state"}

        timestamp, slot, price, reserve, volume, fdv = row
        history = self.connection.execute(
            """
            SELECT price_usd
            FROM historical_events
            WHERE token_address = ? AND _event_id <= ?
            ORDER BY _event_id DESC
            LIMIT 100
            """,
            [token_address, self.current_event_id],
        ).fetchall()
        closes = np.asarray([float(item[0]) for item in reversed(history)], dtype=np.float64)
        indicators = calculate_indicators(closes)
        reserve_value = float(reserve)
        volume_value = float(volume)
        passed = (
            reserve_value >= self.settings.min_pool_liquidity_usd
            and volume_value >= self.settings.min_pool_volume_24h_usd
        )
        reason = "ok" if passed else (
            f"insufficient_liquidity:${reserve_value:.0f}"
            if reserve_value < self.settings.min_pool_liquidity_usd
            else f"insufficient_24h_vol:${volume_value:.0f}"
        )
        return {
            "network": "solana",
            "pool_address": f"historical-pool-{token_address}",
            "price_usd": float(price),
            "reserve_in_usd": reserve_value,
            "volume_in_usd": volume_value,
            "volume_usd_h24": volume_value,
            "volume_24h_usd": volume_value,
            "fdv_usd": float(fdv),
            "rsi_14": indicators["rsi_14"],
            "macd_hist": indicators["macd_hist"],
            "atr_14": indicators["atr_14"],
            "passed": passed,
            "reason": reason,
            "historical_timestamp": timestamp,
            "historical_slot": int(slot),
        }

    def close(self) -> None:
        """Close the owned DuckDB connection."""
        if self._owns_connection:
            self.connection.close()

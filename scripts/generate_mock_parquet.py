"""Generate a deterministic, event-level Solana pool dataset for backtesting."""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "mock_solana_pools.parquet"
TOKEN_ADDRESSES = (
    "MockSolanaMeme1111111111111111111111111111111",
    "MockSolanaMeme2222222222222222222222222222222",
    "MockSolanaMeme3333333333333333333333333333333",
)
EVENT_COUNT = 1_000
PERIOD = timedelta(hours=4)
SLOT_SECONDS = 0.4


def generate_mock_parquet(
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    seed: int = 20261103,
    row_count: int = EVENT_COUNT,
) -> Path:
    """Write chronological pool updates and social trade signals to Parquet."""
    if row_count <= 0:
        raise ValueError("row_count must be positive")

    rng = random.Random(seed)
    start = datetime.now(timezone.utc) - PERIOD
    offsets = sorted(rng.randrange(int(PERIOD.total_seconds() * 1_000)) / 1_000 for _ in range(row_count))
    prices = [rng.uniform(0.00001, 0.0001) for _ in TOKEN_ADDRESSES]
    reserves = [rng.uniform(40_000, 120_000) for _ in TOKEN_ADDRESSES]
    active_positions = [False for _ in TOKEN_ADDRESSES]
    rows: list[dict[str, object]] = []

    for offset in offsets:
        token_index = rng.randrange(len(TOKEN_ADDRESSES))
        token = TOKEN_ADDRESSES[token_index]
        timestamp = start + timedelta(seconds=offset)
        prices[token_index] *= 1 + rng.gauss(0, 0.045)
        if rng.random() < 0.04:
            prices[token_index] *= 1 + rng.choice((-1, 1)) * rng.uniform(0.08, 0.22)
        prices[token_index] = max(prices[token_index], 1e-8)
        reserves[token_index] = max(
            15_000.0,
            reserves[token_index] * (1 + rng.gauss(0, 0.035)),
        )

        signal_type = "POOL_UPDATE"
        if active_positions[token_index] and rng.random() < 0.24:
            signal_type = "FOMO_SELL"
            active_positions[token_index] = False
        elif not active_positions[token_index] and rng.random() < 0.24:
            signal_type = "FOMO_BUY"
            active_positions[token_index] = True

        fdv = rng.uniform(100_000, 900_000)
        volume = fdv * rng.uniform(0.55, 1.5)
        rows.append(
            {
                "timestamp": timestamp,
                "slot": int(offset / SLOT_SECONDS),
                "token_address": token,
                "signal_type": signal_type,
                "price_usd": prices[token_index],
                "reserve_in_usd": reserves[token_index],
                "volume_usd_h24": volume,
                "fdv_usd": fdv,
            }
        )

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["slot"] = frame["slot"].astype("int64")
    for column in ("token_address", "signal_type"):
        frame[column] = frame[column].astype("string")
    for column in ("price_usd", "reserve_in_usd", "volume_usd_h24", "fdv_usd"):
        frame[column] = frame[column].astype("float64")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, destination, compression="zstd")
    logger.info("Wrote %d historical events to %s", len(frame), destination)
    return destination


def main() -> None:
    """Generate the default mock dataset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generate_mock_parquet()


if __name__ == "__main__":
    main()

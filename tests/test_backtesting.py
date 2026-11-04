from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from src.execution.routers import OrderIntent
from src.execution.simulated_router import SimulatedOnChainRouter
from src.ingestion.historical_replayer import HistoricalReplayer
from scripts.generate_mock_parquet import generate_mock_parquet


TOKEN = "MockSolanaMeme1111111111111111111111111111111"


def _write_events(path: Path) -> Path:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    frame = pd.DataFrame(
        [
            {
                "timestamp": start + timedelta(seconds=index),
                "slot": slot,
                "token_address": TOKEN,
                "signal_type": signal,
                "price_usd": price,
                "reserve_in_usd": 50_000.0,
                "volume_usd_h24": 100_000.0,
                "fdv_usd": 200_000.0,
            }
            for index, (slot, signal, price) in enumerate(
                [(10, "FOMO_BUY", 0.001), (20, "POOL_UPDATE", 0.003), (30, "POOL_UPDATE", 0.004)]
            )
        ]
    )
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    return path


def test_mock_generator_writes_one_thousand_chronological_rows(tmp_path):
    output = generate_mock_parquet(tmp_path / "mock.parquet")

    frame = pd.read_parquet(output)
    assert len(frame) == 1_000
    assert list(frame.columns) == [
        "timestamp",
        "slot",
        "token_address",
        "signal_type",
        "price_usd",
        "reserve_in_usd",
        "volume_usd_h24",
        "fdv_usd",
    ]
    assert str(frame["timestamp"].dtype).endswith(", UTC]")
    assert frame["timestamp"].is_monotonic_increasing
    assert frame["token_address"].nunique() == 3
    assert {"POOL_UPDATE", "FOMO_BUY", "FOMO_SELL"} <= set(frame["signal_type"])


@pytest.mark.asyncio
async def test_replayer_streams_live_shaped_signals_and_peeks_at_future_slot(tmp_path):
    replayer = HistoricalReplayer(_write_events(tmp_path / "events.parquet"))
    try:
        iterator = replayer.stream_events()
        signal = await anext(iterator)

        assert signal["signal_type"] == "FOMO_BUY"
        assert signal["userHandle"]
        assert signal["network"] == "solana"
        assert signal["tokenAddress"] == TOKEN
        assert signal["side"] == "BUY"
        assert replayer.current_slot == 10
        assert replayer.get_pool_state_at_slot(TOKEN, 13)["slot"] == 20
        await iterator.aclose()
    finally:
        replayer.close()


@pytest.mark.asyncio
async def test_simulated_router_uses_latency_price_and_deducts_fees(tmp_path):
    replayer = HistoricalReplayer(_write_events(tmp_path / "events.parquet"))
    try:
        iterator = replayer.stream_events()
        await anext(iterator)
        router = SimulatedOnChainRouter(
            replayer,
            latency_slots=3,
            starting_balance_usd=1_000.0,
            seed=0,
        )
        intent = OrderIntent(
            client_order_id="test-order",
            trader_handle="test-trader",
            network="solana",
            token_address=TOKEN,
            side="BUY",
            notional_usd=100.0,
            price_usd=0.001,
            pool_reserve_usd=50_000.0,
        )

        result = await router.submit(intent)

        assert result.status == "OPEN"
        assert result.average_price_usd > 0.003
        assert result.fee_usd == pytest.approx(
            (0.001 + 0.00005 + router.attempts[0]["priority_fee_sol"]) * 150
        )
        assert router.cash_balance_usd == pytest.approx(900 - result.fee_usd)
        assert router.attempts[0]["execution_slot"] == 13
    finally:
        replayer.close()


@pytest.mark.asyncio
async def test_dropped_bundle_still_deducts_fixed_fees(tmp_path):
    replayer = HistoricalReplayer(_write_events(tmp_path / "events.parquet"))
    try:
        iterator = replayer.stream_events()
        await anext(iterator)
        router = SimulatedOnChainRouter(replayer, seed=4)
        intent = OrderIntent(
            client_order_id="dropped-order",
            trader_handle="test-trader",
            network="solana",
            token_address=TOKEN,
            side="BUY",
            notional_usd=100.0,
            price_usd=0.001,
            pool_reserve_usd=50_000.0,
        )

        with pytest.raises(RuntimeError, match="bundle dropped"):
            await router.submit(intent)

        assert router.total_fees_paid_usd == pytest.approx(
            (0.001 + 0.00005 + router.attempts[0]["priority_fee_sol"]) * 150
        )
        assert router.cash_balance_usd == pytest.approx(
            1_000 - router.total_fees_paid_usd
        )
        assert router.attempts[0]["status"] == "FAILED"
    finally:
        replayer.close()

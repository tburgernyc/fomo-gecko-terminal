from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import duckdb
import httpx
import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.db import Database
from src.execution.gatekeeper import ExecutionGatekeeper
from src.execution.routers import (
    JITO_TIP_ACCOUNTS,
    DirectOnChainRouter,
    DryRunRouter,
    build_jupiter_swap_payload,
    compute_fee_split,
    jitodontfront_instruction,
    select_jito_tip_account,
)
from src.ingestion.gecko_scanner import GeckoTerminalScanner
from src.main import monitor_open_positions
from src.quant.backtest_vbt import (
    constant_product_slippage,
    run_slippage_adjusted_backtest,
    sweep_exit_parameters,
    target_metrics_met,
)
from src.quant.indicators import evaluate_microstructure_alpha_gate


@pytest.fixture
def db() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    connection.execute(Path("schema.sql").read_text(encoding="utf-8"))
    return connection


def test_microstructure_gate_enforces_liquidity_fdv_flow_drift_and_wick_limits():
    base = {
        "fdv_usd": 200_000,
        "liquidity_usd": 25_000,
        "volume_24h_usd": 150_000,
        "buys_m5": 16,
        "sells_m5": 10,
        "candle_open": 1.0,
        "candle_high": 1.1,
        "candle_low": 0.9,
        "candle_close": 1.05,
        "leader_entry_price": 1.05,
    }

    assert evaluate_microstructure_alpha_gate(**base) == (True, "PASS")
    assert "LOW_LIQUIDITY" in evaluate_microstructure_alpha_gate(
        **{**base, "liquidity_usd": 24_999}
    )[1]
    assert "PRE_MIGRATION_DUMP_ZONE" in evaluate_microstructure_alpha_gate(
        **{**base, "fdv_usd": 50_000}
    )[1]
    assert "WEAK_FLOW" in evaluate_microstructure_alpha_gate(
        **{**base, "fdv_usd": 100_000, "buys_m5": 15, "sells_m5": 10}
    )[1]
    assert "LEADER_DRIFT" in evaluate_microstructure_alpha_gate(
        **{**base, "candle_close": 1.074, "candle_high": 1.08}
    )[1]
    assert "UPPER_WICK_DISTRIBUTION" in evaluate_microstructure_alpha_gate(
        **{**base, "candle_high": 1.2, "candle_close": 1.05}
    )[1]


def test_scanner_exposes_geckoterminal_five_minute_transaction_counts(monkeypatch):
    candles = [
        [index * 300, 1.0, 1.02, 0.98, 1.0, 100]
        for index in range(1, 51)
    ]
    responses = [
        {
            "data": [{
                "attributes": {
                    "address": "pool",
                    "reserve_in_usd": "50000",
                    "volume_usd": {"h24": "100000"},
                    "base_token_price_usd": "1",
                    "fdv_usd": "100000",
                    "transactions": {"m5": {"buys": "16", "sells": "10"}},
                }
            }]
        },
        {"data": {"attributes": {"ohlcv_list": list(reversed(candles))}}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses.pop(0), request=request)

    scanner = GeckoTerminalScanner(Settings())

    async def acquire() -> None:
        return None

    monkeypatch.setattr(scanner.limiter, "acquire", acquire)

    async def run() -> dict[str, Any]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await scanner.evaluate_token(client, "solana", "Token")

    import asyncio

    result = asyncio.run(run())
    assert result["buys_m5"] == 16
    assert result["sells_m5"] == 10
    assert result["order_flow_ratio"] == pytest.approx(1.6)


def test_jito_fee_split_tip_rotation_and_jupiter_payload(monkeypatch):
    assert compute_fee_split(150_000) == (105_000, 45_000)
    assert compute_fee_split(1_000) == (700, 1_000)
    monkeypatch.setattr("src.execution.routers.random.choice", lambda choices: choices[-1])
    assert select_jito_tip_account() == JITO_TIP_ACCOUNTS[-1]

    payload = build_jupiter_swap_payload({"outAmount": "42"}, "wallet")
    assert payload["quoteResponse"]["outAmount"] == "42"
    assert payload["prioritizationFeeLamports"]["priorityLevelWithMaxLamports"] == {
        "priorityLevel": "veryHigh",
        "maxLamports": 105_000,
    }
    protection_instruction = jitodontfront_instruction()
    assert str(protection_instruction.accounts[0].pubkey) == (
        "jitodontfront111111111111111111111111111111"
    )
    assert not protection_instruction.accounts[0].is_signer
    assert not protection_instruction.accounts[0].is_writable


def test_jito_fee_builder_rejects_invalid_budget():
    with pytest.raises(ValueError, match="positive"):
        compute_fee_split(0)


@pytest.mark.asyncio
async def test_jito_bundle_sends_base64_transaction_to_configured_endpoint():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"result": "bundle-id"}, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    router = DirectOnChainRouter(
        Settings(jito_block_engine_url="https://jito.example/bundles"), client
    )
    try:
        assert await router._send_jito_bundle(b"signed transaction") == "bundle-id"
        request = requests[0]
        assert str(request.url) == "https://jito.example/bundles"
        payload = json.loads(request.content)
        assert payload["method"] == "sendBundle"
        assert payload["params"][0][0] == "c2lnbmVkIHRyYW5zYWN0aW9u"
    finally:
        await client.aclose()


def test_database_readonly_snapshots_are_isolated_and_read_only(tmp_path):
    database = Database(str(tmp_path / "terminal.duckdb"), "schema.sql")
    try:
        with database.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO leaderboard_traders (user_id, user_handle)
                VALUES ('snapshot-a', 'before')
                """
            )
        with database.get_readonly_snapshot_conn() as snapshot:
            assert snapshot.execute(
                "SELECT user_handle FROM leaderboard_traders WHERE user_id = 'snapshot-a'"
            ).fetchone() == ("before",)
            with pytest.raises(duckdb.Error):
                snapshot.execute(
                    """
                    INSERT INTO leaderboard_traders (user_id, user_handle)
                    VALUES ('snapshot-b', 'blocked')
                    """
                )
        from src.reporting.terminal_ui import _read_state

        assert _read_state(database)["traders"][0]["handle"] == "before"
        with database.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO leaderboard_traders (user_id, user_handle)
                VALUES ('snapshot-b', 'after')
                """
            )
        with database.get_readonly_snapshot_conn() as snapshot:
            assert snapshot.execute(
                "SELECT COUNT(*) FROM leaderboard_traders"
            ).fetchone() == (2,)
    finally:
        database.close()


def test_gatekeeper_persists_microstructure_inputs_and_entry_atr(db):
    class _EntryMarket(_Market):
        async def evaluate_token(
            self, client: httpx.AsyncClient, network: str, token: str
        ) -> dict[str, Any]:
            return {
                "passed": True,
                "network": network,
                "pool_address": "pool",
                "reserve_in_usd": 50_000.0,
                "volume_24h_usd": 100_000.0,
                "fdv_usd": 200_000.0,
                "buys_m5": 20,
                "sells_m5": 10,
                "price_usd": 1.0,
                "rsi_14": 50.0,
                "macd_hist": 0.02,
                "atr_14": 0.01,
            }

    import asyncio

    gatekeeper = ExecutionGatekeeper(
        db, _EntryMarket(1.0), DryRunRouter(), Settings(starting_bankroll_usd=10_000)
    )

    async def submit() -> str:
        async with httpx.AsyncClient() as client:
            return await gatekeeper.process_copy_signal(
                client, "trader", "solana", "Token", "BUY"
            )

    assert asyncio.run(submit()).startswith("EXECUTED:")
    assert db.execute(
        """
        SELECT fdv_usd, buys_m5, sells_m5, rsi_14, macd_hist, atr_14
        FROM market_snapshots
        """
    ).fetchone() == pytest.approx((200_000, 20, 10, 50, 0.02, 0.01))
    assert db.execute(
        "SELECT entry_atr_usd, exit_campaign_id FROM execution_orders WHERE side = 'BUY'"
    ).fetchone()[0] == pytest.approx(0.01)


def test_slippage_backtest_models_nonlinear_amm_impact_and_targets():
    impact = constant_product_slippage(50_000, 100, pd.Index(range(3)))
    assert impact.iloc[0] == pytest.approx((1 + 2 * 0.002) ** 2 - 1)
    with pytest.raises(ValueError, match="one side"):
        constant_product_slippage(100, 50, pd.Index(range(3)))

    index = pd.date_range("2026-01-01", periods=30, freq="400ms")
    close = pd.Series(np.linspace(1.0, 1.2, len(index)), index=index)
    entries = pd.Series(False, index=index)
    exits = pd.Series(False, index=index)
    entries.iloc[1] = True
    exits.iloc[10] = True
    result = run_slippage_adjusted_backtest(
        close, entries, exits, 50_000, 100, latency_slots=1
    )
    assert result["metrics"]["latency_seconds"] == pytest.approx(0.4)
    assert result["metrics"]["friction_per_leg"] == pytest.approx(0.008)
    assert result["metrics"]["mean_amm_slippage"] > 0
    assert target_metrics_met(1.46, -0.149)
    assert not target_metrics_met(1.45, -0.10)
    assert not target_metrics_met(2.0, -0.15)
    sweep = sweep_exit_parameters(
        close,
        entries,
        trailing_stops=(0.1,),
        time_stops=(5,),
        liquidity_usd=50_000,
        position_size_usd=100,
    )
    assert bool(sweep.iloc[0]["target_metrics_met"]) == target_metrics_met(
        float(sweep.iloc[0]["profit_factor"]), float(sweep.iloc[0]["max_drawdown"])
    )


class _Market:
    def __init__(self, price: float):
        self.price = price

    async def evaluate_token(
        self, client: httpx.AsyncClient, network: str, token: str
    ) -> dict[str, Any]:
        return {
            "passed": True,
            "reason": "ok",
            "network": network,
            "pool_address": "pool",
            "reserve_in_usd": 50_000.0,
            "volume_24h_usd": 100_000.0,
            "price_usd": self.price,
            "rsi_14": 50.0,
            "atr_14": 0.1,
        }


def _seed_open_position(connection: duckdb.DuckDBPyConnection, **overrides: Any) -> None:
    values = {
        "client_order_id": "buy-1",
        "exit_campaign_id": "campaign-1",
        "trader_handle": "trader",
        "network": "solana",
        "token_address": "Token",
        "side": "BUY",
        "status": "DRY_RUN_OPEN",
        "requested_usd": 100.0,
        "filled_usd": 100.0,
        "filled_token_amount": 100.0,
        "remaining_token_amount": 100.0,
        "entry_price_usd": 1.0,
        "high_water_price_usd": 1.0,
        "entry_atr_usd": 0.1,
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        "tranche_1_sold_amount": 0.0,
        "tranche_2_sold_amount": 0.0,
    }
    values.update(overrides)
    connection.execute(
        """
        INSERT INTO execution_orders (
            client_order_id, exit_campaign_id, trader_handle, network, token_address,
            side, status, requested_usd, filled_usd, filled_token_amount,
            remaining_token_amount, entry_price_usd, high_water_price_usd, entry_atr_usd,
            created_at, tranche_1_sold_amount, tranche_2_sold_amount
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        list(values.values()),
    )


@pytest.mark.asyncio
async def test_position_exit_engine_sells_first_40_percent_tranche(db):
    _seed_open_position(db)
    settings = Settings(starting_bankroll_usd=10_000)
    gatekeeper = ExecutionGatekeeper(db, _Market(1.65), DryRunRouter(), settings)
    async with httpx.AsyncClient() as client:
        await monitor_open_positions(
            client, db, _Market(1.65), gatekeeper, settings
        )

    buy = db.execute(
        """
        SELECT remaining_token_amount, tranche_1_sold_amount
        FROM execution_orders WHERE client_order_id = 'buy-1'
        """
    ).fetchone()
    sell = db.execute(
        "SELECT exit_tranche, filled_token_amount FROM execution_orders WHERE side = 'SELL'"
    ).fetchone()
    assert buy == pytest.approx((60.0, 40.0))
    assert sell[0] == "TAKE_PROFIT_1"
    assert sell[1] == pytest.approx(40.0)


@pytest.mark.asyncio
async def test_position_exit_engine_uses_atr_trail_and_twelve_minute_time_stop(db):
    _seed_open_position(
        db,
        tranche_1_sold_amount=40.0,
        tranche_2_sold_amount=40.0,
        remaining_token_amount=20.0,
        high_water_price_usd=2.5,
    )
    settings = Settings(starting_bankroll_usd=10_000)
    gatekeeper = ExecutionGatekeeper(db, _Market(2.2), DryRunRouter(), settings)
    async with httpx.AsyncClient() as client:
        await monitor_open_positions(client, db, _Market(2.2), gatekeeper, settings)
    trail = db.execute(
        "SELECT exit_tranche, filled_token_amount FROM execution_orders WHERE side = 'SELL'"
    ).fetchone()
    assert trail == ("TRAILING_STOP", pytest.approx(20.0))

    db.execute("DELETE FROM execution_orders")
    _seed_open_position(
        db,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=13),
    )
    time_stop_gatekeeper = ExecutionGatekeeper(
        db, _Market(1.05), DryRunRouter(), settings
    )
    async with httpx.AsyncClient() as client:
        await monitor_open_positions(
            client, db, _Market(1.05), time_stop_gatekeeper, settings
        )
    stop = db.execute(
        "SELECT exit_tranche, filled_token_amount FROM execution_orders WHERE side = 'SELL'"
    ).fetchone()
    assert stop == ("TIME_STOP", pytest.approx(100.0))

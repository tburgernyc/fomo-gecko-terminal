from pathlib import Path
import time
from typing import Any

import duckdb
import httpx
import pytest

from src.config import Settings
from src.execution.gatekeeper import (
    ExecutionGatekeeper,
    passes_memecoin_microstructure_gates,
)
from src.execution.routers import (
    DirectOnChainRouter,
    DryRunRouter,
    ExecutionResult,
    OrderIntent,
    SubmissionUncertain,
)
from src.ingestion.fomo_client import FomoLeaderboardClient
from src.ingestion.gecko_scanner import GeckoTerminalScanner
from src.main import build_router, sync_initial_top_10


class StubScanner:
    def __init__(self, response: dict[str, Any]):
        self.response = response
        self.calls = 0

    async def evaluate_token(self, client: httpx.AsyncClient, network: str, token: str) -> dict[str, Any]:
        self.calls += 1
        return self.response


class UnresolvedRouter(DryRunRouter):
    def __init__(self):
        self.reconciled: list[str] = []

    async def reconcile(self, client_order_id: str, tx_id: str | None) -> str:
        self.reconciled.append(client_order_id)
        return "UNCERTAIN"


@pytest.fixture
def db() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    connection.execute(Path("schema.sql").read_text(encoding="utf-8"))
    return connection


@pytest.fixture
def market() -> dict[str, Any]:
    return {
        "passed": True,
        "reason": "ok",
        "network": "solana",
        "pool_address": "pool_1",
        "reserve_in_usd": 50000.0,
        "volume_24h_usd": 120000.0,
        "price_usd": 1.25,
        "rsi_14": 54.0,
        "macd_hist": 0.02,
        "atr_14": 0.01,
    }


@pytest.mark.asyncio
async def test_idempotency_blocks_duplicate_orders(db, market):
    scanner = StubScanner(market)
    gatekeeper = ExecutionGatekeeper(
        db, scanner, DryRunRouter(), Settings(starting_bankroll_usd=10_000)
    )
    async with httpx.AsyncClient() as client:
        first = await gatekeeper.process_copy_signal(
            client, "whale_alpha", "solana", "TokenMint111", "BUY", 0.10
        )
        second = await gatekeeper.process_copy_signal(
            client, "whale_alpha", "solana", "TokenMint111", "BUY", 0.10
        )
    assert first.startswith("EXECUTED:DRY_RUN_OPEN:")
    assert second == "DUPLICATE_SKIPPED:DRY_RUN_OPEN"
    assert scanner.calls == 1


@pytest.mark.asyncio
async def test_chain_aliases_share_the_same_idempotency_key(db, market):
    scanner = StubScanner(market)
    gatekeeper = ExecutionGatekeeper(
        db, scanner, DryRunRouter(), Settings(starting_bankroll_usd=10_000)
    )
    async with httpx.AsyncClient() as client:
        first = await gatekeeper.process_copy_signal(
            client, "whale_alpha", "1399811149", "TokenMint111", "BUY", 0.10
        )
        second = await gatekeeper.process_copy_signal(
            client, "whale_alpha", "solana", "TokenMint111", "BUY", 0.10
        )
    assert first.startswith("EXECUTED:DRY_RUN_OPEN:")
    assert second == "DUPLICATE_SKIPPED:DRY_RUN_OPEN"
    assert scanner.calls == 1


@pytest.mark.asyncio
async def test_duplicate_uncertain_order_is_reconciled_not_resubmitted(db, market):
    bucket = int(time.time() // 300)
    order_id = ExecutionGatekeeper.order_id(
        "trader", "solana", "Token", "BUY", bucket
    )
    db.execute(
        """
        INSERT INTO execution_orders (
            client_order_id, trader_handle, network, token_address, side, status, requested_usd
        ) VALUES (?, 'trader', 'solana', 'Token', 'BUY', 'UNCERTAIN', 100)
        """,
        [order_id],
    )
    scanner = StubScanner(market)
    router = UnresolvedRouter()
    gatekeeper = ExecutionGatekeeper(db, scanner, router, Settings())
    async with httpx.AsyncClient() as client:
        result = await gatekeeper.process_copy_signal(
            client, "trader", "1399811149", "Token", "BUY", 0.10
        )
    assert result == "BLOCKED_UNRESOLVED_ORDER"
    assert router.reconciled == [order_id]
    assert scanner.calls == 0


@pytest.mark.asyncio
async def test_rejects_low_liquidity_and_overbought_rsi(db, market):
    scanner = StubScanner({**market, "passed": False, "reason": "insufficient_liquidity:$4500"})
    gatekeeper = ExecutionGatekeeper(
        db, scanner, DryRunRouter(), Settings(starting_bankroll_usd=10_000)
    )
    async with httpx.AsyncClient() as client:
        low_liquidity = await gatekeeper.process_copy_signal(
            client, "trader_1", "solana", "LowLiqToken", "BUY", 0.10
        )
    assert low_liquidity == "REJECTED:insufficient_liquidity:$4500"

    scanner.response = {**market, "rsi_14": 83.5}
    async with httpx.AsyncClient() as client:
        overbought = await gatekeeper.process_copy_signal(
            client, "trader_2", "solana", "OverboughtToken", "BUY", 0.10
        )
    assert overbought == "REJECTED:rsi_overbought:83.5"


@pytest.mark.asyncio
async def test_two_trader_consensus_allows_overbought_buy(db, market):
    scanner = StubScanner({**market, "rsi_14": 83.5})
    gatekeeper = ExecutionGatekeeper(
        db, scanner, DryRunRouter(), Settings(starting_bankroll_usd=10_000)
    )
    async with httpx.AsyncClient() as client:
        rejected = await gatekeeper.process_copy_signal(
            client, "trader_1", "solana", "HotToken", "BUY", 0.10
        )
        accepted = await gatekeeper.process_copy_signal(
            client, "trader_2", "solana", "HotToken", "BUY", 0.10
        )
    assert rejected == "REJECTED:rsi_overbought:83.5"
    assert accepted.startswith("EXECUTED:DRY_RUN_OPEN:")


def test_liquidity_clamp_and_fee_floor():
    from src.quant.allocator import PortfolioAllocator

    ticket = PortfolioAllocator.size_ticket_usd(
        bankroll_usd=10000, sleeve_weight=0.10, pool_liquidity_usd=15000
    )
    micro_ticket = PortfolioAllocator.size_ticket_usd(
        bankroll_usd=10000, sleeve_weight=0.10, pool_liquidity_usd=2000
    )
    assert ticket == 300
    assert micro_ticket == 0


def test_allocator_caps_each_sleeve_and_redistributes_available_weight():
    import numpy as np
    import pandas as pd

    from src.quant.allocator import PortfolioAllocator

    rng = np.random.default_rng(42)
    returns = pd.DataFrame(
        rng.normal(size=(60, 10)) * np.geomspace(0.1, 10, 10),
        columns=[f"trader-{index}" for index in range(10)],
    )
    weights = PortfolioAllocator.compute_sleeve_weights(returns)
    assert len(weights) == 10
    assert max(weights.values()) <= 0.15
    assert sum(weights.values()) == pytest.approx(1.0)


def test_live_router_requires_explicit_confirmation():
    settings = Settings(dry_run=False, live_trading_confirmed=False)
    with pytest.raises(RuntimeError, match="Live routing is locked"):
        build_router(settings, httpx.AsyncClient())


@pytest.mark.asyncio
async def test_direct_router_blocks_unsupported_network_before_market_scan(db, market):
    router = DirectOnChainRouter(Settings())
    scanner = StubScanner(market)
    gatekeeper = ExecutionGatekeeper(db, scanner, router, Settings())
    async with httpx.AsyncClient() as client:
        result = await gatekeeper.process_copy_signal(
            client, "trader", "143", "MonadToken", "BUY", 0.10
        )
    assert result == "REJECTED:unsupported_execution_network:monad"
    assert scanner.calls == 0
    assert router.supports_network("eth")
    assert router.supports_network("bsc")


@pytest.mark.asyncio
async def test_uncertain_evm_approval_is_tagged_as_non_order_transaction(monkeypatch):
    from eth_account import Account

    class FakeAccount:
        address = "0x0000000000000000000000000000000000000001"

    router = DirectOnChainRouter(
        Settings(
            dry_run=False,
            live_trading_confirmed=True,
            evm_private_key="unused",
            zerox_api_key="test-key",
        )
    )
    monkeypatch.setattr(Account, "from_key", lambda _: FakeAccount())

    async def token_decimals(token: str, rpc_url: str | None = None) -> int:
        return 6

    async def quote(url: str, *, headers: dict[str, str] | None = None) -> dict[str, Any]:
        return {
            "transaction": {"to": "0x0000000000000000000000000000000000000002"},
            "issues": {"allowance": {"spender": "0x0000000000000000000000000000000000000003"}},
        }

    async def allowance(*args: Any) -> str:
        return "0x0"

    async def broadcast(*args: Any, **kwargs: Any) -> str:
        raise SubmissionUncertain("approval broadcast uncertain", "0xapproval")

    monkeypatch.setattr(router, "_evm_token_decimals", token_decimals)
    monkeypatch.setattr(router, "_get_json", quote)
    monkeypatch.setattr(router, "_evm_rpc", allowance)
    monkeypatch.setattr(router, "_broadcast_evm_transaction", broadcast)
    intent = OrderIntent(
        "order", "trader", "base", "0x0000000000000000000000000000000000000004",
        "BUY", 100, 1, 50_000,
    )

    with pytest.raises(SubmissionUncertain) as caught:
        await router.submit(intent)
    assert caught.value.tx_id == "approval:8453:0xapproval"


@pytest.mark.asyncio
async def test_evm_reconciliation_uses_chain_rpc_and_never_opens_on_approval(monkeypatch):
    router = DirectOnChainRouter(Settings())
    requests: list[tuple[str, str]] = []

    async def rpc(url: str, method: str, params: list[Any]) -> dict[str, str]:
        requests.append((url, params[0]))
        return {"status": "0x1"}

    monkeypatch.setattr(router, "_evm_rpc", rpc)
    approval_state = await router.reconcile("order", "approval:1:0xapproval")
    swap_state = await router.reconcile("order", "eip155:1:0xswap")

    assert approval_state == "FAILED"
    assert swap_state == "OPEN"
    assert requests == [
        ("https://ethereum-rpc.publicnode.com", "0xapproval"),
        ("https://ethereum-rpc.publicnode.com", "0xswap"),
    ]


@pytest.mark.asyncio
async def test_direct_evm_routes_use_chain_specific_quotes_and_rpc(monkeypatch):
    from eth_account import Account

    class FakeAccount:
        address = "0x0000000000000000000000000000000000000001"

    router = DirectOnChainRouter(
        Settings(
            dry_run=False,
            live_trading_confirmed=True,
            evm_private_key="unused",
            zerox_api_key="test-key",
        )
    )
    monkeypatch.setattr(Account, "from_key", lambda _: FakeAccount())
    quoted: list[str] = []
    broadcast_rpc_urls: list[str] = []
    token_decimal_calls: list[tuple[str, str]] = []

    async def token_decimals(token: str, rpc_url: str | None = None) -> int:
        token_decimal_calls.append((token, str(rpc_url)))
        return 6

    async def quote(url: str, *, headers: dict[str, str] | None = None) -> dict[str, Any]:
        quoted.append(url)
        return {
            "transaction": {"to": "0x0000000000000000000000000000000000000002"},
            "buyAmount": "1000000",
        }

    async def broadcast(
        account: Any, transaction: dict[str, Any], *, rpc_url: str | None = None
    ) -> str:
        broadcast_rpc_urls.append(str(rpc_url))
        return "0xswap"

    monkeypatch.setattr(router, "_evm_token_decimals", token_decimals)
    monkeypatch.setattr(router, "_get_json", quote)
    monkeypatch.setattr(router, "_broadcast_evm_transaction", broadcast)
    intents = [
        OrderIntent("base", "trader", "base", "token", "BUY", 100, 1, 50_000),
        OrderIntent("eth", "trader", "eth", "token", "BUY", 100, 1, 50_000),
        OrderIntent("bsc", "trader", "bsc", "token", "BUY", 100, 1, 50_000),
    ]

    results = [await router.submit(intent) for intent in intents]

    assert [result.tx_id for result in results] == [
        "eip155:8453:0xswap", "eip155:1:0xswap", "eip155:56:0xswap",
    ]
    assert ["chainId=8453" in quoted[0], "chainId=1" in quoted[1], "chainId=56" in quoted[2]]
    assert broadcast_rpc_urls == [
        "https://mainnet.base.org",
        "https://ethereum-rpc.publicnode.com",
        "https://bsc-rpc.publicnode.com",
    ]
    assert len(token_decimal_calls) == 6


def test_fomo_client_loads_persisted_top_10_in_rank_order(db):
    db.executemany(
        """
        INSERT INTO leaderboard_traders (
            user_id, user_handle, rank_24h, sleeve_weight
        ) VALUES (?, ?, ?, ?)
        """,
        [
            ("id-2", "trader-2", 2, 0.12),
            ("id-1", "trader-1", 1, 0.11),
        ],
    )
    fomo = FomoLeaderboardClient(db, Settings())

    assert fomo.load_cached_top_10() == {"trader-1": 0.11, "trader-2": 0.12}


@pytest.mark.asyncio
async def test_initial_leaderboard_uses_persisted_top_10_when_api_fails():
    class CachedFomo:
        async def refresh_top_10_traders(self, client: httpx.AsyncClient) -> None:
            raise httpx.ConnectError("FOMO is unavailable")

        def load_cached_top_10(self) -> dict[str, float]:
            return {"persisted-trader": 0.1}

    async with httpx.AsyncClient() as client:
        await sync_initial_top_10(CachedFomo(), client)


@pytest.mark.asyncio
async def test_initial_leaderboard_still_fails_without_persisted_data():
    class EmptyFomo:
        async def refresh_top_10_traders(self, client: httpx.AsyncClient) -> None:
            raise httpx.ConnectError("FOMO is unavailable")

        def load_cached_top_10(self) -> dict[str, float]:
            return {}

    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.ConnectError, match="FOMO is unavailable"):
            await sync_initial_top_10(EmptyFomo(), client)


@pytest.mark.asyncio
async def test_gecko_scanner_parses_chronological_verified_candles(monkeypatch):
    candles = [
        [timestamp, 1 + index / 1000, 1.01 + index / 1000,
         0.99 + index / 1000, 1 + index / 1000, 100]
        for index, timestamp in enumerate(range(300, 300 * 51, 300))
    ]
    responses = [
        {
            "data": [{
                "attributes": {
                    "address": "pool-address",
                    "reserve_in_usd": "50000",
                    "volume_usd": {"h24": "100000"},
                    "base_token_price_usd": "1.05",
                }
            }]
        },
        {"data": {"attributes": {"ohlcv_list": list(reversed(candles))}}},
    ]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=responses.pop(0))

    scanner = GeckoTerminalScanner(Settings())

    async def acquire() -> None:
        return None

    monkeypatch.setattr(scanner.limiter, "acquire", acquire)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await scanner.evaluate_token(client, "8453", "Token Address")

    assert result["passed"] is True
    assert result["pool_address"] == "pool-address"
    assert len(result["candles"]) == 50
    assert result["candles"][0]["timestamp"].timestamp() == 300
    assert result["candles"][-1]["timestamp"].timestamp() == 15_000
    assert result["rsi_14"] is not None
    assert requests[0].headers["accept"] == "application/json;version=20230302"
    assert "/networks/base/tokens/Token%20Address/pools" in str(requests[0].url)


def test_terminal_ui_serves_bundled_chart_library(db):
    from urllib.request import urlopen

    from src.reporting.terminal_ui import start_terminal_ui

    server = start_terminal_ui(db, port=0)
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/") as response:
            page = response.read().decode()
        with urlopen(
            f"http://127.0.0.1:{server.server_port}/lightweight-charts.js"
        ) as response:
            library = response.read()
            content_type = response.headers["Content-Type"]
        assert "/lightweight-charts.js" in page
        assert "unpkg.com" not in page
        assert len(library) > 100_000
        assert content_type.startswith("text/javascript")
    finally:
        server.shutdown()
        server.server_close()


def test_state_api_exposes_live_chart_and_risk_dashboard_data(db):
    from src.reporting.terminal_ui import _read_state

    db.execute(
        """
        INSERT INTO leaderboard_traders (
            user_id, user_handle, rank_24h, pnl_24h_usd, win_rate, sleeve_weight
        ) VALUES ('id-1', 'leader', 1, 250, 0.6, 0.1)
        """
    )
    db.execute(
        """
        INSERT INTO market_snapshots (
            snapshot_id, network, token_address, pool_address, passed_gate
        ) VALUES ('snapshot', 'solana', 'Token', 'pool', true)
        """
    )
    db.execute(
        """
        INSERT INTO market_candles (
            network, token_address, candle_ts, open, high, low, close, volume
        ) VALUES ('solana', 'Token', CURRENT_TIMESTAMP, 1, 2, 0.5, 1.5, 100)
        """
    )
    db.execute(
        """
        INSERT INTO execution_orders (
            client_order_id, trader_handle, network, token_address, side, status,
            requested_usd, filled_usd, entry_price_usd, realized_pnl_usd, fee_usd
        ) VALUES (
            'buy', 'leader', 'solana', 'Token', 'BUY', 'DRY_RUN_OPEN',
            100, 100, 1, 0, 0.95
        )
        """
    )

    state = _read_state(db)

    assert state["token"] == "Token"
    assert state["tokens"] == ["Token"]
    assert len(state["candles"]) == 1
    assert state["markers"][0]["text"] == "BUY leader DRY_RUN_OPEN"
    assert state["traders"][0]["handle"] == "leader"
    assert state["risk"]["open_positions"] == 1
    assert state["risk"]["net_pnl_24h_usd"] == pytest.approx(-0.95)


def test_chart_dashboard_converts_candles_and_trade_markers():
    from src.reporting.chart_dashboard import add_trade_markers, candle_frame

    frame = candle_frame([
        {"time": 300, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10}
    ])
    assert list(frame.columns) == ["time", "open", "high", "low", "close", "volume"]
    assert frame.iloc[0]["close"] == 1.5

    class MarkerChart:
        def __init__(self) -> None:
            self.markers: list[dict[str, Any]] = []

        def marker(self, **kwargs: Any) -> None:
            self.markers.append(kwargs)

    chart = MarkerChart()
    add_trade_markers(
        chart,
        [
            {"time": 300, "side": "BUY", "trader": "alpha", "color": "#26a69a"},
            {"time": 600, "side": "SELL", "trader": "beta", "color": "#ef5350"},
        ],
    )
    assert [marker["position"] for marker in chart.markers] == ["below", "above"]
    assert [marker["shape"] for marker in chart.markers] == ["arrow_up", "arrow_down"]
    assert [marker["text"] for marker in chart.markers] == [
        "BUY | alpha", "SELL | beta",
    ]


@pytest.mark.asyncio
async def test_solana_router_uses_solana_py_for_signed_transaction_broadcast(monkeypatch):
    from solana.rpc.async_api import AsyncClient as SolanaAsyncClient

    class FakeResponse:
        value = "signature-from-rpc"

    class FakeSolanaClient:
        def __init__(self, endpoint: str, **kwargs: Any) -> None:
            self.endpoint = endpoint
            self.options: Any = None
            self.transaction: bytes | None = None

        async def send_raw_transaction(self, transaction: bytes, opts: Any) -> FakeResponse:
            self.transaction = transaction
            self.options = opts
            return FakeResponse()

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        "src.execution.routers.SolanaAsyncClient",
        FakeSolanaClient,
    )
    router = DirectOnChainRouter(Settings(solana_rpc_url="https://rpc.example"))

    signature = await router._send_solana_transaction(b"signed transaction")

    assert signature == "signature-from-rpc"
    assert router._solana_client.endpoint == "https://rpc.example"
    assert router._solana_client.transaction == b"signed transaction"
    assert router._solana_client.options.skip_preflight is False
    assert router._solana_client.options.max_retries == 2
    await router.close()


@pytest.mark.asyncio
async def test_direct_onchain_router_refuses_submission_in_dry_run_mode():
    router = DirectOnChainRouter(Settings())
    intent = OrderIntent(
        "order", "trader", "solana", "TokenMint111", "BUY", 100, 1, 50_000
    )
    with pytest.raises(RuntimeError, match="DRY_RUN=false"):
        await router.submit(intent)
    await router.close()


@pytest.mark.asyncio
async def test_chart_streamlit_server_starts_and_serves_http(db):
    import socket

    from src.reporting.terminal_ui import start_chart_dashboard, start_terminal_ui

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    state_server = start_terminal_ui(db, port=0)
    dashboard = None
    try:
        dashboard = await start_chart_dashboard(
            port=port,
            state_api_url=f"http://127.0.0.1:{state_server.server_port}",
            startup_timeout_seconds=45,
        )
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://127.0.0.1:{port}/")
        assert response.status_code == 200
        assert "streamlit" in response.text.lower()
        assert dashboard.process.poll() is None
    finally:
        if dashboard is not None:
            dashboard.close()
        state_server.shutdown()
        state_server.server_close()


@pytest.mark.asyncio
async def test_circuit_breaker_halts_buy_before_market_scan(db, market):
    db.execute(
        """
        INSERT INTO execution_orders (
            client_order_id, trader_handle, network, token_address, side, status,
            requested_usd, realized_pnl_usd
        ) VALUES ('loss_1', 'trader', 'solana', 'token', 'BUY', 'CLOSED', 500, -101)
        """
    )
    scanner = StubScanner(market)
    gatekeeper = ExecutionGatekeeper(db, scanner, DryRunRouter(), Settings())
    async with httpx.AsyncClient() as client:
        result = await gatekeeper.process_copy_signal(
            client, "trader", "solana", "newtoken", "BUY", 0.10
        )
    assert result == "HALTED_CIRCUIT_BREAKER"
    assert scanner.calls == 0


@pytest.mark.asyncio
async def test_circuit_breaker_includes_unrealized_loss_on_old_open_positions(db, market):
    db.execute(
        """
        INSERT INTO execution_orders (
            client_order_id, trader_handle, network, token_address, side, status,
            requested_usd, unrealized_pnl_usd, created_at
        ) VALUES (
            'old_position', 'trader', 'solana', 'token', 'BUY', 'OPEN', 500, -101,
            CURRENT_TIMESTAMP - INTERVAL 25 HOUR
        )
        """
    )
    scanner = StubScanner(market)
    gatekeeper = ExecutionGatekeeper(db, scanner, DryRunRouter(), Settings())
    async with httpx.AsyncClient() as client:
        result = await gatekeeper.process_copy_signal(
            client, "trader", "solana", "newtoken", "BUY", 0.10
        )
    assert result == "HALTED_CIRCUIT_BREAKER"
    assert scanner.calls == 0


@pytest.mark.asyncio
async def test_unresolved_order_blocks_submission(db, market):
    db.execute(
        """
        INSERT INTO execution_orders (
            client_order_id, trader_handle, network, token_address, side, status, requested_usd
        ) VALUES ('pending_1', 'trader', 'solana', 'token', 'BUY', 'UNCERTAIN', 100)
        """
    )
    scanner = StubScanner(market)
    gatekeeper = ExecutionGatekeeper(db, scanner, UnresolvedRouter(), Settings())
    async with httpx.AsyncClient() as client:
        result = await gatekeeper.process_copy_signal(
            client, "trader", "solana", "newtoken", "BUY", 0.10
        )
    assert result == "BLOCKED_UNRESOLVED_ORDER"
    assert scanner.calls == 0


@pytest.mark.asyncio
async def test_gatekeeper_alerts_for_execution_and_market_rejection(
    db, market, monkeypatch
):
    from src.execution import gatekeeper as gatekeeper_module

    alerts = []

    async def record_alert(message: str, severity: str = "INFO") -> None:
        alerts.append((message, severity))

    monkeypatch.setattr(gatekeeper_module, "send_alert", record_alert)
    scanner = StubScanner(market)
    gatekeeper = ExecutionGatekeeper(
        db, scanner, DryRunRouter(), Settings(starting_bankroll_usd=10_000)
    )
    async with httpx.AsyncClient() as client:
        executed = await gatekeeper.process_copy_signal(
            client, "trader", "solana", "GoodToken", "BUY", 0.10
        )
        scanner.response = {
            **market,
            "passed": False,
            "reason": "insufficient_liquidity:$4500",
        }
        rejected = await gatekeeper.process_copy_signal(
            client, "trader", "solana", "LowLiqToken", "BUY", 0.10
        )

    assert executed.startswith("EXECUTED:DRY_RUN_OPEN:")
    assert rejected == "REJECTED:insufficient_liquidity:$4500"
    assert [severity for _, severity in alerts] == ["SUCCESS", "WARNING"]
    assert "GoodToken" in alerts[0][0]
    assert "LowLiqToken" in alerts[1][0]


@pytest.mark.asyncio
async def test_gatekeeper_alerts_when_daily_circuit_breaker_trips(
    db, market, monkeypatch
):
    from src.execution import gatekeeper as gatekeeper_module

    alerts = []

    async def record_alert(message: str, severity: str = "INFO") -> None:
        alerts.append((message, severity))

    monkeypatch.setattr(gatekeeper_module, "send_alert", record_alert)
    db.execute(
        """
        INSERT INTO execution_orders (
            client_order_id, trader_handle, network, token_address, side, status,
            requested_usd, realized_pnl_usd
        ) VALUES ('loss_for_alert', 'trader', 'solana', 'token', 'BUY', 'CLOSED', 500, -101)
        """
    )
    scanner = StubScanner(market)
    gatekeeper = ExecutionGatekeeper(db, scanner, DryRunRouter(), Settings())
    async with httpx.AsyncClient() as client:
        result = await gatekeeper.process_copy_signal(
            client, "trader", "solana", "newtoken", "BUY", 0.10
        )

    assert result == "HALTED_CIRCUIT_BREAKER"
    assert len(alerts) == 1
    assert alerts[0][1] == "ALERT"
    assert "circuit breaker" in alerts[0][0].lower()


@pytest.mark.asyncio
async def test_sell_closes_the_open_dry_run_position_and_realizes_pnl(db, market):
    gatekeeper = ExecutionGatekeeper(
        db, StubScanner(market), DryRunRouter(),
        Settings(starting_bankroll_usd=10_000),
    )
    async with httpx.AsyncClient() as client:
        buy = await gatekeeper.process_copy_signal(
            client, "buyer", "1399811149", "TokenMint111", "BUY", 0.10
        )
        sell = await gatekeeper.process_copy_signal(
            client, "buyer", "solana", "TokenMint111", "SELL", 0.10
        )
    assert buy.startswith("EXECUTED:DRY_RUN_OPEN:")
    assert sell.startswith("EXECUTED:DRY_RUN_OPEN:")
    position = db.execute(
        """
        SELECT status, remaining_token_amount
        FROM execution_orders
        WHERE side = 'BUY'
        """
    ).fetchone()
    realized = db.execute(
        "SELECT realized_pnl_usd FROM execution_orders WHERE side = 'SELL'"
    ).fetchone()[0]
    assert position == ("CLOSED", 0.0)
    assert realized != 0


@pytest.mark.asyncio
async def test_sell_ticket_does_not_close_more_than_its_clamped_quantity(db, market):
    db.execute(
        """
        INSERT INTO execution_orders (
            client_order_id, trader_handle, network, token_address, side, status,
            requested_usd, filled_usd, filled_token_amount, remaining_token_amount,
            entry_price_usd, high_water_price_usd
        ) VALUES ('large_position', 'buyer', 'solana', 'TokenMint111', 'BUY',
                  'DRY_RUN_OPEN', 2500, 2500, 2000, 2000, 1.25, 1.25)
        """
    )
    db.execute(
        """
        INSERT INTO execution_orders (
            client_order_id, trader_handle, network, token_address, side, status,
            requested_usd, filled_usd, filled_token_amount, remaining_token_amount,
            entry_price_usd, high_water_price_usd
        ) VALUES ('other_trader_position', 'other', 'solana', 'TokenMint111', 'BUY',
                  'DRY_RUN_OPEN', 1250, 1250, 1000, 1000, 1.25, 1.25)
        """
    )
    gatekeeper = ExecutionGatekeeper(
        db, StubScanner(market), DryRunRouter(), Settings(starting_bankroll_usd=10_000)
    )
    async with httpx.AsyncClient() as client:
        result = await gatekeeper.process_copy_signal(
            client, "buyer", "solana", "TokenMint111", "SELL", 0.10
        )
    assert result.startswith("EXECUTED:DRY_RUN_OPEN:")
    remaining = db.execute(
        "SELECT remaining_token_amount FROM execution_orders WHERE client_order_id = 'large_position'"
    ).fetchone()[0]
    other_remaining = db.execute(
        "SELECT remaining_token_amount FROM execution_orders WHERE client_order_id = 'other_trader_position'"
    ).fetchone()[0]
    sold = db.execute(
        "SELECT filled_token_amount FROM execution_orders WHERE side = 'SELL'"
    ).fetchone()[0]
    assert sold <= 480
    assert remaining >= 1520
    assert other_remaining == 1000


@pytest.mark.asyncio
async def test_gecko_terminal_etag_returns_cached_json():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, headers={"ETag": '"market-v1"'}, json={"data": []})
        return httpx.Response(304, request=request)

    scanner = GeckoTerminalScanner(Settings())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        first = await scanner._get_json(client, "https://example.test/market")
        second = await scanner._get_json(client, "https://example.test/market")

    assert first == second == {"data": []}
    assert requests[1].headers["If-None-Match"] == '"market-v1"'


@pytest.mark.asyncio
async def test_fomo_positions_etag_is_cached_per_page(db):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200, headers={"ETag": '"positions-v1"'}, json={"data": []}
            )
        return httpx.Response(304, request=request)

    fomo = FomoLeaderboardClient(db, Settings())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        first = await fomo._fetch_positions_page(client, "trader", None)
        second = await fomo._fetch_positions_page(client, "trader", None)

    assert first == second == {"data": []}
    assert requests[1].headers["If-None-Match"] == '"positions-v1"'


@pytest.mark.asyncio
async def test_fomo_rest_fallback_preserves_explicit_sell_side(db, monkeypatch):
    fomo = FomoLeaderboardClient(db, Settings(fomo_api_key="test-key"))
    fomo.top_10_handles = {"trader": 0.1}

    async def positions_page(
        client: httpx.AsyncClient, handle: str, cursor: str | None
    ) -> dict[str, Any]:
        return {
            "data": [{"id": "sell-1", "tokenAddress": "Token", "side": "SELL", "amount": 5}],
            "pagination": {},
        }

    monkeypatch.setattr(fomo, "_fetch_positions_page", positions_page)
    signals: list[tuple[str, str, str, str, float]] = []

    async def capture(*signal: Any) -> str:
        signals.append(signal)
        return "accepted"

    async with httpx.AsyncClient() as client:
        await fomo.poll_trader_positions_fallback(client, capture)

    assert signals == [("trader", "solana", "Token", "SELL", 0.1)]


@pytest.mark.parametrize(
    "api_key",
    ["", "your_pumpfomo_or_fomoradar_api_key_here"],
)
@pytest.mark.asyncio
async def test_fomo_leaderboard_uses_keyless_seed_without_request(db, api_key):
    fomo = FomoLeaderboardClient(db, Settings(fomo_api_key=api_key))

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"unexpected FOMO request: {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rows = await fomo.fetch_leaderboard(client, "24h")

    assert len(rows) == 10
    assert [row["rank"] for row in rows] == list(range(1, 11))
    assert all(row["is_demo"] and not row["verified"] for row in rows)
    assert {row["chain"] for row in rows} == {"solana", "base"}


@pytest.mark.parametrize("status_code", [401, 403])
@pytest.mark.asyncio
async def test_fomo_leaderboard_uses_keyless_seed_after_unauthorized_response(db, status_code):
    fomo = FomoLeaderboardClient(db, Settings(fomo_api_key="test-key"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        rows = await fomo.fetch_leaderboard(client, "24h")

    assert len(rows) == 10
    assert rows[0]["handle"] == "keyless_demo_01"


@pytest.mark.asyncio
async def test_keyless_fallback_maps_gecko_trending_pools_to_demo_traders(db):
    requests: list[httpx.Request] = []
    pools = [
        {
            "id": f"solana_pool_{index}",
            "attributes": {"address": f"pool-{index}"},
            "relationships": {
                "base_token": {"data": {"id": f"solana_token-{index}"}}
            },
        }
        for index in range(2)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if "trending_pools" in str(request.url):
            return httpx.Response(200, json={"data": pools}, request=request)
        return httpx.Response(403, request=request)

    fomo = FomoLeaderboardClient(db, Settings(fomo_api_key="test-key"))
    fomo.top_10_handles = {"existing_trader": 0.1}
    signals: list[tuple[str, str, str, str, float]] = []

    async def capture(*signal: Any) -> str:
        signals.append(signal)
        return "accepted"

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await fomo.poll_trader_positions_fallback(client, capture)

    assert [signal[:4] for signal in signals] == [
        ("keyless_demo_01", "solana", "token-0", "BUY"),
        ("keyless_demo_02", "solana", "token-1", "BUY"),
    ]
    assert all(signal[4] == 0.1 for signal in signals)
    gecko_request = next(request for request in requests if "trending_pools" in str(request.url))
    assert gecko_request.headers["Accept"] == "application/json;version=20230203"


@pytest.mark.asyncio
async def test_missing_fomo_key_polls_gecko_trending_pools_directly(db):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "attributes": {"address": "pool-1", "base_token_address": "mint-1"}
                    }
                ]
            },
            request=request,
        )

    fomo = FomoLeaderboardClient(db, Settings(fomo_api_key=""))
    signals: list[tuple[Any, ...]] = []

    async def capture(*signal: Any) -> str:
        signals.append(signal)
        return "accepted"

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await fomo.poll_trader_positions_fallback(client, capture)

    assert len(requests) == 1
    assert "trending_pools" in str(requests[0].url)
    assert signals == [("keyless_demo_01", "solana", "mint-1", "BUY", 0.1)]


@pytest.mark.asyncio
async def test_keyless_gecko_signal_fallback_is_disabled_outside_dry_run(db):
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"unexpected keyless fallback request: {request.url}")

    fomo = FomoLeaderboardClient(db, Settings(fomo_api_key="", dry_run=False))
    signals: list[tuple[Any, ...]] = []

    async def capture(*signal: Any) -> str:
        signals.append(signal)
        return "accepted"

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await fomo.poll_trader_positions_fallback(client, capture)

    assert signals == []


def test_microstructure_rejects_migration_fdv_and_low_turnover():
    rejected_fdv = passes_memecoin_microstructure_gates(
        {"fdv_usd": 50_000, "volume_usd_h24": 100_000, "reserve_in_usd": 50_000}
    )
    rejected_volume = passes_memecoin_microstructure_gates(
        {"fdv_usd": 200_000, "volume_24h_usd": 50_000, "liquidity_usd": 50_000}
    )

    assert rejected_fdv[0] is False
    assert "pre-migration" in rejected_fdv[1]
    assert rejected_volume[0] is False
    assert "Weak Volume/FDV" in rejected_volume[1]


def test_microstructure_rejects_stair_step_patterns_and_green_spikes():
    import pandas as pd

    stair_step = pd.DataFrame(
        {
            "open": [1.0 + index * 0.001 for index in range(12)],
            "close": [1.001 + index * 0.001 for index in range(12)],
        }
    )
    spike = pd.DataFrame(
        {
            "open": [1.0 + (index % 2) * 0.03 for index in range(12)],
            "close": [0.98 + (index % 3) * 0.02 for index in range(11)] + [1.10],
        }
    )
    accepted_pullback = pd.DataFrame(
        {
            "open": [1.0 + index * 0.01 for index in range(12)],
            "close": [0.99 + index * 0.01 for index in range(11)] + [1.10],
        }
    )

    assert "stair-step" in passes_memecoin_microstructure_gates({}, stair_step)[1]
    assert "chase" in passes_memecoin_microstructure_gates({}, spike)[1]
    assert passes_memecoin_microstructure_gates({}, accepted_pullback)[0] is True


@pytest.mark.asyncio
async def test_microstructure_rejection_is_persisted_and_alerted(db, market, monkeypatch):
    scanner = StubScanner(
        {
            **market,
            "fdv_usd": 50_000,
            "volume_usd_h24": 100_000,
        }
    )
    alerts: list[tuple[str, str]] = []

    async def record_alert(message: str, severity: str) -> None:
        alerts.append((message, severity))

    monkeypatch.setattr("src.execution.gatekeeper.send_alert", record_alert)
    gatekeeper = ExecutionGatekeeper(
        db, scanner, DryRunRouter(), Settings(starting_bankroll_usd=10_000)
    )
    async with httpx.AsyncClient() as client:
        result = await gatekeeper.process_copy_signal(
            client, "trader", "solana", "MigrationZoneToken", "BUY", 0.1
        )

    assert result.startswith("REJECTED:Rejected: Token in pre-migration")
    assert alerts and alerts[0][1] == "WARNING"
    assert db.execute(
        "SELECT status, failure_reason FROM execution_orders"
    ).fetchone() == ("REJECTED", result.removeprefix("REJECTED:"))


@pytest.mark.asyncio
async def test_gatekeeper_normalizes_volume_and_liquidity_aliases(db, market, monkeypatch):
    scanner = StubScanner(
        {
            **market,
            "fdv_usd": 100_000,
            "volume_usd_h24": None,
            "volume_24h_usd": 20_000,
            "reserve_in_usd": None,
            "liquidity_usd": 50_000,
        }
    )

    async def record_alert(message: str, severity: str) -> None:
        return None

    monkeypatch.setattr("src.execution.gatekeeper.send_alert", record_alert)
    gatekeeper = ExecutionGatekeeper(
        db, scanner, DryRunRouter(), Settings(starting_bankroll_usd=10_000)
    )
    async with httpx.AsyncClient() as client:
        result = await gatekeeper.process_copy_signal(
            client, "trader", "solana", "LowTurnoverToken", "BUY", 0.1
        )

    assert result.startswith("REJECTED:Rejected: Weak Volume/FDV turnover")


@pytest.mark.asyncio
async def test_fomo_selects_top_qualifying_traders_by_daily_pnl(db, monkeypatch):
    fomo = FomoLeaderboardClient(db, Settings())
    daily = [
        {"user_id": "unqualified", "user_handle": "unqualified", "pnl_24h_usd": 1000,
         "win_rate": 0, "num_trades": 100},
        *[
            {"user_id": f"user-{rank}", "user_handle": f"trader-{rank}",
             "pnl_24h_usd": float(100 - rank), "win_rate": 0.5, "num_trades": 5}
            for rank in range(12)
        ],
    ]
    weekly = [
        {"user_id": f"user-{rank}", "user_handle": f"trader-{rank}",
         "pnl_7d_usd": 500.0, "win_rate": 0.5, "num_trades": 5}
        for rank in range(12)
    ]

    async def fetch_leaderboard(client: httpx.AsyncClient, window: str) -> list[dict[str, Any]]:
        return daily if window == "24h" else weekly

    monkeypatch.setattr(fomo, "fetch_leaderboard", fetch_leaderboard)
    async with httpx.AsyncClient() as client:
        selected = await fomo.refresh_top_10_traders(client)

    assert list(selected) == [f"trader-{rank}" for rank in range(10)]
    assert len(selected) == 10
    assert max(selected.values()) <= 0.15

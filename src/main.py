import asyncio
import logging
import os
import signal
from datetime import datetime, timezone
from typing import Any

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from rich.console import Console
from rich.logging import RichHandler

from src.config import get_settings
from src.db import close_database, initialize_database
from src.execution.gatekeeper import ExecutionGatekeeper
from src.execution.routers import (
    DirectOnChainRouter,
    DryRunRouter,
    FreqtradeWebhookRouter,
    OrderRouter,
    close_router,
)
from src.ingestion.fomo_client import FomoLeaderboardClient
from src.ingestion.gecko_scanner import GeckoTerminalScanner
from src.reporting.morning_audit import run_morning_audit
from src.reporting.terminal_ui import (
    run_terminal_dashboard,
    start_chart_dashboard,
    start_terminal_ui,
)

console = Console(stderr=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
            markup=False,
        )
    ],
)
logger = logging.getLogger("fomo_terminal")


def build_router(settings: Any, client: httpx.AsyncClient) -> OrderRouter:
    if settings.dry_run:
        return DryRunRouter()
    if not settings.live_trading_confirmed:
        raise RuntimeError(
            "Live routing is locked. Set DRY_RUN=false and LIVE_TRADING_CONFIRMED=true "
            "after explicitly confirming live execution."
        )
    if settings.execution_router == "freqtrade":
        return FreqtradeWebhookRouter(settings, client)
    return DirectOnChainRouter(settings, client)


async def sync_initial_top_10(fomo: FomoLeaderboardClient, client: httpx.AsyncClient) -> None:
    try:
        await fomo.refresh_top_10_traders(client)
    except httpx.HTTPError:
        cached_traders = fomo.load_cached_top_10()
        if not cached_traders:
            logger.exception(
                "Initial FOMO leaderboard sync failed and no persisted Top-10 is available."
            )
            raise
        logger.exception(
            "Initial FOMO leaderboard sync failed; continuing with %d persisted traders.",
            len(cached_traders),
        )


async def monitor_open_positions(
    client: httpx.AsyncClient,
    database: Any,
    scanner: GeckoTerminalScanner,
    gatekeeper: ExecutionGatekeeper,
    settings: Any,
) -> None:
    with database.cursor() as cur:
        orders = cur.execute(
            """
            SELECT o.client_order_id, o.trader_handle, o.network, o.token_address,
                   o.remaining_token_amount, o.entry_price_usd, o.high_water_price_usd,
                   o.created_at, t.sleeve_weight
            FROM execution_orders o
            LEFT JOIN leaderboard_traders t ON t.user_handle = o.trader_handle
            WHERE o.side = 'BUY' AND o.status IN ('OPEN', 'DRY_RUN_OPEN')
              AND o.filled_token_amount > 0
            """
        ).fetchall()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for cid, trader, network, token, remaining_amount, entry, high_water, created_at, weight in orders:
        if not entry or float(entry) <= 0:
            continue
        market = await scanner.evaluate_token(client, network, token)
        current = float(market.get("price_usd") or 0)
        if current <= 0:
            continue
        latest_high = max(float(high_water or entry), current)
        age_minutes = (now - created_at).total_seconds() / 60
        drawdown = (latest_high - current) / latest_high
        pnl = (current - float(entry)) / float(entry)
        if drawdown >= settings.trailing_stop_pct or (
            age_minutes >= settings.time_stop_minutes and pnl < 0.03
        ):
            result = await gatekeeper.process_copy_signal(
                client, trader, network, token, "SELL", float(weight or 0.10)
            )
            logger.info("Risk exit for %s: %s", token, result)
        else:
            with database.transaction() as cur:
                cur.execute(
                    """
                    UPDATE execution_orders
                    SET high_water_price_usd = ?, unrealized_pnl_usd = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE client_order_id = ?
                    """,
                    [latest_high, float(remaining_amount) * (current - float(entry)), cid],
                )


async def main() -> None:
    settings = get_settings()
    if not settings.dry_run and not settings.live_trading_confirmed:
        raise RuntimeError(
            "Live routing is locked. Set DRY_RUN=false and LIVE_TRADING_CONFIRMED=true "
            "only after explicitly confirming live execution."
        )
    database = initialize_database(settings)
    scanner = GeckoTerminalScanner(settings)
    fomo = FomoLeaderboardClient(database, settings)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            logger.warning("Signal handlers are unavailable on this event loop.")

    ui_server = None
    chart_dashboard = None
    terminal_dashboard_task: asyncio.Task[None] | None = None
    stream_task: asyncio.Task[None] | None = None
    scheduler: AsyncIOScheduler | None = None
    async with httpx.AsyncClient() as client:
        router: OrderRouter | None = None
        try:
            router = build_router(settings, client)
            gatekeeper = ExecutionGatekeeper(database, scanner, router, settings)
            await sync_initial_top_10(fomo, client)

            scheduler = AsyncIOScheduler(timezone="UTC")
            scheduler.add_job(
                fomo.refresh_top_10_traders, "interval", minutes=60,
                args=[client], id="refresh_top_10", max_instances=1, coalesce=True,
            )
            scheduler.add_job(
                monitor_open_positions, "interval", seconds=60,
                args=[client, database, scanner, gatekeeper, settings],
                id="monitor_open_positions", max_instances=1, coalesce=True,
            )
            scheduler.add_job(
                run_morning_audit, "cron", hour=7, minute=0,
                args=[database, settings.starting_bankroll_usd],
                id="morning_audit", max_instances=1, coalesce=True,
            )
            scheduler.start()
            ui_server = start_terminal_ui(database)
            external_chart_dashboard = os.getenv(
                "FOMO_EXTERNAL_CHART_DASHBOARD", ""
            ).strip().lower() in {"1", "true", "yes"}
            if not external_chart_dashboard:
                chart_dashboard = await start_chart_dashboard(
                    state_api_url=f"http://127.0.0.1:{ui_server.server_port}"
                )
            terminal_dashboard_task = asyncio.create_task(
                run_terminal_dashboard(
                    database,
                    settings,
                    stop_event,
                    console=console,
                ),
                name="terminal-risk-dashboard",
            )

            async def handle_signal(
                trader_handle: str, network: str, token_address: str, side: str, weight: float
            ) -> str:
                return await gatekeeper.process_copy_signal(
                    client, trader_handle, network, token_address, side, weight
                )

            stream_task = asyncio.create_task(
                fomo.stream_top_10_trades(client, handle_signal, stop_event),
                name="fomo-trade-stream",
            )
            completed, _ = await asyncio.wait(
                {stream_task, terminal_dashboard_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if terminal_dashboard_task in completed and not stop_event.is_set():
                terminal_dashboard_task.result()
                raise RuntimeError("Terminal risk dashboard stopped unexpectedly.")
            await stream_task
        finally:
            stop_event.set()
            if stream_task is not None and not stream_task.done():
                stream_task.cancel()
            if stream_task is not None:
                await asyncio.gather(stream_task, return_exceptions=True)
            try:
                if terminal_dashboard_task is not None:
                    await terminal_dashboard_task
            finally:
                try:
                    if chart_dashboard is not None:
                        chart_dashboard.close()
                finally:
                    try:
                        if scheduler is not None:
                            scheduler.shutdown(wait=False)
                        if ui_server is not None:
                            ui_server.shutdown()
                            ui_server.server_close()
                        if router is not None:
                            await close_router(router)
                    finally:
                        close_database()


if __name__ == "__main__":
    asyncio.run(main())

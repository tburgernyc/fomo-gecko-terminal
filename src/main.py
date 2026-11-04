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
from src.db import close_database, cursor_scope, initialize_database, transaction_scope
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
    await gatekeeper.reconcile_pending_orders()
    with cursor_scope(database) as cur:
        orders = cur.execute(
            """
            SELECT o.exit_campaign_id, o.trader_handle, o.network, o.token_address,
                   SUM(o.filled_token_amount),
                   SUM(o.remaining_token_amount),
                   SUM(o.filled_token_amount * o.entry_price_usd)
                       / NULLIF(SUM(o.filled_token_amount), 0),
                   MAX(o.high_water_price_usd), MIN(o.created_at),
                   MAX(o.entry_atr_usd), SUM(o.tranche_1_sold_amount),
                   SUM(o.tranche_2_sold_amount), t.sleeve_weight
            FROM execution_orders o
            LEFT JOIN leaderboard_traders t ON t.user_handle = o.trader_handle
            WHERE o.side = 'BUY'
              AND o.status IN ('OPEN', 'DRY_RUN_OPEN', 'CLOSED', 'DRY_RUN_CLOSED')
              AND o.filled_token_amount > 0
            GROUP BY o.exit_campaign_id, o.trader_handle, o.network, o.token_address,
                     t.sleeve_weight
            HAVING SUM(o.remaining_token_amount) > 0
            """
        ).fetchall()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for (
        campaign_id,
        trader,
        network,
        token,
        filled_amount,
        remaining_amount,
        entry,
        high_water,
        created_at,
        entry_atr,
        tranche_1_sold,
        tranche_2_sold,
        weight,
    ) in orders:
        if not entry or float(entry) <= 0 or float(remaining_amount or 0) <= 0:
            continue
        market = await scanner.evaluate_token(client, network, token)
        current = float(market.get("price_usd") or 0)
        if current <= 0:
            continue
        latest_high = max(float(high_water or entry), current)
        age_minutes = (now - created_at).total_seconds() / 60
        pnl = (current - float(entry)) / float(entry)
        initial_amount = float(filled_amount or 0)
        remaining = float(remaining_amount or 0)
        tranche_1_target = initial_amount * 0.40
        tranche_2_target = initial_amount * 0.40
        tranche_1_done = float(tranche_1_sold or 0) >= tranche_1_target - 1e-12
        tranche_2_done = float(tranche_2_sold or 0) >= tranche_2_target - 1e-12
        exit_amount: float | None = None
        exit_tranche: str | None = None

        if age_minutes >= settings.exit_time_stop_minutes and pnl < 0.08:
            exit_amount, exit_tranche = remaining, "TIME_STOP"
        elif not tranche_1_done and pnl + 1e-9 >= 0.65:
            exit_amount = min(remaining, tranche_1_target - float(tranche_1_sold or 0))
            exit_tranche = "TAKE_PROFIT_1"
        elif tranche_1_done and not tranche_2_done and pnl + 1e-9 >= 1.50:
            exit_amount = min(remaining, tranche_2_target - float(tranche_2_sold or 0))
            exit_tranche = "TAKE_PROFIT_2"
        else:
            stop_price: float | None = None
            if tranche_2_done and float(entry_atr or 0) > 0:
                stop_price = latest_high - 2.2 * float(entry_atr)
                if tranche_1_done:
                    stop_price = max(stop_price, float(entry))
            elif tranche_1_done:
                stop_price = float(entry)
            elif latest_high > 0 and (
                latest_high - current
            ) / latest_high >= settings.trailing_stop_pct:
                stop_price = current
            if stop_price is not None and current <= stop_price:
                exit_amount, exit_tranche = remaining, "TRAILING_STOP"

        with transaction_scope(database) as cur:
            cur.execute(
                """
                UPDATE execution_orders
                SET high_water_price_usd = GREATEST(COALESCE(high_water_price_usd, ?), ?),
                    unrealized_pnl_usd = remaining_token_amount * (? - entry_price_usd),
                    updated_at = CURRENT_TIMESTAMP
                WHERE side = 'BUY' AND trader_handle = ? AND network = ? AND token_address = ?
                  AND exit_campaign_id IS NOT DISTINCT FROM ?
                  AND status IN ('OPEN', 'DRY_RUN_OPEN') AND remaining_token_amount > 0
                """,
                [latest_high, latest_high, current, trader, network, token, campaign_id],
            )
        if exit_amount is not None and exit_amount > 0:
            result = await gatekeeper.process_copy_signal(
                client,
                trader,
                network,
                token,
                "SELL",
                float(weight or 0.10),
                exit_token_amount=exit_amount,
                exit_tranche=exit_tranche,
            )
            logger.info("Position exit %s for %s: %s", exit_tranche, token, result)


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
                trader_handle: str, network: str, token_address: str, side: str,
                weight: float, leader_entry_price: float = 0.0,
            ) -> str:
                return await gatekeeper.process_copy_signal(
                    client, trader_handle, network, token_address, side, weight,
                    leader_entry_price,
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

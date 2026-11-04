"""Run the historical Solana event simulation through the live gatekeeper."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import httpx
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.execution.gatekeeper as gatekeeper_module
from src.config import Settings
from src.execution.gatekeeper import ExecutionGatekeeper
from src.execution.simulated_router import SimulatedOnChainRouter
from src.ingestion.historical_replayer import HistoricalReplayer
from src.main import monitor_open_positions

logger = logging.getLogger(__name__)
DEFAULT_PARQUET = ROOT / "data" / "mock_solana_pools.parquet"


class EventClock:
    """Expose the current dataset timestamp to gatekeeper idempotency buckets."""

    def __init__(self) -> None:
        self.timestamp = 0.0
        self._counter = 0

    def time(self) -> float:
        return self.timestamp

    def time_ns(self) -> int:
        self._counter += 1
        return int(self.timestamp * 1_000_000_000) + self._counter


def _append_equity_snapshot(
    ledger: duckdb.DuckDBPyConnection,
    event_id: int,
    timestamp: datetime,
    router: SimulatedOnChainRouter,
    replayer: HistoricalReplayer,
) -> None:
    pool_states = replayer.current_pool_states()
    with ledger.cursor() as cur:
        open_orders = cur.execute(
            """
            SELECT token_address, remaining_token_amount
            FROM execution_orders
            WHERE side = 'BUY' AND status = 'OPEN' AND remaining_token_amount > 0
            """
        ).fetchall()
    positions_value = sum(
        float(amount) * float(pool_states[token]["price_usd"])
        for token, amount in open_orders
        if token in pool_states
    )
    ledger.execute(
        "INSERT INTO simulation_equity (event_id, event_timestamp, equity_usd) VALUES (?, ?, ?)",
        [event_id, timestamp, router.cash_balance_usd + positions_value],
    )


def _finalize_open_positions(
    ledger: duckdb.DuckDBPyConnection, replayer: HistoricalReplayer
) -> None:
    pool_states = replayer.current_pool_states()
    with ledger.cursor() as cur:
        open_orders = cur.execute(
            """
            SELECT client_order_id, token_address, remaining_token_amount, entry_price_usd
            FROM execution_orders
            WHERE side = 'BUY' AND status = 'OPEN' AND remaining_token_amount > 0
            """
        ).fetchall()
    for client_order_id, token, amount, entry_price in open_orders:
        state = pool_states.get(str(token))
        if state is None:
            continue
        unrealized = float(amount) * (
            float(state["price_usd"]) - float(entry_price or 0)
        )
        ledger.execute(
            """
            UPDATE execution_orders
            SET unrealized_pnl_usd = ?, updated_at = CURRENT_TIMESTAMP
            WHERE client_order_id = ?
            """,
            [unrealized, client_order_id],
        )


def _write_attempts(
    ledger: duckdb.DuckDBPyConnection, router: SimulatedOnChainRouter
) -> None:
    ledger.execute(
        """
        CREATE TABLE simulation_attempts (
            attempt_id BIGINT,
            client_order_id VARCHAR,
            side VARCHAR,
            signal_slot BIGINT,
            execution_slot BIGINT,
            notional_usd DOUBLE,
            priority_fee_sol DOUBLE,
            jito_tip_multiplier DOUBLE,
            fee_usd DOUBLE,
            status VARCHAR,
            fill_price_usd DOUBLE,
            failure_reason VARCHAR
        )
        """
    )
    if router.attempts:
        ledger.executemany(
            """
            INSERT INTO simulation_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    index,
                    attempt["client_order_id"],
                    attempt["side"],
                    attempt["signal_slot"],
                    attempt["execution_slot"],
                    attempt["notional_usd"],
                    attempt["priority_fee_sol"],
                    attempt["jito_tip_multiplier"],
                    attempt["fee_usd"],
                    attempt["status"],
                    attempt["fill_price_usd"],
                    attempt["failure_reason"],
                )
                for index, attempt in enumerate(router.attempts, start=1)
            ],
        )


def _print_report(ledger: duckdb.DuckDBPyConnection) -> dict[str, float | int]:
    campaigns = ledger.execute(
        """
        SELECT exit_campaign_id,
               SUM(campaign_realized_pnl_usd + unrealized_pnl_usd) AS pnl_usd,
               SUM(campaign_fees_usd) AS campaign_fees_usd,
               SUM(remaining_token_amount) AS remaining_token_amount
        FROM execution_orders
        WHERE side = 'BUY' AND filled_token_amount > 0
        GROUP BY exit_campaign_id
        """
    ).fetchall()
    fees = float(
        ledger.execute("SELECT COALESCE(SUM(fee_usd), 0.0) FROM simulation_attempts").fetchone()[0]
    )
    drawdown = float(
        ledger.execute(
            """
            WITH equity_peaks AS (
                SELECT equity_usd,
                       max(equity_usd) OVER (
                           ORDER BY event_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                       ) AS peak_usd
                FROM simulation_equity
            )
            SELECT COALESCE(
                max((peak_usd - equity_usd) / NULLIF(peak_usd, 0)), 0.0
            )
            FROM equity_peaks
            """
        ).fetchone()[0]
    )
    closed_campaign_pnls = [
        float(pnl or 0.0) - float(campaign_fees or 0.0)
        for _, pnl, campaign_fees, remaining in campaigns
        if float(remaining or 0.0) <= 1e-12
    ]
    campaign_gross_pnls = [float(pnl or 0.0) for _, pnl, _, _ in campaigns]
    wins = sum(pnl > 0 for pnl in closed_campaign_pnls)
    win_rate = wins / len(closed_campaign_pnls) if closed_campaign_pnls else 0.0
    gross_pnl = sum(campaign_gross_pnls)
    positive_pnl = sum(pnl for pnl in closed_campaign_pnls if pnl > 0)
    negative_pnl = sum(pnl for pnl in closed_campaign_pnls if pnl < 0)
    profit_factor = positive_pnl / abs(negative_pnl) if negative_pnl < 0 else float("inf")

    table = Table(title="Institutional Campaign Backtest Audit")
    table.add_column("Metric", style="cyan")
    table.add_column("Result", justify="right", style="green")
    table.add_row("Total Campaigns", str(len(campaigns)))
    table.add_row(
        "Win Rate",
        f"{win_rate:.2%} ({wins}/{len(closed_campaign_pnls)} closed campaigns)",
    )
    table.add_row("Gross PnL", f"${gross_pnl:,.2f}")
    table.add_row("Total Fees Paid", f"${fees:,.4f}")
    table.add_row("Net PnL", f"${gross_pnl - fees:,.2f}")
    table.add_row("Profit Factor", f"{profit_factor:.3f}")
    table.add_row("Maximum Drawdown", f"{drawdown:.2%}")
    Console().print(table)
    return {
        "total_campaigns": len(campaigns),
        "closed_campaigns": len(closed_campaign_pnls),
        "win_rate": win_rate,
        "gross_pnl_usd": gross_pnl,
        "total_fees_usd": fees,
        "net_pnl_usd": gross_pnl - fees,
        "profit_factor": profit_factor,
        "max_drawdown": drawdown,
    }


async def run_backtest(
    parquet_path: str | Path = DEFAULT_PARQUET,
    *,
    latency_slots: int = 3,
    seed: int = 20261103,
) -> dict[str, Any]:
    """Replay each event through ExecutionGatekeeper and return terminal metrics."""
    settings = Settings(dry_run=True, live_trading_confirmed=False)
    ledger = duckdb.connect(":memory:")
    replayer: HistoricalReplayer | None = None
    original_clock = gatekeeper_module.time
    try:
        ledger.execute((ROOT / "schema.sql").read_text(encoding="utf-8"))
        ledger.execute(
            """
            CREATE TABLE simulation_equity (
                event_id BIGINT PRIMARY KEY,
                event_timestamp TIMESTAMPTZ,
                equity_usd DOUBLE NOT NULL
            )
            """
        )
        replayer = HistoricalReplayer(parquet_path, settings=settings)
        router = SimulatedOnChainRouter(
            replayer,
            latency_slots=latency_slots,
            starting_balance_usd=settings.starting_bankroll_usd,
            seed=seed,
        )
        gatekeeper = ExecutionGatekeeper(ledger, replayer, router, settings)
        clock = EventClock()
        gatekeeper_module.time = clock
        event_count = 0
        signals_processed = 0
        first_timestamp: datetime | None = None

        async with httpx.AsyncClient() as client:
            async for event in replayer.stream_events():
                event_count += 1
                timestamp = event["timestamp"]
                if first_timestamp is None:
                    first_timestamp = timestamp
                    ledger.execute(
                        "INSERT INTO simulation_equity VALUES (0, ?, ?)",
                        [timestamp, router.cash_balance_usd],
                    )
                clock.timestamp = timestamp.timestamp()

                if event["signal_type"] in {"FOMO_BUY", "FOMO_SELL"}:
                    result = await gatekeeper.process_copy_signal(
                        client,
                        event["userHandle"],
                        event["network"],
                        event["tokenAddress"],
                        event["side"],
                        float(event["sleeve_weight"]),
                    )
                    signals_processed += 1
                    logger.debug(
                        "Event %d slot %d %s %s: %s",
                        event_count,
                        event["slot"],
                        event["side"],
                        event["tokenAddress"],
                        result,
                    )

                await monitor_open_positions(
                    client, ledger, replayer, gatekeeper, settings
                )
                _append_equity_snapshot(
                    ledger, event_count, timestamp, router, replayer
                )

        _finalize_open_positions(ledger, replayer)
        _write_attempts(ledger, router)
        audit_metrics = _print_report(ledger)
        logger.info(
            "Backtest complete: %d events, %d FOMO signals, %d execution attempts",
            event_count,
            signals_processed,
            len(router.attempts),
        )
        return {
            "event_count": event_count,
            "signals_processed": signals_processed,
            "attempt_count": len(router.attempts),
            "cash_balance_usd": router.cash_balance_usd,
            "audit_metrics": audit_metrics,
        }
    finally:
        gatekeeper_module.time = original_clock
        if replayer is not None:
            replayer.close()
        ledger.close()


def main() -> None:
    """Parse command-line options and run the backtest."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--latency-slots", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    asyncio.run(run_backtest(args.parquet, latency_slots=args.latency_slots, seed=args.seed))


if __name__ == "__main__":
    main()

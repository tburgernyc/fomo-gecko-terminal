import logging
from pathlib import Path
from typing import Any

import pandas as pd
import quantstats as qs

from src.db import cursor_scope, transaction_scope

logger = logging.getLogger(__name__)


def generate_morning_reconstruction(connection: Any) -> pd.DataFrame:
    with cursor_scope(connection) as cur:
        return cur.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN side = 'SELL' AND status IN ('OPEN', 'CLOSED', 'DRY_RUN_OPEN', 'DRY_RUN_CLOSED')
                                  THEN realized_pnl_usd ELSE 0.0 END), 0.0) AS closed_profit_usd,
                COUNT(CASE WHEN side = 'BUY' AND status IN ('OPEN', 'DRY_RUN_OPEN') THEN 1 END)
                    AS open_positions_count,
                COALESCE(SUM(CASE WHEN side = 'BUY' AND status IN ('OPEN', 'DRY_RUN_OPEN')
                                  THEN remaining_token_amount * COALESCE(entry_price_usd, 0)
                                       + unrealized_pnl_usd ELSE 0.0 END), 0.0)
                    AS open_exposure_usd,
                COALESCE(SUM(CASE WHEN side = 'BUY' AND status IN ('OPEN', 'DRY_RUN_OPEN')
                                  THEN unrealized_pnl_usd ELSE 0.0 END), 0.0)
                    AS open_unrealized_pnl_usd,
                COALESCE(SUM(fee_usd), 0.0) AS total_fees_usd,
                COUNT(CASE WHEN status IN ('FAILED', 'REJECTED') THEN 1 END)
                    AS failed_or_rejected_orders,
                COUNT(CASE WHEN status IN ('UNCERTAIN', 'SUBMITTED') THEN 1 END)
                    AS unresolved_uncertain_orders
            FROM execution_orders
            WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL 24 HOUR
            """
        ).df()


def run_morning_audit(
    connection: Any,
    starting_bankroll_usd: float,
    output_path: str = "morning_quantstats_tearsheet.html",
) -> pd.DataFrame:
    summary = generate_morning_reconstruction(connection)
    closed_profit = float(summary["closed_profit_usd"].iloc[0])
    open_exposure = float(summary["open_exposure_usd"].iloc[0])
    open_unrealized = float(summary["open_unrealized_pnl_usd"].iloc[0])
    fees = float(summary["total_fees_usd"].iloc[0])
    total_equity = starting_bankroll_usd + closed_profit + open_unrealized - fees
    with transaction_scope(connection) as cur:
        cur.execute(
            """
            INSERT OR REPLACE INTO equity_snapshots (
                ts, cash_usd, open_exposure_usd, total_equity_usd,
                daily_realized_pnl_usd, daily_fees_usd
            ) VALUES (CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
            """,
            [total_equity - open_exposure, open_exposure, total_equity, closed_profit, fees],
        )
    with cursor_scope(connection) as cur:
        equity = cur.execute(
            "SELECT ts, total_equity_usd FROM equity_snapshots ORDER BY ts"
        ).df()
    if len(equity) >= 3:
        equity["ts"] = pd.to_datetime(equity["ts"])
        returns = equity.set_index("ts")["total_equity_usd"].pct_change().dropna()
        if not returns.empty:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            qs.reports.html(
                returns,
                output=str(destination),
                title="FOMO + GeckoTerminal Daily Audit",
            )
            logger.info("Wrote QuantStats tear-sheet to %s", destination)
    logger.info("Morning reconstruction:\n%s", summary.to_string(index=False))
    return summary

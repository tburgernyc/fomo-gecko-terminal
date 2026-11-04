import asyncio
import hashlib
import logging
import time
from typing import Any

import duckdb
import httpx
import pandas as pd

from src.config import Settings, get_settings
from src.db import cursor_scope, transaction_scope
from src.execution.routers import ExecutionResult, OrderIntent, OrderRouter, SubmissionUncertain
from src.ingestion.gecko_scanner import CHAIN_TO_GECKO
from src.quant.allocator import PortfolioAllocator
from src.quant.indicators import evaluate_microstructure_alpha_gate
from src.utils.notifications import send_alert

logger = logging.getLogger(__name__)


def passes_memecoin_microstructure_gates(
    pool_metrics: dict[str, Any],
    ohlcv_df: pd.DataFrame | None = None,
    leader_entry_price: float = 0.0,
) -> tuple[bool, str]:
    """Apply anti-rug and anti-exit-liquidity rules to a prospective entry."""
    fdv = float(pool_metrics.get("fdv_usd") or 0.0)
    vol_24h = float(pool_metrics.get("volume_usd_h24") or 0.0)
    reserve_usd = float(pool_metrics.get("reserve_in_usd") or 0.0)

    if 30_000 <= fdv <= 75_000:
        return False, f"Rejected: Token in pre-migration sniper dump zone (FDV=${fdv:,.0f})"

    if fdv > 0 and fdv < 1_000_000 and (vol_24h / fdv) < 0.50:
        return False, f"Rejected: Weak Volume/FDV turnover ({vol_24h / fdv:.2f}x < 0.50x)"

    if ohlcv_df is not None and len(ohlcv_df) >= 12:
        recent = ohlcv_df.tail(12)
        green_ratio = (recent["close"] >= recent["open"]).mean()
        pct_changes = recent["close"].pct_change().dropna()
        if green_ratio > 0.85 and pct_changes.std() < 0.008:
            return False, "Rejected: Deceptive stair-step candlestick pattern detected"

        last_candle = recent.iloc[-1]
        if last_candle["close"] > last_candle["open"] * 1.03:
            return False, "Rejected: Refusing to chase >3% green spike candle; waiting for pullback"

    if "reserve_in_usd" in pool_metrics:
        if ohlcv_df is not None and not ohlcv_df.empty:
            candle = ohlcv_df.iloc[-1]
            candle_open = float(candle["open"])
            candle_high = float(candle["high"])
            candle_low = float(candle["low"])
            candle_close = float(candle["close"])
        else:
            candle_open = candle_high = candle_low = candle_close = float(
                pool_metrics.get("price_usd") or 0.0
            )
        return evaluate_microstructure_alpha_gate(
            fdv_usd=fdv,
            liquidity_usd=reserve_usd,
            volume_24h_usd=vol_24h,
            buys_m5=int(pool_metrics.get("buys_m5") or 0),
            sells_m5=int(pool_metrics.get("sells_m5") or 0),
            candle_open=candle_open,
            candle_high=candle_high,
            candle_low=candle_low,
            candle_close=candle_close,
            leader_entry_price=leader_entry_price,
        )

    return True, "Passed microstructure gates"


class ExecutionGatekeeper:
    """The sole order submission boundary: gate, persist intent, submit, then reconcile."""

    def __init__(
        self,
        connection: Any,
        scanner: Any,
        router: OrderRouter,
        settings: Settings | None = None,
    ):
        self.db = connection
        self.scanner = scanner
        self.router = router
        self.settings = settings or get_settings()
        self._submission_lock = asyncio.Lock()

    @staticmethod
    async def _notify(message: str, severity: str) -> None:
        try:
            await send_alert(message, severity)
        except Exception:
            logger.exception("Could not dispatch %s gatekeeper alert", severity)

    @staticmethod
    def order_id(trader_handle: str, network: str, token_address: str, side: str, bucket: int) -> str:
        material = f"{trader_handle}:{network}:{token_address}:{side}:{bucket}"
        return hashlib.sha256(material.encode()).hexdigest()[:24]

    def _circuit_breaker_tripped(self) -> tuple[bool, float]:
        with cursor_scope(self.db) as cur:
            row = cur.execute(
                """
                SELECT COALESCE(SUM(
                    CASE WHEN created_at >= CURRENT_TIMESTAMP - INTERVAL 24 HOUR
                         THEN realized_pnl_usd - fee_usd ELSE 0.0 END
                ), 0.0)
                + COALESCE(SUM(
                    CASE WHEN side = 'BUY' AND status IN ('OPEN', 'DRY_RUN_OPEN')
                         THEN unrealized_pnl_usd ELSE 0.0 END
                ), 0.0)
                FROM execution_orders
                """
            ).fetchone()
        net_pnl = float(row[0]) if row else 0.0
        return net_pnl <= -(self.settings.starting_bankroll_usd * self.settings.max_daily_loss_pct), net_pnl

    async def _reconcile_pending_orders(self) -> bool:
        with cursor_scope(self.db) as cur:
            pending = cur.execute(
                """
                SELECT client_order_id, tx_hash_or_exchange_id, trader_handle, side, network,
                       token_address, entry_price_usd, filled_token_amount, exit_tranche
                FROM execution_orders
                WHERE status IN ('INTENT', 'UNCERTAIN', 'SUBMITTED')
                ORDER BY created_at
                """
            ).fetchall()
        unresolved = False
        for (
            client_order_id,
            tx_id,
            trader_handle,
            side,
            network,
            token_address,
            entry_price,
            filled_amount,
            exit_tranche,
        ) in pending:
            reconciliation = await self.router.reconcile(client_order_id, tx_id)
            result = reconciliation if isinstance(reconciliation, ExecutionResult) else None
            state = result.status if result else reconciliation
            if state not in {"OPEN", "FAILED", "UNCERTAIN"}:
                raise ValueError(f"router returned invalid reconciliation state: {state}")
            with transaction_scope(self.db) as cur:
                if state != "UNCERTAIN":
                    cur.execute(
                        """
                        UPDATE execution_orders
                        SET status = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE client_order_id = ?
                        """,
                        [state, client_order_id],
                    )
                    if result and state == "OPEN" and result.filled_token_amount > 0:
                        cur.execute(
                            """
                            UPDATE execution_orders
                            SET filled_usd = ?, filled_token_amount = ?, fee_usd = ?,
                                tx_hash_or_exchange_id = COALESCE(?, tx_hash_or_exchange_id),
                                execution_detail = COALESCE(?, execution_detail),
                                entry_price_usd = COALESCE(NULLIF(?, 0), entry_price_usd),
                                remaining_token_amount = CASE WHEN side = 'BUY' THEN ? ELSE 0 END,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE client_order_id = ?
                            """,
                            [
                                result.filled_usd, result.filled_token_amount, result.fee_usd,
                                result.tx_id, result.detail, result.average_price_usd,
                                result.filled_token_amount if side == "BUY" else 0,
                                client_order_id,
                            ],
                        )
                        if side == "SELL":
                            self._apply_sell_fill(
                                cur, trader_handle, network, token_address,
                                result.filled_token_amount,
                                result.average_price_usd or None,
                                client_order_id,
                                exit_tranche,
                            )
                    elif state == "OPEN" and side == "SELL":
                        self._apply_sell_fill(
                            cur, trader_handle, network, token_address,
                            float(filled_amount or 0), entry_price, client_order_id,
                            exit_tranche,
                        )
            unresolved |= state == "UNCERTAIN"
        return unresolved

    async def reconcile_pending_orders(self) -> bool:
        async with self._submission_lock:
            return await self._reconcile_pending_orders()

    def _consensus_count(self, network: str, token_address: str) -> int:
        with cursor_scope(self.db) as cur:
            row = cur.execute(
                """
                SELECT COUNT(DISTINCT trader_handle)
                FROM copy_signals
                WHERE network = ? AND token_address = ? AND side = 'BUY'
                  AND received_at >= CURRENT_TIMESTAMP - (? * INTERVAL '1 second')
                """,
                [network, token_address, self.settings.consensus_window_seconds],
            ).fetchone()
        return int(row[0]) if row else 0

    def _open_token_amount(
        self, trader_handle: str, network: str, token_address: str
    ) -> float:
        with cursor_scope(self.db) as cur:
            row = cur.execute(
                """
                SELECT COALESCE(SUM(remaining_token_amount), 0.0)
                FROM execution_orders
                WHERE side = 'BUY' AND trader_handle = ? AND network = ? AND token_address = ?
                  AND status IN ('OPEN', 'DRY_RUN_OPEN')
                """,
                [trader_handle, network, token_address],
            ).fetchone()
        return max(0.0, float(row[0])) if row else 0.0

    @staticmethod
    def _apply_sell_fill(
        cur: duckdb.DuckDBPyConnection,
        trader_handle: str,
        network: str,
        token_address: str,
        sold_amount: float,
        exit_price: float | None,
        sell_order_id: str,
        exit_tranche: str | None = None,
    ) -> None:
        remaining_to_close = sold_amount
        realized_pnl = 0.0
        lots = cur.execute(
            """
            SELECT client_order_id, remaining_token_amount, entry_price_usd, unrealized_pnl_usd,
                   tranche_1_sold_amount, tranche_2_sold_amount
            FROM execution_orders
            WHERE side = 'BUY' AND trader_handle = ? AND network = ? AND token_address = ?
              AND status IN ('OPEN', 'DRY_RUN_OPEN') AND remaining_token_amount > 0
            ORDER BY created_at
            """,
            [trader_handle, network, token_address],
        ).fetchall()
        for buy_id, available, entry_price, unrealized, tranche_1_sold, tranche_2_sold in lots:
            if remaining_to_close <= 0:
                break
            close_amount = min(float(available), remaining_to_close)
            new_remaining = float(available) - close_amount
            if new_remaining <= 1e-12:
                new_remaining = 0.0
            cur.execute(
                """
                UPDATE execution_orders
                SET remaining_token_amount = ?,
                    unrealized_pnl_usd = ?,
                    tranche_1_sold_amount = ?,
                    tranche_2_sold_amount = ?,
                    status = CASE WHEN ? <= 1e-12 THEN 'CLOSED' ELSE status END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE client_order_id = ?
                """,
                [
                    max(new_remaining, 0),
                    float(unrealized or 0) * new_remaining / float(available)
                    if available and new_remaining > 1e-12 else 0,
                    float(tranche_1_sold or 0) + (
                        close_amount if exit_tranche == "TAKE_PROFIT_1" else 0.0
                    ),
                    float(tranche_2_sold or 0) + (
                        close_amount if exit_tranche == "TAKE_PROFIT_2" else 0.0
                    ),
                    new_remaining,
                    buy_id,
                ],
            )
            if exit_price is not None and entry_price is not None:
                realized_pnl += close_amount * (exit_price - float(entry_price))
            remaining_to_close -= close_amount
        cur.execute(
            "UPDATE execution_orders SET realized_pnl_usd = ? WHERE client_order_id = ?",
            [realized_pnl, sell_order_id],
        )

    async def process_copy_signal(
        self,
        client: httpx.AsyncClient,
        trader_handle: str,
        network: str,
        token_address: str,
        side: str,
        sleeve_weight: float = 0.10,
        leader_entry_price: float = 0.0,
        exit_token_amount: float | None = None,
        exit_tranche: str | None = None,
    ) -> str:
        normalized_side = side.upper()
        if normalized_side not in {"BUY", "SELL"}:
            raise ValueError(f"unsupported order side: {side}")
        if not trader_handle or not network or not token_address:
            raise ValueError("trader, network, and token are required")
        if exit_token_amount is not None and (
            normalized_side != "SELL" or not 0 < exit_token_amount
        ):
            raise ValueError("exit token amount must be positive and apply only to SELL signals")
        normalized_network = str(network).strip().lower()
        execution_network = CHAIN_TO_GECKO.get(normalized_network, normalized_network)

        async with self._submission_lock:
            bucket = int(time.time() // 300)
            client_order_id = self.order_id(
                trader_handle, execution_network, token_address, normalized_side, bucket
            )
            with cursor_scope(self.db) as cur:
                existing = cur.execute(
                    "SELECT status FROM execution_orders WHERE client_order_id = ?",
                    [client_order_id],
                ).fetchone()
            if existing:
                if existing[0] in {"INTENT", "SUBMITTED", "UNCERTAIN"}:
                    await self._reconcile_pending_orders()
                    with cursor_scope(self.db) as cur:
                        existing = cur.execute(
                            "SELECT status FROM execution_orders WHERE client_order_id = ?",
                            [client_order_id],
                        ).fetchone()
                    if existing and existing[0] in {"INTENT", "SUBMITTED", "UNCERTAIN"}:
                        return "BLOCKED_UNRESOLVED_ORDER"
                if existing:
                    return f"DUPLICATE_SKIPPED:{existing[0]}"

            supports_network = getattr(self.router, "supports_network", None)
            if supports_network is not None and not supports_network(execution_network):
                return await self._reject(
                    client_order_id, trader_handle, execution_network, token_address,
                    "", normalized_side, f"unsupported_execution_network:{execution_network}",
                )

            if await self._reconcile_pending_orders():
                return "BLOCKED_UNRESOLVED_ORDER"

            tripped, net_pnl = self._circuit_breaker_tripped()
            if normalized_side == "BUY" and tripped:
                logger.error("Circuit breaker tripped: rolling 24h net PnL %.2f USD", net_pnl)
                await self._notify(
                    f"Daily circuit breaker halted {trader_handle}'s BUY signal for "
                    f"{token_address} on {execution_network}. Rolling 24h net PnL: "
                    f"${net_pnl:.2f}.",
                    "ALERT",
                )
                return "HALTED_CIRCUIT_BREAKER"

            signal_id = hashlib.sha256(
                f"{client_order_id}:{trader_handle}:{time.time_ns()}".encode()
            ).hexdigest()[:24]
            with transaction_scope(self.db) as cur:
                cur.execute(
                    """
                    INSERT INTO copy_signals (signal_id, trader_handle, network, token_address, side)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [signal_id, trader_handle, execution_network, token_address, normalized_side],
                )

            market = dict(
                await self.scanner.evaluate_token(client, execution_network, token_address)
            )
            execution_network = str(market.get("network") or execution_network)
            pool_metrics = {
                "fdv_usd": market.get("fdv_usd"),
                "volume_usd_h24": (
                    market.get("volume_usd_h24")
                    if market.get("volume_usd_h24") is not None
                    else market.get("volume_24h_usd")
                ),
                "reserve_in_usd": (
                    market.get("reserve_in_usd")
                    if market.get("reserve_in_usd") is not None
                    else market.get("liquidity_usd")
                ),
                "buys_m5": market.get("buys_m5", 0),
                "sells_m5": market.get("sells_m5", 0),
                "price_usd": market.get("price_usd"),
            }
            market["volume_24h_usd"] = pool_metrics["volume_usd_h24"]
            market["reserve_in_usd"] = pool_metrics["reserve_in_usd"]
            consensus = self._consensus_count(execution_network, token_address)
            reject_reason = ""
            if not market.get("passed"):
                reject_reason = str(market.get("reason") or "market_gate_failed")
            elif normalized_side == "BUY" and market.get("rsi_14") is None:
                reject_reason = "missing_rsi"
            elif (
                normalized_side == "BUY"
                and float(market["rsi_14"]) > self.settings.max_entry_rsi_14
                and consensus < 2
            ):
                reject_reason = f"rsi_overbought:{float(market['rsi_14']):.1f}"
            if normalized_side == "BUY" and not reject_reason:
                ohlcv_df = market.get("ohlcv_df")
                if not isinstance(ohlcv_df, pd.DataFrame):
                    candles = market.get("candles")
                    ohlcv_df = (
                        pd.DataFrame(candles)
                        if isinstance(candles, list) and candles
                        else None
                    )
                passed, microstructure_reason = passes_memecoin_microstructure_gates(
                    pool_metrics, ohlcv_df, leader_entry_price
                )
                if not passed:
                    reject_reason = microstructure_reason
                    logger.warning(
                        "Microstructure gate rejected %s (%s on %s): %s",
                        trader_handle,
                        token_address,
                        execution_network,
                        reject_reason,
                    )

            with transaction_scope(self.db) as cur:
                cur.execute(
                    """
                    INSERT INTO market_snapshots (
                        snapshot_id, network, token_address, pool_address, price_usd,
                        reserve_in_usd, volume_24h_usd, fdv_usd, buys_m5, sells_m5,
                        rsi_14, macd_hist, atr_14, passed_gate, reject_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        signal_id, execution_network, token_address, market.get("pool_address", ""),
                        market.get("price_usd"), market.get("reserve_in_usd"),
                        market.get("volume_24h_usd"), pool_metrics["fdv_usd"],
                        pool_metrics["buys_m5"],
                        pool_metrics["sells_m5"], market.get("rsi_14"),
                        market.get("macd_hist"),
                        market.get("atr_14"),
                        not reject_reason, reject_reason or None,
                    ],
                )
                for candle in market.get("candles", []):
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO market_candles (
                            network, token_address, candle_ts, open, high, low, close, volume
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            execution_network, token_address, candle["timestamp"], candle["open"],
                            candle["high"], candle["low"], candle["close"], candle["volume"],
                        ],
                    )
                if reject_reason:
                    cur.execute(
                        """
                        INSERT INTO execution_orders (
                            client_order_id, trader_handle, network, token_address, pool_address,
                            side, status, requested_usd, failure_reason
                        ) VALUES (?, ?, ?, ?, ?, ?, 'REJECTED', 0, ?)
                        """,
                        [
                            client_order_id, trader_handle, execution_network, token_address,
                            market.get("pool_address", ""), normalized_side, reject_reason,
                        ],
                    )
            if reject_reason:
                await self._notify(
                    f"Copy signal rejected for {trader_handle} ({normalized_side} "
                    f"{token_address} on {execution_network}): {reject_reason}.",
                    "WARNING",
                )
                return f"REJECTED:{reject_reason}"

            price = float(market.get("price_usd") or 0)
            reserve = float(market.get("reserve_in_usd") or 0)
            if price <= 0 or reserve <= 0:
                return await self._reject(
                    client_order_id, trader_handle, execution_network, token_address,
                    market.get("pool_address", ""), normalized_side, "invalid_market_price_or_liquidity",
                )
            if normalized_side == "SELL":
                available_tokens = self._open_token_amount(
                    trader_handle, execution_network, token_address
                )
                if available_tokens <= 0:
                    return await self._reject(
                        client_order_id, trader_handle, execution_network, token_address,
                        market.get("pool_address", ""), normalized_side, "no_open_copied_position",
                    )
                sell_amount = min(
                    available_tokens,
                    exit_token_amount if exit_token_amount is not None else available_tokens,
                )
                requested_notional = min(
                    sell_amount * price,
                    reserve * self.settings.max_pool_liquidity_impact_pct,
                )
                token_amount = min(sell_amount, requested_notional / price)
            else:
                requested_notional = 0.0
                token_amount = 0.0

            ticket = PortfolioAllocator.size_ticket_usd(
                bankroll_usd=self.settings.starting_bankroll_usd,
                sleeve_weight=min(max(sleeve_weight, 0), 0.15),
                pool_liquidity_usd=reserve,
                max_ticket_pct=self.settings.max_single_ticket_pct,
                max_pool_impact_pct=self.settings.max_pool_liquidity_impact_pct,
                platform_fee_rate=self.router.fee_rate,
                minimum_fee_usd=self.router.minimum_fee_usd,
            )
            if normalized_side == "SELL":
                ticket = min(ticket, requested_notional)
            if ticket <= 0:
                return await self._reject(
                    client_order_id, trader_handle, execution_network, token_address,
                    market.get("pool_address", ""), normalized_side, "ticket_exceeds_fee_limit_or_below_floor",
                )
            if normalized_side == "SELL":
                token_amount = min(token_amount, ticket / price)

            intent = OrderIntent(
                client_order_id=client_order_id,
                trader_handle=trader_handle,
                network=execution_network,
                token_address=token_address,
                side=normalized_side,
                notional_usd=ticket,
                price_usd=price,
                pool_reserve_usd=reserve,
                token_amount=token_amount,
            )
            exit_campaign_id: str | None = None
            if normalized_side == "BUY":
                with cursor_scope(self.db) as cur:
                    active_campaign = cur.execute(
                        """
                        SELECT exit_campaign_id
                        FROM execution_orders
                        WHERE side = 'BUY' AND trader_handle = ? AND network = ?
                          AND token_address = ? AND status IN ('OPEN', 'DRY_RUN_OPEN')
                          AND remaining_token_amount > 0
                        ORDER BY created_at
                        LIMIT 1
                        """,
                        [trader_handle, execution_network, token_address],
                    ).fetchone()
                exit_campaign_id = (
                    str(active_campaign[0]) if active_campaign and active_campaign[0]
                    else client_order_id
                )
            with transaction_scope(self.db) as cur:
                cur.execute(
                    """
                    INSERT INTO execution_orders (
                        client_order_id, trader_handle, network, token_address, pool_address,
                        side, status, requested_usd, entry_atr_usd, exit_campaign_id,
                        exit_tranche
                    ) VALUES (?, ?, ?, ?, ?, ?, 'INTENT', ?, ?, ?, ?)
                    """,
                    [
                        client_order_id, trader_handle, execution_network, token_address,
                        market.get("pool_address", ""), normalized_side, ticket,
                        market.get("atr_14") if normalized_side == "BUY" else None,
                        exit_campaign_id, exit_tranche,
                    ],
                )
                cur.execute(
                    """
                    UPDATE execution_orders
                    SET status = 'SUBMITTED', updated_at = CURRENT_TIMESTAMP
                    WHERE client_order_id = ? AND status = 'INTENT'
                    """,
                    [client_order_id],
                )
            try:
                result = await self.router.submit(intent)
            except SubmissionUncertain as exc:
                logger.exception("Order submission outcome is uncertain for %s", client_order_id)
                with transaction_scope(self.db) as cur:
                    cur.execute(
                        """
                        UPDATE execution_orders SET status = 'UNCERTAIN', failure_reason = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE client_order_id = ?
                        """,
                        [str(exc), client_order_id],
                    )
                    cur.execute(
                        "UPDATE execution_orders SET tx_hash_or_exchange_id = ? WHERE client_order_id = ?",
                        [exc.tx_id, client_order_id],
                    )
                return f"UNCERTAIN:{client_order_id}"
            except (httpx.HTTPError, asyncio.TimeoutError) as exc:
                logger.exception("Order submission outcome is uncertain for %s", client_order_id)
                with transaction_scope(self.db) as cur:
                    cur.execute(
                        """
                        UPDATE execution_orders SET status = 'UNCERTAIN', failure_reason = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE client_order_id = ?
                        """,
                        [str(exc), client_order_id],
                    )
                return f"UNCERTAIN:{client_order_id}"
            except (RuntimeError, ValueError) as exc:
                logger.error("Order submission failed for %s: %s", client_order_id, exc)
                with transaction_scope(self.db) as cur:
                    cur.execute(
                        """
                        UPDATE execution_orders SET status = 'FAILED', failure_reason = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE client_order_id = ?
                        """,
                        [str(exc), client_order_id],
                    )
                return f"FAILED:{client_order_id}"
            if result.status not in {"SUBMITTED", "OPEN", "DRY_RUN_OPEN"}:
                raise ValueError(f"router returned invalid submission status: {result.status}")
            with transaction_scope(self.db) as cur:
                cur.execute(
                    """
                    UPDATE execution_orders
                    SET status = ?, filled_usd = ?, filled_token_amount = ?, entry_price_usd = ?,
                        high_water_price_usd = ?, fee_usd = ?, tx_hash_or_exchange_id = ?,
                        execution_detail = ?,
                        remaining_token_amount = CASE WHEN side = 'BUY' THEN ? ELSE 0 END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE client_order_id = ?
                    """,
                    [
                        result.status, result.filled_usd, result.filled_token_amount,
                        result.average_price_usd or price,
                        result.average_price_usd or price, result.fee_usd, result.tx_id,
                        result.detail,
                        result.filled_token_amount if normalized_side == "BUY" else 0,
                        client_order_id,
                    ],
                )
                if normalized_side == "SELL" and result.status in {"OPEN", "DRY_RUN_OPEN"}:
                    self._apply_sell_fill(
                        cur, trader_handle, execution_network, token_address,
                        result.filled_token_amount, result.average_price_usd or price, client_order_id,
                        exit_tranche,
                    )
            await self._notify(
                f"Copy signal executed for {trader_handle}: {normalized_side} "
                f"{token_address} on {execution_network}, ${ticket:.2f}, "
                f"status={result.status}, order={client_order_id}.",
                "SUCCESS",
            )
            return f"EXECUTED:{result.status}:{client_order_id}:${ticket:.2f}"

    async def _reject(
        self, client_order_id: str, trader_handle: str, network: str, token_address: str,
        pool_address: str, side: str, reason: str,
    ) -> str:
        with transaction_scope(self.db) as cur:
            cur.execute(
                """
                INSERT INTO execution_orders (
                    client_order_id, trader_handle, network, token_address, pool_address,
                    side, status, requested_usd, failure_reason
                ) VALUES (?, ?, ?, ?, ?, ?, 'REJECTED', 0, ?)
                """,
                [client_order_id, trader_handle, network, token_address, pool_address, side, reason],
            )
        if "liquidity" in reason.lower() or "rsi" in reason.lower():
            await self._notify(
                f"Copy signal rejected for {trader_handle} ({side} {token_address} "
                f"on {network}): {reason}.",
                "WARNING",
            )
        return f"REJECTED:{reason}"

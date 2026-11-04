"""Latency-aware simulated Solana router for event-driven historical tests."""

from __future__ import annotations

import logging
import random
from typing import Any

from src.execution.routers import ExecutionResult, OrderIntent
from src.ingestion.historical_replayer import HistoricalReplayer

logger = logging.getLogger(__name__)


class SimulatedOnChainRouter:
    """Simulate slot latency, MEV slippage, priority bidding, and dropped bundles."""

    fee_rate = 0.0
    minimum_fee_usd = 0.0
    JITO_TIP_SOL = 0.001
    COMPUTE_FEE_SOL = 0.00005
    PRIORITY_FEE_RANGE_SOL = (0.00001, 0.00005)

    def __init__(
        self,
        replayer: HistoricalReplayer,
        *,
        latency_slots: int = 2,
        starting_balance_usd: float = 1_000.0,
        sol_price_usd: float = 150.0,
        failure_probability: float = 0.15,
        seed: int | None = 20261103,
    ) -> None:
        if latency_slots < 0:
            raise ValueError("latency_slots cannot be negative")
        if starting_balance_usd <= 0 or sol_price_usd <= 0:
            raise ValueError("starting balance and SOL price must be positive")
        if not 0 <= failure_probability <= 1:
            raise ValueError("failure_probability must be between zero and one")
        self.replayer = replayer
        self.latency_slots = latency_slots
        self.sol_price_usd = sol_price_usd
        self.failure_probability = failure_probability
        self._rng = random.Random(seed)
        self._priority_rng = random.Random(None if seed is None else seed + 1)
        self.cash_balance_usd = starting_balance_usd
        self.total_fees_paid_usd = 0.0
        self.attempts: list[dict[str, Any]] = []

    def supports_network(self, network: str) -> bool:
        """Accept only Solana aliases."""
        return str(network).strip().lower() in {"solana", "sol", "1399811149"}

    async def submit(self, intent: OrderIntent) -> ExecutionResult:
        """Implement the live OrderRouter submission contract used by the gatekeeper."""
        if intent.side == "BUY":
            return await self.execute_buy(intent)
        if intent.side == "SELL":
            return await self.execute_sell(intent)
        raise ValueError(f"unsupported simulated order side: {intent.side}")

    async def execute_buy(self, intent: OrderIntent) -> ExecutionResult:
        """Simulate a buy routed through a competing Jito bundle market."""
        if intent.side != "BUY":
            raise ValueError("execute_buy requires a BUY intent")
        return await self._execute(intent)

    async def execute_sell(self, intent: OrderIntent) -> ExecutionResult:
        """Simulate a sell routed through a competing Jito bundle market."""
        if intent.side != "SELL":
            raise ValueError("execute_sell requires a SELL intent")
        return await self._execute(intent)

    async def _execute(self, intent: OrderIntent) -> ExecutionResult:
        if intent.notional_usd <= 0 or intent.pool_reserve_usd <= 0 or intent.price_usd <= 0:
            raise ValueError("simulated order requires positive notional, price, and liquidity")

        priority_fee_sol = self._priority_rng.uniform(*self.PRIORITY_FEE_RANGE_SOL)
        fee_usd = (
            self.JITO_TIP_SOL + self.COMPUTE_FEE_SOL + priority_fee_sol
        ) * self.sol_price_usd
        self.cash_balance_usd -= fee_usd
        self.total_fees_paid_usd += fee_usd
        target_slot = self.replayer.current_slot + self.latency_slots
        attempt: dict[str, Any] = {
            "client_order_id": intent.client_order_id,
            "side": intent.side,
            "signal_slot": self.replayer.current_slot,
            "execution_slot": target_slot,
            "notional_usd": intent.notional_usd,
            "priority_fee_sol": priority_fee_sol,
            "fee_usd": fee_usd,
            "status": "FAILED",
            "fill_price_usd": None,
            "failure_reason": None,
        }
        try:
            future_state = self.replayer.get_pool_state_at_slot(intent.token_address, target_slot)
            if future_state is None:
                raise RuntimeError(f"no historical pool state at or after execution slot {target_slot}")

            base_slippage_pct = intent.notional_usd / float(future_state["reserve_in_usd"]) * 100
            volatility_penalty_pct = self._rng.uniform(0.5, 2.0)
            total_slippage_pct = base_slippage_pct + volatility_penalty_pct
            price_direction = 1 if intent.side == "BUY" else -1
            fill_price = float(future_state["price_usd"]) * (
                1 + price_direction * total_slippage_pct / 100
            )
            if fill_price <= 0:
                raise RuntimeError("simulated slippage produced a non-positive fill price")

            if self._rng.random() < self.failure_probability:
                raise RuntimeError("simulated Jito bundle dropped during slot congestion")

            if intent.side == "BUY":
                self.cash_balance_usd -= intent.notional_usd
                filled_tokens = intent.notional_usd / fill_price
                filled_usd = intent.notional_usd
            else:
                filled_tokens = intent.token_amount
                if filled_tokens <= 0:
                    raise ValueError("simulated SELL requires a positive token amount")
                filled_usd = filled_tokens * fill_price
                self.cash_balance_usd += filled_usd

            attempt.update(
                {
                    "status": "OPEN",
                    "fill_price_usd": fill_price,
                    "failure_reason": None,
                }
            )
            logger.info(
                "Simulated %s %s at slot %d: price=%.10f slippage=%.3f%% fee=$%.4f",
                intent.side,
                intent.token_address,
                target_slot,
                fill_price,
                total_slippage_pct,
                fee_usd,
            )
            return ExecutionResult(
                status="OPEN",
                filled_usd=filled_usd,
                filled_token_amount=filled_tokens,
                fee_usd=fee_usd,
                average_price_usd=fill_price,
                tx_id=f"sim-{intent.client_order_id}",
                detail=(
                    f"signal_slot={self.replayer.current_slot};execution_slot={target_slot};"
                    f"base_slippage_pct={base_slippage_pct:.6f};"
                    f"volatility_penalty_pct={volatility_penalty_pct:.6f};"
                    f"priority_fee_sol={priority_fee_sol:.8f}"
                ),
            )
        except (RuntimeError, ValueError) as exc:
            attempt["failure_reason"] = str(exc)
            logger.warning(
                "Simulated %s attempt failed for %s: %s",
                intent.side,
                intent.token_address,
                exc,
            )
            raise
        finally:
            self.attempts.append(attempt)

    async def reconcile(self, client_order_id: str, tx_id: str | None) -> str:
        """Simulated submissions resolve synchronously and never remain pending."""
        return "UNCERTAIN"

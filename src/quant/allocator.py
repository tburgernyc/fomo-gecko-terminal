import numpy as np
import pandas as pd
from skfolio.optimization import EqualWeighted, InverseVolatility


class PortfolioAllocator:
    @staticmethod
    def compute_sleeve_weights(returns: pd.DataFrame, max_weight: float = 0.15) -> dict[str, float]:
        if not 0 < max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")
        if returns.empty or returns.shape[1] == 0:
            return {}
        if returns.shape[0] < 30:
            model = EqualWeighted()
            model.fit(returns)
            base = np.asarray(model.weights_, dtype=float)
        else:
            try:
                model = InverseVolatility()
                model.fit(returns)
                base = np.asarray(model.weights_, dtype=float)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                model = EqualWeighted()
                model.fit(returns)
                base = np.asarray(model.weights_, dtype=float)
        base = np.maximum(base, 0)
        if base.shape != (returns.shape[1],) or not np.isfinite(base).all() or base.sum() <= 0:
            raise ValueError("skfolio returned invalid portfolio weights")
        base /= base.sum()
        weights = np.zeros_like(base)
        active = base > 0
        remaining = 1.0
        while active.any() and remaining > 1e-12:
            proposed = remaining * base[active] / base[active].sum()
            active_indices = np.flatnonzero(active)
            capped = proposed > max_weight
            if not capped.any():
                weights[active_indices] = proposed
                break
            capped_indices = active_indices[capped]
            weights[capped_indices] = max_weight
            remaining -= max_weight * len(capped_indices)
            active[capped_indices] = False
        return {name: float(weight) for name, weight in zip(returns.columns, weights, strict=True)}

    @staticmethod
    def size_ticket_usd(
        bankroll_usd: float,
        sleeve_weight: float,
        pool_liquidity_usd: float,
        max_ticket_pct: float = 0.06,
        max_pool_impact_pct: float = 0.02,
        platform_fee_rate: float = 0.0045,
        minimum_fee_usd: float = 0.95,
        maximum_fee_pct: float = 0.015,
    ) -> float:
        values = (
            bankroll_usd, sleeve_weight, pool_liquidity_usd, max_ticket_pct,
            max_pool_impact_pct, platform_fee_rate, minimum_fee_usd, maximum_fee_pct,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("ticket sizing inputs must be finite")
        if bankroll_usd <= 0 or sleeve_weight < 0 or pool_liquidity_usd <= 0:
            raise ValueError("bankroll and liquidity must be positive and weight non-negative")
        ticket = min(
            bankroll_usd * sleeve_weight,
            bankroll_usd * max_ticket_pct,
            pool_liquidity_usd * max_pool_impact_pct,
        )
        fee = max(minimum_fee_usd, ticket * platform_fee_rate)
        if ticket <= 0 or fee / ticket > maximum_fee_pct:
            return 0.0
        return round(ticket, 2)

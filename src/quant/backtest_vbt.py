from collections.abc import Iterable
from math import isfinite

import pandas as pd
import vectorbt as vbt
import numpy as np


DEFAULT_FRICTION_PER_LEG = 0.008
TARGET_PROFIT_FACTOR = 1.45
MAXIMUM_DRAWDOWN = 0.15


def constant_product_slippage(
    liquidity_usd: float | pd.Series,
    position_size_usd: float,
    index: pd.Index,
) -> pd.Series:
    """Estimate conservative CPMM price impact assuming equal-value pool reserves."""
    if position_size_usd <= 0 or not isfinite(position_size_usd):
        raise ValueError("position size must be finite and positive")
    liquidity = (
        liquidity_usd.reindex(index).astype(float)
        if isinstance(liquidity_usd, pd.Series)
        else pd.Series(float(liquidity_usd), index=index)
    )
    if not np.isfinite(liquidity.to_numpy()).all() or (liquidity <= 0).any():
        raise ValueError("liquidity must be finite, positive, and aligned to close")
    reserve_fraction = position_size_usd / liquidity
    if (reserve_fraction >= 0.5).any():
        raise ValueError("position size must be smaller than one side of pool liquidity")
    return (1 + 2 * reserve_fraction) ** 2 - 1


def target_metrics_met(profit_factor: float, max_drawdown: float) -> bool:
    return (
        not np.isnan(profit_factor)
        and profit_factor > TARGET_PROFIT_FACTOR
        and isfinite(max_drawdown)
        and abs(max_drawdown) < MAXIMUM_DRAWDOWN
    )


def run_slippage_adjusted_backtest(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    liquidity_usd: float | pd.Series,
    position_size_usd: float,
    *,
    starting_cash_usd: float = 10_000.0,
    latency_slots: int = 1,
    friction_per_leg: float = DEFAULT_FRICTION_PER_LEG,
) -> dict[str, object]:
    if close.empty or len(close) != len(entries) or len(close) != len(exits):
        raise ValueError("close, entries, and exits must be non-empty series of equal length")
    if not close.index.equals(entries.index) or not close.index.equals(exits.index):
        raise ValueError("close, entries, and exits must have identical indexes")
    if not np.isfinite(close.to_numpy(dtype=float)).all() or (close <= 0).any():
        raise ValueError("close prices must be finite and positive")
    if latency_slots < 0:
        raise ValueError("latency_slots must be non-negative")
    if not 0 <= friction_per_leg < 1:
        raise ValueError("friction_per_leg must be in [0, 1)")
    if starting_cash_usd <= 0 or not isfinite(starting_cash_usd):
        raise ValueError("starting cash must be finite and positive")

    entry_signals = entries.astype(bool).shift(latency_slots, fill_value=False)
    exit_signals = exits.astype(bool).shift(latency_slots, fill_value=False)
    impact = constant_product_slippage(liquidity_usd, position_size_usd, close.index)
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=entry_signals,
        exits=exit_signals,
        size=position_size_usd,
        size_type="value",
        init_cash=starting_cash_usd,
        fees=friction_per_leg,
        slippage=impact,
        freq="400ms",
    )
    raw_profit_factor = portfolio.trades.profit_factor()
    profit_factor = float(raw_profit_factor) if raw_profit_factor is not None else float("nan")
    max_drawdown = float(portfolio.max_drawdown())
    metrics: dict[str, object] = {
        "total_return": float(portfolio.total_return()),
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "target_metrics_met": target_metrics_met(profit_factor, max_drawdown),
        "latency_slots": latency_slots,
        "latency_seconds": latency_slots * 0.4,
        "friction_per_leg": friction_per_leg,
        "mean_amm_slippage": float(impact.mean()),
    }
    return {"portfolio": portfolio, "metrics": metrics}


def sweep_exit_parameters(
    close: pd.Series,
    entries: pd.Series,
    trailing_stops: Iterable[float] = (0.08, 0.12, 0.16),
    time_stops: Iterable[int] = (30, 60, 90),
    *,
    liquidity_usd: float | pd.Series | None = None,
    position_size_usd: float = 0.0,
    latency_slots: int = 1,
    friction_per_leg: float = DEFAULT_FRICTION_PER_LEG,
) -> pd.DataFrame:
    if close.empty or len(close) != len(entries):
        raise ValueError("close and entries must be non-empty series of equal length")
    if not close.index.equals(entries.index):
        raise ValueError("close and entries must have identical indexes")
    rows = []
    for stop_pct in trailing_stops:
        if not 0 < stop_pct < 1:
            raise ValueError("trailing stops must be in (0, 1)")
        for max_bars in time_stops:
            if max_bars <= 0:
                raise ValueError("time stops must be positive")
            exit_signal = pd.Series(False, index=close.index)
            entry_positions = np.flatnonzero(entries.to_numpy(dtype=bool))
            for start in entry_positions:
                exit_at = min(start + max_bars, len(close) - 1)
                exit_signal.iloc[exit_at] = True
            delayed_entries = entries.astype(bool).shift(latency_slots, fill_value=False)
            delayed_exits = exit_signal.shift(latency_slots, fill_value=False)
            slippage = (
                constant_product_slippage(liquidity_usd, position_size_usd, close.index)
                if liquidity_usd is not None
                else 0.0
            )
            portfolio = vbt.Portfolio.from_signals(
                close,
                entries=delayed_entries,
                exits=delayed_exits,
                **(
                    {"size": position_size_usd, "size_type": "value"}
                    if position_size_usd > 0 else {}
                ),
                sl_stop=stop_pct,
                sl_trail=True,
                fees=friction_per_leg,
                slippage=slippage,
                freq="400ms",
            )
            raw_profit_factor = portfolio.trades.profit_factor()
            rows.append({
                "trailing_stop_pct": stop_pct,
                "time_stop_bars": max_bars,
                "total_return": float(portfolio.total_return()),
                "sharpe_ratio": float(portfolio.sharpe_ratio()),
                "max_drawdown": float(portfolio.max_drawdown()),
                "profit_factor": (
                    float(raw_profit_factor) if raw_profit_factor is not None else float("nan")
                ),
                "target_metrics_met": target_metrics_met(
                    float(raw_profit_factor) if raw_profit_factor is not None else float("nan"),
                    float(portfolio.max_drawdown()),
                ),
            })
    return pd.DataFrame(rows).sort_values("total_return", ascending=False, ignore_index=True)

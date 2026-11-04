from collections.abc import Iterable
from math import isfinite

import pandas as pd
import vectorbt as vbt
import numpy as np


DEFAULT_FRICTION_PER_LEG = 0.008
TARGET_PROFIT_FACTOR = 1.65
MAXIMUM_DRAWDOWN = 0.15
CAMPAIGN_TAKE_PROFIT_1 = 0.28
CAMPAIGN_TAKE_PROFIT_2 = 0.75
CAMPAIGN_HARD_STOP = 0.11
CAMPAIGN_TRAILING_ATR_MULTIPLIER = 2.2
CAMPAIGN_EXPIRATION_SLOTS = 2_000


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


def run_three_tranche_campaign_backtest(
    close: pd.Series,
    entries: pd.Series,
    *,
    atr: pd.Series | None = None,
    slots: pd.Series | None = None,
    position_size_usd: float = 100.0,
    friction_per_leg: float = DEFAULT_FRICTION_PER_LEG,
    expiration_slots: int = CAMPAIGN_EXPIRATION_SLOTS,
) -> dict[str, object]:
    """Simulate deterministic 40/40/20 exits, hard stops, ATR trails, and expiry.

    One campaign is active at a time. `slots` may supply real slot numbers; when omitted,
    consecutive series positions are treated as slots.
    """
    if close.empty or len(close) != len(entries) or not close.index.equals(entries.index):
        raise ValueError("close and entries must be non-empty series with identical indexes")
    if not np.isfinite(close.to_numpy(dtype=float)).all() or (close <= 0).any():
        raise ValueError("close prices must be finite and positive")
    if position_size_usd <= 0 or not isfinite(position_size_usd):
        raise ValueError("position size must be finite and positive")
    if not 0 <= friction_per_leg < 1:
        raise ValueError("friction_per_leg must be in [0, 1)")
    if expiration_slots <= 0:
        raise ValueError("expiration_slots must be positive")
    if atr is not None and (
        len(atr) != len(close) or not close.index.equals(atr.index)
        or not np.isfinite(atr.to_numpy(dtype=float)).all()
        or (atr < 0).any()
    ):
        raise ValueError("ATR must be finite, non-negative, and aligned to close")
    if slots is not None and (
        len(slots) != len(close) or not close.index.equals(slots.index)
        or not np.isfinite(slots.to_numpy(dtype=float)).all()
        or (slots.diff().dropna() < 0).any()
    ):
        raise ValueError("slots must be finite, non-decreasing, and aligned to close")

    slot_values = (
        slots.to_numpy(dtype=np.int64)
        if slots is not None
        else np.arange(len(close), dtype=np.int64)
    )
    prices = close.to_numpy(dtype=float)
    atr_values = atr.to_numpy(dtype=float) if atr is not None else None
    entry_values = entries.to_numpy(dtype=bool)
    fills: list[dict[str, object]] = []
    campaign_number = 0
    index = 0
    while index < len(close):
        if not entry_values[index]:
            index += 1
            continue
        campaign_number += 1
        entry_index = index
        entry_slot = int(slot_values[index])
        entry_price = float(prices[index])
        high_water = entry_price
        remaining_fraction = 1.0
        tranche_1_done = False
        tranche_2_done = False
        exit_index = index

        for exit_index in range(index + 1, len(close)):
            price = float(prices[exit_index])
            held_slots = int(slot_values[exit_index]) - entry_slot
            high_water = max(high_water, price)
            exit_fraction: float | None = None
            tranche: str | None = None
            if price <= entry_price * (1 - CAMPAIGN_HARD_STOP):
                exit_fraction, tranche = remaining_fraction, "HARD_STOP_LOSS"
            elif held_slots >= expiration_slots:
                exit_fraction, tranche = remaining_fraction, "TIME_STOP"
            elif tranche_2_done:
                current_atr = (
                    float(atr_values[exit_index]) if atr_values is not None else 0.0
                )
                trailing_stop = (
                    high_water - CAMPAIGN_TRAILING_ATR_MULTIPLIER * current_atr
                    if current_atr > 0
                    else high_water * (1 - 0.12)
                )
                if price <= trailing_stop:
                    exit_fraction, tranche = remaining_fraction, "TRAILING_ATR"
            elif tranche_1_done and price <= entry_price:
                exit_fraction, tranche = remaining_fraction, "BREAK_EVEN_STOP"
            elif not tranche_1_done and price >= entry_price * (1 + CAMPAIGN_TAKE_PROFIT_1):
                exit_fraction, tranche = min(0.40, remaining_fraction), "TAKE_PROFIT_1"
            elif tranche_1_done and not tranche_2_done and (
                price >= entry_price * (1 + CAMPAIGN_TAKE_PROFIT_2)
            ):
                exit_fraction, tranche = min(0.40, remaining_fraction), "TAKE_PROFIT_2"

            if exit_fraction is None or tranche is None:
                continue
            exit_notional = position_size_usd * exit_fraction
            quantity = exit_notional / entry_price
            gross_pnl = quantity * (price - entry_price)
            entry_fee = exit_notional * friction_per_leg
            exit_fee = quantity * price * friction_per_leg
            fills.append(
                {
                    "campaign": campaign_number,
                    "entry_index": entry_index,
                    "exit_index": exit_index,
                    "entry_slot": entry_slot,
                    "exit_slot": int(slot_values[exit_index]),
                    "tranche": tranche,
                    "position_fraction": exit_fraction,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "gross_pnl_usd": gross_pnl,
                    "fees_usd": entry_fee + exit_fee,
                    "net_pnl_usd": gross_pnl - entry_fee - exit_fee,
                }
            )
            remaining_fraction -= exit_fraction
            if tranche == "TAKE_PROFIT_1":
                tranche_1_done = True
            elif tranche == "TAKE_PROFIT_2":
                tranche_2_done = True
            if remaining_fraction <= 1e-12:
                remaining_fraction = 0.0
                break

        if remaining_fraction > 1e-12:
            price = float(prices[exit_index])
            exit_notional = position_size_usd * remaining_fraction
            quantity = exit_notional / entry_price
            gross_pnl = quantity * (price - entry_price)
            fees = exit_notional * friction_per_leg + quantity * price * friction_per_leg
            fills.append(
                {
                    "campaign": campaign_number,
                    "entry_index": entry_index,
                    "exit_index": exit_index,
                    "entry_slot": entry_slot,
                    "exit_slot": int(slot_values[exit_index]),
                    "tranche": "END_OF_DATA",
                    "position_fraction": remaining_fraction,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "gross_pnl_usd": gross_pnl,
                    "fees_usd": fees,
                    "net_pnl_usd": gross_pnl - fees,
                }
            )
        index = max(entry_index + 1, exit_index + 1)

    trades = pd.DataFrame(fills)
    if trades.empty:
        metrics: dict[str, object] = {
            "total_campaigns": 0,
            "win_rate": 0.0,
            "gross_pnl_usd": 0.0,
            "net_pnl_usd": 0.0,
            "total_fees_usd": 0.0,
            "profit_factor": float("nan"),
            "max_drawdown": 0.0,
        }
        return {"trades": trades, "metrics": metrics}

    campaign_results = trades.groupby("campaign", sort=True)["net_pnl_usd"].sum()
    gains = float(campaign_results[campaign_results > 0].sum())
    losses = float(campaign_results[campaign_results < 0].sum())
    equity_changes = campaign_results.to_numpy(dtype=float)
    equity = np.cumsum(equity_changes) + position_size_usd
    peaks = np.maximum.accumulate(equity)
    drawdowns = np.divide(
        peaks - equity,
        peaks,
        out=np.zeros_like(peaks),
        where=peaks != 0,
    )
    metrics = {
        "total_campaigns": int(len(campaign_results)),
        "win_rate": float((campaign_results > 0).mean()),
        "gross_pnl_usd": float(trades["gross_pnl_usd"].sum()),
        "net_pnl_usd": float(trades["net_pnl_usd"].sum()),
        "total_fees_usd": float(trades["fees_usd"].sum()),
        "profit_factor": gains / abs(losses) if losses < 0 else float("inf"),
        "max_drawdown": float(drawdowns.max(initial=0.0)),
    }
    return {"trades": trades, "metrics": metrics}


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

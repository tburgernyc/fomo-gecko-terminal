from collections.abc import Sequence
import math
from typing import Any

import numpy as np
import talib


def evaluate_microstructure_alpha_gate(
    fdv_usd: float,
    liquidity_usd: float,
    volume_24h_usd: float,
    buys_m5: int,
    sells_m5: int,
    candle_open: float,
    candle_high: float,
    candle_low: float,
    candle_close: float,
    leader_entry_price: float = 0.0,
) -> tuple[bool, str]:
    values = (
        fdv_usd,
        liquidity_usd,
        volume_24h_usd,
        candle_open,
        candle_high,
        candle_low,
        candle_close,
        leader_entry_price,
    )
    if not all(math.isfinite(value) for value in values) or buys_m5 < 0 or sells_m5 < 0:
        return False, "REJECT_INVALID_MARKET_DATA"
    if liquidity_usd < 25_000.0:
        return False, f"REJECT_LOW_LIQUIDITY: ${liquidity_usd:,.0f} below $25k floor"
    if 30_000.0 <= fdv_usd <= 75_000.0:
        return False, f"REJECT_PRE_MIGRATION_DUMP_ZONE: FDV ${fdv_usd:,.0f}"

    order_flow_ratio = buys_m5 / max(1, sells_m5)
    if 75_000.0 < fdv_usd <= 115_000.0 and order_flow_ratio < 1.60:
        return (
            False,
            f"REJECT_POST_GRADUATION_WEAK_FLOW: OFI {order_flow_ratio:.2f} below 1.60",
        )
    if leader_entry_price > 0:
        price_drift_pct = (candle_close - leader_entry_price) / leader_entry_price * 100.0
        if price_drift_pct > 2.20:
            return False, f"REJECT_LEADER_DRIFT: +{price_drift_pct:.2f}% above whale entry"

    candle_range = candle_high - candle_low
    if candle_range < 0 or not candle_low <= candle_close <= candle_high:
        return False, "REJECT_INVALID_CANDLE_RANGE"
    if candle_range > 0:
        upper_wick_ratio = (candle_high - candle_close) / candle_range
        if upper_wick_ratio > 0.45:
            return (
                False,
                f"REJECT_UPPER_WICK_DISTRIBUTION: ratio {upper_wick_ratio:.2f}",
            )
    if fdv_usd > 0 and volume_24h_usd / max(1.0, fdv_usd) < 0.50:
        return False, "REJECT_LOW_TURNOVER_RATIO"
    return True, "PASS"


def calculate_indicators(
    closes: Sequence[float] | np.ndarray,
    highs: Sequence[float] | np.ndarray | None = None,
    lows: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    values = np.asarray(closes, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("closing prices must be a finite one-dimensional sequence")
    rsi: float | None = None
    macd_hist: float | None = None
    atr: float | None = None
    if values.size >= 15:
        value = talib.RSI(values, timeperiod=14)[-1]
        if np.isfinite(value):
            rsi = float(value)
    if values.size >= 35:
        _, _, histogram = talib.MACD(values, fastperiod=12, slowperiod=26, signalperiod=9)
        if np.isfinite(histogram[-1]):
            macd_hist = float(histogram[-1])
    if highs is not None and lows is not None:
        high_values = np.asarray(highs, dtype=np.float64)
        low_values = np.asarray(lows, dtype=np.float64)
        if high_values.shape != values.shape or low_values.shape != values.shape:
            raise ValueError("high, low, and close arrays must have matching shapes")
        if not np.isfinite(high_values).all() or not np.isfinite(low_values).all():
            raise ValueError("OHLC values must be finite")
        if values.size >= 15:
            latest_atr = talib.ATR(high_values, low_values, values, timeperiod=14)[-1]
            if np.isfinite(latest_atr):
                atr = float(latest_atr)
    return {"rsi_14": rsi, "macd_hist": macd_hist, "atr_14": atr}

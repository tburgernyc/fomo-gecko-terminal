from collections.abc import Sequence
from typing import Any

import numpy as np
import talib


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

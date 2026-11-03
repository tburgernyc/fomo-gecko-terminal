from collections.abc import Iterable

import pandas as pd
import vectorbt as vbt
import numpy as np


def sweep_exit_parameters(
    close: pd.Series,
    entries: pd.Series,
    trailing_stops: Iterable[float] = (0.08, 0.12, 0.16),
    time_stops: Iterable[int] = (30, 60, 90),
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
            portfolio = vbt.Portfolio.from_signals(
                close,
                entries=entries.astype(bool),
                exits=exit_signal,
                sl_stop=stop_pct,
                sl_trail=True,
                freq="5min",
            )
            rows.append({
                "trailing_stop_pct": stop_pct,
                "time_stop_bars": max_bars,
                "total_return": float(portfolio.total_return()),
                "sharpe_ratio": float(portfolio.sharpe_ratio()),
                "max_drawdown": float(portfolio.max_drawdown()),
            })
    return pd.DataFrame(rows).sort_values("total_return", ascending=False, ignore_index=True)

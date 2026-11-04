# fomo-gecko-terminal

When `FOMO_API_KEY` is missing or unauthorized, dry-run mode uses clearly labeled
synthetic demo trader profiles and polls GeckoTerminal's public Solana trending
pools. These profiles are not verified real traders and have no wallet addresses;
keyless trending-pool signals are disabled when `DRY_RUN=false`.

BUY signals also pass FDV/volume, migration-zone, and recent-candle
microstructure checks in addition to the existing liquidity and RSI gates.
The entry gate enforces a $25,000 liquidity floor, rejects the $30,000-$75,000
FDV dump zone, checks 5-minute buy/sell flow for $75,000-$115,000 FDV tokens,
and rejects excessive leader-fill drift or upper wicks when those inputs are
available.

The terminal and local state API read isolated, read-only DuckDB snapshots.
Live Solana routing builds a Jupiter swap transaction, adds a read-only
`jitodontfront` marker and a randomly selected Jito tip, and submits it as a
bundle; it does not fall back to public-RPC broadcast. Open positions use
40%/40% profit-taking tranches, a break-even stop after tranche one, a 2.2x
ATR trail after tranche two, and a 12-minute time stop below +8%.

`src/quant/backtest_vbt.py` models one-slot (400 ms) delay, nonlinear
constant-product price impact, and 0.80% friction per leg. Its profit-factor
and drawdown thresholds are diagnostics, not a guarantee of future performance.

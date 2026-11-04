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
bundle; it does not fall back to public-RPC broadcast. Explicitly rejected live
SELL bundles are retried in the next observed slot up to four times, escalating
the Jito tip by 1.75x per retry. Ambiguous submissions remain unresolved rather
than being blindly resent.

The gatekeeper enforces a 1,200-slot token re-entry cooldown, an 1,800-slot
trend hold, and a five-observation expanding-liquidity/higher-low/SMA entry
filter. Position exits use a hard -11% stop and deterministic 40%/40%/20%
tranches at +28% and +75%, followed by break-even and 2.2x ATR trailing stops;
the remaining position expires after 2,000 slots.

For qualifying two-wallet BUY consensus, `MeteoraDLMMHybridRouter` can use
`METEORA_DLMM_SIDECAR_URL` (configured as `meteora_dlmm_sidecar_url`) to resolve
an active pool and atomically acquire tokens plus seed the single-sided
`activeBin + 2` through `activeBin + 14` ladder. The sidecar must support the
`find_active_pool`, `open_jit_position`, and `get_order_status` operations and
confirm both atomic fill and an armed high-water-mark reversal withdrawal.
Pools below `METEORA_DLMM_MIN_LIQUIDITY_USD` fall back to the base router;
sidecar errors are surfaced, not treated as fills.

`python3 scripts/run_backtest.py` runs the event replay and prints campaign
count, closed-campaign win rate, gross/net PnL, fees, profit factor, and maximum
drawdown. The checked-in default Parquet file is synthetic demo data, so its
results are a regression check, not evidence of live-market profitability.
`src/quant/backtest_vbt.py` also models one-slot (400 ms) delay, nonlinear
constant-product price impact, and 0.80% friction per leg. Its 1.65 profit
factor and drawdown thresholds are diagnostics, not a guarantee of future
performance.

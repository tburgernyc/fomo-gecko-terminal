# fomo-gecko-terminal

When `FOMO_API_KEY` is missing or unauthorized, dry-run mode uses clearly labeled
synthetic demo trader profiles and polls GeckoTerminal's public Solana trending
pools. These profiles are not verified real traders and have no wallet addresses;
keyless trending-pool signals are disabled when `DRY_RUN=false`.

BUY signals also pass FDV/volume, migration-zone, and recent-candle
microstructure checks in addition to the existing liquidity and RSI gates.

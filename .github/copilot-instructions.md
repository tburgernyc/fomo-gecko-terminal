# Copilot Instructions

## Build and test

Requires Python 3.11+. From the repository root:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests -v
python -m pytest tests/test_gatekeeper.py::test_idempotency_blocks_duplicate_orders
python -m src.main
```

There is no configured lint command. Tests use an in-memory DuckDB database and stub scanner data; they should not require external APIs.

## Architecture

- `src/ingestion/fomo_client.py` fetches and ETag-caches FOMO leaderboards, persists the qualified Top-10, and streams trades over WebSocket with paginated REST fallback.
- `src/ingestion/gecko_scanner.py` maps supported chain IDs to GeckoTerminal networks, rate-limits V2 requests, and returns verified pool metrics and chronological OHLCV.
- `src/execution/gatekeeper.py` is the only order-submission boundary: it persists signals and intents, applies market/consensus/fee/portfolio/circuit-breaker gates, enforces five-minute idempotency, and reconciles pending orders.
- `src/execution/routers.py` contains dry-run, HMAC webhook, and direct Jupiter/0x routers. Direct execution supports Solana, Base, Ethereum, and BNB; Monad is rejected before scanning for direct live execution unless another configured router supports it. EVM RPC endpoints are set by `BASE_RPC_URL`, `ETHEREUM_RPC_URL`, and `BNB_RPC_URL`. `DRY_RUN=true` is the default; live execution additionally requires explicit `LIVE_TRADING_CONFIRMED=true`.
- `src/db.py` owns the shared DuckDB connection. Use its cursor and transaction scopes for serialized database access; keep schema changes aligned with `schema.sql`.
- `src/main.py` wires the async ingestion flow and APScheduler jobs, starts a Rich risk/log dashboard in the terminal, and serves the DuckDB chart-state API. `src/reporting/chart_dashboard.py` runs the Lightweight Charts Python Streamlit view at `http://localhost:8050`; the terminal and browser share the same state API. Indicator, allocation, and VectorBT calibration code live in `src/quant/`.

## Repository-specific conventions

- Derive order IDs from `sha256(f"{trader_id}:{network}:{token_address}:{side}:{time_bucket_5m}")[:24]`. Reconcile all `UNCERTAIN`/`SUBMITTED` orders before new submissions; never blindly retry.
- Apply pool reserve ($12,000 minimum), 24-hour volume ($40,000 minimum), RSI(14) (78 maximum except for two distinct Top-10 BUY traders inside 600 seconds), 2% pool-impact, 6% bankroll-ticket, and 1.5% maximum fee/notional gates.
- GeckoTerminal requests must remain at or below 28 per minute. Use `httpx.AsyncClient` with Tenacity backoff for retryable requests; avoid retries of non-idempotent order submissions unless the idempotency key is honored.
- Load secrets only from environment settings. Never persist private keys in DuckDB or source files. Keep live execution disabled unless explicitly requested and confirmed.
- The Lightweight Charts browser bundle and Apache 2.0 license are packaged under `src/reporting/static/`; the local terminal must not depend on a CDN at runtime.
- The scanner calculates RSI(14), MACD(12,26,9), and ATR(14) from OHLCV. Keep indicator inputs chronological and use real high/low data for ATR.

CREATE TABLE IF NOT EXISTS leaderboard_traders (
    user_id VARCHAR PRIMARY KEY,
    user_handle VARCHAR NOT NULL UNIQUE,
    display_name VARCHAR,
    rank_24h INTEGER,
    pnl_24h_usd DOUBLE,
    pnl_7d_usd DOUBLE,
    win_rate DOUBLE,
    num_trades INTEGER,
    sleeve_weight DOUBLE NOT NULL DEFAULT 0.10,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trader_pnl_history (
    user_handle VARCHAR NOT NULL,
    observed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    pnl_24h_usd DOUBLE NOT NULL,
    PRIMARY KEY (user_handle, observed_at)
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    snapshot_id VARCHAR PRIMARY KEY,
    network VARCHAR NOT NULL,
    token_address VARCHAR NOT NULL,
    pool_address VARCHAR NOT NULL,
    price_usd DOUBLE,
    reserve_in_usd DOUBLE,
    volume_24h_usd DOUBLE,
    fdv_usd DOUBLE,
    buys_m5 INTEGER NOT NULL DEFAULT 0,
    sells_m5 INTEGER NOT NULL DEFAULT 0,
    rsi_14 DOUBLE,
    macd_hist DOUBLE,
    atr_14 DOUBLE,
    passed_gate BOOLEAN NOT NULL,
    reject_reason VARCHAR,
    captured_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS market_candles (
    network VARCHAR NOT NULL,
    token_address VARCHAR NOT NULL,
    candle_ts TIMESTAMP NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume DOUBLE,
    PRIMARY KEY (network, token_address, candle_ts)
);

CREATE TABLE IF NOT EXISTS copy_signals (
    signal_id VARCHAR PRIMARY KEY,
    trader_handle VARCHAR NOT NULL,
    network VARCHAR NOT NULL,
    token_address VARCHAR NOT NULL,
    side VARCHAR NOT NULL CHECK (side IN ('BUY', 'SELL')),
    received_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS execution_orders (
    client_order_id VARCHAR PRIMARY KEY,
    trader_handle VARCHAR NOT NULL,
    network VARCHAR NOT NULL,
    token_address VARCHAR NOT NULL,
    pool_address VARCHAR,
    side VARCHAR NOT NULL CHECK (side IN ('BUY', 'SELL')),
    status VARCHAR NOT NULL CHECK (
        status IN ('INTENT', 'SUBMITTED', 'UNCERTAIN', 'OPEN', 'CLOSED', 'FAILED',
                   'REJECTED', 'DRY_RUN_OPEN', 'DRY_RUN_CLOSED')
    ),
    requested_usd DOUBLE NOT NULL,
    filled_usd DOUBLE NOT NULL DEFAULT 0,
    filled_token_amount DOUBLE NOT NULL DEFAULT 0,
    remaining_token_amount DOUBLE NOT NULL DEFAULT 0,
    entry_price_usd DOUBLE,
    exit_price_usd DOUBLE,
    high_water_price_usd DOUBLE,
    entry_atr_usd DOUBLE,
    exit_campaign_id VARCHAR,
    exit_tranche VARCHAR,
    tranche_1_sold_amount DOUBLE NOT NULL DEFAULT 0,
    tranche_2_sold_amount DOUBLE NOT NULL DEFAULT 0,
    fee_usd DOUBLE NOT NULL DEFAULT 0,
    realized_pnl_usd DOUBLE NOT NULL DEFAULT 0,
    unrealized_pnl_usd DOUBLE NOT NULL DEFAULT 0,
    tx_hash_or_exchange_id VARCHAR,
    execution_detail VARCHAR,
    failure_reason VARCHAR,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE market_snapshots ADD COLUMN IF NOT EXISTS fdv_usd DOUBLE;
ALTER TABLE market_snapshots ADD COLUMN IF NOT EXISTS buys_m5 INTEGER DEFAULT 0;
ALTER TABLE market_snapshots ADD COLUMN IF NOT EXISTS sells_m5 INTEGER DEFAULT 0;
ALTER TABLE execution_orders ADD COLUMN IF NOT EXISTS entry_atr_usd DOUBLE;
ALTER TABLE execution_orders ADD COLUMN IF NOT EXISTS exit_campaign_id VARCHAR;
ALTER TABLE execution_orders ADD COLUMN IF NOT EXISTS exit_tranche VARCHAR;
ALTER TABLE execution_orders ADD COLUMN IF NOT EXISTS tranche_1_sold_amount DOUBLE DEFAULT 0;
ALTER TABLE execution_orders ADD COLUMN IF NOT EXISTS tranche_2_sold_amount DOUBLE DEFAULT 0;
ALTER TABLE execution_orders ADD COLUMN IF NOT EXISTS execution_detail VARCHAR;
UPDATE execution_orders
SET exit_campaign_id = client_order_id
WHERE side = 'BUY' AND exit_campaign_id IS NULL;

CREATE TABLE IF NOT EXISTS equity_snapshots (
    ts TIMESTAMP PRIMARY KEY,
    cash_usd DOUBLE NOT NULL,
    open_exposure_usd DOUBLE NOT NULL,
    total_equity_usd DOUBLE NOT NULL,
    daily_realized_pnl_usd DOUBLE NOT NULL,
    daily_fees_usd DOUBLE NOT NULL
);

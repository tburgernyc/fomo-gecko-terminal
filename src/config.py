from functools import lru_cache

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    dry_run: bool = True
    live_trading_confirmed: bool = False
    starting_bankroll_usd: float = 1000.0
    max_daily_loss_pct: float = 0.10
    max_single_ticket_pct: float = 0.06
    max_pool_liquidity_impact_pct: float = 0.02
    execution_router: str = "onchain"
    min_pool_liquidity_usd: float = 25000.0
    min_pool_volume_24h_usd: float = 40000.0
    max_entry_rsi_14: float = 78.0
    consensus_window_seconds: int = 600
    time_stop_minutes: int = 90
    exit_time_stop_minutes: int = 12
    trailing_stop_pct: float = 0.12

    duckdb_path: str = "fomo_gecko_terminal.duckdb"
    schema_path: str = "schema.sql"
    fomo_api_base_url: str = "https://api.pumpfomo.com"
    fomo_api_key: SecretStr = SecretStr("")
    fomo_ws_url: str = "wss://fomoradar.io/ws/trades?board=24h&top=10"
    geckoterminal_base_url: str = "https://api.geckoterminal.com/api/v2"

    solana_private_key: SecretStr = SecretStr("")
    solana_rpc_url: str = "https://api.mainnet-beta.solana.com"
    evm_private_key: SecretStr = SecretStr("")
    ethereum_rpc_url: str = "https://ethereum-rpc.publicnode.com"
    base_rpc_url: str = "https://mainnet.base.org"
    bnb_rpc_url: str = "https://bsc-rpc.publicnode.com"
    jupiter_base_url: str = "https://quote-api.jup.ag/v6"
    jito_block_engine_url: str = "https://mainnet.block-engine.jito.wtf/api/v1/bundles"
    solana_fee_budget_lamports: int = 150_000
    meteora_dlmm_sidecar_url: str = ""
    meteora_dlmm_min_liquidity_usd: float = 25_000.0
    zerox_base_url: str = "https://api.0x.org"
    zerox_api_key: SecretStr = SecretStr("")
    freqtrade_webhook_url: str = ""
    freqtrade_webhook_secret: SecretStr = SecretStr("")
    freqtrade_order_status_url: str = ""
    telegram_bot_token: SecretStr = SecretStr("")
    telegram_chat_id: str = ""
    discord_webhook_url: SecretStr = SecretStr("")

    @field_validator("max_daily_loss_pct", "max_single_ticket_pct", "max_pool_liquidity_impact_pct",
                     "trailing_stop_pct")
    @classmethod
    def validate_fraction(cls, value: float) -> float:
        if not 0 < value <= 1:
            raise ValueError("percentage settings must be in (0, 1]")
        return value

    @field_validator(
        "starting_bankroll_usd",
        "min_pool_liquidity_usd",
        "min_pool_volume_24h_usd",
        "meteora_dlmm_min_liquidity_usd",
    )
    @classmethod
    def validate_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("max_entry_rsi_14")
    @classmethod
    def validate_rsi_limit(cls, value: float) -> float:
        if not 0 <= value <= 100:
            raise ValueError("RSI limit must be in [0, 100]")
        return value

    @field_validator("consensus_window_seconds", "time_stop_minutes", "exit_time_stop_minutes")
    @classmethod
    def validate_positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("solana_fee_budget_lamports")
    @classmethod
    def validate_fee_budget(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Solana fee budget must be positive")
        return value

    @field_validator("execution_router")
    @classmethod
    def validate_router(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"onchain", "freqtrade"}:
            raise ValueError("execution_router must be 'onchain' or 'freqtrade'")
        return normalized

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

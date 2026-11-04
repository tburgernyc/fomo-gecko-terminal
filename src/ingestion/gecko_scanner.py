import asyncio
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx
import numpy as np
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.config import Settings, get_settings
from src.quant.indicators import calculate_indicators

CHAIN_TO_GECKO = {
    "solana": "solana",
    "sol": "solana",
    "1399811149": "solana",
    "base": "base",
    "8453": "base",
    "bnb": "bsc",
    "bsc": "bsc",
    "56": "bsc",
    "ethereum": "eth",
    "eth": "eth",
    "1": "eth",
    "monad": "monad",
    "143": "monad",
}


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)) or (
        isinstance(exc, httpx.HTTPStatusError)
        and (exc.response.status_code == 429 or exc.response.status_code >= 500)
    )


class TokenBucket:
    def __init__(self, capacity: int = 28, period_seconds: float = 60.0):
        if capacity <= 0 or period_seconds <= 0:
            raise ValueError("token bucket capacity and period must be positive")
        self.capacity = float(capacity)
        self.period_seconds = period_seconds
        self.tokens = float(capacity)
        self.updated_at = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self.lock:
                now = time.monotonic()
                self.tokens = min(
                    self.capacity,
                    self.tokens + (now - self.updated_at) * self.capacity / self.period_seconds,
                )
                self.updated_at = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait_seconds = (1 - self.tokens) * self.period_seconds / self.capacity
            await asyncio.sleep(wait_seconds)


class GeckoTerminalScanner:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.base_url = self.settings.geckoterminal_base_url.rstrip("/")
        self.headers = {"Accept": "application/json;version=20230302"}
        self.limiter = TokenBucket(28, 60)
        self._etags: dict[str, str] = {}
        self._responses: dict[str, dict[str, Any]] = {}

    @retry(
        retry=retry_if_exception(_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=12),
        reraise=True,
    )
    async def _get_json(self, client: httpx.AsyncClient, url: str) -> dict[str, Any]:
        await self.limiter.acquire()
        headers = dict(self.headers)
        if url in self._etags:
            headers["If-None-Match"] = self._etags[url]
        response = await client.get(url, headers=headers, timeout=15)
        if response.status_code == 304:
            if url not in self._responses:
                raise RuntimeError(f"GeckoTerminal returned 304 without a cached response for {url}")
            return self._responses[url]
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError(f"GeckoTerminal returned a non-object response for {url}")
        if etag := response.headers.get("ETag"):
            self._etags[url] = etag
        self._responses[url] = body
        return body

    async def evaluate_token(
        self, client: httpx.AsyncClient, network: str, token_address: str
    ) -> dict[str, Any]:
        slug = CHAIN_TO_GECKO.get(str(network).lower())
        if not slug:
            return {"passed": False, "reason": f"unsupported_network:{network}"}
        pools_response = await self._get_json(
            client, f"{self.base_url}/networks/{slug}/tokens/{quote(token_address, safe='')}/pools"
        )
        pools = pools_response.get("data", [])
        if not isinstance(pools, list) or not pools:
            return {"passed": False, "reason": "no_geckoterminal_pools"}
        attributes = pools[0].get("attributes", {})
        pool_address = str(attributes.get("address") or "")
        reserve = float(attributes.get("reserve_in_usd") or 0)
        volume = float((attributes.get("volume_usd") or {}).get("h24") or 0)
        price = float(attributes.get("base_token_price_usd") or 0)
        fdv = float(attributes.get("fdv_usd") or 0)
        transactions = attributes.get("transactions") or {}
        m5_transactions = transactions.get("m5") or {}
        buys_m5 = int(m5_transactions.get("buys") or 0)
        sells_m5 = int(m5_transactions.get("sells") or 0)
        result: dict[str, Any] = {
            "network": slug,
            "pool_address": pool_address,
            "reserve_in_usd": reserve,
            "volume_24h_usd": volume,
            "volume_usd_h24": volume,
            "fdv_usd": fdv,
            "price_usd": price,
            "buys_m5": buys_m5,
            "sells_m5": sells_m5,
            "order_flow_ratio": buys_m5 / max(1, sells_m5),
            "passed": False,
            "reason": "",
        }
        if reserve < self.settings.min_pool_liquidity_usd:
            result["reason"] = f"insufficient_liquidity:${reserve:.0f}"
            return result
        if volume < self.settings.min_pool_volume_24h_usd:
            result["reason"] = f"insufficient_24h_vol:${volume:.0f}"
            return result
        if not pool_address:
            result["reason"] = "missing_pool_address"
            return result
        ohlcv = await self._get_json(
            client,
            f"{self.base_url}/networks/{slug}/pools/{quote(pool_address, safe='')}/ohlcv/minute?aggregate=5&limit=50",
        )
        candles = (
            ohlcv.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
            if isinstance(ohlcv.get("data"), dict)
            else []
        )
        if not isinstance(candles, list) or any(
            not isinstance(candle, (list, tuple)) or len(candle) < 5
            for candle in candles
        ):
            raise ValueError("GeckoTerminal returned malformed OHLCV candles")
        chronological = list(reversed(candles))
        closes = np.asarray([float(candle[4]) for candle in chronological], dtype=np.float64)
        highs = np.asarray([float(candle[2]) for candle in chronological], dtype=np.float64)
        lows = np.asarray([float(candle[3]) for candle in chronological], dtype=np.float64)
        indicators = calculate_indicators(closes, highs, lows)
        result["candles"] = [
            {
                "timestamp": datetime.fromtimestamp(float(candle[0]), timezone.utc).replace(tzinfo=None),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]) if len(candle) > 5 else 0.0,
            }
            for candle in chronological
        ]
        result.update(indicators)
        if indicators["rsi_14"] is None:
            result["reason"] = "insufficient_ohlcv_for_rsi"
            return result
        result["passed"] = True
        result["reason"] = "ok"
        return result

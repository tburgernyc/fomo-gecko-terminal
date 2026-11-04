import asyncio
import json
import logging
import math
from typing import Any, Awaitable, Callable
from urllib.parse import quote

import httpx
import pandas as pd
import websockets
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.config import Settings, get_settings
from src.db import cursor_scope, transaction_scope
from src.quant.allocator import PortfolioAllocator

logger = logging.getLogger(__name__)
TradeCallback = Callable[..., Awaitable[str]]
FOMO_API_KEY_PLACEHOLDER = "your_pumpfomo_or_fomoradar_api_key_here"
GECKOTERMINAL_TRENDING_POOLS = "/networks/solana/trending_pools"


def _seed_trader_profiles() -> list[dict[str, Any]]:
    return [
        {
            "user_id": f"keyless-demo-{rank:02d}",
            "userHandle": f"keyless_demo_{rank:02d}",
            "handle": f"keyless_demo_{rank:02d}",
            "display_name": f"Keyless Demo Trader {rank:02d}",
            "rank": rank,
            "wallet_address": None,
            "chain": "solana" if rank % 2 else "base",
            "pnl24h": float(1200 - rank * 75),
            "pnl_24h_usd": float(1200 - rank * 75),
            "pnl7d": float(6000 - rank * 350),
            "win_rate": 0.62 + (rank % 4) * 0.04,
            "trades": 12 + rank,
            "numTrades": 12 + rank,
            "is_demo": True,
            "verified": False,
        }
        for rank in range(1, 11)
    ]


def _retryable_http_error(exc: BaseException) -> bool:
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)) or (
        isinstance(exc, httpx.HTTPStatusError)
        and (exc.response.status_code == 429 or exc.response.status_code >= 500)
    )


class FomoLeaderboardClient:
    def __init__(self, connection: Any, settings: Settings | None = None):
        self.con = connection
        self.settings = settings or get_settings()
        self.base_url = self.settings.fomo_api_base_url.rstrip("/")
        self._etags: dict[str, str] = {}
        self._payloads: dict[str, Any] = {}
        self._seen_trades: set[str] = set()
        self.top_10_handles: dict[str, float] = {}

    @staticmethod
    def _field(row: dict[str, Any], *names: str, default: Any = None) -> Any:
        for name in names:
            if name in row and row[name] is not None:
                return row[name]
        return default

    @classmethod
    def _leader_entry_price(cls, row: dict[str, Any]) -> float | None:
        value = cls._field(
            row, "priceUsd", "price_usd", "entryPrice", "entry_price",
            "executionPrice", "execution_price", "fillPrice", "fill_price", "price",
        )
        if value is None:
            return None
        try:
            price = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("leader fill price must be numeric") from exc
        if not math.isfinite(price) or price <= 0:
            raise ValueError("leader fill price must be finite and positive")
        return price

    def _headers(self, cache_key: str) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "FomoGeckoTerminal/1.0"}
        api_key = self.settings.fomo_api_key.get_secret_value().strip()
        if api_key.lower() == FOMO_API_KEY_PLACEHOLDER:
            api_key = ""
        if api_key:
            headers["X-API-Key"] = api_key
        if cache_key in self._etags:
            headers["If-None-Match"] = self._etags[cache_key]
        return headers

    def _has_fomo_api_key(self) -> bool:
        api_key = self.settings.fomo_api_key.get_secret_value().strip()
        return bool(api_key and api_key.lower() != FOMO_API_KEY_PLACEHOLDER)

    @staticmethod
    def _warn_keyless_fallback(status_code: int | str) -> None:
        logger.warning(
            "FOMO API key missing or unauthorized (HTTP %s); activating keyless GeckoTerminal V2 fallback mode.",
            status_code,
        )

    async def _poll_geckoterminal_trending(
        self, client: httpx.AsyncClient, on_signal: TradeCallback
    ) -> None:
        if not self.settings.dry_run:
            logger.error("Keyless GeckoTerminal signal fallback is disabled outside dry-run mode.")
            return
        self.top_10_handles = {
            row["handle"]: 0.10 for row in _seed_trader_profiles()
        }

        url = f"{self.settings.geckoterminal_base_url.rstrip('/')}{GECKOTERMINAL_TRENDING_POOLS}"
        try:
            response = await client.get(
                url,
                headers={"Accept": "application/json;version=20230203"},
                timeout=15.0,
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError:
            logger.exception("Could not fetch GeckoTerminal trending pools for keyless fallback.")
            return
        except ValueError:
            logger.exception("GeckoTerminal returned invalid JSON for keyless fallback.")
            return
        pools = body.get("data", []) if isinstance(body, dict) else None
        if not isinstance(pools, list):
            raise ValueError("GeckoTerminal trending pools response is not a list")

        profiles = _seed_trader_profiles()
        handles = list(self.top_10_handles)
        for index, pool in enumerate(pools[:10]):
            if not isinstance(pool, dict):
                continue
            attributes = pool.get("attributes")
            attributes = attributes if isinstance(attributes, dict) else {}
            token = attributes.get("base_token_address") or attributes.get("token_address")
            if not token:
                relationships = pool.get("relationships")
                base_token = (
                    relationships.get("base_token", {}).get("data")
                    if isinstance(relationships, dict)
                    and isinstance(relationships.get("base_token"), dict)
                    else None
                )
                if isinstance(base_token, dict):
                    token = str(base_token.get("id") or "")
                    if token.startswith("solana_"):
                        token = token[len("solana_"):]
            if not token:
                logger.warning("Skipping GeckoTerminal trending pool without a base-token address.")
                continue

            handle = handles[index % len(handles)] if handles else profiles[index]["handle"]
            weight = self.top_10_handles.get(handle, 0.10)
            pool_id = str(attributes.get("address") or pool.get("id") or token)
            trade_id = f"gecko-trending:solana:{pool_id}:{token}"
            if trade_id in self._seen_trades:
                continue
            await on_signal(handle, "solana", str(token), "BUY", weight)
            self._seen_trades.add(trade_id)

    def load_cached_top_10(self) -> dict[str, float]:
        with cursor_scope(self.con) as cur:
            rows = cur.execute(
                """
                SELECT user_handle, sleeve_weight FROM leaderboard_traders
                ORDER BY rank_24h LIMIT 10
                """
            ).fetchall()
        self.top_10_handles = {
            str(handle): float(weight)
            for handle, weight in rows
            if handle and weight is not None
        }
        return self.top_10_handles

    @retry(
        retry=retry_if_exception(_retryable_http_error),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=12),
        reraise=True,
    )
    async def fetch_leaderboard(self, client: httpx.AsyncClient, window: str) -> list[dict[str, Any]]:
        if window not in {"24h", "7d"}:
            raise ValueError(f"unsupported leaderboard window: {window}")
        if not self._has_fomo_api_key():
            self._warn_keyless_fallback("missing")
            return _seed_trader_profiles()
        cache_key = f"leaderboard:{window}"
        try:
            response = await client.get(
                f"{self.base_url}/v1/fomo/leaderboard/{window}",
                headers=self._headers(cache_key),
                timeout=15.0,
            )
            if response.status_code == 304:
                if cache_key not in self._payloads:
                    raise RuntimeError(f"FOMO returned 304 without a cached {window} leaderboard")
                return self._payloads[cache_key]
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in {401, 403}:
                self._warn_keyless_fallback(e.response.status_code)
                return _seed_trader_profiles()
            raise
        etag = response.headers.get("ETag")
        if etag:
            self._etags[cache_key] = etag
        body = response.json()
        rows = body.get("data", body) if isinstance(body, dict) else body
        if not isinstance(rows, list):
            raise ValueError(f"FOMO {window} leaderboard response is not a list")
        normalized = [row for row in rows if isinstance(row, dict)]
        self._payloads[cache_key] = normalized
        return normalized

    async def refresh_top_10_traders(self, client: httpx.AsyncClient) -> dict[str, float]:
        day, week = await asyncio.gather(
            self.fetch_leaderboard(client, "24h"),
            self.fetch_leaderboard(client, "7d"),
        )
        week_by_handle = {
            str(self._field(row, "userHandle", "user_handle", "handle", "userId", "user_id")): row
            for row in week
            if self._field(row, "userHandle", "user_handle", "handle", "userId", "user_id")
        }
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        ranked_day = sorted(
            day,
            key=lambda row: float(self._field(
                row, "pnl24h", "pnl24hUsd", "pnl_24h", "pnl_24h_usd", "pnlUsd", "pnl_usd",
                default=0,
            )),
            reverse=True,
        )
        for rank, row in enumerate(ranked_day, start=1):
            handle = str(self._field(
                row, "userHandle", "user_handle", "handle", "userId", "user_id", default=""
            ))
            if not handle or handle in seen:
                continue
            seen.add(handle)
            weekly = week_by_handle.get(handle, {})
            pnl_24h = float(self._field(
                row, "pnl24h", "pnl24hUsd", "pnl_24h", "pnl_24h_usd", "pnlUsd", "pnl_usd",
                default=0,
            ))
            pnl_7d = float(self._field(
                weekly, "pnl7d", "pnl7dUsd", "pnl_7d", "pnl_7d_usd", "pnlUsd", "pnl_usd",
                default=0,
            ))
            win_rate = float(self._field(
                row, "winRate", "win_rate",
                default=self._field(weekly, "winRate", "win_rate", default=0),
            ))
            trades = int(self._field(
                row, "numTrades", "num_trades", "trades",
                default=self._field(weekly, "numTrades", "num_trades", "trades", default=0),
            ))
            if pnl_24h <= 0 or pnl_7d <= 0 or win_rate <= 0 or trades < 5:
                continue
            selected.append({
                "user_id": str(self._field(row, "userId", "user_id", default=handle)),
                "handle": handle,
                "display_name": str(self._field(row, "displayName", "display_name", default=handle)),
                "rank": rank,
                "pnl_24h": pnl_24h,
                "pnl_7d": pnl_7d,
                "win_rate": win_rate,
                "trades": trades,
            })
            if len(selected) == 10:
                break

        if not selected:
            logger.warning("FOMO returned no qualifying traders; retaining the last leaderboard.")
            if not self.top_10_handles:
                self.load_cached_top_10()
            return self.top_10_handles

        handles = [row["handle"] for row in selected]
        placeholders = ", ".join("?" for _ in handles)
        with cursor_scope(self.con) as cur:
            history = cur.execute(
                f"""
                SELECT user_handle, observed_at, pnl_24h_usd
                FROM trader_pnl_history
                WHERE user_handle IN ({placeholders})
                  AND observed_at >= CURRENT_TIMESTAMP - INTERVAL 90 DAY
                ORDER BY observed_at
                """,
                handles,
            ).df()
        if not history.empty:
            pnl_history = history.pivot(
                index="observed_at", columns="user_handle", values="pnl_24h_usd"
            ).sort_index()
            current_observation = pd.DataFrame(
                [{"user_handle": row["handle"], "pnl_24h_usd": row["pnl_24h"]}
                 for row in selected]
            ).assign(observed_at=pd.Timestamp.now())
            pnl_history = pd.concat(
                [
                    pnl_history,
                    current_observation.pivot(
                        index="observed_at", columns="user_handle", values="pnl_24h_usd"
                    ),
                ]
            ).sort_index().ffill()
            returns = pnl_history.diff().div(self.settings.starting_bankroll_usd).dropna()
        else:
            returns = pd.DataFrame(columns=handles, dtype=float)
        weights = PortfolioAllocator.compute_sleeve_weights(returns.reindex(columns=handles))
        if not weights:
            weights = {handle: min(1.0 / len(handles), 0.15) for handle in handles}
        with transaction_scope(self.con) as cur:
            cur.execute(
                "DELETE FROM trader_pnl_history WHERE observed_at < CURRENT_TIMESTAMP - INTERVAL 90 DAY"
            )
            for row in selected:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO trader_pnl_history (user_handle, observed_at, pnl_24h_usd)
                    VALUES (?, CURRENT_TIMESTAMP, ?)
                    """,
                    [row["handle"], row["pnl_24h"]],
                )
                cur.execute(
                    """
                    DELETE FROM leaderboard_traders
                    WHERE user_handle = ? AND user_id <> ?
                    """,
                    [row["handle"], row["user_id"]],
                )
                cur.execute(
                    """
                    INSERT INTO leaderboard_traders (
                        user_id, user_handle, display_name, rank_24h, pnl_24h_usd,
                        pnl_7d_usd, win_rate, num_trades, sleeve_weight, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id) DO UPDATE SET
                        user_handle = excluded.user_handle,
                        display_name = excluded.display_name,
                        rank_24h = excluded.rank_24h,
                        pnl_24h_usd = excluded.pnl_24h_usd,
                        pnl_7d_usd = excluded.pnl_7d_usd,
                        win_rate = excluded.win_rate,
                        num_trades = excluded.num_trades,
                        sleeve_weight = excluded.sleeve_weight,
                        updated_at = excluded.updated_at
                    """,
                    [
                        row["user_id"], row["handle"], row["display_name"], row["rank"],
                        row["pnl_24h"], row["pnl_7d"], row["win_rate"], row["trades"],
                        weights[row["handle"]],
                    ],
                )
        self.top_10_handles = weights
        logger.info("Refreshed FOMO Top-10: %s", list(weights))
        return weights

    @retry(
        retry=retry_if_exception(_retryable_http_error),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=12),
        reraise=True,
    )
    async def _fetch_positions_page(
        self, client: httpx.AsyncClient, handle: str, cursor: str | None
    ) -> dict[str, Any]:
        params = {"limit": 100}
        if cursor:
            params["cursor"] = cursor
        cache_key = f"positions:{handle}:{cursor or ''}"
        response = await client.get(
            f"{self.base_url}/v1/fomo/users/{quote(handle, safe='')}/positions",
            params=params,
            headers=self._headers(cache_key),
            timeout=15.0,
        )
        if response.status_code == 304:
            if cache_key not in self._payloads:
                raise RuntimeError(f"FOMO returned 304 without a cached positions page for {handle}")
            return self._payloads[cache_key]
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("FOMO positions response must be an object")
        if etag := response.headers.get("ETag"):
            self._etags[cache_key] = etag
        self._payloads[cache_key] = body
        return body

    async def poll_trader_positions_fallback(
        self, client: httpx.AsyncClient, on_signal: TradeCallback
    ) -> None:
        if not self._has_fomo_api_key():
            self._warn_keyless_fallback("missing")
            await self._poll_geckoterminal_trending(client, on_signal)
            return
        for handle, weight in tuple(self.top_10_handles.items()):
            cursor: str | None = None
            seen_cursors: set[str] = set()
            while True:
                try:
                    body = await self._fetch_positions_page(client, handle, cursor)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in {401, 403}:
                        self._warn_keyless_fallback(e.response.status_code)
                        await self._poll_geckoterminal_trending(client, on_signal)
                        return
                    logger.exception("Could not poll FOMO positions for %s", handle)
                    break
                except httpx.HTTPError:
                    logger.exception("Could not poll FOMO positions for %s", handle)
                    break
                rows = body.get("data", [])
                if not isinstance(rows, list):
                    raise ValueError(f"FOMO positions for {handle} are not a list")
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    trade_id = str(row.get("id") or "")
                    token = str(row.get("tokenAddress") or row.get("mint") or "")
                    if not token:
                        continue
                    if not trade_id:
                        trade_id = f"{handle}:{token}:{row.get('updatedAt', '')}"
                    if trade_id in self._seen_trades:
                        continue
                    network = str(row.get("network") or row.get("chainId") or "solana")
                    side = str(self._field(
                        row, "side", "action", "type", "tradeSide", default=""
                    )).upper()
                    if side not in {"BUY", "SELL"}:
                        amount = float(self._field(row, "amount", "quantity", default=1.0))
                        side = "SELL" if amount <= 0 else "BUY"
                    signal = (handle, network, token, side, weight)
                    leader_entry_price = self._leader_entry_price(row)
                    await on_signal(
                        *signal,
                        *([leader_entry_price] if leader_entry_price is not None else []),
                    )
                    self._seen_trades.add(trade_id)
                page = body.get("pagination", {})
                next_cursor = (
                    page.get("nextCursor") or page.get("next_cursor")
                    if isinstance(page, dict)
                    else body.get("nextCursor")
                )
                next_cursor = str(next_cursor) if next_cursor else ""
                if not next_cursor or next_cursor == cursor or next_cursor in seen_cursors:
                    break
                seen_cursors.add(next_cursor)
                cursor = next_cursor

    async def stream_top_10_trades(
        self, client: httpx.AsyncClient, on_signal: TradeCallback, stop_event: asyncio.Event
    ) -> None:
        if not self._has_fomo_api_key():
            while not stop_event.is_set():
                self._warn_keyless_fallback("missing")
                try:
                    await self._poll_geckoterminal_trending(client, on_signal)
                except httpx.HTTPError:
                    logger.exception("Could not poll GeckoTerminal trending pools.")
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=60)
                except asyncio.TimeoutError:
                    pass
            return
        backoff = 2.0
        while not stop_event.is_set():
            headers = {}
            key = self.settings.fomo_api_key.get_secret_value()
            if key:
                headers["Authorization"] = "Bearer " + key
            try:
                async with websockets.connect(
                    self.settings.fomo_ws_url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=10,
                ) as ws:
                    backoff = 2.0
                    while not stop_event.is_set():
                        raw = await asyncio.wait_for(ws.recv(), timeout=45)
                        event = json.loads(raw)
                        if not isinstance(event, dict):
                            continue
                        handle = str(event.get("userHandle") or event.get("handle") or "")
                        token = str(event.get("tokenAddress") or event.get("token") or "")
                        side = str(event.get("side") or "BUY").upper()
                        if handle not in self.top_10_handles or not token or side not in {"BUY", "SELL"}:
                            continue
                        signal = (
                            handle,
                            str(event.get("network") or event.get("chain") or "solana"),
                            token,
                            side,
                            self.top_10_handles[handle],
                        )
                        leader_entry_price = self._leader_entry_price(event)
                        result = await on_signal(
                            *signal,
                            *([leader_entry_price] if leader_entry_price is not None else []),
                        )
                        logger.info("FOMO signal %s %s -> %s", handle, side, result)
            except (
                websockets.WebSocketException,
                asyncio.TimeoutError,
                json.JSONDecodeError,
                OSError,
            ) as exc:
                logger.warning("FOMO stream interrupted (%s); polling REST before reconnect.", exc)
                if stop_event.is_set():
                    break
                await self.poll_trader_positions_fallback(client, on_signal)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 60.0)

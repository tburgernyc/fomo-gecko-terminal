import asyncio
import base64
import hashlib
import hmac
import json
import logging
import math
import random
import struct
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from eth_account import Account
from solana.rpc.async_api import AsyncClient as SolanaAsyncClient
from solana.rpc.models import TxOpts
from solders.address_lookup_table_account import AddressLookupTable, AddressLookupTableAccount
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction
from solders.compute_budget import set_compute_unit_price
from solders.system_program import TransferParams, transfer
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)
SOLANA_USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
JITO_TIP_ACCOUNTS = (
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
)
JITO_DONT_FRONT_PUBKEY = "jitodontfront111111111111111111111111111111"
JITO_MEMO_PROGRAM_ID = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"
MIN_JITO_TIP_LAMPORTS = 1_000
DEFAULT_SOLANA_FEE_BUDGET_LAMPORTS = 150_000


def compute_fee_split(total_budget_lamports: int) -> tuple[int, int]:
    if total_budget_lamports <= 0:
        raise ValueError("fee budget must be positive")
    priority_fee = int(total_budget_lamports * 0.70)
    jito_tip = max(MIN_JITO_TIP_LAMPORTS, total_budget_lamports - priority_fee)
    return priority_fee, jito_tip


def select_jito_tip_account() -> str:
    return random.choice(JITO_TIP_ACCOUNTS)


def jitodontfront_instruction() -> Instruction:
    return Instruction(
        Pubkey.from_string(JITO_MEMO_PROGRAM_ID),
        b"jitodontfront",
        [AccountMeta(Pubkey.from_string(JITO_DONT_FRONT_PUBKEY), False, False)],
    )


def build_jupiter_swap_payload(
    quote_response: dict[str, Any],
    user_pubkey: str,
    total_fee_budget_lamports: int = DEFAULT_SOLANA_FEE_BUDGET_LAMPORTS,
) -> dict[str, Any]:
    priority_fee_lamports, _ = compute_fee_split(total_fee_budget_lamports)
    return {
        "quoteResponse": quote_response,
        "userPublicKey": user_pubkey,
        "wrapAndUnwrapSol": True,
        "dynamicComputeUnitLimit": True,
        "dynamicSlippage": {"minBps": 50, "maxBps": 250},
        "prioritizationFeeLamports": {
            "priorityLevelWithMaxLamports": {
                "priorityLevel": "veryHigh",
                "maxLamports": priority_fee_lamports,
            }
        },
    }


def _jupiter_instruction(payload: dict[str, Any]) -> Instruction:
    accounts = [
        AccountMeta(
            Pubkey.from_string(str(account["pubkey"])),
            bool(account.get("isSigner", False)),
            bool(account.get("isWritable", False)),
        )
        for account in payload.get("accounts", [])
    ]
    return Instruction(
        Pubkey.from_string(str(payload["programId"])),
        base64.b64decode(payload["data"]),
        accounts,
    )


def _compute_unit_limit(instructions: list[Instruction]) -> int:
    for instruction in instructions:
        data = bytes(instruction.data)
        if (
            str(instruction.program_id) == "ComputeBudget111111111111111111111111111111"
            and len(data) >= 5
            and data[0] == 2
        ):
            return struct.unpack_from("<I", data, 1)[0]
    return 1_400_000
BASE_USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
EVM_NETWORKS = {
    "base": (8453, BASE_USDC, "base_rpc_url"),
    "8453": (8453, BASE_USDC, "base_rpc_url"),
    "eth": (1, "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "ethereum_rpc_url"),
    "ethereum": (1, "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "ethereum_rpc_url"),
    "1": (1, "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "ethereum_rpc_url"),
    "bsc": (56, "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d", "bnb_rpc_url"),
    "bnb": (56, "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d", "bnb_rpc_url"),
    "56": (56, "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d", "bnb_rpc_url"),
}
EVM_NETWORKS_BY_CHAIN_ID = {
    chain_id: (stablecoin, rpc_setting)
    for chain_id, stablecoin, rpc_setting in EVM_NETWORKS.values()
}


@dataclass(frozen=True)
class OrderIntent:
    client_order_id: str
    trader_handle: str
    network: str
    token_address: str
    side: str
    notional_usd: float
    price_usd: float
    pool_reserve_usd: float
    token_amount: float = 0.0
    token_amount_raw: int | None = None
    high_conviction: bool = False
    meteora_dlmm_pool_address: str | None = None
    meteora_active_bin_id: int | None = None
    meteora_dlmm_liquidity_usd: float | None = None


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    filled_usd: float = 0.0
    filled_token_amount: float = 0.0
    fee_usd: float = 0.0
    average_price_usd: float = 0.0
    tx_id: str | None = None
    detail: str | None = None


class SubmissionUncertain(RuntimeError):
    def __init__(self, message: str, tx_id: str | None):
        super().__init__(message)
        self.tx_id = tx_id


class JitoBundleRejected(RuntimeError):
    """A definitive block-engine rejection that is safe to retry with a higher tip."""


class OrderRouter(Protocol):
    fee_rate: float
    minimum_fee_usd: float

    def supports_network(self, network: str) -> bool: ...

    async def submit(self, intent: OrderIntent) -> ExecutionResult: ...

    async def reconcile(self, client_order_id: str, tx_id: str | None) -> str | ExecutionResult: ...


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)) or (
        isinstance(exc, httpx.HTTPStatusError)
        and (exc.response.status_code == 429 or exc.response.status_code >= 500)
    )


class DryRunRouter:
    fee_rate = 0.0045
    minimum_fee_usd = 0.95

    def supports_network(self, network: str) -> bool:
        return network.lower() in {
            "solana", "sol", "1399811149", "base", "8453",
            "eth", "ethereum", "1", "bsc", "bnb", "56", "monad", "143",
        }

    async def submit(self, intent: OrderIntent) -> ExecutionResult:
        if intent.notional_usd <= 0 or intent.price_usd <= 0 or intent.pool_reserve_usd <= 0:
            raise ValueError("dry-run order requires positive notional, price, and pool liquidity")
        # A conservative constant-product approximation, not an order-book fill claim.
        impact = min(intent.notional_usd / intent.pool_reserve_usd, 0.99)
        fill_price = intent.price_usd * (1 + impact / 2 if intent.side == "BUY" else 1 - impact / 2)
        fee = max(self.minimum_fee_usd, intent.notional_usd * self.fee_rate)
        quantity = (
            intent.notional_usd / fill_price
            if intent.side == "BUY"
            else intent.token_amount or intent.notional_usd / fill_price
        )
        logger.info(
            "DRY RUN %s %s notional=%.2f fill_price=%.8f impact=%.4f fee=%.2f",
            intent.side, intent.token_address, intent.notional_usd, fill_price, impact, fee,
        )
        return ExecutionResult(
            "DRY_RUN_OPEN",
            filled_usd=intent.notional_usd,
            filled_token_amount=quantity,
            fee_usd=fee,
            average_price_usd=fill_price,
            detail=f"estimated_impact={impact:.6f}",
        )

    async def reconcile(self, client_order_id: str, tx_id: str | None) -> str:
        return "UNCERTAIN"


class FreqtradeWebhookRouter:
    """Idempotency-keyed HMAC webhook adapter; status lookup is required for reconciliation."""

    fee_rate = 0.0045
    minimum_fee_usd = 0.95

    def supports_network(self, network: str) -> bool:
        return network.lower() in {
            "solana", "sol", "1399811149", "base", "8453",
            "eth", "ethereum", "1", "bsc", "bnb", "56", "monad", "143",
        }

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None):
        self.settings = settings or get_settings()
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient()

    @retry(
        retry=retry_if_exception(_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        reraise=True,
    )
    async def submit(self, intent: OrderIntent) -> ExecutionResult:
        url = self.settings.freqtrade_webhook_url
        secret = self.settings.freqtrade_webhook_secret.get_secret_value()
        if not url or not secret:
            raise RuntimeError("live Freqtrade routing requires webhook URL and secret")
        body = json.dumps({
            "client_order_id": intent.client_order_id,
            "pair": intent.token_address,
            "side": intent.side.lower(),
            "network": intent.network,
            "stake_amount": intent.notional_usd,
            "price": intent.price_usd,
        }, separators=(",", ":"), sort_keys=True).encode()
        signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        response = await self.client.post(
            url,
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Signature-SHA256": signature,
                "Idempotency-Key": intent.client_order_id,
            },
            timeout=20,
        )
        response.raise_for_status()
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise SubmissionUncertain(
                "Freqtrade response could not be reconciled", intent.client_order_id
            ) from exc
        if not isinstance(data, dict):
            raise SubmissionUncertain("Freqtrade response could not be reconciled", intent.client_order_id)
        state = str(data.get("status", "SUBMITTED")).upper()
        if state in {"FAILED", "REJECTED", "ERROR"}:
            raise RuntimeError(f"Freqtrade rejected order: {data}")
        if state not in {"SUBMITTED", "OPEN", "FILLED"}:
            raise SubmissionUncertain(
                f"Unrecognized Freqtrade order status: {state}", intent.client_order_id
            )
        try:
            filled_amount = float(data.get("filled_amount") or 0)
            filled_usd = float(data.get("filled_usd") or 0)
            average_price = float(data.get("average_price") or intent.price_usd)
        except (TypeError, ValueError) as exc:
            raise SubmissionUncertain(
                "Freqtrade response could not be reconciled", intent.client_order_id
            ) from exc
        resolved_status = (
            "OPEN" if filled_amount > 0 and state in {"OPEN", "FILLED"} else "SUBMITTED"
        )
        return ExecutionResult(
            resolved_status,
            filled_usd=filled_usd,
            filled_token_amount=filled_amount,
            fee_usd=max(self.minimum_fee_usd, intent.notional_usd * self.fee_rate),
            average_price_usd=average_price,
            tx_id=str(data.get("order_id") or intent.client_order_id),
        )

    @retry(
        retry=retry_if_exception(_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=12),
        reraise=True,
    )
    async def _get_status(self, url: str, secret: str, client_order_id: str) -> dict[str, Any]:
        signature = hmac.new(secret.encode(), client_order_id.encode(), hashlib.sha256).hexdigest()
        response = await self.client.get(
            url,
            params={"client_order_id": client_order_id},
            headers={"X-Signature-SHA256": signature},
            timeout=15,
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Freqtrade status response must be an object")
        return body

    @retry(
        retry=retry_if_exception(_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        reraise=True,
    )
    async def _post_json(
        self, url: str, payload: dict[str, Any], *, headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        response = await self.client.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError(f"expected a JSON object from {url}")
        return body

    async def reconcile(self, client_order_id: str, tx_id: str | None) -> str | ExecutionResult:
        url = self.settings.freqtrade_order_status_url
        secret = self.settings.freqtrade_webhook_secret.get_secret_value()
        if not url or not secret:
            return "UNCERTAIN"
        result = await self._get_status(url, secret, client_order_id)
        state = str(result.get("status", "")).upper()
        mapped = {"FILLED": "OPEN", "OPEN": "OPEN", "FAILED": "FAILED", "REJECTED": "FAILED"}.get(
            state, "UNCERTAIN"
        )
        filled_amount = float(result.get("filled_amount") or 0)
        if mapped == "OPEN" and filled_amount <= 0:
            return "UNCERTAIN"
        if mapped == "UNCERTAIN":
            return mapped
        return ExecutionResult(
            mapped,
            filled_usd=float(result.get("filled_usd") or 0),
            filled_token_amount=filled_amount,
            fee_usd=float(result.get("fee_usd") or 0),
            average_price_usd=float(result.get("average_price") or 0),
            tx_id=str(result.get("order_id") or tx_id or client_order_id),
        )

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()


class DirectOnChainRouter:
    """Jupiter V6 on Solana and 0x Swap API v2 on supported EVM networks."""

    fee_rate = 0.0
    minimum_fee_usd = 0.0

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None):
        self.settings = settings or get_settings()
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient()
        self._solana_client: SolanaAsyncClient | None = None

    def supports_network(self, network: str) -> bool:
        normalized = network.lower()
        return normalized in {"solana", "sol", "1399811149", *EVM_NETWORKS}

    async def submit(self, intent: OrderIntent) -> ExecutionResult:
        if self.settings.dry_run or not self.settings.live_trading_confirmed:
            raise RuntimeError(
                "Direct on-chain submission requires DRY_RUN=false and "
                "LIVE_TRADING_CONFIRMED=true."
            )
        network = intent.network.lower()
        if network in {"solana", "sol", "1399811149"}:
            return await self._submit_solana(intent)
        if network in EVM_NETWORKS:
            return await self._submit_evm(intent)
        raise ValueError(f"direct on-chain routing is not configured for {intent.network}")

    @retry(
        retry=retry_if_exception(_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=12),
        reraise=True,
    )
    async def _get_json(self, url: str, *, headers: dict[str, str] | None = None) -> dict[str, Any]:
        response = await self.client.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("swap quote response must be an object")
        return body

    @retry(
        retry=retry_if_exception(_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        reraise=True,
    )
    async def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self.client.post(url, json=payload, timeout=20)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("swap instruction response must be an object")
        return body

    async def _submit_solana(self, intent: OrderIntent) -> ExecutionResult:
        retry_limit = 4 if intent.side == "SELL" else 0
        last_error: JitoBundleRejected | None = None
        for retry_index in range(retry_limit + 1):
            try:
                return await self._submit_solana_once(
                    intent,
                    tip_multiplier=1.75**retry_index,
                )
            except JitoBundleRejected as exc:
                if intent.side != "SELL":
                    raise
                last_error = exc
                if retry_index == retry_limit:
                    break
                logger.warning(
                    "Jito rejected SELL %s; retry %d/%d will use a %.2fx tip in the "
                    "next slot",
                    intent.client_order_id,
                    retry_index + 1,
                    retry_limit,
                    1.75 ** (retry_index + 1),
                )
                await self._wait_for_next_solana_slot()
        if last_error is None:
            raise RuntimeError("Jito SELL retry loop exited without a result")
        raise RuntimeError(
            f"Jito rejected SELL after {retry_limit} escalated retries: {last_error}"
        ) from last_error

    async def _wait_for_next_solana_slot(self) -> None:
        """Wait for a strictly newer processed slot before rebuilding a rejected bundle."""
        current = await self._solana_rpc("getSlot", [{"commitment": "processed"}])
        if not isinstance(current, int) or current < 0:
            raise RuntimeError("Solana RPC returned an invalid processed slot")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            await asyncio.sleep(0.05)
            next_slot = await self._solana_rpc("getSlot", [{"commitment": "processed"}])
            if isinstance(next_slot, int) and next_slot > current:
                return
        raise TimeoutError(f"Solana slot did not advance beyond {current} within 2 seconds")

    async def _submit_solana_once(
        self, intent: OrderIntent, *, tip_multiplier: float
    ) -> ExecutionResult:
        private_key = self.settings.solana_private_key.get_secret_value()
        if not private_key:
            raise RuntimeError("SOLANA_PRIVATE_KEY is required for live Solana execution")
        wallet = Keypair.from_base58_string(private_key)
        is_buy = intent.side == "BUY"
        input_mint = SOLANA_USDC if is_buy else intent.token_address
        output_mint = intent.token_address if is_buy else SOLANA_USDC
        amount = (
            int(round(intent.notional_usd * 1_000_000))
            if is_buy
            else intent.token_amount_raw
        )
        if not is_buy and amount is None and intent.token_amount > 0:
            amount = int(intent.token_amount * 10**await self._solana_token_decimals(input_mint))
        if amount is None or amount <= 0:
            raise ValueError("SELL requires a positive raw token amount")
        base_url = self.settings.jupiter_base_url.rstrip("/")
        quote = await self._get_json(
            f"{base_url}/quote?inputMint={input_mint}&outputMint={output_mint}"
            f"&amount={amount}&slippageBps=100"
        )
        fee_budget = self.settings.solana_fee_budget_lamports
        priority_fee_lamports, jito_tip_lamports = compute_fee_split(fee_budget)
        jito_tip_lamports = max(
            MIN_JITO_TIP_LAMPORTS,
            int(jito_tip_lamports * tip_multiplier),
        )
        swap_instructions = await self._post_json(
            f"{base_url}/swap-instructions",
            build_jupiter_swap_payload(quote, str(wallet.pubkey()), fee_budget),
        )
        compute_instructions = [
            _jupiter_instruction(instruction)
            for instruction in swap_instructions.get("computeBudgetInstructions", [])
        ]
        compute_unit_limit = _compute_unit_limit(compute_instructions)
        compute_instructions = [
            instruction for instruction in compute_instructions
            if not (
                str(instruction.program_id) == "ComputeBudget111111111111111111111111111111"
                and bytes(instruction.data)[:1] == b"\x03"
            )
        ]
        compute_instructions.append(
            set_compute_unit_price(
                priority_fee_lamports * 1_000_000 // compute_unit_limit
            )
        )
        instructions = compute_instructions + [
            _jupiter_instruction(instruction)
            for instruction in swap_instructions.get("setupInstructions", [])
        ]
        if token_ledger_instruction := swap_instructions.get("tokenLedgerInstruction"):
            instructions.append(_jupiter_instruction(token_ledger_instruction))
        swap_instruction = swap_instructions.get("swapInstruction")
        if not isinstance(swap_instruction, dict):
            raise ValueError("Jupiter did not return a swap instruction")
        instructions.append(_jupiter_instruction(swap_instruction))
        if cleanup_instruction := swap_instructions.get("cleanupInstruction"):
            instructions.append(_jupiter_instruction(cleanup_instruction))
        instructions.extend(
            _jupiter_instruction(instruction)
            for instruction in swap_instructions.get("otherInstructions", [])
        )
        instructions.append(jitodontfront_instruction())
        instructions.append(
            transfer(
                TransferParams(
                    from_pubkey=wallet.pubkey(),
                    to_pubkey=Pubkey.from_string(select_jito_tip_account()),
                    lamports=jito_tip_lamports,
                )
            )
        )
        lookup_tables = await self._solana_address_lookup_tables(
            swap_instructions.get("addressLookupTableAddresses", [])
        )
        latest_blockhash = await self._solana_rpc(
            "getLatestBlockhash", [{"commitment": "processed"}]
        )
        blockhash = (
            latest_blockhash.get("value", {}).get("blockhash")
            if isinstance(latest_blockhash, dict)
            else None
        )
        if not blockhash:
            raise ValueError("Solana RPC returned no recent blockhash")
        message = MessageV0.try_compile(
            wallet.pubkey(),
            instructions,
            lookup_tables,
            Hash.from_string(blockhash),
        )
        signed = VersionedTransaction(message, [wallet])
        signature = str(signed.signatures[0])
        out_amount = int(quote.get("outAmount") or 0)
        out_mint = output_mint if is_buy else input_mint
        decimals = await self._solana_token_decimals(out_mint)
        try:
            bundle_id = await self._send_jito_bundle(bytes(signed))
        except JitoBundleRejected:
            raise
        except Exception as exc:
            raise SubmissionUncertain(
                f"Jito bundle submission outcome is unknown: {exc}", signature
            ) from exc
        filled_tokens = out_amount / (10**decimals) if is_buy else amount / (10**decimals)
        return ExecutionResult(
            "SUBMITTED",
            filled_usd=intent.notional_usd,
            filled_token_amount=filled_tokens,
            tx_id=signature,
            detail=(
                f"jito_bundle_id={bundle_id};priority_fee_lamports={priority_fee_lamports};"
                f"jito_tip_lamports={jito_tip_lamports};"
                f"jito_tip_multiplier={tip_multiplier:.6f}"
            ),
        )

    async def _solana_address_lookup_tables(
        self, addresses: list[str]
    ) -> list[AddressLookupTableAccount]:
        lookup_tables: list[AddressLookupTableAccount] = []
        for address in addresses:
            account = await self._solana_rpc(
                "getAccountInfo", [address, {"encoding": "base64"}]
            )
            value = account.get("value") if isinstance(account, dict) else None
            data = value.get("data") if isinstance(value, dict) else None
            if not isinstance(data, list) or not data or not isinstance(data[0], str):
                raise ValueError(f"Solana RPC returned no address lookup table data for {address}")
            table = AddressLookupTable.deserialize(base64.b64decode(data[0]))
            lookup_tables.append(
                AddressLookupTableAccount(Pubkey.from_string(address), table.addresses)
            )
        return lookup_tables

    async def _send_jito_bundle(self, signed_transaction: bytes) -> str:
        response = await self.client.post(
            self.settings.jito_block_engine_url,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "sendBundle",
                "params": [[base64.b64encode(signed_transaction).decode("ascii")], {"encoding": "base64"}],
            },
            timeout=20,
        )
        if response.status_code == 429 or 400 <= response.status_code < 500:
            raise JitoBundleRejected(
                f"Jito rejected bundle with HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or body.get("error"):
            raise JitoBundleRejected(f"Jito rejected bundle submission: {body}")
        bundle_id = body.get("result")
        if not isinstance(bundle_id, str) or not bundle_id:
            raise ValueError("Jito returned no bundle ID")
        return bundle_id

    async def _solana_rpc(self, method: str, params: list[Any]) -> Any:
        return await self._solana_rpc_request(method, params)

    async def _send_solana_transaction(self, signed_transaction: bytes) -> str:
        if self._solana_client is None:
            self._solana_client = SolanaAsyncClient(
                self.settings.solana_rpc_url,
                timeout=20,
                max_transport_retries=0,
            )
        response = await self._solana_client.send_raw_transaction(
            signed_transaction,
            opts=TxOpts(skip_preflight=False, max_retries=2),
        )
        signature = getattr(response, "value", None)
        if signature is None:
            raise RuntimeError("Solana RPC returned no signature for the submitted transaction")
        return str(signature)

    @retry(
        retry=retry_if_exception(_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        reraise=True,
    )
    async def _solana_rpc_request(self, method: str, params: list[Any]) -> Any:
        response = await self.client.post(
            self.settings.solana_rpc_url,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            timeout=20,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("error"):
            raise RuntimeError(f"Solana RPC {method} failed: {body['error']}")
        return body.get("result")

    async def _solana_token_decimals(self, mint: str) -> int:
        supply = await self._solana_rpc("getTokenSupply", [mint])
        value = (supply or {}).get("value", {})
        if "decimals" not in value:
            raise ValueError(f"Solana RPC returned no decimals for token {mint}")
        return int(value["decimals"])

    async def _submit_evm(self, intent: OrderIntent) -> ExecutionResult:
        private_key = self.settings.evm_private_key.get_secret_value()
        if not private_key:
            raise RuntimeError("EVM_PRIVATE_KEY is required for live EVM execution")
        account = Account.from_key(private_key)
        chain_id, stablecoin, rpc_setting = EVM_NETWORKS[intent.network.lower()]
        rpc_url = getattr(self.settings, rpc_setting)
        stablecoin_decimals = await self._evm_token_decimals(stablecoin, rpc_url)
        is_buy = intent.side == "BUY"
        sell_token = stablecoin if is_buy else intent.token_address
        buy_token = intent.token_address if is_buy else stablecoin
        amount = (
            int(round(intent.notional_usd * 10**stablecoin_decimals))
            if is_buy else intent.token_amount_raw
        )
        if not is_buy and amount is None and intent.token_amount > 0:
            amount = int(intent.token_amount * 10**await self._evm_token_decimals(sell_token, rpc_url))
        if amount is None or amount <= 0:
            raise ValueError("SELL requires a positive raw token amount")
        headers = {
            "0x-api-key": self.settings.zerox_api_key.get_secret_value(),
            "0x-version": "v2",
        }
        if not headers["0x-api-key"]:
            raise RuntimeError("ZEROX_API_KEY is required for live EVM execution")
        quote_url = (
            f"{self.settings.zerox_base_url.rstrip('/')}/swap/allowance-holder/quote"
            f"?chainId={chain_id}&sellToken={sell_token}&buyToken={buy_token}&sellAmount={amount}"
            f"&taker={account.address}&slippageBps=100"
        )
        quote = await self._get_json(quote_url, headers=headers)
        transaction = quote.get("transaction")
        if not isinstance(transaction, dict):
            raise ValueError("0x did not return a transaction")
        allowance = quote.get("issues", {}).get("allowance", {})
        spender = allowance.get("spender") if isinstance(allowance, dict) else None
        if spender:
            current_allowance = int(
                await self._evm_rpc(
                    rpc_url,
                    "eth_call",
                    [{"to": sell_token, "data": "0xdd62ed3e" + account.address[2:].lower().rjust(64, "0")
                      + str(spender)[2:].lower().rjust(64, "0")}, "latest"],
                ),
                16,
            )
            if current_allowance < amount:
                approval_data = "0x095ea7b3" + str(spender)[2:].lower().rjust(64, "0") + hex(amount)[2:].rjust(64, "0")
                try:
                    approval_hash = await self._broadcast_evm_transaction(
                        account, {"to": sell_token, "value": "0x0", "data": approval_data},
                        rpc_url=rpc_url,
                    )
                except SubmissionUncertain as exc:
                    tagged_tx_id = (
                        f"approval:{chain_id}:{exc.tx_id}" if exc.tx_id else None
                    )
                    raise SubmissionUncertain(str(exc), tagged_tx_id) from exc
                try:
                    approval_receipt = await self._wait_evm_receipt(approval_hash, rpc_url=rpc_url)
                except (httpx.HTTPError, TimeoutError, RuntimeError, ValueError) as exc:
                    raise SubmissionUncertain(
                        f"EVM token approval outcome is unknown: {exc}",
                        f"approval:{chain_id}:{approval_hash}",
                    ) from exc
                if int(approval_receipt.get("status", "0x0"), 16) != 1:
                    raise RuntimeError("EVM token approval transaction failed")
        out_amount = int(quote.get("buyAmount") or 0)
        out_token = buy_token if is_buy else sell_token
        decimals = await self._evm_token_decimals(out_token, rpc_url)
        try:
            tx_hash = await self._broadcast_evm_transaction(account, transaction, rpc_url=rpc_url)
        except SubmissionUncertain as exc:
            tagged_tx_id = f"eip155:{chain_id}:{exc.tx_id}" if exc.tx_id else None
            raise SubmissionUncertain(str(exc), tagged_tx_id) from exc
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            raise SubmissionUncertain(
                f"EVM transaction broadcast outcome is unknown: {exc}", None
            ) from exc
        filled_tokens = out_amount / (10**decimals) if is_buy else amount / (10**decimals)
        return ExecutionResult(
            "SUBMITTED",
            filled_usd=intent.notional_usd,
            filled_token_amount=filled_tokens,
            tx_id=f"eip155:{chain_id}:{tx_hash}",
        )

    async def _broadcast_evm_transaction(
        self, account: Any, transaction: dict[str, Any], *, rpc_url: str | None = None
    ) -> str:
        rpc = rpc_url or self.settings.base_rpc_url
        nonce = int(await self._evm_rpc(rpc, "eth_getTransactionCount", [account.address, "pending"]), 16)
        chain_id = int(await self._evm_rpc(rpc, "eth_chainId", []), 16)
        tx: dict[str, Any] = {
            "chainId": chain_id,
            "nonce": nonce,
            "to": transaction["to"],
            "value": int(transaction.get("value", "0x0"), 16)
            if isinstance(transaction.get("value", "0x0"), str)
            else int(transaction.get("value", 0)),
            "data": transaction.get("data", "0x"),
            "gas": int(transaction.get("gas", "0x0"), 16)
            if isinstance(transaction.get("gas", "0x0"), str)
            else int(transaction.get("gas", 0)),
        }
        gas_price = transaction.get("gasPrice")
        tx["gasPrice"] = (
            int(gas_price, 16) if isinstance(gas_price, str) and gas_price.startswith("0x")
            else int(gas_price) if gas_price is not None
            else int(await self._evm_rpc(rpc, "eth_gasPrice", []), 16)
        )
        if tx["gas"] <= 0:
            tx["gas"] = int(await self._evm_rpc(
                rpc, "eth_estimateGas", [{
                    "from": account.address, "to": tx["to"], "value": hex(tx["value"]), "data": tx["data"]
                }]
            ), 16)
        signed = Account.sign_transaction(tx, self.settings.evm_private_key.get_secret_value())
        expected_hash = signed.hash.hex()
        try:
            tx_hash = await self._evm_rpc(rpc, "eth_sendRawTransaction", [signed.raw_transaction.hex()])
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            raise SubmissionUncertain(f"Base transaction broadcast outcome is unknown: {exc}", expected_hash) from exc
        return str(tx_hash)

    async def _evm_rpc(self, url: str, method: str, params: list[Any]) -> Any:
        return await self._evm_rpc_request(url, method, params)

    @retry(
        retry=retry_if_exception(_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        reraise=True,
    )
    async def _evm_rpc_request(self, url: str, method: str, params: list[Any]) -> Any:
        response = await self.client.post(
            url,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            timeout=25,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("error"):
            raise RuntimeError(f"EVM RPC {method} failed: {body['error']}")
        return body.get("result")

    async def _evm_token_decimals(self, token: str, rpc_url: str | None = None) -> int:
        result = await self._evm_rpc(
            rpc_url or self.settings.base_rpc_url,
            "eth_call", [{"to": token, "data": "0x313ce567"}, "latest"]
        )
        if not isinstance(result, str):
            raise ValueError(f"EVM RPC returned no decimals for token {token}")
        return int(result, 16)

    async def _wait_evm_receipt(
        self, tx_hash: str, timeout_seconds: int = 120, *, rpc_url: str | None = None
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            receipt = await self._evm_rpc(
                rpc_url or self.settings.base_rpc_url, "eth_getTransactionReceipt", [tx_hash]
            )
            if receipt is not None:
                return receipt
            await asyncio.sleep(2)
        raise TimeoutError(f"Timed out waiting for EVM transaction {tx_hash}")

    async def reconcile(self, client_order_id: str, tx_id: str | None) -> str:
        if not tx_id:
            return "UNCERTAIN"
        try:
            if tx_id.startswith("approval:"):
                _, chain_id_raw, approval_hash = tx_id.split(":", 2)
                evm_network = EVM_NETWORKS_BY_CHAIN_ID.get(int(chain_id_raw))
                if evm_network is None:
                    return "UNCERTAIN"
                rpc_url = getattr(self.settings, evm_network[1])
                receipt = await self._evm_rpc(
                    rpc_url,
                    "eth_getTransactionReceipt",
                    [approval_hash],
                )
                if receipt is None:
                    return "UNCERTAIN"
                logger.warning(
                    "Approval transaction for %s resolved; no swap was broadcast, requiring a fresh signal",
                    client_order_id,
                )
                return "FAILED"
            if tx_id.startswith("eip155:"):
                _, chain_id_raw, tx_hash = tx_id.split(":", 2)
                evm_network = EVM_NETWORKS_BY_CHAIN_ID.get(int(chain_id_raw))
                if evm_network is None:
                    return "UNCERTAIN"
                receipt = await self._evm_rpc(
                    getattr(self.settings, evm_network[1]),
                    "eth_getTransactionReceipt",
                    [tx_hash],
                )
                if receipt is None:
                    return "UNCERTAIN"
                return "OPEN" if int(receipt.get("status", "0x0"), 16) == 1 else "FAILED"
            if not tx_id.startswith("0x"):
                status = await self._solana_rpc("getSignatureStatuses", [[tx_id], {"searchTransactionHistory": True}])
                value = (status or {}).get("value", [None])[0]
                if value and value.get("err") is not None:
                    return "FAILED"
                if value and value.get("confirmationStatus") in {"confirmed", "finalized"}:
                    return "OPEN"
                return "UNCERTAIN"
            receipt = await self._evm_rpc(
                self.settings.base_rpc_url, "eth_getTransactionReceipt", [tx_id]
            )
            if receipt is None:
                return "UNCERTAIN"
            return "OPEN" if int(receipt.get("status", "0x0"), 16) == 1 else "FAILED"
        except (httpx.HTTPError, RuntimeError, ValueError):
            logger.exception("Could not reconcile on-chain order %s", client_order_id)
            return "UNCERTAIN"

    async def close(self) -> None:
        if self._solana_client is not None:
            await self._solana_client.close()
            self._solana_client = None
        if self._owns_client:
            await self.client.aclose()


async def route_jit_dlmm_order(
    intent: OrderIntent,
    *,
    sidecar_url: str,
    pool_address: str,
    active_bin_id: int,
    pool_liquidity_usd: float,
    minimum_liquidity_usd: float,
    client: httpx.AsyncClient | None = None,
) -> ExecutionResult:
    """Atomically buy and seed a DLMM bin ladder through a configured sidecar.

    The sidecar must return an OPEN, atomically confirmed fill and confirm that its
    high-water-mark reversal withdrawal is armed. No success-shaped fallback is used.
    """
    if intent.side != "BUY":
        raise ValueError("Meteora DLMM JIT routing supports BUY intents only")
    if intent.network.lower() not in {"solana", "sol", "1399811149"}:
        raise ValueError("Meteora DLMM routing is only supported on Solana")
    if not sidecar_url.strip():
        raise RuntimeError("METEORA_DLMM_SIDECAR_URL is required for DLMM routing")
    if not pool_address.strip():
        raise ValueError("an active DLMM pool address is required")
    if (
        not math.isfinite(pool_liquidity_usd)
        or not math.isfinite(minimum_liquidity_usd)
        or minimum_liquidity_usd <= 0
        or pool_liquidity_usd < minimum_liquidity_usd
    ):
        raise ValueError("DLMM pool liquidity is below the configured minimum")
    if intent.notional_usd <= 0 or intent.price_usd <= 0:
        raise ValueError("DLMM BUY requires positive notional and reference price")

    owns_client = client is None
    http_client = client or httpx.AsyncClient()
    try:
        response = await http_client.post(
            sidecar_url,
            json={
                "operation": "open_jit_position",
                "client_order_id": intent.client_order_id,
                "token_mint": intent.token_address,
                "pool_address": pool_address,
                "input_amount_usd": intent.notional_usd,
                "active_bin_id": active_bin_id,
                "bin_ladder": {
                    "from_bin_id": active_bin_id + 2,
                    "to_bin_id": active_bin_id + 14,
                    "strategy": "single_sided_upward_bid_ask",
                },
                "atomic": True,
                "withdraw_on_high_water_reversal": {
                    "enabled": True,
                    "trigger": "price_below_active_high_water_mark",
                    "action": "withdraw_all",
                },
            },
            headers={
                "Content-Type": "application/json",
                "Idempotency-Key": intent.client_order_id,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    finally:
        if owns_client:
            await http_client.aclose()

    if not isinstance(data, dict):
        raise ValueError("Meteora sidecar response must be a JSON object")
    if data.get("error"):
        raise RuntimeError(f"Meteora sidecar rejected JIT order: {data['error']}")
    if data.get("atomic") is not True:
        raise RuntimeError("Meteora sidecar did not confirm atomic acquisition and seeding")
    if data.get("hwm_withdrawal_armed") is not True:
        raise RuntimeError("Meteora sidecar did not arm the high-water-mark withdrawal")
    if str(data.get("status", "")).upper() != "OPEN":
        raise RuntimeError(f"Meteora sidecar returned non-final status: {data.get('status')}")
    try:
        filled_usd = float(data["filled_usd"])
        filled_token_amount = float(data["filled_token_amount"])
        fee_usd = float(data.get("fee_usd", 0.0))
        average_price_usd = float(data["average_price_usd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Meteora sidecar returned incomplete fill metrics") from exc
    if (
        not all(
            math.isfinite(value)
            for value in (filled_usd, filled_token_amount, fee_usd, average_price_usd)
        )
        or filled_usd <= 0
        or filled_token_amount <= 0
        or fee_usd < 0
        or average_price_usd <= 0
    ):
        raise ValueError("Meteora sidecar returned invalid fill metrics")
    tx_id = data.get("tx_id")
    if not isinstance(tx_id, str) or not tx_id:
        raise ValueError("Meteora sidecar returned no transaction ID")
    return ExecutionResult(
        status="OPEN",
        filled_usd=filled_usd,
        filled_token_amount=filled_token_amount,
        fee_usd=fee_usd,
        average_price_usd=average_price_usd,
        tx_id=tx_id,
        detail=(
            f"meteora_dlmm=true;atomic=true;pool={pool_address};"
            f"bin_ladder={active_bin_id + 2}:{active_bin_id + 14};"
            "hwm_withdrawal_armed=true"
        ),
    )


class MeteoraDLMMHybridRouter:
    """Route qualifying whale buys through DLMM and delegate all other orders."""

    def __init__(
        self, base_router: OrderRouter, settings: Settings, client: httpx.AsyncClient
    ) -> None:
        self.base_router = base_router
        self.settings = settings
        self.client = client
        self.fee_rate = base_router.fee_rate
        self.minimum_fee_usd = base_router.minimum_fee_usd

    def supports_network(self, network: str) -> bool:
        return self.base_router.supports_network(network)

    async def _discover_active_pool(
        self, token_address: str
    ) -> tuple[str, int, float] | None:
        if not self.settings.meteora_dlmm_sidecar_url:
            return None
        response = await self.client.post(
            self.settings.meteora_dlmm_sidecar_url,
            json={"operation": "find_active_pool", "token_mint": token_address},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Meteora pool discovery response must be a JSON object")
        status = str(data.get("status", "")).upper()
        if status in {"NOT_FOUND", "INACTIVE"}:
            return None
        if status != "FOUND" or data.get("active") is not True:
            raise ValueError(f"Meteora sidecar returned invalid pool discovery state: {status}")
        pool_address = data.get("pool_address")
        try:
            active_bin_id = int(data["active_bin_id"])
            liquidity_usd = float(data["liquidity_usd"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Meteora sidecar returned incomplete pool metadata") from exc
        if (
            not isinstance(pool_address, str)
            or not pool_address
            or not math.isfinite(liquidity_usd)
            or liquidity_usd <= 0
        ):
            raise ValueError("Meteora sidecar returned invalid pool metadata")
        return pool_address, active_bin_id, liquidity_usd

    async def submit(self, intent: OrderIntent) -> ExecutionResult:
        if intent.side == "BUY" and intent.high_conviction:
            pool_address = intent.meteora_dlmm_pool_address
            active_bin_id = intent.meteora_active_bin_id
            liquidity_usd = intent.meteora_dlmm_liquidity_usd
            if pool_address is None and self.settings.meteora_dlmm_sidecar_url:
                discovered_pool = await self._discover_active_pool(intent.token_address)
                if discovered_pool is not None:
                    pool_address, active_bin_id, liquidity_usd = discovered_pool
            if pool_address is not None:
                if active_bin_id is None:
                    raise ValueError("active Meteora DLMM bin ID is missing")
                if liquidity_usd is None:
                    liquidity_usd = intent.pool_reserve_usd
                if liquidity_usd < self.settings.meteora_dlmm_min_liquidity_usd:
                    logger.info(
                        "Meteora DLMM pool %s is below minimum liquidity; using base router",
                        pool_address,
                    )
                    return await self.base_router.submit(intent)
                if not self.settings.meteora_dlmm_sidecar_url:
                    raise RuntimeError(
                        "METEORA_DLMM_SIDECAR_URL is required for an active DLMM pool"
                    )
                return await route_jit_dlmm_order(
                    intent,
                    sidecar_url=self.settings.meteora_dlmm_sidecar_url,
                    pool_address=pool_address,
                    active_bin_id=active_bin_id,
                    pool_liquidity_usd=liquidity_usd,
                    minimum_liquidity_usd=self.settings.meteora_dlmm_min_liquidity_usd,
                    client=self.client,
                )
        return await self.base_router.submit(intent)

    async def reconcile(
        self, client_order_id: str, tx_id: str | None
    ) -> str | ExecutionResult:
        if not self.settings.meteora_dlmm_sidecar_url:
            return await self.base_router.reconcile(client_order_id, tx_id)
        response = await self.client.post(
            self.settings.meteora_dlmm_sidecar_url,
            json={"operation": "get_order_status", "client_order_id": client_order_id},
            headers={"Idempotency-Key": client_order_id},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Meteora sidecar status response must be a JSON object")
        status = str(data.get("status", "")).upper()
        if status in {"NOT_FOUND", "UNSUPPORTED"}:
            return await self.base_router.reconcile(client_order_id, tx_id)
        if status == "UNCERTAIN":
            return "UNCERTAIN"
        if status not in {"OPEN", "FAILED"}:
            raise ValueError(f"Meteora sidecar returned invalid order state: {status}")
        if status == "FAILED":
            return "FAILED"
        try:
            filled_usd = float(data["filled_usd"])
            filled_token_amount = float(data["filled_token_amount"])
            fee_usd = float(data.get("fee_usd", 0.0))
            average_price_usd = float(data["average_price_usd"])
            resolved_tx_id = str(data.get("tx_id") or tx_id or client_order_id)
            if (
                not all(
                    math.isfinite(value)
                    for value in (filled_usd, filled_token_amount, fee_usd, average_price_usd)
                )
                or filled_usd <= 0
                or filled_token_amount <= 0
                or fee_usd < 0
                or average_price_usd <= 0
                or not resolved_tx_id
            ):
                raise ValueError("Meteora sidecar returned invalid reconciliation metrics")
            return ExecutionResult(
                status="OPEN",
                filled_usd=filled_usd,
                filled_token_amount=filled_token_amount,
                fee_usd=fee_usd,
                average_price_usd=average_price_usd,
                tx_id=resolved_tx_id,
                detail="meteora_dlmm=true;reconciled=true",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Meteora sidecar returned incomplete reconciliation data") from exc

    async def close(self) -> None:
        await close_router(self.base_router)


async def close_router(router: OrderRouter) -> None:
    close = getattr(router, "close", None)
    if close is not None:
        await close()

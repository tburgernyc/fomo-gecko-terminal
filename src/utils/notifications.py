import asyncio
import logging

import httpx

from src.config import Settings

logger = logging.getLogger("fomo_terminal.notifications")


async def send_alert(message: str, severity: str = "INFO") -> None:
    """Send an alert to each configured Telegram and Discord destination."""
    if not message.strip():
        raise ValueError("alert message must not be empty")

    severity = severity.upper()
    if severity not in {"INFO", "SUCCESS", "WARNING", "ALERT", "ERROR", "CRITICAL"}:
        raise ValueError(f"unsupported alert severity: {severity}")

    settings = Settings()
    telegram_token = settings.telegram_bot_token.get_secret_value()
    telegram_chat_id = settings.telegram_chat_id
    discord_webhook_url = settings.discord_webhook_url.get_secret_value()

    if bool(telegram_token) != bool(telegram_chat_id):
        raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be configured together")

    destinations = []
    if telegram_token and telegram_chat_id:
        destinations.append(
            (
                "Telegram",
                f"https://api.telegram.org/bot{telegram_token}/sendMessage",
                {"chat_id": telegram_chat_id, "text": f"[{severity}] FOMO Gecko Terminal\n{message}"},
            )
        )
    if discord_webhook_url:
        destinations.append(
            (
                "Discord",
                discord_webhook_url,
                {"content": f"**[{severity}] FOMO Gecko Terminal**\n{message}"},
            )
        )

    if not destinations:
        logger.info("Alert not sent because no notification destinations are configured")
        return

    async def post(client: httpx.AsyncClient, url: str, payload: dict[str, str]) -> None:
        response = await client.post(url, json=payload)
        response.raise_for_status()

    async with httpx.AsyncClient(timeout=5.0) as client:
        results = await asyncio.gather(
            *(post(client, url, payload) for _, url, payload in destinations),
            return_exceptions=True,
        )

    failures = []
    for (name, _, _), result in zip(destinations, results):
        if isinstance(result, Exception):
            logger.error("%s alert delivery failed (%s)", name, type(result).__name__)
            failures.append(name)

    if failures:
        raise RuntimeError(f"alert delivery failed for: {', '.join(failures)}")

import httpx
import pytest

from src.utils import notifications


class FakeResponse:
    def raise_for_status(self) -> None:
        return None


class FakeAsyncClient:
    def __init__(self, timeout: float):
        self.timeout = timeout
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def post(self, url: str, json: dict[str, str]) -> FakeResponse:
        self.requests.append((url, json))
        return FakeResponse()


@pytest.mark.asyncio
async def test_send_alert_dispatches_to_configured_channels(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "test-chat")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/webhook")
    fake_client = FakeAsyncClient(timeout=5.0)
    monkeypatch.setattr(
        notifications.httpx, "AsyncClient", lambda timeout: fake_client
    )

    await notifications.send_alert("Order completed", "success")

    assert fake_client.timeout == 5.0
    assert len(fake_client.requests) == 2
    telegram_url, telegram_payload = fake_client.requests[0]
    discord_url, discord_payload = fake_client.requests[1]
    assert telegram_url.endswith("/bottest-token/sendMessage")
    assert telegram_payload["chat_id"] == "test-chat"
    assert "[SUCCESS]" in telegram_payload["text"]
    assert discord_url == "https://discord.example/webhook"
    assert "[SUCCESS]" in discord_payload["content"]


@pytest.mark.asyncio
async def test_send_alert_rejects_partial_telegram_configuration(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)

    with pytest.raises(ValueError, match="must be configured together"):
        await notifications.send_alert("Order completed")


@pytest.mark.asyncio
async def test_send_alert_surfaces_http_failures_without_leaking_webhook(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.example/private-secret")

    class FailedClient(FakeAsyncClient):
        async def post(self, url: str, json: dict[str, str]) -> FakeResponse:
            request = httpx.Request("POST", url)
            response = httpx.Response(503, request=request)
            raise httpx.HTTPStatusError("unavailable", request=request, response=response)

    monkeypatch.setattr(
        notifications.httpx, "AsyncClient", lambda timeout: FailedClient(timeout)
    )

    with pytest.raises(RuntimeError, match="Discord"):
        await notifications.send_alert("Order completed")

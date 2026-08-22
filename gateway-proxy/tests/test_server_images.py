"""Remote image hydration (_hydrate_remote_images) with a mocked HTTP transport."""

import asyncio

import httpx

import server


def _body(url: str) -> dict:
    return {
        "model": "zai/glm-5.2",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
    }


def _run(coro):
    asyncio.run(coro)


def test_remote_image_becomes_data_url():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"\x89PNG-fake", headers={"content-type": "image/png"}
        )

    body = _body("https://img.example.com/a.png")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await server._hydrate_remote_images(client, body)

    _run(run())

    url = body["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")


def test_failed_fetch_leaves_url_passthrough():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    body = _body("https://img.example.com/b.png")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await server._hydrate_remote_images(client, body)

    _run(run())
    assert body["messages"][0]["content"][1]["image_url"]["url"] == (
        "https://img.example.com/b.png"
    )


def test_oversized_image_rejected_and_passthrough(monkeypatch):
    monkeypatch.setattr(server, "IMAGE_FETCH_MAX_BYTES", 4)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"1234567890", headers={"content-type": "image/png"}
        )

    body = _body("https://img.example.com/big.png")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await server._hydrate_remote_images(client, body)

    _run(run())
    assert body["messages"][0]["content"][1]["image_url"]["url"] == (
        "https://img.example.com/big.png"
    )
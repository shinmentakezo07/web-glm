"""
fx-style AI Gateway proxy — v3 AI SDK protocol.

Exposes OpenAI-compatible endpoints (/v1/chat/completions, /v1/models,
/v1/responses, /v1/embeddings) and forwards to the Vercel AI Gateway using
the **same v3 AI SDK protocol** the native fx CLI uses:

    POST https://ai-gateway.vercel.sh/v3/ai/language-model

    Headers:
      Authorization: Bearer <gateway_key>
      User-Agent: fx/0.0.3
      HTTP-Referer: https://github.com/vercel-labs/fx
      X-Title: fx
      ai-gateway-protocol-version: 0.0.1
      ai-language-model-specification-version: 4
      ai-language-model-id: <model>
      ai-language-model-streaming: true

    Body (AI SDK v3 format):
      {prompt, tools, toolChoice, responseFormat, reasoning, providerOptions}

The proxy translates incoming OpenAI-format requests to AI SDK v3 format via
`converter.openai_to_v3` and converts the v3 SSE stream back to OpenAI-format
chunks via `converter.v3_stream_iter` (stream) or
`converter.v3_sse_stream_to_openai` (non-stream).

Features beyond plain forwarding:
  * client-side tool-history validation (clear 400s instead of opaque
    gateway "Invalid input" errors)
  * upstream error bodies normalized to the OpenAI error shape
  * upstream stream cancelled when the client disconnects
  * optional usage reporting on the final streaming chunk
  * configurable upstream timeouts, HTTP/2, and model-list caching
  * request logging
"""

from __future__ import annotations

import json
import os
import time
import uuid
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urljoin

# Load .env file if present (python-dotenv ships with uvicorn[standard]).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import httpx
from fastapi import FastAPI, Request, HTTPException, Depends, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import StreamingResponse, JSONResponse

from converter import (
    openai_to_v3,
    validate_tool_history,
    v3_stream_iter,
    v3_sse_stream_to_openai,
    responses_input_to_messages,
    openai_to_responses,
    openai_chunk_to_responses_sse,
    _ResponsesStreamState,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

GATEWAY_BASE_URL = os.getenv("GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh")
GATEWAY_API_KEY = os.getenv("AI_GATEWAY_API_KEY", "")
GATEWAY_TEAM = os.getenv("GATEWAY_TEAM", "")
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "zai/glm-5.2")

GATEWAY_TIMEOUT_CONNECT = float(os.getenv("GATEWAY_TIMEOUT_CONNECT", "15"))
GATEWAY_TIMEOUT_READ = float(os.getenv("GATEWAY_TIMEOUT_READ", "300"))
GATEWAY_TIMEOUT_WRITE = float(os.getenv("GATEWAY_TIMEOUT_WRITE", "60"))
GATEWAY_TIMEOUT_POOL = float(os.getenv("GATEWAY_TIMEOUT_POOL", "60"))
GATEWAY_HTTP2 = os.getenv("GATEWAY_HTTP2", "1").lower() in ("1", "true", "yes")
MODELS_CACHE_TTL = float(os.getenv("MODELS_CACHE_TTL", "300"))

GATEWAY_V3_CHAT = "/v3/ai/language-model"
GATEWAY_V1_MODELS = "/v1/models"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("gateway-proxy")

bearer = HTTPBearer(auto_error=False)


def _build_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=GATEWAY_TIMEOUT_CONNECT,
            read=GATEWAY_TIMEOUT_READ,
            write=GATEWAY_TIMEOUT_WRITE,
            pool=GATEWAY_TIMEOUT_POOL,
        ),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        http2=GATEWAY_HTTP2,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Reuse a client that was injected ahead of time (e.g. by tests).
    if getattr(app.state, "client", None) is None:
        app.state.client = _build_client()
    if not hasattr(app.state, "models_cache"):
        app.state.models_cache = {"data": None, "expires": 0.0}
    client: httpx.AsyncClient = app.state.client
    log.info("proxy ready  gateway=%s  default_model=%s  http2=%s",
             GATEWAY_BASE_URL, DEFAULT_MODEL, GATEWAY_HTTP2)
    if not GATEWAY_API_KEY:
        log.warning("AI_GATEWAY_API_KEY not set — upstream will reject requests")
    if not PROXY_API_KEY:
        log.warning("PROXY_API_KEY not set — proxy is open to all callers")
    yield
    await client.aclose()


app = FastAPI(title="fx-style AI Gateway proxy", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Request logging middleware
# --------------------------------------------------------------------------- #


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("%s %s failed", request.method, request.url.path)
        raise
    duration_ms = (time.monotonic() - start) * 1000
    log.info("%s %s -> %d (%.0fms)", request.method, request.url.path,
             response.status_code, duration_ms)
    return response


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #


def verify_proxy_key(
    creds: HTTPAuthorizationCredentials | None = Security(bearer),
) -> str:
    if not PROXY_API_KEY:
        return "anonymous"
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if creds.credentials != PROXY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return creds.credentials


def _get_client() -> httpx.AsyncClient:
    client: httpx.AsyncClient | None = getattr(app.state, "client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="client not ready")
    return client


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _v3_headers(model: str, streaming: bool) -> dict[str, str]:
    headers: dict[str, str] = {
        "Authorization": f"Bearer {GATEWAY_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "fx/0.0.3",
        "HTTP-Referer": "https://github.com/vercel-labs/fx",
        "X-Title": "fx",
        "ai-gateway-protocol-version": "0.0.1",
        "ai-language-model-specification-version": "4",
        "ai-language-model-id": model,
        "ai-language-model-streaming": "true" if streaming else "false",
    }
    if streaming:
        headers["Accept"] = "text/event-stream"
    if GATEWAY_TEAM:
        headers["x-vercel-ai-gateway-team"] = GATEWAY_TEAM
    return headers


def _parse_sse_line(line: str) -> str | None:
    """Strip the SSE `data: ` prefix. Returns None for done/empty lines."""
    if not line:
        return None
    stripped = line.strip()
    if stripped in ("DONE", "[DONE]"):
        return None
    if stripped.startswith("data: "):
        return stripped[6:]
    if stripped.startswith("data:"):
        return stripped[5:]
    return stripped


async def _v3_lines_to_events(lines: AsyncIterator[str]) -> AsyncIterator[dict]:
    """Parse raw upstream SSE lines into event dicts."""
    async for raw in lines:
        data = _parse_sse_line(raw)
        if data is None:
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def _client_error(resp: httpx.Response) -> JSONResponse:
    """Normalize an upstream error into an OpenAI-shaped error body."""
    try:
        body = resp.json()
    except Exception:
        body = None
    if isinstance(body, dict) and "error" in body:
        return JSONResponse(status_code=resp.status_code, content=body)
    message = None
    if isinstance(body, dict):
        message = body.get("message") or json.dumps(body)[:2000]
    if not message:
        message = (resp.text or f"upstream error (HTTP {resp.status_code})")[:2000]
    return JSONResponse(status_code=resp.status_code, content={
        "error": {"message": message, "type": "upstream_error"},
    })


def _invalid_request(message: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={
        "error": {"message": message, "type": "invalid_request_error"},
    })


def _warn_dropped_params(body: dict) -> None:
    """The gateway only supports the mapped params; log anything we drop."""
    unsupported = ("seed", "logprobs", "top_logprobs", "presence_penalty",
                   "frequency_penalty", "user", "logit_bias")
    for key in unsupported:
        if key in body:
            log.warning("dropping unsupported request param: %s", key)
    if body.get("n", 1) != 1:
        log.warning("n=%s unsupported by the gateway; forcing 1", body.get("n"))


async def _parse_body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


async def _send_upstream(
    client: httpx.AsyncClient, url: str, headers: dict, v3_body: dict
) -> httpx.Response:
    req = client.build_request("POST", url, headers=headers, json=v3_body)
    resp = await client.send(req, stream=True)
    if resp.status_code != 200:
        await resp.aread()
    return resp


# --------------------------------------------------------------------------- #
# Streaming response generators
# --------------------------------------------------------------------------- #


async def _chat_stream(
    resp: httpx.Response, model: str, include_usage: bool
) -> AsyncIterator[str]:
    """Consume the upstream v3 stream and yield OpenAI SSE chunks.

    The upstream response is closed in a finally block, so if the client
    disconnects (generator cancelled) the upstream request is cancelled too.
    """
    try:
        async for chunk in v3_stream_iter(
            _v3_lines_to_events(resp.aiter_lines()), model, include_usage
        ):
            yield chunk
    finally:
        await resp.aclose()


async def _responses_stream(resp: httpx.Response, model: str) -> AsyncIterator[str]:
    """Consume the upstream v3 stream and yield Responses API SSE events."""
    state = _ResponsesStreamState(model)
    try:
        async for chunk in v3_stream_iter(
            _v3_lines_to_events(resp.aiter_lines()), model, include_usage=True
        ):
            out = openai_chunk_to_responses_sse(chunk, state)
            if out:
                yield out
    finally:
        await resp.aclose()


async def _collect_response(resp: httpx.Response, model: str) -> dict:
    events: list[dict] = []
    async for raw in resp.aiter_lines():
        data = _parse_sse_line(raw)
        if data is None:
            continue
        try:
            events.append(json.loads(data))
        except json.JSONDecodeError:
            continue
    return v3_sse_stream_to_openai(iter(events), model)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "gateway": GATEWAY_BASE_URL, "protocol": "v3"}


@app.get("/v1/models")
async def list_models(_: str = Depends(verify_proxy_key)):
    client = _get_client()
    cache: dict = getattr(app.state, "models_cache", {})
    now = time.monotonic()
    if cache.get("expires", 0) > now and cache.get("data") is not None:
        log.info("GET /v1/models -> cached (%d models)", len(cache["data"].get("data", [])))
        return JSONResponse(content=cache["data"])

    url = urljoin(GATEWAY_BASE_URL + "/", GATEWAY_V1_MODELS.lstrip("/"))
    headers = {"Accept": "application/json"}
    if GATEWAY_API_KEY:
        headers["Authorization"] = f"Bearer {GATEWAY_API_KEY}"
    resp = await client.get(url, headers=headers)
    if resp.status_code != 200:
        return _client_error(resp)
    data = resp.json()
    cache["data"] = data
    cache["expires"] = now + MODELS_CACHE_TTL
    log.info("GET /v1/models -> upstream (%d models, cached %.0fs)",
             len(data.get("data", [])), MODELS_CACHE_TTL)
    return JSONResponse(content=data)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, _: str = Depends(verify_proxy_key)):
    """Proxy chat completions via the v3 AI SDK endpoint."""
    client = _get_client()

    body = await _parse_body(request)
    if not body:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = body.get("model") or DEFAULT_MODEL
    if not isinstance(model, str) or not model:
        return _invalid_request("model is required")

    # Client-side validation: catch bad tool histories before the gateway
    # rejects them with an opaque "Invalid input".
    err = validate_tool_history(body.get("messages", []))
    if err:
        log.warning("rejected request: %s", err)
        return _invalid_request(f"Invalid messages: {err}")

    _warn_dropped_params(body)

    stream = bool(body.get("stream", False))
    stream_options = body.get("stream_options")
    include_usage = bool(
        (stream_options or {}).get("include_usage", True)
        if isinstance(stream_options, dict) else True
    )

    v3_body = openai_to_v3(body)
    url = urljoin(GATEWAY_BASE_URL + "/", GATEWAY_V3_CHAT.lstrip("/"))
    headers = _v3_headers(model, streaming=True)

    if stream:
        resp = await _send_upstream(client, url, headers, v3_body)
        if resp.status_code != 200:
            err_resp = _client_error(resp)
            await resp.aclose()
            return err_resp
        return StreamingResponse(
            _chat_stream(resp, model, include_usage),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    resp = await _send_upstream(client, url, headers, v3_body)
    if resp.status_code != 200:
        err_resp = _client_error(resp)
        await resp.aclose()
        return err_resp
    try:
        result = await _collect_response(resp, model)
    finally:
        await resp.aclose()
    return JSONResponse(content=result)


@app.post("/v1/responses")
async def responses_route(request: Request, _: str = Depends(verify_proxy_key)):
    """Proxy the OpenAI Responses API (input items) via the v3 endpoint."""
    client = _get_client()

    body = await _parse_body(request)
    if not body:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = body.get("model") or DEFAULT_MODEL
    if not isinstance(model, str) or not model:
        return _invalid_request("model is required")

    messages = responses_input_to_messages(body.get("input"))
    err = validate_tool_history(messages)
    if err:
        log.warning("rejected /v1/responses request: %s", err)
        return _invalid_request(f"Invalid input: {err}")

    _warn_dropped_params(body)

    stream = bool(body.get("stream", False))
    chat_body = dict(body)
    chat_body["messages"] = messages
    chat_body.pop("input", None)
    chat_body.pop("stream", None)

    v3_body = openai_to_v3(chat_body)
    url = urljoin(GATEWAY_BASE_URL + "/", GATEWAY_V3_CHAT.lstrip("/"))
    headers = _v3_headers(model, streaming=True)

    if stream:
        resp = await _send_upstream(client, url, headers, v3_body)
        if resp.status_code != 200:
            err_resp = _client_error(resp)
            await resp.aclose()
            return err_resp
        return StreamingResponse(
            _responses_stream(resp, model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    resp = await _send_upstream(client, url, headers, v3_body)
    if resp.status_code != 200:
        err_resp = _client_error(resp)
        await resp.aclose()
        return err_resp
    try:
        result = await _collect_response(resp, model)
    finally:
        await resp.aclose()
    return JSONResponse(content=openai_to_responses(result, model))


@app.post("/v1/embeddings")
async def embeddings(request: Request, _: str = Depends(verify_proxy_key)):
    client = _get_client()
    body = await _parse_body(request)
    if not body:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = body.get("model") or "openai/text-embedding-3-large"
    body["model"] = model
    url = urljoin(GATEWAY_BASE_URL + "/", "v1/embeddings")
    headers = {
        "Authorization": f"Bearer {GATEWAY_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = await client.post(url, headers=headers, json=body)
    if resp.status_code != 200:
        return _client_error(resp)
    return JSONResponse(content=resp.json())


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def main() -> None:
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8787"))
    log.info("starting on %s:%d", host, port)
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=bool(os.getenv("RELOAD")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()

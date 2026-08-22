"""
fx-style AI Gateway proxy — v3 AI SDK protocol.

Exposes OpenAI-compatible endpoints (/v1/chat/completions, /v1/models,
/v1/responses, /v1/embeddings) and forwards to the Vercel AI Gateway using
the **same v3 AI SDK protocol** the native fx CLI uses:

    POST https://ai-gateway.vercel.sh/v3/ai/language-model

    Headers:
      Authorization: Bearer <gateway_key>
      User-Agent: fx/0.0.4
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
import asyncio
import base64
import secrets
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
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

from converter.request import openai_to_v3
from converter.validation import validate_tool_history
from converter.streaming import v3_stream_iter, v3_sse_stream_to_openai
from converter.responses import (
    responses_input_to_messages,
    openai_to_responses,
    openai_chunk_to_responses_sse,
    _ResponsesStreamState,
)

import identity
from keys import KeyPool, load_keys, mask
from usage import UsageTracker

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

GATEWAY_BASE_URL = os.getenv("GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh")
GATEWAY_KEYS = load_keys()
KEY_POOL = KeyPool(GATEWAY_KEYS)
USAGE = UsageTracker(
    enabled=os.getenv("USAGE_TRACKING", "1").lower() in ("1", "true", "yes", "on")
)
GATEWAY_TEAM = os.getenv("GATEWAY_TEAM", "")
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "zai/glm-5.2")
GATEWAY_SESSION_ID = os.getenv("GATEWAY_SESSION_ID", "")
GATEWAY_SESSION_AFFINITY = os.getenv("GATEWAY_SESSION_AFFINITY", "")

# Models that receive the body-level headers.user-agent (fx scopes it to
# zai/glm-5.2). "*" = all models, "" = none, else comma-separated list.
_raw_pua_models = os.getenv("PRODUCT_USER_AGENT_MODELS", "zai/glm-5.2")
if _raw_pua_models == "*":
    PRODUCT_USER_AGENT_MODELS: frozenset[str] | None = None
elif _raw_pua_models == "":
    PRODUCT_USER_AGENT_MODELS = frozenset()
else:
    PRODUCT_USER_AGENT_MODELS = frozenset(
        m.strip() for m in _raw_pua_models.split(",") if m.strip()
    )

GATEWAY_TIMEOUT_CONNECT = float(os.getenv("GATEWAY_TIMEOUT_CONNECT", "15"))
GATEWAY_TIMEOUT_READ = float(os.getenv("GATEWAY_TIMEOUT_READ", "300"))
GATEWAY_TIMEOUT_WRITE = float(os.getenv("GATEWAY_TIMEOUT_WRITE", "60"))
GATEWAY_TIMEOUT_POOL = float(os.getenv("GATEWAY_TIMEOUT_POOL", "60"))
GATEWAY_HTTP2 = os.getenv("GATEWAY_HTTP2", "1").lower() in ("1", "true", "yes")
MODELS_CACHE_TTL = float(os.getenv("MODELS_CACHE_TTL", "300"))

# fx parity: download remote image_url parts into data URLs pre-conversion
# (fx fetches/verifies attachments locally; the gateway cannot fetch URLs).
IMAGE_FETCH = os.getenv("IMAGE_FETCH", "1").lower() in ("1", "true", "yes", "on")
IMAGE_FETCH_TIMEOUT = float(os.getenv("IMAGE_FETCH_TIMEOUT", "10"))
IMAGE_FETCH_MAX_BYTES = int(os.getenv("IMAGE_FETCH_MAX_BYTES", str(5 * 1024 * 1024)))

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
    client_injected = getattr(app.state, "client", None) is not None
    if not client_injected:
        app.state.client = _build_client()
    if not hasattr(app.state, "models_cache"):
        app.state.models_cache = {"data": None, "expires": 0.0}
    if not len(KEY_POOL):
        log.warning("no AI_GATEWAY_API_KEY / AI_GATEWAY_API_KEY_N set — upstream will reject requests")
    else:
        log.info("key pool: %d key(s)  rotation=%s  failover=%s  cooldown=%.0fs",
                 len(KEY_POOL), KEY_POOL.rotation, KEY_POOL.failover,
                 KEY_POOL.cooldown_seconds)
        for _k in KEY_POOL.keys:
            log.info("  gateway key %s", mask(_k))
    if not PROXY_API_KEY:
        log.warning("PROXY_API_KEY not set — proxy is open to all callers")
    if not client_injected:
        app.state.identity_task = identity.start(app.state)
    yield
    task: asyncio.Task | None = getattr(app.state, "identity_task", None)
    if task is not None:
        task.cancel()
    await app.state.client.aclose()


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
    if not secrets.compare_digest(creds.credentials.encode(), PROXY_API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return creds.credentials


def _get_client() -> httpx.AsyncClient:
    client: httpx.AsyncClient | None = getattr(app.state, "client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="client not ready")
    return client


# --------------------------------------------------------------------------- #
def _v3_headers(
    model: str,
    streaming: bool,
    *,
    api_key: str,
    session_id: str | None = None,
    session_affinity: str | None = None,
) -> dict[str, str]:
    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": identity.state["user_agent"],
        "HTTP-Referer": "https://github.com/vercel-labs/fx",
        "X-Title": "fx",
        "ai-gateway-protocol-version": identity.state["protocol_version"],
        "ai-language-model-specification-version": identity.state["specification_version"],
        "ai-language-model-id": model,
        "ai-language-model-streaming": "true" if streaming else "false",
    }
    if streaming:
        headers["Accept"] = "text/event-stream"
    if GATEWAY_TEAM:
        headers["x-vercel-ai-gateway-team"] = GATEWAY_TEAM
    sid = session_id or GATEWAY_SESSION_ID
    affinity = session_affinity or GATEWAY_SESSION_AFFINITY
    if sid:
        headers["x-session-id"] = sid
    if affinity:
        headers["x-session-affinity"] = affinity
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


def _int_tokens(value) -> int:
    """Coerce an upstream usage token count to int (0 on anything odd)."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _iter_remote_image_parts(messages: list) -> list[tuple[dict, int]]:
    """Collect (message, index) pairs of remote http(s) image_url parts."""
    targets: list[tuple[dict, int]] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for i, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            inner = part.get("image_url")
            url = inner.get("url") if isinstance(inner, dict) else None
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                targets.append((msg, i))
    return targets


async def _hydrate_remote_images(client: httpx.AsyncClient, body: dict) -> None:
    """fx parity: download remote images into data URLs before conversion.

    Data URLs flow straight into the converter's `{type:"file"}` part shape;
    failures leave the original URL untouched so the request still goes out
    (matching pre-hydration behaviour instead of failing the request).
    """
    if not IMAGE_FETCH or not isinstance(body, dict):
        return
    for msg, idx in _iter_remote_image_parts(body.get("messages", [])):
        url = msg["content"][idx]["image_url"]["url"]
        try:
            img_resp = await client.get(url, timeout=IMAGE_FETCH_TIMEOUT, follow_redirects=True)
            img_resp.raise_for_status()
            data = img_resp.content
            if len(data) > IMAGE_FETCH_MAX_BYTES:
                raise ValueError(f"image exceeds {IMAGE_FETCH_MAX_BYTES} byte limit")
            media_type = (
                img_resp.headers.get("content-type", "").split(";")[0].strip()
                or "application/octet-stream"
            )
            encoded = base64.b64encode(data).decode("ascii")
            msg["content"][idx]["image_url"]["url"] = f"data:{media_type};base64,{encoded}"
            log.info("hydrated remote image (%d bytes)", len(data))
        except Exception as exc:
            log.warning(
                "remote image fetch failed (%s); passing URL through: %s",
                exc,
                url[:120],
            )


async def _tracked_stream(
    aiter: AsyncIterator[str], caller: str, model: str
) -> AsyncIterator[str]:
    """Pass-through SSE generator that extracts usage from the final chunk."""
    saw_usage = False
    async for sse in aiter:
        if '"usage"' in sse and not saw_usage:
            try:
                data = json.loads(sse[sse.index("{"):sse.rindex("}") + 1])
                u = data.get("usage") or {}
                USAGE.record(
                    caller,
                    model=model,
                    prompt_tokens=_int_tokens(u.get("prompt_tokens")),
                    completion_tokens=_int_tokens(u.get("completion_tokens")),
                )
                saw_usage = True
            except Exception:
                pass
        yield sse
    if not saw_usage:
        USAGE.record(caller, model=model)


async def _send_upstream(
    client: httpx.AsyncClient, url: str, headers: dict, v3_body: dict
) -> httpx.Response:
    req = client.build_request("POST", url, headers=headers, json=v3_body)
    resp = await client.send(req, stream=True)
    if resp.status_code != 200:
        await resp.aread()
    return resp


async def _upstream_pooled(
    build: Callable[[str], Awaitable[httpx.Response]],
) -> tuple[httpx.Response, str]:
    """Run `build(api_key)` once per pooled key until one attempt succeeds.

    Keys come from KEY_POOL (round-robin when KEY_ROTATION is on). A
    key-attributable upstream status (401/402/403/408/429/5xx when
    KEY_FAILOVER is on) or a network error transparently retries the next
    key; failed responses are closed before retrying. When every key is
    exhausted the last failing response is returned so callers render their
    normal error path. With no keys configured, one unauthenticated attempt
    is made (legacy behaviour).
    """
    tried: set[str] = set()
    last_resp: httpx.Response | None = None
    last_key = ""
    last_exc: Exception | None = None
    while True:
        key = KEY_POOL.next(exclude=tried)
        if key is None:
            if tried or len(KEY_POOL):
                break
            key = ""  # no keys configured: single legacy attempt
        tried.add(key)
        try:
            resp = await build(key)
        except httpx.RequestError as exc:
            last_exc, last_key = exc, key
            KEY_POOL.report_failure(key)
            log.warning("gateway key %s network error (%s); failing over",
                        mask(key), type(exc).__name__)
            continue
        if resp.status_code == 200 or not KEY_POOL.should_failover(resp.status_code):
            if resp.status_code == 200:
                KEY_POOL.report_success(key)
            return resp, key
        KEY_POOL.report_failure(key)
        log.warning("gateway key %s -> HTTP %d; failing over", mask(key), resp.status_code)
        if last_resp is not None:
            await last_resp.aclose()
        last_resp, last_key = resp, key
    if last_resp is not None:
        return last_resp, last_key
    assert last_exc is not None
    raise last_exc


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
    return {
        "status": "ok",
        "gateway": GATEWAY_BASE_URL,
        "protocol": "v3",
        "fx": dict(identity.state),
        "keys": KEY_POOL.stats(),
        "usage": USAGE.totals(),
    }


@app.get("/v1/usage")
async def usage_stats(_: str = Depends(verify_proxy_key)):
    """Per-caller request/token counters recorded since process start."""
    return USAGE.snapshot()


@app.get("/v1/models")
async def list_models(_: str = Depends(verify_proxy_key)):
    client = _get_client()
    cache: dict = getattr(app.state, "models_cache", {})
    now = time.monotonic()
    if cache.get("expires", 0) > now and cache.get("data") is not None:
        log.info("GET /v1/models -> cached (%d models)", len(cache["data"].get("data", [])))
        return JSONResponse(content=cache["data"])

    url = urljoin(GATEWAY_BASE_URL + "/", GATEWAY_V1_MODELS.lstrip("/"))

    async def send(key: str) -> httpx.Response:
        headers = {"Accept": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return await client.get(url, headers=headers)

    resp, used_key = await _upstream_pooled(send)
    log.debug("GET /v1/models via gateway key %s", mask(used_key))
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

    # fx parity: pull remote images into data URLs before translating.
    await _hydrate_remote_images(client, body)

    stream = bool(body.get("stream", False))
    stream_options = body.get("stream_options")
    include_usage = bool(
        (stream_options or {}).get("include_usage", True)
        if isinstance(stream_options, dict) else True
    )

    v3_body = openai_to_v3(
        body,
        product_user_agent=identity.state["user_agent"],
        product_user_agent_models=PRODUCT_USER_AGENT_MODELS,
    )
    url = urljoin(GATEWAY_BASE_URL + "/", GATEWAY_V3_CHAT.lstrip("/"))
    session_id = request.headers.get("x-session-id") or GATEWAY_SESSION_ID
    session_affinity = request.headers.get("x-session-affinity") or GATEWAY_SESSION_AFFINITY

    async def send(key: str) -> httpx.Response:
        headers = _v3_headers(
            model,
            streaming=True,
            api_key=key,
            session_id=session_id,
            session_affinity=session_affinity,
        )
        return await _send_upstream(client, url, headers, v3_body)

    resp, used_key = await _upstream_pooled(send)
    log.debug("POST /v1/chat/completions via gateway key %s", mask(used_key))

    caller = request.client.host if request.client else "unknown"
    if resp.status_code != 200:
        USAGE.record(caller, model=model, error=True)

    if stream:
        if resp.status_code != 200:
            err_resp = _client_error(resp)
            await resp.aclose()
            return err_resp
        return StreamingResponse(
            _tracked_stream(_chat_stream(resp, model, include_usage), caller, model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if resp.status_code != 200:
        err_resp = _client_error(resp)
        await resp.aclose()
        return err_resp
    try:
        result = await _collect_response(resp, model)
    finally:
        await resp.aclose()
    _u = result.get("usage") or {}
    USAGE.record(
        caller,
        model=model,
        prompt_tokens=_int_tokens(_u.get("prompt_tokens")),
        completion_tokens=_int_tokens(_u.get("completion_tokens")),
    )
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

    # fx parity: pull remote images into data URLs before translating.
    await _hydrate_remote_images(client, chat_body)

    v3_body = openai_to_v3(
        chat_body,
        product_user_agent=identity.state["user_agent"],
        product_user_agent_models=PRODUCT_USER_AGENT_MODELS,
    )
    url = urljoin(GATEWAY_BASE_URL + "/", GATEWAY_V3_CHAT.lstrip("/"))
    session_id = request.headers.get("x-session-id") or GATEWAY_SESSION_ID
    session_affinity = request.headers.get("x-session-affinity") or GATEWAY_SESSION_AFFINITY

    async def send(key: str) -> httpx.Response:
        headers = _v3_headers(
            model,
            streaming=True,
            api_key=key,
            session_id=session_id,
            session_affinity=session_affinity,
        )
        return await _send_upstream(client, url, headers, v3_body)

    resp, used_key = await _upstream_pooled(send)
    log.debug("POST /v1/responses via gateway key %s", mask(used_key))

    if stream:
        if resp.status_code != 200:
            err_resp = _client_error(resp)
            await resp.aclose()
            return err_resp
        return StreamingResponse(
            _responses_stream(resp, model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if resp.status_code != 200:
        err_resp = _client_error(resp)
        await resp.aclose()
        return err_resp
    try:
        result = await _collect_response(resp, model)
    finally:
        await resp.aclose()
    caller = request.client.host if request.client else "unknown"
    _u = result.get("usage") or {}
    USAGE.record(
        caller,
        model=model,
        prompt_tokens=_int_tokens(_u.get("prompt_tokens")),
        completion_tokens=_int_tokens(_u.get("completion_tokens")),
    )
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

    async def send(key: str) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        return await client.post(url, headers=headers, json=body)

    resp, used_key = await _upstream_pooled(send)
    log.debug("POST /v1/embeddings via gateway key %s", mask(used_key))
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


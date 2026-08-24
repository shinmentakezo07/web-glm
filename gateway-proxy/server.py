"""
fx-style AI Gateway proxy — v3 AI SDK protocol.

Exposes OpenAI-compatible endpoints (/v1/chat/completions, /v1/models,
/v1/responses, /v1/embeddings) and forwards to the Vercel AI Gateway using
the **same v3 AI SDK protocol** the native fx CLI uses:

    POST https://ai-gateway.vercel.sh/v3/ai/language-model

    Headers:
      Authorization: Bearer <gateway_key>
      User-Agent: fx/<latest>           (auto-synced from GitHub at startup)
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
from logging.handlers import RotatingFileHandler
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
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from converter.request import openai_to_v3
from converter.validation import validate_tool_history
from converter.streaming import v3_stream_iter, v3_sse_stream_to_openai
from converter.responses import (
    responses_input_to_messages,
    openai_to_responses,
    openai_chunk_to_responses_sse,
    _ResponsesStreamState,
)
from converter.anthropic import (
    anthropic_to_openai,
    openai_to_anthropic,
    anthropic_stream_iter,
    count_anthropic_tokens,
)

import healer
import identity
from keys import KeyPool, load_keys, mask
from db import UsageStore

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

GATEWAY_BASE_URL = os.getenv("GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh")
GATEWAY_KEYS = load_keys()
KEY_POOL = KeyPool(GATEWAY_KEYS)
USAGE = UsageStore(
    enabled=os.getenv("USAGE_TRACKING", "1").lower() in ("1", "true", "yes", "on")
)
GATEWAY_TEAM = os.getenv("GATEWAY_TEAM", "")
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "zai/glm-5.2")
GATEWAY_SESSION_ID = os.getenv("GATEWAY_SESSION_ID", "")
GATEWAY_SESSION_AFFINITY = os.getenv("GATEWAY_SESSION_AFFINITY", "")

# --- fx.sh free web-gateway provider -------------------------------------- #
# A second upstream backend: the free fx.sh web gateway. It speaks the SAME
# v3 AI SDK protocol as the Vercel AI Gateway. The in-browser fx WASM terminal
# sets ``AI_GATEWAY_API_KEY = "fx-demo-proxy"``; the page's fetch adapter then
# rewrites the URL from ``ai-gateway.vercel.sh/<path>`` to
# ``fx.sh/fx-wasm/gateway/<path>`` and forwards the Bearer token to the fx.sh
# server, which proxies to the real gateway with that demo key.
#
#   POST https://fx.sh/fx-wasm/gateway/v3/ai/language-model
#   Authorization: Bearer fx-demo-proxy
#
# FXWEB_FALLBACK=1 (default) makes the proxy try the fx.sh free endpoint
# automatically when the Vercel gateway has no keys configured OR every key
# fails over. So API-key callers keep working, and free-tier users with no
# key also get served. Set FXWEB_FALLBACK=0 to disable the free fallback.
FXWEB_FALLBACK = os.getenv("FXWEB_FALLBACK", "1").strip().lower() in (
    "1", "true", "yes", "on",
)
FXWEB_BASE_URL = os.getenv("FXWEB_BASE_URL", "https://fx.sh")
FXWEB_V3_CHAT = os.getenv(
    "FXWEB_V3_CHAT", "/fx-wasm/gateway/v3/ai/language-model"
)
# The demo API key the fx.sh web terminal uses (found in the page's JS:
# ``let l = "fx-demo-proxy"``). This is NOT a secret — it is the public
# free-tier key baked into the fx.sh /try page.
FXWEB_API_KEY = os.getenv("FXWEB_API_KEY", "fx-demo-proxy")
# A plausible desktop Chrome User-Agent. The fx.sh gateway is browser-served,
# so the HTTP-level User-Agent should look like a real browser rather than
# fx/<version> (which lives in the body-level headers.user-agent instead).
_FXWEB_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
)

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
MAX_REQUEST_BODY_BYTES = int(os.getenv("MAX_REQUEST_BODY_BYTES", str(10 * 1024 * 1024)))

GATEWAY_V3_CHAT = "/v3/ai/language-model"
GATEWAY_V1_MODELS = "/v1/models"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("gateway-proxy")

# Also write logs to a file so the dashboard can read them back.
LOG_FILE = os.getenv("LOG_FILE", os.path.join(os.getcwd(), "proxy.log"))
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MiB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "3"))
try:
    _file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
    )
    _file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(_file_handler)
except OSError:
    pass

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
        if FXWEB_FALLBACK:
            log.info("no AI_GATEWAY_API_KEY set — fx.sh free web-gateway fallback enabled")
        else:
            log.warning("no AI_GATEWAY_API_KEY / AI_GATEWAY_API_KEY_N set — upstream will reject requests")
    else:
        log.info("key pool: %d key(s)  failover=%s  cooldown=%.0fs",
                 len(KEY_POOL), KEY_POOL.failover,
                 KEY_POOL.cooldown_seconds)
        for _k in KEY_POOL.keys:
            log.info("  gateway key %s", mask(_k))
    if not PROXY_API_KEY:
        log.warning("PROXY_API_KEY not set — proxy is open to all callers")
    if FXWEB_FALLBACK:
        log.info("fx.sh free web-gateway fallback enabled (POST %s%s)",
                 FXWEB_BASE_URL, FXWEB_V3_CHAT)
    if not client_injected:
        # Fetch the live fx identity (GitHub + local binary) before serving
        # any request, then start the periodic background refresher.
        await identity.initialize(app.state)
        app.state.identity_task = identity.start(app.state)
        # Key healer: probe cooling keys in the background and restore
        # healthy ones before their full cooldown elapses.
        app.state.healer_task = healer.start(
            app.state,
            pool=KEY_POOL,
            models_url=urljoin(GATEWAY_BASE_URL + "/", GATEWAY_V1_MODELS.lstrip("/")),
        )
    yield
    for _name in ("identity_task", "healer_task"):
        _task: asyncio.Task | None = getattr(app.state, _name, None)
        if _task is not None:
            _task.cancel()
    await app.state.client.aclose()


app = FastAPI(title="fx-style AI Gateway proxy", lifespan=lifespan)

# Serve the dashboard UI (pre-built Vite React app) from /dashboard/static/.
_DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "dashboard", "dist")
if os.path.isdir(_DASHBOARD_DIR):
    app.mount("/dashboard/static", StaticFiles(directory=_DASHBOARD_DIR), name="dashboard-static")


@app.get("/dashboard")
async def dashboard_page():
    """Serve the dashboard SPA (or a fallback if not built yet)."""
    index = os.path.join(_DASHBOARD_DIR, "index.html")
    if os.path.isfile(index):
        return FileResponse(index)
    return JSONResponse(
        status_code=404,
        content={"error": "Dashboard not built. Run: cd dashboard && npm run build"},
    )


# --------------------------------------------------------------------------- #
# Error types
# --------------------------------------------------------------------------- #


class AnthropicError(Exception):
    """Exception that renders as the Anthropic error JSON shape.

    Raised by Anthropic-route auth/validation so error bodies match what
    Claude Code and the Anthropic SDKs expect:
    ``{"type": "error", "error": {"type": ..., "message": ...}}``
    """

    def __init__(self, message: str, error_type: str = "api_error", status: int = 400):
        self.message = message
        self.error_type = error_type
        self.status = status
        super().__init__(message)


class _BodyTooLarge(Exception):
    """Raised when a request body exceeds MAX_REQUEST_BODY_BYTES."""


@app.exception_handler(AnthropicError)
async def _anthropic_error_handler(request: Request, exc: AnthropicError):
    return JSONResponse(
        status_code=exc.status,
        content={
            "type": "error",
            "error": {"type": exc.error_type, "message": exc.message},
        },
    )


# --------------------------------------------------------------------------- #
# Request logging middleware
# --------------------------------------------------------------------------- #


@app.exception_handler(_BodyTooLarge)
async def _body_too_large_handler(request: Request, exc: _BodyTooLarge):
    return JSONResponse(
        status_code=413,
        content={
            "error": {
                "message": "Request body too large",
                "type": "invalid_request_error",
            }
        },
    )


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
# fx.sh web-gateway session helpers
# --------------------------------------------------------------------------- #


def _generate_fxweb_session() -> tuple[str, str]:
    """Generate a random fx.sh-style session id and affinity pair.

    The in-browser fx terminal derives both from the current millisecond
    timestamp plus a 16-hex-char nonce: ``<ms>-<ms*1000000>-<hex16>``. The
    exact value is not validated server-side, so we use ``secrets`` for the
    nonce (stronger than the WASM RNG) and keep the same shape so traffic
    looks like the real web client.
    """
    now_ms = int(time.time() * 1000)
    nonce = secrets.token_hex(8)  # 16 hex chars
    sid = f"{now_ms}-{now_ms * 1000000}-{nonce}"
    return sid, sid


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


def verify_anthropic_key(request: Request) -> str:
    """Auth for Anthropic-style requests: x-api-key header or Bearer token.

    Claude Code and the Anthropic SDKs send ``x-api-key: <key>``. We also
    accept ``Authorization: Bearer <key>`` so non-Anthropic clients work too.
    When no PROXY_API_KEY is set, the proxy is open (same as OpenAI routes).

    Raises ``AnthropicError`` (not ``HTTPException``) so the response body
    matches the Anthropic error shape ``{"type": "error", "error": {...}}``.
    """
    if not PROXY_API_KEY:
        return "anonymous"
    api_key = request.headers.get("x-api-key", "")
    if not api_key:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            api_key = auth[7:]
    if not api_key:
        raise AnthropicError("Missing API key", "authentication_error", 401)
    if not secrets.compare_digest(api_key.encode(), PROXY_API_KEY.encode()):
        raise AnthropicError("Invalid API key", "authentication_error", 401)
    return api_key


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
        "User-Agent": identity.state["user_agent"] or identity._FALLBACK_USER_AGENT,
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


def _fxweb_headers(
    model: str,
    streaming: bool = True,
    *,
    session_id: str | None = None,
    session_affinity: str | None = None,
) -> dict[str, str]:
    """Build headers for the fx.sh free web-gateway fallback provider.

    Mirrors what the in-browser fx WASM terminal sends:
      - ``Authorization: Bearer fx-demo-proxy`` — the public demo key baked
        into the fx.sh /try page JS (NOT a secret; it is the free-tier key
        the page sets as ``AI_GATEWAY_API_KEY`` and the fetch adapter forwards
        to the fx.sh server-side proxy).
      - desktop-browser User-Agent (the fx version lives in the body-level
        ``headers.user-agent`` instead, matching the WASM terminal)
      - ``Origin``/``Referer`` pointing at fx.sh
      - the same v3 protocol headers as the Vercel gateway
      - per-request session pinning (``x-session-id``/``x-session-affinity``),
        auto-generated in the fx.sh wire shape when not supplied
    """
    headers: dict[str, str] = {
        "Authorization": f"Bearer {FXWEB_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": _FXWEB_BROWSER_UA,
        "Accept": "text/event-stream" if streaming else "*/*",
        "Accept-Language": "en-US,en;q=0.6",
        "Origin": "https://fx.sh",
        "Referer": "https://fx.sh/",
        # Vercel's fx.sh edge checks sec-fetch-* to confirm same-origin
        # browser requests. Without these the gateway returns 403.
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "HTTP-Referer": "https://github.com/vercel-labs/fx",
        "X-Title": "fx",
        "ai-gateway-protocol-version": identity.state["protocol_version"],
        "ai-language-model-specification-version": identity.state["specification_version"],
        "ai-language-model-id": model,
        # fx.sh always streams on the wire (same invariant as the Vercel
        # backend); non-streaming client requests still open a streaming
        # upstream connection and collect deltas internally.
        "ai-language-model-streaming": "true",
    }
    # Session pinning: prefer caller-provided, then env, then generate fresh.
    sid = session_id or GATEWAY_SESSION_ID
    affinity = session_affinity or GATEWAY_SESSION_AFFINITY
    if not sid:
        sid, affinity = _generate_fxweb_session()
    headers["x-session-id"] = sid
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
    cl = request.headers.get("content-length")
    if cl and int(cl) > MAX_REQUEST_BODY_BYTES:
        raise _BodyTooLarge()
    try:
        raw = await request.body()
    except Exception:
        return {}
    if len(raw) > MAX_REQUEST_BODY_BYTES:
        raise _BodyTooLarge()
    try:
        body = json.loads(raw)
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


def _int_tokens(value) -> int:
    """Coerce an upstream usage token count to int (0 on anything odd)."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _extract_usage(u: dict) -> dict:
    """Extract all token fields from an OpenAI-shaped usage dict."""
    pt = u.get("prompt_tokens_details") or {}
    ct = u.get("output_tokens_details") or {}
    return {
        "prompt_tokens": _int_tokens(u.get("prompt_tokens")),
        "completion_tokens": _int_tokens(u.get("completion_tokens")),
        "cached_tokens": _int_tokens(pt.get("cached_tokens")),
        "reasoning_tokens": _int_tokens(ct.get("reasoning_tokens")),
    }


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


def _merge_usage(dst: dict, src: dict) -> dict:
    """Merge non-zero token counts from src into dst (max wins)."""
    for k in ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens"):
        v = int(src.get(k) or 0)
        if v > 0:
            dst[k] = max(dst.get(k, 0), v)
    return dst


async def _tracked_stream(
    aiter: AsyncIterator[str], caller: str, model: str, endpoint: str = ""
) -> AsyncIterator[str]:
    """Pass-through SSE generator that extracts usage from usage-bearing chunks.

    Understands all three response dialects the proxy emits:
      * OpenAI chat:      top-level ``usage`` on the final chunk
      * Responses API:    ``response.usage`` inside ``response.completed``
      * Anthropic Messages: ``message.usage`` (input) + Anthropic-shaped
        top-level ``usage`` with ``input_tokens``/``output_tokens``

    Token counts are accumulated across chunks, so dialects that split
    input/output usage across events are recorded exactly once and fully.
    A chunk merely *mentioning* "usage" (e.g. echoed model text) no longer
    suppresses real usage capture.
    """
    start = time.monotonic()
    acc: dict = {}
    recorded = False
    async for sse in aiter:
        if '"usage"' in sse and not recorded:
            try:
                payload = sse.strip()
                if payload.startswith("data: "):
                    payload = payload[6:]
                elif payload.startswith("data:"):
                    payload = payload[5:]
                data = json.loads(payload)
                raw = data.get("usage")
                if not isinstance(raw, dict):
                    rsp = data.get("response")
                    raw = rsp.get("usage") if isinstance(rsp, dict) else None
                if not isinstance(raw, dict):
                    msg = data.get("message")
                    raw = msg.get("usage") if isinstance(msg, dict) else None
                cand = _extract_usage(raw) if isinstance(raw, dict) else {}
                if isinstance(raw, dict) and not cand.get("completion_tokens"):
                    # Anthropic dialect: input_tokens / output_tokens naming.
                    cand["prompt_tokens"] = (
                        cand.get("prompt_tokens") or _int_tokens(raw.get("input_tokens"))
                    )
                    cand["completion_tokens"] = (
                        cand.get("completion_tokens") or _int_tokens(raw.get("output_tokens"))
                    )
                merged = _merge_usage(dict(acc), cand)
                if merged != acc:
                    acc = merged
                    # Record as soon as completion tokens are known; the
                    # fallback below covers streams that never carry them.
                    if acc.get("completion_tokens"):
                        USAGE.record(caller, model=model, endpoint=endpoint,
                                     duration_ms=(time.monotonic() - start) * 1000, **acc)
                        recorded = True
            except Exception:
                pass
        yield sse
    if not recorded:
        USAGE.record(
            caller, model=model, endpoint=endpoint,
            duration_ms=(time.monotonic() - start) * 1000,
            prompt_tokens=acc.get("prompt_tokens", 0),
            completion_tokens=acc.get("completion_tokens", 0),
            cached_tokens=acc.get("cached_tokens", 0),
            reasoning_tokens=acc.get("reasoning_tokens", 0),
        )



async def _send_upstream(
    client: httpx.AsyncClient, url: str, headers: dict, v3_body: dict
) -> httpx.Response:
    req = client.build_request("POST", url, headers=headers, json=v3_body)
    resp = await client.send(req, stream=True)
    if resp.status_code != 200:
        await resp.aread()
    return resp


async def _fxweb_send(
    client: httpx.AsyncClient, v3_body: dict, model: str,
    session_id: str | None = None, session_affinity: str | None = None,
) -> httpx.Response:
    """Send a v3 request to the free fx.sh web-gateway fallback.

    No API key, no Authorization header — the same endpoint the in-browser
    fx WASM terminal uses. Returns the streaming httpx.Response.
    """
    url = urljoin(FXWEB_BASE_URL + "/", FXWEB_V3_CHAT.lstrip("/"))
    headers = _fxweb_headers(
        model, streaming=True, session_id=session_id, session_affinity=session_affinity,
    )
    return await _send_upstream(client, url, headers, v3_body)


async def _upstream_pooled(
    build: Callable[[str], Awaitable[httpx.Response]],
    *,
    fxweb_build: Callable[[], Awaitable[httpx.Response]] | None = None,
) -> tuple[httpx.Response, str]:
    """Run `build(api_key)` once per pooled key until one attempt succeeds.

    The first key in priority order is preferred and reused until it fails;
    a key-attributable upstream status (401/402/403/408/429/5xx when
    KEY_FAILOVER is on) or a network error transparently retries the next
    key; failed responses are closed before retrying. When every key is
    exhausted the last failing response is returned so callers render their
    normal error path. With no keys configured, one unauthenticated attempt
    is made (legacy behaviour).

    If ``fxweb_build`` is provided (and FXWEB_FALLBACK is on), it is tried as
    a last resort after every pooled Vercel key fails — or immediately when no
    keys are configured. This lets free-tier users without a gateway API key
    reach the model via the fx.sh free web-gateway, while API-key callers keep
    using the paid Vercel gateway as the primary path.
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
            # Close any previously held response before continuing so we
            # don't leak a connection from a prior failed attempt.
            if last_resp is not None:
                await last_resp.aclose()
                last_resp = None
            continue
        if resp.status_code == 200 or not KEY_POOL.should_failover(resp.status_code):
            if resp.status_code == 200:
                KEY_POOL.report_success(key)
            # Close any prior failed response we were holding.
            if last_resp is not None:
                await last_resp.aclose()
            return resp, key
        KEY_POOL.report_failure(key)
        log.warning("gateway key %s -> HTTP %d; failing over", mask(key), resp.status_code)
        if last_resp is not None:
            await last_resp.aclose()
        last_resp, last_key = resp, key
    # All Vercel keys exhausted (or none configured). Try the fx.sh free
    # web-gateway fallback before giving up, so free-tier callers without a
    # gateway key still reach the model.
    if fxweb_build is not None and FXWEB_FALLBACK:
        # Close the last failed Vercel response before falling through.
        if last_resp is not None:
            await last_resp.aclose()
            last_resp = None
        try:
            resp = await fxweb_build()
        except httpx.RequestError as exc:
            log.warning("fx.sh fallback network error (%s)", type(exc).__name__)
            if last_exc is not None:
                raise last_exc
            raise
        if resp.status_code == 200:
            log.info("served via fx.sh free web-gateway fallback")
            return resp, "fxweb"
        # fx.sh also failed — return its error response.
        log.warning("fx.sh fallback -> HTTP %d", resp.status_code)
        return resp, "fxweb"
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


async def _anthropic_stream(resp: httpx.Response, model: str) -> AsyncIterator[str]:
    """Consume the upstream v3 stream and yield Anthropic Messages SSE events.

    The upstream v3 stream is first translated to OpenAI chunks (reusing
    ``_chat_stream``), then those chunks are translated to Anthropic SSE
    events by ``anthropic_stream_iter``.
    """
    try:
        async for chunk in anthropic_stream_iter(
            _chat_stream(resp, model, include_usage=True), model
        ):
            yield chunk
    finally:
        pass  # _chat_stream closes resp in its own finally


async def _send_to_v3(
    client: httpx.AsyncClient,
    request: Request,
    chat_body: dict,
    model: str,
) -> tuple[httpx.Response, str]:
    """Shared upstream-send: hydrate images, build v3 body, pooled send.

    Returns (upstream_response, used_key). Caller is responsible for closing
    the response when stream=True (via the stream generator) or after
    collecting.
    """
    await _hydrate_remote_images(client, chat_body)
    v3_body = openai_to_v3(
        chat_body,
        product_user_agent=identity.state["user_agent"] or identity._FALLBACK_USER_AGENT,
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

    async def fxweb_send() -> httpx.Response:
        return await _fxweb_send(client, v3_body, model, session_id, session_affinity)

    return await _upstream_pooled(send, fxweb_build=fxweb_send)


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
        "fxweb_fallback": {
            "enabled": FXWEB_FALLBACK,
            "base_url": FXWEB_BASE_URL,
            "endpoint": FXWEB_V3_CHAT,
        },
    }


@app.get("/v1/usage")
async def usage_stats(_: str = Depends(verify_proxy_key)):
    """Per-caller request/token counters recorded since process start."""
    return USAGE.snapshot()


@app.get("/v1/dashboard")
async def dashboard_stats():
    """Dashboard payload: totals, time series, per-model/caller breakdowns.

    No auth — internal monitoring tool for the operator, not exposed to
    proxy callers.
    """
    return USAGE.dashboard()


@app.get("/v1/logs")
async def logs_api(limit: int = 200):
    """Recent proxy log lines for the dashboard. No auth."""
    limit = max(1, min(limit, 1000))
    lines: list[str] = []
    try:
        with open(LOG_FILE, "r", errors="replace") as f:
            # Read the last `limit` lines efficiently using a rolling deque.
            from collections import deque
            buf: deque[str] = deque(maxlen=limit)
            for line in f:
                buf.append(line)
            lines = list(buf)
    except (OSError, FileNotFoundError):
        pass
    return {"lines": lines, "file": LOG_FILE}


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

    async def fxweb_send() -> httpx.Response:
        fxweb_url = urljoin(FXWEB_BASE_URL + "/", "fx-wasm/gateway/v1/models")
        return await client.get(
            fxweb_url,
            headers={"Accept": "application/json", "Authorization": f"Bearer {FXWEB_API_KEY}"},
        )

    resp, used_key = await _upstream_pooled(send, fxweb_build=fxweb_send)
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
        return _invalid_request("Invalid JSON body")

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
        product_user_agent=identity.state["user_agent"] or identity._FALLBACK_USER_AGENT,
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

    async def fxweb_send() -> httpx.Response:
        return await _fxweb_send(client, v3_body, model, session_id, session_affinity)

    resp, used_key = await _upstream_pooled(send, fxweb_build=fxweb_send)
    log.debug("POST /v1/chat/completions via gateway key %s", mask(used_key))

    caller = request.client.host if request.client else "unknown"
    _req_start = time.monotonic()
    if resp.status_code != 200:
        USAGE.record(caller, model=model, endpoint="/v1/chat/completions", error=True,
                     duration_ms=(time.monotonic() - _req_start) * 1000)

    if stream:
        if resp.status_code != 200:
            err_resp = _client_error(resp)
            await resp.aclose()
            return err_resp
        return StreamingResponse(
            _tracked_stream(_chat_stream(resp, model, include_usage), caller, model, "/v1/chat/completions"),
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
    _u = _extract_usage(result.get("usage") or {})
    USAGE.record(caller, model=model, endpoint="/v1/chat/completions",
                 duration_ms=(time.monotonic() - _req_start) * 1000, **_u)
    return JSONResponse(content=result)


@app.post("/v1/responses")
async def responses_route(request: Request, _: str = Depends(verify_proxy_key)):
    """Proxy the OpenAI Responses API (input items) via the v3 endpoint."""
    client = _get_client()

    body = await _parse_body(request)
    if not body:
        return _invalid_request("Invalid JSON body")

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
        product_user_agent=identity.state["user_agent"] or identity._FALLBACK_USER_AGENT,
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

    async def fxweb_send() -> httpx.Response:
        return await _fxweb_send(client, v3_body, model, session_id, session_affinity)

    resp, used_key = await _upstream_pooled(send, fxweb_build=fxweb_send)
    log.debug("POST /v1/responses via gateway key %s", mask(used_key))

    caller = request.client.host if request.client else "unknown"
    _req_start = time.monotonic()
    if resp.status_code != 200:
        USAGE.record(caller, model=model, endpoint="/v1/responses", error=True,
                     duration_ms=(time.monotonic() - _req_start) * 1000)

    if stream:
        if resp.status_code != 200:
            err_resp = _client_error(resp)
            await resp.aclose()
            return err_resp
        return StreamingResponse(
            _tracked_stream(_responses_stream(resp, model), caller, model, "/v1/responses"),
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
    _u = _extract_usage(result.get("usage") or {})
    USAGE.record(caller, model=model, endpoint="/v1/responses",
                 duration_ms=(time.monotonic() - _req_start) * 1000, **_u)
    return JSONResponse(content=openai_to_responses(result, model))


@app.post("/v1/embeddings")
async def embeddings(request: Request, _: str = Depends(verify_proxy_key)):
    client = _get_client()
    body = await _parse_body(request)
    if not body:
        return _invalid_request("Invalid JSON body")

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
    caller = request.client.host if request.client else "unknown"
    _req_start = time.monotonic()
    if resp.status_code != 200:
        USAGE.record(caller, model=model, endpoint="/v1/embeddings", error=True,
                     duration_ms=(time.monotonic() - _req_start) * 1000)
        return _client_error(resp)
    data = resp.json()
    _u = _extract_usage(data.get("usage") or {})
    USAGE.record(caller, model=model, endpoint="/v1/embeddings",
                 duration_ms=(time.monotonic() - _req_start) * 1000, **_u)
    return JSONResponse(content=data)


# --------------------------------------------------------------------------- #
# Anthropic Messages API (/v1/messages)
# --------------------------------------------------------------------------- #


def _anthropic_error(resp: httpx.Response) -> JSONResponse:
    """Normalize an upstream error into the Anthropic error shape."""
    try:
        body = resp.json()
    except Exception:
        body = None
    message = None
    if isinstance(body, dict) and "error" in body:
        err = body["error"]
        message = err.get("message") if isinstance(err, dict) else str(err)
    elif isinstance(body, dict):
        message = body.get("message") or json.dumps(body)[:2000]
    if not message:
        message = (resp.text or f"upstream error (HTTP {resp.status_code})")[:2000]
    status = resp.status_code if resp.status_code >= 400 else 500
    err_type = "invalid_request_error" if status == 400 else "api_error"
    return JSONResponse(
        status_code=status,
        content={
            "type": "error",
            "error": {"type": err_type, "message": message},
        },
    )


def _anthropic_invalid_request(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "type": "error",
            "error": {"type": "invalid_request_error", "message": message},
        },
    )


@app.post("/v1/messages")
async def anthropic_messages(request: Request, _: str = Depends(verify_anthropic_key)):
    """Proxy the Anthropic Messages API via the v3 AI SDK endpoint.

    Translates Anthropic Messages requests -> OpenAI -> v3, forwards to the
    gateway, and translates the response back to the Anthropic Message shape.
    Supports streaming (SSE) and non-streaming, tools, images, and system
    prompts.
    """
    client = _get_client()

    body = await _parse_body(request)
    if not body:
        return _anthropic_invalid_request("Invalid JSON body")

    model = body.get("model") or DEFAULT_MODEL
    if not isinstance(model, str) or not model:
        return _anthropic_invalid_request("model is required")

    if "max_tokens" not in body:
        return _anthropic_invalid_request("max_tokens is required")

    # Convert Anthropic request -> OpenAI chat-completions body.
    oai_body = anthropic_to_openai(body)

    # Validate tool history (now in OpenAI shape).
    err = validate_tool_history(oai_body.get("messages", []))
    if err:
        log.warning("rejected /v1/messages request: %s", err)
        return _anthropic_invalid_request(f"Invalid messages: {err}")

    _warn_dropped_params(body)

    stream = bool(body.get("stream", False))

    resp, used_key = await _send_to_v3(client, request, oai_body, model)
    log.debug("POST /v1/messages via gateway key %s", mask(used_key))

    caller = request.client.host if request.client else "unknown"
    _req_start = time.monotonic()
    if resp.status_code != 200:
        USAGE.record(caller, model=model, endpoint="/v1/messages", error=True,
                     duration_ms=(time.monotonic() - _req_start) * 1000)
        err_resp = _anthropic_error(resp)
        await resp.aclose()
        return err_resp

    if stream:
        return StreamingResponse(
            _tracked_stream(_anthropic_stream(resp, model), caller, model, "/v1/messages"),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await _collect_response(resp, model)
    finally:
        await resp.aclose()
    _u = _extract_usage(result.get("usage") or {})
    USAGE.record(caller, model=model, endpoint="/v1/messages",
                 duration_ms=(time.monotonic() - _req_start) * 1000, **_u)
    return JSONResponse(content=openai_to_anthropic(result, model))


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(
    request: Request, _: str = Depends(verify_anthropic_key)
):
    """Estimate token count for an Anthropic Messages API request.

    Uses a character-based heuristic (no tokenizer dependency). Returns the
    same ``{input_tokens}`` shape the Anthropic API does.
    """
    body = await _parse_body(request)
    if not body:
        return _anthropic_invalid_request("Invalid JSON body")
    token_count = count_anthropic_tokens(body)
    return JSONResponse(content={"input_tokens": token_count})


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


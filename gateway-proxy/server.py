"""
fx-style AI Gateway proxy — v3 AI SDK protocol.

Exposes OpenAI-compatible endpoints (/v1/chat/completions, /v1/models)
and forwards to the Vercel AI Gateway using the **same v3 AI SDK protocol**
the native fx CLI uses:

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
      {prompt, tools, toolChoice, headers}

The proxy translates incoming OpenAI-format requests to AI SDK v3 format
via `converter.openai_to_v3` and converts the v3 SSE stream back to
OpenAI-format chunks via `converter.v3_sse_stream_to_openai` (non-stream)
and the streaming helper below (stream).

Authenticates incoming requests with a proxy API key (PROXY_API_KEY env var).
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
    _sse_chunk,
    _v3_finish_reason,
    _v3_usage_to_openai,
    v3_sse_stream_to_openai,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

GATEWAY_BASE_URL = os.getenv("GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh")
GATEWAY_API_KEY = os.getenv("AI_GATEWAY_API_KEY", "")
GATEWAY_TEAM = os.getenv("GATEWAY_TEAM", "")
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "zai/glm-5.2")

GATEWAY_V3_CHAT = "/v3/ai/language-model"
GATEWAY_V1_MODELS = "/v1/models"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("gateway-proxy")

bearer = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=15.0, read=300.0, write=60.0, pool=60.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        http2=True,
    )
    app.state.client = client
    log.info("proxy ready  gateway=%s  default_model=%s", GATEWAY_BASE_URL, DEFAULT_MODEL)
    if not GATEWAY_API_KEY:
        log.warning("AI_GATEWAY_API_KEY not set — upstream will reject requests")
    if not PROXY_API_KEY:
        log.warning("PROXY_API_KEY not set — proxy is open to all callers")
    yield
    await client.aclose()


app = FastAPI(title="fx-style AI Gateway proxy", lifespan=lifespan)


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


def _client_error(resp: httpx.Response) -> JSONResponse:
    try:
        body = resp.json()
    except Exception:
        body = {"error": {"message": resp.text, "type": "upstream_error"}}
    return JSONResponse(status_code=resp.status_code, content=body)


def _usage_or_none(usage_data: dict) -> dict | None:
    if not usage_data:
        return None
    return _v3_usage_to_openai(usage_data)


# --------------------------------------------------------------------------- #
# Streaming response generator
# --------------------------------------------------------------------------- #


async def _stream_response(resp: httpx.Response, model: str) -> AsyncIterator[str]:
    """Consume the upstream v3 SSE stream line-by-line and yield OpenAI SSE chunks."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    yield _sse_chunk(chat_id, model, role="assistant")

    tool_input_buffers: dict[str, str] = {}
    tool_call_index: dict[str, int] = {}
    next_tool_index = 0

    async for raw in resp.aiter_lines():
        data = _parse_sse_line(raw)
        if data is None:
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")

        if etype == "text-delta":
            delta = event.get("delta", "")
            if delta:
                yield _sse_chunk(chat_id, model, delta_text=delta)

        elif etype == "tool-input-delta":
            tool_id = event.get("toolCallId", "") or event.get("id", "")
            if tool_id:
                tool_input_buffers[tool_id] = tool_input_buffers.get(tool_id, "") + event.get("delta", "")

        elif etype == "tool-call":
            tool_id = event.get("toolCallId", "") or event.get("id", "")
            tool_name = event.get("toolName", "")
            tool_args = event.get("input", "") or tool_input_buffers.get(tool_id, "")
            if not isinstance(tool_args, str):
                try:
                    tool_args = json.dumps(tool_args)
                except (TypeError, ValueError):
                    tool_args = ""
            tool_args = tool_args or tool_input_buffers.get(tool_id, "")

            idx = tool_call_index.get(tool_id)
            if idx is None:
                idx = next_tool_index
                tool_call_index[tool_id] = idx
                next_tool_index += 1

            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": idx,
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_args},
                        }]
                    },
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            tool_input_buffers.pop(tool_id, None)

        elif etype == "finish":
            yield _sse_chunk(
                chat_id, model,
                finish_reason=_v3_finish_reason(event.get("finishReason", "stop")),
                usage=_usage_or_none(event.get("usage", {})),
            )
            break

        elif etype == "error":
            yield _sse_chunk(chat_id, model, finish_reason="stop")
            break

    yield "data: [DONE]\n\n"


# --------------------------------------------------------------------------- #
# Non-streaming response collector
# --------------------------------------------------------------------------- #


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
    url = urljoin(GATEWAY_BASE_URL + "/", GATEWAY_V1_MODELS.lstrip("/"))
    headers = {"Accept": "application/json"}
    if GATEWAY_API_KEY:
        headers["Authorization"] = f"Bearer {GATEWAY_API_KEY}"
    resp = await client.get(url, headers=headers)
    if resp.status_code != 200:
        return _client_error(resp)
    return JSONResponse(content=resp.json())


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, _: str = Depends(verify_proxy_key)):
    """Proxy chat completions via the v3 AI SDK endpoint."""
    client = _get_client()

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = body.get("model") or DEFAULT_MODEL
    stream = bool(body.get("stream", False))

    v3_body = openai_to_v3(body)
    url = urljoin(GATEWAY_BASE_URL + "/", GATEWAY_V3_CHAT.lstrip("/"))
    headers = _v3_headers(model, streaming=True)

    if stream:
        req = client.build_request("POST", url, headers=headers, json=v3_body)
        resp = await client.send(req, stream=True)
        if resp.status_code != 200:
            await resp.aread()
            err = _client_error(resp)
            await resp.aclose()
            return err
        return StreamingResponse(
            _stream_response(resp, model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        req = client.build_request("POST", url, headers=headers, json=v3_body)
        resp = await client.send(req, stream=True)
        if resp.status_code != 200:
            await resp.aread()
            err = _client_error(resp)
            await resp.aclose()
            return err

        try:
            result = await _collect_response(resp, model)
        finally:
            await resp.aclose()
        return JSONResponse(content=result)


@app.post("/v1/embeddings")
async def embeddings(request: Request, _: str = Depends(verify_proxy_key)):
    client = _get_client()
    try:
        body = await request.json()
    except Exception:
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

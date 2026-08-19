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
and converts the v3 SSE stream back to OpenAI-format chunks, so any
OpenAI-compatible CLI can use it.

Authenticates incoming requests with a proxy API key (PROXY_API_KEY env var).
"""

from __future__ import annotations

import os
import json
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

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

GATEWAY_BASE_URL = os.getenv("GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh")
GATEWAY_API_KEY = os.getenv("AI_GATEWAY_API_KEY", "")
GATEWAY_TEAM = os.getenv("GATEWAY_TEAM", "")  # optional team slug
PROXY_API_KEY = os.getenv("PROXY_API_KEY", "")  # key clients must present
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "zai/glm-5.2")

# The v3 AI SDK language-model endpoint.
GATEWAY_V3_CHAT = "/v3/ai/language-model"
GATEWAY_V1_MODELS = "/v1/models"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("gateway-proxy")

bearer = HTTPBearer(auto_error=False)

# A shared async client for connection pooling.
_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _client
    _client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=15.0, read=300.0, write=60.0, pool=60.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        http2=True,
    )
    log.info("proxy ready  gateway=%s  default_model=%s", GATEWAY_BASE_URL, DEFAULT_MODEL)
    if not GATEWAY_API_KEY:
        log.warning("AI_GATEWAY_API_KEY not set — upstream will reject requests")
    if not PROXY_API_KEY:
        log.warning("PROXY_API_KEY not set — proxy is open to all callers")
    yield
    await _client.aclose()


app = FastAPI(title="fx-style AI Gateway proxy", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #


def verify_proxy_key(
    creds: HTTPAuthorizationCredentials | None = Security(bearer),
) -> str:
    """Validate the incoming Bearer token against PROXY_API_KEY."""
    if not PROXY_API_KEY:
        return "anonymous"
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if creds.credentials != PROXY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return creds.credentials


# --------------------------------------------------------------------------- #
# v3 AI SDK headers (same as fx CLI)
# --------------------------------------------------------------------------- #


def _v3_headers(model: str, streaming: bool) -> dict[str, str]:
    """Build the exact headers fx sends to the v3 endpoint."""
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


# --------------------------------------------------------------------------- #
# OpenAI → AI SDK v3 translation
# --------------------------------------------------------------------------- #


def _openai_content_to_v3_parts(content) -> list[dict]:
    """Convert OpenAI message content to v3 array-of-parts format.

    Handles str, list, None, and other scalar types. Always returns a
    non-empty list (empty string → single empty text part) so the v3
    protocol never receives a bare null or empty array.
    """
    if content is None or content == "":
        return [{"type": "text", "text": ""}]

    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append({"type": "text", "text": part})
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append({"type": "text", "text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    parts.append({"type": "image", "image": url})
                else:
                    # Passthrough for unknown types
                    parts.append(part)
            else:
                parts.append({"type": "text", "text": str(part)})
        return parts if parts else [{"type": "text", "text": ""}]

    return [{"type": "text", "text": str(content)}]


def _openai_tool_call_to_v3(tool_call: dict) -> dict:
    """Convert a single OpenAI tool call to v3 top-level toolCalls format.

    v3 toolCalls (top-level on assistant message) use:
        {"toolCallId": "...", "toolName": "...", "input": "..."}
    """
    fn = tool_call.get("function", {}) if tool_call.get("type") == "function" else {}
    args = fn.get("arguments", "")
    if not isinstance(args, str):
        try:
            args = json.dumps(args)
        except (TypeError, ValueError):
            args = ""
    return {
        "toolCallId": tool_call.get("id", ""),
        "toolName": fn.get("name", ""),
        "input": args,
    }


def _openai_tool_msg_to_v3(msg: dict) -> dict:
    """Convert an OpenAI tool role message to v3 tool-result format.

    v3 tool-result parts (inside content array) use:
        {"type": "tool-result", "toolCallId": "...", "toolName": "...",
         "output": {"type": "text", "value": "..."}}
    """
    tool_call_id = msg.get("tool_call_id", "")
    tool_name = ""
    content = msg.get("content")

    # Try to extract tool name from content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("name"):
                tool_name = part["name"]
                break

    text_parts: list[str] = []
    if isinstance(content, str):
        text_parts = [content]
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
    elif content is not None:
        text_parts = [str(content)]

    return {
        "role": "tool",
        "content": [{
            "type": "tool-result",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "output": {"type": "text", "value": "".join(text_parts)},
        }],
    }


# --------------------------------------------------------------------------- #
# OpenAI → AI SDK v3 translation
# --------------------------------------------------------------------------- #


def _openai_to_v3(body: dict) -> dict:
    """Convert an OpenAI chat-completion request to AI SDK v3 format."""
    messages = body.get("messages", [])

    prompt = []
    for msg in messages:
        role = msg.get("role", "user")

        # System messages: content as plain string
        if role == "system":
            content = msg.get("content") or ""
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = "".join(text_parts)
            prompt.append({"role": "system", "content": str(content)})
            continue

        # Tool messages: v3 tool-result part format
        if role == "tool":
            prompt.append(_openai_tool_msg_to_v3(msg))
            continue

        # User / assistant messages
        v3_msg: dict
        if role == "assistant" and msg.get("tool_calls"):
            # Assistant with tool_calls: content parts + top-level toolCalls
            tool_calls = [_openai_tool_call_to_v3(tc) for tc in msg["tool_calls"]]
            v3_msg = {
                "role": "assistant",
                "content": _openai_content_to_v3_parts(msg.get("content")),
                "toolCalls": tool_calls,
            }
        else:
            v3_msg = {
                "role": role,
                "content": _openai_content_to_v3_parts(msg.get("content")),
            }
        prompt.append(v3_msg)

    v3_body: dict = {
        "prompt": prompt,
        # The Gateway requires tools + toolChoice to return 200 instead of 503.
        # This was discovered by capturing fx's exact request.
        "tools": [],
        "toolChoice": {"type": "auto"},
        "headers": {"user-agent": "fx/0.0.3"},
    }

    # Pass through temperature / maxTokens if provided.
    if "temperature" in body:
        v3_body["temperature"] = body["temperature"]
    if "max_tokens" in body:
        v3_body["maxOutputTokens"] = body["max_tokens"]
    elif "maxOutputTokens" in body:
        v3_body["maxOutputTokens"] = body["maxOutputTokens"]
    if "top_p" in body:
        v3_body["topP"] = body["top_p"]
    if "stop" in body:
        v3_body["stopSequences"] = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]

    # Pass through tools if the caller provided them.
    openai_tools = body.get("tools")
    if openai_tools:
        v3_tools = []
        for t in openai_tools:
            if isinstance(t, dict) and "function" in t:
                fn = t["function"]
                v3_tools.append({
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "inputSchema": fn.get("parameters", {}),
                })
        if v3_tools:
            v3_body["tools"] = v3_tools

    return v3_body


# --------------------------------------------------------------------------- #
# AI SDK v3 SSE → OpenAI SSE translation
# --------------------------------------------------------------------------- #


def _sse_chunk(
    chat_id: str,
    model: str,
    delta_text: str = "",
    finish_reason: str | None = None,
    role: str | None = None,
    usage: dict | None = None,
) -> str:
    """Build an OpenAI-format SSE chunk."""
    delta: dict = {}
    if role:
        delta["role"] = role
    if delta_text:
        delta["content"] = delta_text
    chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n"


async def _v3_stream_to_openai(
    resp: httpx.Response,
    model: str,
) -> AsyncIterator[str]:
    """Convert a v3 AI SDK SSE stream into OpenAI-format SSE chunks."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    # First chunk: role only
    yield _sse_chunk(chat_id, model, role="assistant")

    # Track incremental tool-input deltas by toolCallId so we can stream
    # arguments as they arrive, then emit a final tool-call chunk.
    tool_input_buffers: dict[str, str] = {}
    pending_tool_calls: dict[str, dict] = {}  # toolCallId -> v3 event

    buffer = ""
    async for raw in resp.aiter_lines():
        if not raw:
            continue
        line = raw.strip()
        if line.startswith("data: "):
            line = line[6:]
        elif line.startswith("data:"):
            line = line[5:]
        elif line == "DONE":
            break

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")

        if etype == "text-delta":
            delta = event.get("delta", "")
            if delta:
                yield _sse_chunk(chat_id, model, delta_text=delta)

        elif etype == "tool-input-delta":
            # Incremental tool arguments — accumulate by toolCallId
            tool_id = event.get("toolCallId", "") or event.get("id", "")
            if tool_id:
                tool_input_buffers[tool_id] = tool_input_buffers.get(tool_id, "") + event.get("delta", "")

        elif etype == "tool-call":
            # Final tool-call event — emit as OpenAI tool_calls delta.
            tool_id = event.get("toolCallId", "") or event.get("id", "")
            tool_name = event.get("toolName", "")
            # Prefer the v3 "input" field; fall back to accumulated deltas.
            tool_args = event.get("input", "") or tool_input_buffers.get(tool_id, "")
            if not isinstance(tool_args, str):
                try:
                    tool_args = json.dumps(tool_args)
                except (TypeError, ValueError):
                    tool_args = ""
            tool_args = tool_args or tool_input_buffers.get(tool_id, "")
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_args},
                        }]
                    },
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            # Clean up buffers for this tool call
            tool_input_buffers.pop(tool_id, None)
            pending_tool_calls.pop(tool_id, None)

        elif etype == "finish":
            finish_reason_raw = event.get("finishReason", {})
            if isinstance(finish_reason_raw, dict):
                finish_reason = finish_reason_raw.get("unified", "stop")
            else:
                finish_reason = str(finish_reason_raw) if finish_reason_raw else "stop"

            # Map to OpenAI finish reasons
            finish_map = {
                "stop": "stop",
                "length": "length",
                "tool-calls": "tool_calls",
                "content-filter": "content_filter",
            }
            mapped = finish_map.get(finish_reason, "stop")

            # Extract usage if present
            usage_data = event.get("usage", {})
            usage = None
            if usage_data:
                prompt_tokens = usage_data.get("inputTokens", {})
                if isinstance(prompt_tokens, dict):
                    pt = prompt_tokens.get("total", 0)
                else:
                    pt = prompt_tokens
                completion_tokens = usage_data.get("outputTokens", {})
                if isinstance(completion_tokens, dict):
                    ct = completion_tokens.get("total", 0)
                else:
                    ct = completion_tokens
                usage = {
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": pt + ct,
                }

            yield _sse_chunk(chat_id, model, finish_reason=mapped, usage=usage)
            break

        elif etype in ("error",):
            error_msg = event.get("error", {}).get("message", "Unknown error")
            yield _sse_chunk(chat_id, model, finish_reason="stop")
            break

    # Final OpenAI sentinel
    yield "data: [DONE]\n\n"


# --------------------------------------------------------------------------- #
# AI SDK v3 non-streaming → OpenAI format
# --------------------------------------------------------------------------- #


def _v3_response_to_openai(v3_data: dict, model: str) -> dict:
    """Convert a v3 non-streaming response to OpenAI format."""
    content_parts = v3_data.get("content", [])
    text = ""
    for part in content_parts:
        if isinstance(part, dict) and part.get("type") == "text":
            text += part.get("text", "")

    finish_raw = v3_data.get("finishReason", "stop")
    if isinstance(finish_raw, dict):
        finish_reason = finish_raw.get("unified", "stop")
    else:
        finish_reason = str(finish_raw) if finish_raw else "stop"

    usage_data = v3_data.get("usage", {})
    prompt_tokens = 0
    completion_tokens = 0
    if usage_data:
        pt = usage_data.get("inputTokens", {})
        prompt_tokens = pt.get("total", 0) if isinstance(pt, dict) else pt
        ct = usage_data.get("outputTokens", {})
        completion_tokens = ct.get("total", 0) if isinstance(ct, dict) else ct

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _client_error(resp: httpx.Response) -> JSONResponse:
    """Forward upstream error bodies as JSON."""
    try:
        body = resp.json()
    except Exception:
        body = {"error": {"message": resp.text, "type": "upstream_error"}}
    return JSONResponse(status_code=resp.status_code, content=body)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "gateway": GATEWAY_BASE_URL, "protocol": "v3"}


@app.get("/v1/models")
async def list_models(_: str = Depends(verify_proxy_key)):
    """List models from the Gateway's public /v1/models endpoint."""
    url = urljoin(GATEWAY_BASE_URL + "/", GATEWAY_V1_MODELS.lstrip("/"))
    headers = {"Accept": "application/json"}
    if GATEWAY_API_KEY:
        headers["Authorization"] = f"Bearer {GATEWAY_API_KEY}"
    resp = await _client.get(url, headers=headers)
    if resp.status_code != 200:
        return _client_error(resp)
    return JSONResponse(content=resp.json())


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, _: str = Depends(verify_proxy_key)):
    """Proxy chat completions via the v3 AI SDK endpoint.

    Accepts standard OpenAI request body and translates to AI SDK v3 format.
    Supports both streaming (SSE) and non-streaming modes.
    """
    if _client is None:
        raise HTTPException(status_code=503, detail="client not ready")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = body.get("model") or DEFAULT_MODEL
    stream = bool(body.get("stream", False))

    # Convert OpenAI request → AI SDK v3 format
    v3_body = _openai_to_v3(body)

    url = urljoin(GATEWAY_BASE_URL + "/", GATEWAY_V3_CHAT.lstrip("/"))
    headers = _v3_headers(model, streaming=True)  # always use streaming upstream

    if stream:
        # Stream: open v3 upstream (always streaming), convert to OpenAI SSE
        req = _client.build_request("POST", url, headers=headers, json=v3_body)
        resp = await _client.send(req, stream=True)
        if resp.status_code != 200:
            await resp.aread()
            err = _client_error(resp)
            await resp.aclose()
            return err
        return StreamingResponse(
            _v3_stream_to_openai(resp, model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        # Non-stream: collect the v3 SSE stream and assemble a single response
        req = _client.build_request("POST", url, headers=headers, json=v3_body)
        resp = await _client.send(req, stream=True)
        if resp.status_code != 200:
            await resp.aread()
            err = _client_error(resp)
            await resp.aclose()
            return err

        # Read the full SSE stream and extract the final result
        text_parts = []
        tool_calls = []
        finish_reason = "stop"
        usage_data = {}
        async for raw in resp.aiter_lines():
            if not raw:
                continue
            line = raw.strip()
            if line.startswith("data: "):
                line = line[6:]
            elif line.startswith("data:"):
                line = line[5:]
            elif line == "DONE":
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")
            if etype == "text-delta":
                text_parts.append(event.get("delta", ""))
            elif etype == "tool-call":
                tc = {
                    "id": event.get("toolCallId", ""),
                    "type": "function",
                    "function": {
                        "name": event.get("toolName", ""),
                        "arguments": event.get("input", "{}"),
                    },
                }
                tool_calls.append(tc)
            elif etype == "finish":
                fr = event.get("finishReason", {})
                if isinstance(fr, dict):
                    finish_reason = fr.get("unified", "stop")
                else:
                    finish_reason = str(fr) if fr else "stop"
                finish_map = {
                    "stop": "stop",
                    "length": "length",
                    "tool-calls": "tool_calls",
                    "content-filter": "content_filter",
                }
                finish_reason = finish_map.get(finish_reason, finish_reason)
                usage_data = event.get("usage", {})

        await resp.aclose()

        full_text = "".join(text_parts)
        pt = usage_data.get("inputTokens", {})
        prompt_tokens = pt.get("total", 0) if isinstance(pt, dict) else pt
        ct = usage_data.get("outputTokens", {})
        completion_tokens = ct.get("total", 0) if isinstance(ct, dict) else ct

        result = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_text if full_text else (None if tool_calls else ""),
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                },
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        return JSONResponse(content=result)


@app.post("/v1/embeddings")
async def embeddings(request: Request, _: str = Depends(verify_proxy_key)):
    """Proxy embeddings requests (OpenAI-compatible, uses v1 endpoint)."""
    if _client is None:
        raise HTTPException(status_code=503, detail="client not ready")
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
    resp = await _client.post(url, headers=headers, json=body)
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

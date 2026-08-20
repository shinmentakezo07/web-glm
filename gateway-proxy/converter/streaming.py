"""v3 SSE stream -> OpenAI chat.completion SSE stream.

Includes the sync/async streaming converters, the stateful chunk processor,
and the non-streaming helper that collects a v3 SSE stream into one OpenAI
response.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Iterator

from .response import _v3_finish_reason, _v3_usage_to_openai


# --------------------------------------------------------------------------- #
# v3 streaming -> OpenAI streaming
# --------------------------------------------------------------------------- #


def _sse_chunk(
    chat_id: str,
    model: str,
    delta_text: str = "",
    finish_reason: str | None = None,
    role: str | None = None,
    usage: dict | None = None,
    reasoning_delta: str | None = None,
) -> str:
    delta: dict = {}
    if role:
        delta["role"] = role
    if delta_text:
        delta["content"] = delta_text
    if reasoning_delta:
        delta["reasoning_content"] = reasoning_delta
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


class _StreamState:
    """Mutable state shared by sync and async stream converters."""

    def __init__(self, chat_id: str, model: str, include_usage: bool = True):
        self.chat_id = chat_id
        self.model = model
        self.include_usage = include_usage
        self.tool_input_buffers: dict[str, str] = {}
        self.tool_call_index: dict[str, int] = {}
        self.next_tool_index = 0
        self.last_step_usage: dict = {}
        self.finished = False


def _process_stream_event(state: _StreamState, event: dict) -> list[str]:
    """Return OpenAI SSE chunk strings for one v3 stream event."""
    chunks: list[str] = []
    etype = event.get("type", "")

    if etype == "text-delta":
        delta = event.get("delta", "")
        if delta:
            chunks.append(_sse_chunk(state.chat_id, state.model, delta_text=delta))

    elif etype == "tool-input-delta":
        tool_id = event.get("toolCallId", "") or event.get("id", "")
        if tool_id:
            state.tool_input_buffers[tool_id] = (
                state.tool_input_buffers.get(tool_id, "") + event.get("delta", "")
            )

    elif etype == "tool-call":
        tool_id = event.get("toolCallId", "") or event.get("id", "")
        tool_name = event.get("toolName", "")
        tool_args = event.get("input", "") or state.tool_input_buffers.get(tool_id, "")
        if isinstance(tool_args, (dict, list)):
            try:
                tool_args = json.dumps(tool_args)
            except (TypeError, ValueError):
                tool_args = ""
        tool_args = tool_args or state.tool_input_buffers.get(tool_id, "")

        idx = state.tool_call_index.get(tool_id)
        if idx is None:
            idx = state.next_tool_index
            state.tool_call_index[tool_id] = idx
            state.next_tool_index += 1

        chunk = {
            "id": state.chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": state.model,
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
        chunks.append(f"data: {json.dumps(chunk)}\n\n")
        state.tool_input_buffers.pop(tool_id, None)

    elif etype == "finish":
        finish_reason = _v3_finish_reason(event.get("finishReason", "stop"))
        usage_data = event.get("usage", {}) or state.last_step_usage or {}
        usage = _v3_usage_to_openai(usage_data) if (usage_data and state.include_usage) else None
        chunks.append(_sse_chunk(state.chat_id, state.model, finish_reason=finish_reason, usage=usage))
        state.finished = True

    elif etype == "reasoning-delta":
        delta = event.get("delta", "")
        if delta:
            chunks.append(_sse_chunk(state.chat_id, state.model, reasoning_delta=delta))

    elif etype == "finish-step":
        usage_data = event.get("usage", {})
        if usage_data:
            state.last_step_usage = usage_data

    elif etype == "response-metadata":
        if not state.model and event.get("modelId"):
            state.model = event["modelId"]

    elif etype in (
        "start", "start-step", "tool-result", "source", "file", "raw",
        "text-start", "text-end", "reasoning-start", "reasoning-end",
        "tool-input-start", "tool-input-end",
    ):
        pass  # explicitly handled no-ops: no OpenAI chunk for these

    elif etype == "error":
        # Stream ended with an upstream error.
        err = event.get("error", {})
        if isinstance(err, dict) and err.get("message"):
            chunks.append(f"data: {json.dumps({'type': 'error', 'error': {'message': err['message']}})}\n\n")
        chunks.append(_sse_chunk(state.chat_id, state.model, finish_reason="stop"))
        state.finished = True

    return chunks


def v3_stream_to_openai(
    events: list[dict],
    model: str = "",
    include_usage: bool = True,
) -> str:
    """Convert a list of v3 SSE events to an OpenAI SSE stream string.

    Useful for testing / offline conversion. For live HTTP use, the proxy
    server streams incrementally via `v3_stream_iter`.
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    # If the client supplied no model, resolve it from response-metadata before
    # emitting the leading role chunk so every chunk carries the final model.
    if not model:
        for ev in events:
            if ev.get("type") == "response-metadata" and ev.get("modelId"):
                model = ev["modelId"]
                break
    chunks: list[str] = [_sse_chunk(chat_id, model, role="assistant")]
    state = _StreamState(chat_id, model, include_usage)
    for event in events:
        chunks.extend(_process_stream_event(state, event))
        if state.finished:
            break
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks)


async def v3_stream_iter(
    events: AsyncIterator[dict],
    model: str = "",
    include_usage: bool = True,
) -> AsyncIterator[str]:
    """Translate a live v3 SSE event stream into OpenAI SSE chunks.

    Async generator used by the proxy server's streaming handler; yields
    one OpenAI chunk string per upstream event without buffering the whole
    response.
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    yield _sse_chunk(chat_id, model, role="assistant")

    state = _StreamState(chat_id, model, include_usage)
    async for event in events:
        for chunk in _process_stream_event(state, event):
            yield chunk
        if state.finished:
            break
    yield "data: [DONE]\n\n"


# --------------------------------------------------------------------------- #
# Non-streaming helper: collect a v3 SSE stream into one OpenAI response
# --------------------------------------------------------------------------- #


def v3_sse_stream_to_openai(
    events: Iterator[dict],
    model: str = "",
) -> dict:
    """Consume a v3 SSE stream (as an iterator of parsed events) and return
    a single OpenAI chat.completion dict.

    Used by the server for non-streaming mode.
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    finish_reason = "stop"
    usage_data: dict = {}
    step_usage: dict = {}

    for event in events:
        etype = event.get("type", "")
        if etype == "text-delta":
            text_parts.append(event.get("delta", ""))
        elif etype == "tool-call":
            tool_id = event.get("toolCallId", "") or event.get("id", "")
            tool_name = event.get("toolName", "")
            tool_args = event.get("input", "")
            if not isinstance(tool_args, str):
                try:
                    tool_args = json.dumps(tool_args)
                except (TypeError, ValueError):
                    tool_args = "{}"
            tool_calls.append({
                "id": tool_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": tool_args},
            })
        elif etype == "finish-step":
            if event.get("usage"):
                step_usage = event["usage"]
        elif etype == "finish":
            finish_reason = _v3_finish_reason(event.get("finishReason", "stop"))
            usage_data = event.get("usage", {}) or step_usage or {}
            usage_data = usage_data or {}
            break
        elif etype == "error":
            finish_reason = "stop"
            break

    full_text = "".join(text_parts)
    return {
        "id": chat_id,
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
        "usage": _v3_usage_to_openai(usage_data) if usage_data else {},
    }

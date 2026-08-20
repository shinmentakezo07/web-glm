"""
Standalone OpenAI <-> AI SDK v3 format converter.

Translates between the OpenAI chat-completions format (what most CLIs and
agents speak) and the AI SDK v3 format (what the Vercel AI Gateway expects).

Wire format is aligned with the reference implementation (vercel-labs/fx,
src/core/gateway/gateway_json.zig):

  * assistant tool calls are **content parts**:
        {"type": "tool-call", "toolCallId": "...", "toolName": "...", "input": {raw JSON}}
    (NOT a top-level `toolCalls` array; `input` is the raw JSON object, not a string)
  * tool results are content parts:
        {"role": "tool", "content": [{"type": "tool-result", "toolCallId": "...",
            "toolName": "...", "output": {"type": "text", "value": "..."}}]}
  * missing tool names default to "unknown" (fx behaviour)
  * `toolChoice` is always an object shape ({type: auto|none|required|tool, ...})

The gateway's own validation is opaque ("Invalid input"), so this module
also validates tool history client-side before sending (modeled on fx's
validateToolMessageHistory) and surfaces clear errors.

Use it as a library so agents can talk to each other through the gateway
without an HTTP proxy layer:

    from converter import openai_to_v3, v3_to_openai, validate_tool_history

    err = validate_tool_history(body.get("messages", []))
    v3_body = openai_to_v3(openai_request_dict)
    # ...send v3_body to gateway, get v3_response...
    openai_response = v3_to_openai(v3_response_dict)

The conversion functions are pure: no network calls, no side effects.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Iterator


# --------------------------------------------------------------------------- #
# OpenAI -> v3
# --------------------------------------------------------------------------- #


def _openai_content_to_v3_parts(content) -> list[dict]:
    """Convert OpenAI message content to v3 array-of-parts format.

    Returns a non-empty list; None/empty string becomes a single empty text
    part so the v3 protocol never receives a bare null.
    """
    if content is None or content == "":
        return [{"type": "text", "text": ""}]

    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if isinstance(content, list):
        parts: list[dict] = []
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
                    parts.append(part)
            else:
                parts.append({"type": "text", "text": str(part)})
        return parts if parts else [{"type": "text", "text": ""}]

    return [{"type": "text", "text": str(content)}]


def _openai_tool_call_to_v3(tool_call: dict) -> dict:
    """Convert an OpenAI tool call to a v3 content part (fx wire format).

    OpenAI input:
        {"id": "call_1", "type": "function",
         "function": {"name": "read_file", "arguments": "{\"path\":\"...\"}"}}

    v3 output (content part):
        {"type": "tool-call", "toolCallId": "call_1", "toolName": "read_file",
         "input": {"path": "..."}}
    """
    fn = tool_call.get("function", {}) if tool_call.get("type", "function") == "function" else {}
    args = fn.get("arguments", "{}")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            args = {}
    elif args is None:
        args = {}
    return {
        "type": "tool-call",
        "toolCallId": tool_call.get("id", ""),
        "toolName": fn.get("name", ""),
        "input": args,
    }


def _openai_tool_msg_to_v3(msg: dict) -> dict:
    """Convert an OpenAI tool role message to a v3 tool-result content part.

    OpenAI input:
        {"role": "tool", "tool_call_id": "call_1", "content": "result text"}

    v3 output:
        {"role": "tool", "content": [{"type": "tool-result",
            "toolCallId": "call_1", "toolName": "...",
            "output": {"type": "text", "value": "result text"}}]}

    The `toolName` is back-filled by the caller from the preceding assistant
    message, defaulting to "unknown" (fx behaviour).
    """
    tool_call_id = msg.get("tool_call_id", "")
    tool_name = msg.get("name", "") or "unknown"
    content = msg.get("content")

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
                text_parts.append(part.get("text", "") if part.get("text") else "")
            elif isinstance(part, dict) and part.get("name"):
                pass  # metadata part, skip
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


def _normalize_tool_choice(tool_choice) -> dict:
    """Normalize an OpenAI tool_choice to the v3 object shape.

    OpenAI clients send tool_choice as a plain string ("auto" / "none" /
    "required") or as {"type": "function", "function": {"name": "..."}}.
    The v3 gateway only accepts object shapes:
        {"type": "auto"} | {"type": "none"} | {"type": "required"}
        | {"type": "tool", "toolName": "..."}
    """
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            fn = tool_choice.get("function", {})
            return {"type": "tool", "toolName": fn.get("name", "")}
        return tool_choice
    if isinstance(tool_choice, str):
        return {
            "auto": {"type": "auto"},
            "none": {"type": "none"},
            "required": {"type": "required"},
        }.get(tool_choice, {"type": "auto"})
    return {"type": "auto"}


def _openai_response_format_to_v3(response_format) -> dict | None:
    """Map an OpenAI response_format to the v3 responseFormat shape.

    OpenAI:
        {"type": "json_object"}
        {"type": "json_schema", "json_schema": {"name": ..., "schema": ...}}

    v3 (fx wire format):
        {"type": "json", "name": "...", "description": "...", "schema": {...}}
    """
    if not isinstance(response_format, dict):
        return None
    rtype = response_format.get("type")
    if rtype == "json_object":
        return {"type": "json", "name": "", "description": "", "schema": {}}
    if rtype == "json_schema":
        js = response_format.get("json_schema", {})
        if not isinstance(js, dict):
            return None
        return {
            "type": "json",
            "name": js.get("name", ""),
            "description": js.get("description", ""),
            "schema": js.get("schema", {}),
        }
    return None


# --------------------------------------------------------------------------- #
# Client-side tool-history validation (modeled on fx's validateToolMessageHistory)
# --------------------------------------------------------------------------- #


def _parse_tool_args(args) -> str | None:
    """Return an error message if tool args are not valid JSON, else None."""
    if isinstance(args, dict):
        return None
    if isinstance(args, str):
        if not args.strip():
            return "tool call arguments are empty"
        try:
            json.loads(args)
        except json.JSONDecodeError:
            return "tool call arguments are not valid JSON"
        return None
    if args is None:
        return "tool call arguments are empty"
    return "tool call arguments are not valid JSON"


def validate_tool_history(messages: list[dict]) -> str | None:
    """Validate that assistant tool calls are properly paired with results.

    Modeled on fx's validateToolMessageHistory:
      * tool role messages must be preceded by an assistant tool call block
      * every assistant tool call must have a unique id, non-empty name and
        valid JSON arguments
      * the messages immediately following a tool-calling assistant message
        must be tool results covering every call (in any order), with matched
        ids and tool names

    Returns an error message string, or None if the history is valid.
    """
    seen_ids: set[str] = set()

    def _check_call(call: dict, index: int) -> str | None:
        call_id = call.get("id", "")
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        name = fn.get("name", "")
        args = fn.get("arguments", "")
        if call_id in seen_ids:
            return f"duplicate tool call id in assistant message {index}: {call_id}"
        seen_ids.add(call_id)
        if not name:
            return f"tool call {call_id or index} is missing a function name"
        if not isinstance(call.get("type", "function"), str):
            return f"tool call {call_id} has an invalid type"
        return _parse_tool_args(args)

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "user")

        # A tool result must immediately follow the assistant block it answers.
        if role == "tool":
            return f"tool result at index {i} has no preceding assistant tool call"

        if role != "assistant" or not msg.get("tool_calls"):
            i += 1
            continue

        calls = msg.get("tool_calls", [])
        if not isinstance(calls, list) or len(calls) == 0:
            return f"assistant message at index {i} has an empty tool_calls array"
        for idx, call in enumerate(calls):
            if not isinstance(call, dict):
                return f"tool call at index {idx} of assistant message {i} is not an object"
            err = _check_call(call, i)
            if err:
                return err

        # The next block must be tool results covering all calls.
        results = []
        j = i + 1
        while j < len(messages) and messages[j].get("role") == "tool":
            results.append(messages[j])
            j += 1

        if len(results) == 0:
            return f"assistant tool calls at index {i} have no matching tool results"

        matched_ids: set[str] = set()
        for idx, result in enumerate(results):
            rid = result.get("tool_call_id", "")
            if rid not in seen_ids:
                return f"tool result at index {i + 1 + idx} references unknown tool call: {rid}"
            if rid in matched_ids:
                return f"duplicate tool result for tool call: {rid}"
            matched_ids.add(rid)
            if result.get("content") is None:
                return f"tool result for {rid} has no content"

        if len(matched_ids) != len(calls):
            return (f"assistant tool calls at index {i} are not all paired "
                    f"({len(matched_ids)}/{len(calls)} results)")

        i = j  # skip the consumed result block, continue after it

    return None


# --------------------------------------------------------------------------- #
# OpenAI -> v3 (full request)
# --------------------------------------------------------------------------- #


def openai_to_v3(body: dict) -> dict:
    """Convert an OpenAI chat-completions request to AI SDK v3 format.

    Translates:
    - messages: role/content format conversion (tool calls aligned to fx)
    - tools: OpenAI {"type":"function","function":{...}} -> v3 flat {"type":"function","name","description","inputSchema"}
    - temperature, max_tokens, top_p, stop, response_format, reasoning,
      providerOptions parameters

    Does NOT modify the input body.
    """
    messages = body.get("messages", [])

    # Back-fill toolName on tool results from the preceding assistant message.
    tool_call_name_map: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id", "")
                tc_fn = tc.get("function", {})
                if tc_id:
                    tool_call_name_map[tc_id] = tc_fn.get("name", "")

    prompt: list[dict] = []
    for msg in messages:
        role = msg.get("role", "user")

        if role == "system":
            content = msg.get("content") or ""
            if isinstance(content, list):
                text_parts: list[str] = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", "") if part.get("text") else "")
                content = "".join(text_parts)
            prompt.append({"role": "system", "content": str(content)})
            continue

        if role == "tool":
            v3_tool_msg = _openai_tool_msg_to_v3(msg)
            part = v3_tool_msg["content"][0]
            tc_id = msg.get("tool_call_id", "")
            if part["toolName"] == "unknown" and tc_id in tool_call_name_map:
                part["toolName"] = tool_call_name_map[tc_id]
            prompt.append(v3_tool_msg)
            continue

        if role == "assistant" and msg.get("tool_calls"):
            content_parts = _openai_content_to_v3_parts(msg.get("content"))
            tool_parts = [_openai_tool_call_to_v3(tc) for tc in msg["tool_calls"]]
            prompt.append({
                "role": "assistant",
                "content": content_parts + tool_parts,
            })
        else:
            prompt.append({
                "role": role,
                "content": _openai_content_to_v3_parts(msg.get("content")),
            })

    # Build tools array in flat v3 format.
    v3_tools: list[dict] = []
    openai_tools = body.get("tools")
    if openai_tools:
        for t in openai_tools:
            if isinstance(t, dict) and "function" in t:
                fn = t["function"]
                v3_tools.append({
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "inputSchema": fn.get("parameters", {}),
                })

    v3_body: dict = {
        "prompt": prompt,
        "tools": v3_tools,
        "toolChoice": _normalize_tool_choice(body.get("tool_choice", {"type": "auto"})),
        "headers": {"user-agent": "fx-converter"},
    }

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

    # Structured output / JSON mode.
    rf = _openai_response_format_to_v3(body.get("response_format"))
    if rf:
        v3_body["responseFormat"] = rf

    # Reasoning effort (OpenAI) / reasoning (v3, fx).
    if "reasoning" in body:
        v3_body["reasoning"] = body["reasoning"]
    elif "reasoning_effort" in body:
        _effort_map = {"low": "low", "medium": "medium", "high": "high", "minimal": "minimal"}
        v3_body["reasoning"] = _effort_map.get(
            body.get("reasoning_effort"), body.get("reasoning_effort")
        )

    # Provider options passthrough (routing, caching, BYOK, fallbacks...).
    if isinstance(body.get("providerOptions"), dict) and body["providerOptions"]:
        v3_body["providerOptions"] = body["providerOptions"]

    return v3_body


# --------------------------------------------------------------------------- #
# v3 -> OpenAI (non-streaming)
# --------------------------------------------------------------------------- #


_FINISH_REASON_MAP: dict[str, str] = {
    "stop": "stop",
    "length": "length",
    "tool-calls": "tool_calls",
    "tool_calls": "tool_calls",
    "content-filter": "content_filter",
    "content_filter": "content_filter",
}


def _v3_finish_reason(v3_reason) -> str:
    if isinstance(v3_reason, dict):
        v3_reason = v3_reason.get("unified", "stop")
    elif v3_reason is None:
        v3_reason = "stop"
    return _FINISH_REASON_MAP.get(str(v3_reason), "stop")


def _v3_usage_to_openai(usage_data: dict) -> dict:
    pt = usage_data.get("inputTokens", {})
    prompt_tokens = pt.get("total", 0) if isinstance(pt, dict) else (pt or 0)
    ct = usage_data.get("outputTokens", {})
    completion_tokens = ct.get("total", 0) if isinstance(ct, dict) else (ct or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def v3_to_openai(v3_data: dict, model: str = "") -> dict:
    """Convert a v3 gateway response to an OpenAI chat.completion response.

    Handles both the raw v3 response shape and a choices[0].message shape.
    Does not modify the input.
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if "choices" in v3_data:
        choice = v3_data["choices"][0] if v3_data["choices"] else {}
        msg = choice.get("message", {})
        finish_raw = choice.get("finish_reason") or v3_data.get("finishReason")
        usage_data = v3_data.get("usage", {})
    else:
        msg = v3_data
        finish_raw = v3_data.get("finishReason", "stop")
        usage_data = v3_data.get("usage", {})

    text = ""
    tool_calls: list[dict] = []

    for part in msg.get("content", []):
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            text += part.get("text", "")
        elif part.get("type") == "tool-call":
            tool_args = part.get("input", {})
            if not isinstance(tool_args, str):
                try:
                    tool_args = json.dumps(tool_args)
                except (TypeError, ValueError):
                    tool_args = "{}"
            tool_calls.append({
                "id": part.get("toolCallId", ""),
                "type": "function",
                "function": {
                    "name": part.get("toolName", ""),
                    "arguments": tool_args,
                },
            })

    # Also accept top-level toolCalls (older AI SDK style) and OpenAI-style
    # tool_calls for robustness.
    for tc in msg.get("toolCalls", []):
        tool_args = tc.get("input", {})
        if not isinstance(tool_args, str):
            try:
                tool_args = json.dumps(tool_args)
            except (TypeError, ValueError):
                tool_args = "{}"
        tool_calls.append({
            "id": tc.get("toolCallId", ""),
            "type": "function",
            "function": {
                "name": tc.get("toolName", ""),
                "arguments": tool_args,
            },
        })
    for tc in msg.get("tool_calls", []):
        if tc not in tool_calls:
            tool_calls.append(tc)

    finish_reason = _v3_finish_reason(finish_raw)

    result_msg: dict = {"role": "assistant", "content": text if text else None}
    if tool_calls:
        result_msg["tool_calls"] = tool_calls

    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or v3_data.get("model", ""),
        "choices": [{
            "index": 0,
            "message": result_msg,
            "finish_reason": finish_reason,
        }],
        "usage": _v3_usage_to_openai(usage_data) if usage_data else {},
    }


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
) -> str:
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


class _StreamState:
    """Mutable state shared by sync and async stream converters."""

    def __init__(self, chat_id: str, model: str, include_usage: bool = True):
        self.chat_id = chat_id
        self.model = model
        self.include_usage = include_usage
        self.tool_input_buffers: dict[str, str] = {}
        self.tool_call_index: dict[str, int] = {}
        self.next_tool_index = 0
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
        usage_data = event.get("usage", {})
        usage = _v3_usage_to_openai(usage_data) if (usage_data and state.include_usage) else None
        chunks.append(_sse_chunk(state.chat_id, state.model, finish_reason=finish_reason, usage=usage))
        state.finished = True

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
        elif etype == "finish":
            finish_reason = _v3_finish_reason(event.get("finishReason", "stop"))
            usage_data = event.get("usage", {})
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


# --------------------------------------------------------------------------- #
# Responses API (OpenAI new format) <-> v3
# --------------------------------------------------------------------------- #


def responses_input_to_messages(input_items: list | None) -> list[dict]:
    """Convert OpenAI Responses API `input` items to chat-completions messages.

    Handles both flat {role, content} and typed items:
      {"type": "message", "role": ...}
      {"type": "function_call", "call_id", "name", "arguments"}
      {"type": "function_call_output", "call_id", "output"}

    Ordering invariant: an assistant tool-call message must be immediately
    followed by the tool results for those calls (see validate_tool_history),
    so function_call_output items flush any pending calls first.
    """
    messages: list[dict] = []
    if not input_items:
        return messages

    def _flush_calls() -> None:
        if not pending_calls:
            return
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": list(pending_calls),
        })
        pending_calls.clear()

    pending_calls: list[dict] = []
    for item in input_items:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        role = item.get("role", "user")

        if itype == "function_call" or (itype is None and item.get("call_id") and item.get("name")):
            pending_calls.append({
                "id": item.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                },
            })
        elif itype == "function_call_output" or (itype is None and item.get("call_id") and "output" in item):
            _flush_calls()
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": item.get("output", ""),
            })
        else:
            # Flat messages and {"type": "message"} items.
            content = item.get("content", "")
            if isinstance(content, list):
                text_parts: list[str] = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and part.get("type") in ("text", "output_text", "input_text"):
                        text_parts.append(part.get("text", ""))
                content = "".join(text_parts)
            messages.append({
                "role": ("assistant" if role == "assistant" else "user"),
                "content": content,
            })

    # Flush any trailing assistant tool calls that had no outputs yet.
    _flush_calls()
    return messages


def openai_to_responses(openai_resp: dict, model: str = "") -> dict:
    """Convert an OpenAI chat.completion response to the Responses API shape."""
    choice = openai_resp.get("choices", [{}])[0]
    msg = choice.get("message", {})
    text = msg.get("content") or ""
    tool_calls = msg.get("tool_calls", [])

    output: list[dict] = []
    if text:
        output.append({
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        })
    for tc in tool_calls:
        output.append({
            "type": "function_call",
            "id": f"fc_{uuid.uuid4().hex[:24]}",
            "call_id": tc.get("id", ""),
            "name": tc.get("function", {}).get("name", ""),
            "arguments": tc.get("function", {}).get("arguments", ""),
            "status": "completed",
        })

    usage = openai_resp.get("usage", {})
    return {
        "id": f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model or openai_resp.get("model", ""),
        "output": output,
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": usage.get("completion_tokens", 0),
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


class _ResponsesStreamState:
    """State for translating an OpenAI chat.completion.chunk stream into
    Responses API SSE events (mirrors openai_to_responses for streaming)."""

    def __init__(self, model: str):
        self.response_id = f"resp_{uuid.uuid4().hex[:16]}"
        self.model = model
        self.msg_id: str | None = None
        self.tool_ids: dict[str, str] = {}   # call_id -> fc_ item id
        self.tool_index: dict[str, int] = {}  # call_id -> output index
        self.text_parts: list[str] = []
        self.tool_calls: list[dict] = []
        self.started = False
        self.usage: dict = {}

    def _sse(self, obj: dict) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    def start(self) -> str:
        """Emit response.created + output_item.added + content_part.added."""
        self.started = True
        self.msg_id = f"msg_{uuid.uuid4().hex[:16]}"
        response = {
            "id": self.response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "model": self.model,
            "output": [],
        }
        out = self._sse({"type": "response.created", "response": response})
        out += self._sse({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": self.msg_id,
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
        })
        out += self._sse({
            "type": "response.content_part.added",
            "item_id": self.msg_id,
            "output_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        })
        return out

    def text_delta(self, delta: str) -> str:
        self.text_parts.append(delta)
        return self._sse({
            "type": "response.output_text.delta",
            "item_id": self.msg_id,
            "output_index": 0,
            "delta": delta,
        })

    def tool_call(self, call: dict) -> str:
        """Emit output_item.added + function_call_arguments.delta for a
        single-assignment OpenAI tool-call chunk."""
        call_id = call.get("id", "")
        fn = call.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", "")
        if call_id in self.tool_ids:
            item_id = self.tool_ids[call_id]
            idx = self.tool_index[call_id]
            out = ""
        else:
            idx = len(self.tool_calls)
            item_id = f"fc_{uuid.uuid4().hex[:16]}"
            self.tool_ids[call_id] = item_id
            self.tool_index[call_id] = idx
            self.tool_calls.append({
                "id": item_id,
                "type": "function_call",
                "call_id": call_id,
                "name": name or "",
                "arguments": "",
                "status": "in_progress",
            })
            out = self._sse({
                "type": "response.output_item.added",
                "output_index": idx + 1,
                "item": self.tool_calls[-1],
            })
        if args:
            if self.tool_calls[idx]["arguments"]:
                self.tool_calls[idx]["arguments"] += args
            else:
                self.tool_calls[idx]["arguments"] = args
            out += self._sse({
                "type": "response.function_call_arguments.delta",
                "item_id": item_id,
                "output_index": idx + 1,
                "delta": args,
            })
        return out

    def finish(self, finish_reason: str | None, usage: dict | None) -> str:
        """Emit output_text.done / output_item.done for each item + response.completed."""
        out = ""
        if self.text_parts:
            out += self._sse({
                "type": "response.output_text.done",
                "item_id": self.msg_id,
                "output_index": 0,
                "text": "".join(self.text_parts),
            })
            out += self._sse({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": self.msg_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": "".join(self.text_parts),
                        "annotations": [],
                    }],
                },
            })
        for i, tc in enumerate(self.tool_calls):
            item = dict(tc)
            item["status"] = "completed"
            out += self._sse({
                "type": "response.output_item.done",
                "output_index": i + 1,
                "item": item,
            })

        output = []
        if self.text_parts:
            output.append({
                "type": "message",
                "id": self.msg_id,
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "".join(self.text_parts),
                    "annotations": [],
                }],
            })
        for tc in self.tool_calls:
            tc = dict(tc)
            tc["status"] = "completed"
            output.append(tc)

        if usage:
            self.usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": usage.get("completion_tokens", 0),
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": usage.get("total_tokens", 0),
            }

        response = {
            "id": self.response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": self.model,
            "output": output,
            "parallel_tool_calls": True,
            "usage": self.usage,
        }
        out += self._sse({"type": "response.completed", "response": response})
        return out

    def failed(self, message: str) -> str:
        response = {
            "id": self.response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "failed",
            "model": self.model,
            "output": [],
            "error": {"message": message},
        }
        out = self._sse({"type": "response.failed", "response": response})
        out += self._sse({"type": "response.completed", "response": response})
        return out


def openai_chunk_to_responses_sse(
    chunk_str: str,
    state: _ResponsesStreamState,
) -> str | None:
    """Translate one OpenAI SSE data line to Responses SSE events (or None).

    `chunk_str` is a full SSE data line, e.g. "data: {...}" or "data: [DONE]".
    `state` carries the incremental translation state across calls
    (create one `_ResponsesStreamState` per request).
    """
    stripped = chunk_str.strip()
    if not stripped.startswith("data:"):
        return None
    payload = stripped[len("data:"):].strip()
    if not payload:
        return None
    if payload in ("[DONE]", "DONE"):
        return chunk_str if chunk_str.strip() == stripped else None
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return None

    if chunk.get("type") == "error":
        err = chunk.get("error", {})
        return state.failed(err.get("message", "upstream error"))

    choices = chunk.get("choices") or []
    choice = choices[0] if choices else {}
    delta = choice.get("delta") or {}
    finish_reason = choice.get("finish_reason")

    out = ""
    if not state.started:
        out += state.start()
    if delta.get("role") == "assistant":
        pass  # already emitted in start()
    if delta.get("content"):
        out += state.text_delta(delta["content"])
    for tc in delta.get("tool_calls") or []:
        out += state.tool_call(tc)
    if finish_reason is not None:
        out += state.finish(finish_reason, chunk.get("usage"))
    return out or None


def v3_stream_to_responses_sse(
    events: list[dict],
    model: str = "",
) -> str:
    """Convert a list of v3 SSE events directly to Responses SSE (offline use)."""
    state = _ResponsesStreamState(model)
    lines = [state.start()]
    for event in events:
        etype = event.get("type", "")
        if etype == "text-delta":
            delta = event.get("delta", "")
            if delta:
                lines.append(state.text_delta(delta))
        elif etype == "tool-call":
            call = {
                "id": event.get("toolCallId", "") or event.get("id", ""),
                "function": {
                    "name": event.get("toolName", ""),
                    "arguments": event.get("input", ""),
                },
            }
            lines.append(state.tool_call(call))
        elif etype == "finish":
            lines.append(state.finish(
                _v3_finish_reason(event.get("finishReason", "stop")),
                _v3_usage_to_openai(event.get("usage", {})),
            ))
            break
        elif etype == "error":
            err = event.get("error", {})
            lines.append(state.failed(
                err.get("message", "upstream error") if isinstance(err, dict) else str(err)
            ))
            break
    lines.append("data: [DONE]\n\n")
    return "".join(lines)


# --------------------------------------------------------------------------- #
# CLI (optional)
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python converter.py <input.json> [--stream] [--reverse]")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    is_stream = "--stream" in sys.argv
    is_reverse = "--reverse" in sys.argv

    if is_stream:
        events = data if isinstance(data, list) else [data]
        print(v3_stream_to_openai(events))
    elif is_reverse:
        print(json.dumps(v3_to_openai(data), indent=2))
    elif "prompt" in data:
        print(json.dumps(v3_to_openai(data), indent=2))
    else:
        print(json.dumps(openai_to_v3(data), indent=2))

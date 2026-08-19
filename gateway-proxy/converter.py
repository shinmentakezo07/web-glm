"""
Standalone OpenAI <-> AI SDK v3 format converter.

Translates between the OpenAI chat-completions format (what most CLIs and
agents speak) and the AI SDK v3 format (what the Vercel AI Gateway expects).
This is the same protocol the fx CLI uses internally
(see fx/src/core/gateway/gateway_json.zig).

Use it as a library so agents can talk to each other through the gateway
without an HTTP proxy layer:

    from converter import openai_to_v3, v3_to_openai

    v3_body = openai_to_v3(openai_request_dict)
    # ...send v3_body to gateway, get v3_response...
    openai_response = v3_to_openai(v3_response_dict)

The functions are pure: no network calls, no side effects.
"""

from __future__ import annotations

import json
import time
import uuid


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
    """Convert an OpenAI tool call to a v3 tool-call content part.

    v3 tool-call parts go inside the assistant's content array. The `input`
    field is a raw JSON object (not stringified).

    OpenAI input:
        {"id": "call_1", "type": "function",
         "function": {"name": "read_file", "arguments": "{\"path\":\"...\"}"}}

    v3 output:
        {"type": "tool-call", "toolCallId": "call_1",
         "toolName": "read_file", "input": {"path": "..."}}
    """
    fn = tool_call.get("function", {}) if tool_call.get("type") == "function" else {}
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
    """Convert an OpenAI tool role message to a v3 tool-result part.

    OpenAI input:
        {"role": "tool", "tool_call_id": "call_1", "content": "result text"}

    v3 output:
        {"role": "tool", "content": [{"type": "tool-result",
            "toolCallId": "call_1", "toolName": "...",
            "output": {"type": "text", "value": "result text"}}]}
    """
    tool_call_id = msg.get("tool_call_id", "")
    tool_name = msg.get("name", "")
    content = msg.get("content")

    # Extract tool name from content parts if present.
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("name") and not tool_name:
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


def openai_to_v3(body: dict) -> dict:
    """Convert an OpenAI chat-completions request to AI SDK v3 format.

    Translates:
    - messages: role/content format conversion
    - tools: OpenAI {"type":"function","function":{...}} -> v3 flat {"type":"function","name","description","parameters"}
    - temperature, max_tokens, top_p, stop parameters

    Does NOT modify the input body.
    """
    messages = body.get("messages", [])

    # Build a lookup of tool_call_id -> tool_name from assistant messages
    # so we can populate toolName in tool-result messages (OpenAI doesn't
    # carry it on the tool message itself).
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

        # System: content as plain string.
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

        # Tool: v3 tool-result part.
        if role == "tool":
            v3_tool_msg = _openai_tool_msg_to_v3(msg)
            if not v3_tool_msg["content"][0].get("toolName"):
                tc_id = msg.get("tool_call_id", "")
                v3_tool_msg["content"][0]["toolName"] = tool_call_name_map.get(tc_id, "")
            prompt.append(v3_tool_msg)
            continue

        # User / assistant.
        if role == "assistant" and msg.get("tool_calls"):
            content_parts = _openai_content_to_v3_parts(msg.get("content"))
            tool_call_parts = [_openai_tool_call_to_v3(tc) for tc in msg["tool_calls"]]
            prompt.append({
                "role": "assistant",
                "content": content_parts + tool_call_parts,
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
            if isinstance(t, dict):
                if "function" in t:
                    fn = t["function"]
                    v3_tools.append({
                        "type": t.get("type", "function"),
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}),
                    })
                elif "name" in t:
                    # Already flat v3 shape — pass through.
                    v3_tools.append(t)

    v3_body: dict = {
        "prompt": prompt,
        "tools": v3_tools,
        "toolChoice": body.get("tool_choice", {"type": "auto"}),
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

    return v3_body


# --------------------------------------------------------------------------- #
# v3 -> OpenAI (non-streaming)
# --------------------------------------------------------------------------- #


# Mapping of v3 finish reasons to OpenAI finish reasons.
_FINISH_REASON_MAP = {
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

    # v3 responses can come as a flat {"content":[...],"finishReason":...}
    # or wrapped in choices[0].
    if "choices" in v3_data:
        choice = v3_data["choices"][0] if v3_data["choices"] else {}
        msg = choice.get("message", {})
        finish_raw = choice.get("finish_reason") or v3_data.get("finishReason")
        usage_data = v3_data.get("usage", {})
    else:
        msg = v3_data
        finish_raw = v3_data.get("finishReason", "stop")
        usage_data = v3_data.get("usage", {})

    content_parts = msg.get("content", [])
    text = ""
    tool_calls: list[dict] = []

    for part in content_parts:
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

    # Some v3 responses also carry top-level tool_calls (OpenAI-style).
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
# v3 streaming -> OpenAI streaming (generator)
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


def v3_stream_to_openai(
    events: list[dict],
    model: str = "",
) -> str:
    """Convert a list of v3 SSE events to an OpenAI SSE stream string.

    Useful for testing / offline conversion. For live HTTP use, the proxy
    server's streaming handler does this incrementally.

    Returns the full SSE text (multiple chunks separated by blank lines).
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    chunks: list[str] = [_sse_chunk(chat_id, model, role="assistant")]
    tool_input_buffers: dict[str, str] = {}
    finish_reason = "stop"

    for event in events:
        etype = event.get("type", "")

        if etype == "text-delta":
            delta = event.get("delta", "")
            if delta:
                chunks.append(_sse_chunk(chat_id, model, delta_text=delta))

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
            chunks.append(f"data: {json.dumps(chunk)}\n\n")
            tool_input_buffers.pop(tool_id, None)

        elif etype == "finish":
            finish_reason = _v3_finish_reason(event.get("finishReason", "stop"))
            usage_data = event.get("usage", {})
            usage = _v3_usage_to_openai(usage_data) if usage_data else None
            chunks.append(_sse_chunk(chat_id, model, finish_reason=finish_reason, usage=usage))
            break

        elif etype in ("error",):
            chunks.append(_sse_chunk(chat_id, model, finish_reason="stop"))
            break

    chunks.append("data: [DONE]\n\n")
    return "".join(chunks)


# --------------------------------------------------------------------------- #
# CLI (optional)
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python converter.py <input.json> [--stream]")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    is_stream = "--stream" in sys.argv
    if is_stream:
        events = data if isinstance(data, list) else [data]
        print(v3_stream_to_openai(events))
    else:
        # Try v3_to_openai first; if it doesn't look like v3, try openai_to_v3.
        has_prompt = "prompt" in data
        if has_prompt:
            # Already v3, pass through (echo)
            print(json.dumps(data, indent=2))
        else:
            # OpenAI -> v3
            print(json.dumps(openai_to_v3(data), indent=2))

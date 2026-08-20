"""v3 gateway response -> OpenAI chat.completion (non-streaming)."""

from __future__ import annotations

import json
import time
import uuid


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

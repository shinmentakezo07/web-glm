"""Full OpenAI chat-completions request -> AI SDK v3 request body."""

from __future__ import annotations

import json

from .parts import (
    _normalize_tool_choice,
    _openai_content_to_v3_parts,
    _openai_response_format_to_v3,
    _openai_tool_call_to_v3,
    _openai_tool_msg_to_v3,
)


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

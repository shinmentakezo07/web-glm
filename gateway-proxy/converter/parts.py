"""Low-level OpenAI -> v3 part translation.

Pure helpers that convert a single message content / tool call / tool
result / tool_choice / response_format into AI SDK v3 shapes. No I/O, no
side effects.
"""

from __future__ import annotations

import json


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

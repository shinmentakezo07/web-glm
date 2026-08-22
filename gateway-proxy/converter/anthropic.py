"""Anthropic Messages API <-> OpenAI/v3 translation (pure, no I/O).

Converts the Anthropic Messages API wire format (used by Claude Code, the
Anthropic Python/TS SDKs, etc.) into the OpenAI chat-completions shape that
``converter.openai_to_v3`` already understands, and converts the OpenAI
response/stream back into the Anthropic shape.

Public API:
    anthropic_to_openai(body)        -> chat-completions dict
    openai_to_anthropic(resp, model) -> Anthropic Message dict
    anthropic_stream_iter(openai_sse) -> async iter of Anthropic SSE events
    count_anthropic_tokens(body)     -> rough token estimate
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator


# --------------------------------------------------------------------------- #
# Anthropic -> OpenAI
# --------------------------------------------------------------------------- #


def _anthropic_content_to_openai(content) -> tuple[str | list, list[dict] | None]:
    """Convert Anthropic message content to OpenAI (content, tool_calls).

    Returns (openai_content, tool_calls_or_None). Text-only content becomes
    a plain string; mixed content stays a list of OpenAI parts. Assistant
    tool_use blocks collect into the tool_calls list.
    """
    if content is None:
        return None, None
    if isinstance(content, str):
        return content, None
    if not isinstance(content, list):
        return str(content), None

    parts: list[dict] = []
    tool_calls: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append({"type": "text", "text": str(block)})
            continue
        btype = block.get("type", "text")
        if btype == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif btype == "tool_use":
            inp = block.get("input", {})
            args = inp if isinstance(inp, str) else json.dumps(inp)
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": args,
                },
            })
        elif btype == "image":
            source = block.get("source", {})
            media_type = source.get("media_type", "image/png")
            data = source.get("data", "")
            url = f"data:{media_type};base64,{data}" if data else ""
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
        elif btype == "tool_result":
            # tool_result only appears in user messages; handled by caller
            pass
        # Unknown block types are silently skipped (forward-compat)

    if not parts and not tool_calls:
        return None, None
    if not parts:
        return None, tool_calls
    # If all parts are plain text, collapse to a string for cleanliness
    if all(p.get("type") == "text" for p in parts):
        return "".join(p.get("text", "") for p in parts), tool_calls
    return parts, tool_calls


def _extract_tool_result(msg: dict) -> tuple[str | None, str, list[dict] | None]:
    """Extract (tool_call_id, content, content_blocks) from a tool_result block.

    ``content`` is the OpenAI tool message string content. ``content_blocks``
    is the list of original Anthropic blocks for text/JSON results.
    """
    tool_use_id = msg.get("tool_use_id", "")
    content = msg.get("content")

    if isinstance(content, str):
        return tool_use_id, content, None
    if isinstance(content, list):
        parts: list[str] = []
        blocks: list[dict] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif part.get("type") == "json":
                    # Anthropic JSON tool result — serialize for OpenAI wire
                    raw = part.get("json", "")
                    try:
                        parts.append(json.dumps(raw) if not isinstance(raw, str) else raw)
                    except (TypeError, ValueError):
                        parts.append(str(raw))
                else:
                    parts.append(json.dumps(part))
        return tool_use_id, "".join(parts), content if blocks is not None else None
    if content is not None:
        return tool_use_id, str(content), None
    return tool_use_id, "", None


def anthropic_to_openai(body: dict) -> dict:
    """Convert an Anthropic Messages API request to OpenAI chat-completions format.

    Translates:
    - system (string or list of blocks) -> system message
    - messages: content blocks -> OpenAI message parts
    - tools: Anthropic tool schema -> OpenAI function tools
    - tool_choice: Anthropic -> OpenAI tool_choice
    - max_tokens -> max_tokens, temperature, top_p, top_k, stop

    Does NOT modify the input body.
    """
    messages: list[dict] = []

    # System prompt: Anthropic sends it top-level, not in messages.
    system = body.get("system")
    if system:
        if isinstance(system, list):
            text_parts = [
                b.get("text", "") for b in system
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            system = "".join(text_parts)
        if isinstance(system, str) and system:
            messages.append({"role": "system", "content": system})

    # Messages
    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content")

        if role == "assistant":
            oai_content, tool_calls = _anthropic_content_to_openai(content)
            entry: dict = {"role": "assistant"}
            if oai_content is not None:
                entry["content"] = oai_content
            else:
                entry["content"] = None
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
            continue

        # user
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        if not isinstance(content, list):
            messages.append({"role": "user", "content": str(content)})
            continue

        # Scan for tool_result blocks — each becomes its own OpenAI tool message.
        # Non-tool_result blocks (text, image) merge into one user message.
        user_parts: list[dict] = []
        for block in content:
            if not isinstance(block, dict):
                user_parts.append({"type": "text", "text": str(block)})
                continue
            if block.get("type") == "tool_result":
                tc_id, tc_content, _ = _extract_tool_result(block)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tc_content,
                })
            else:
                user_parts.append(block)

        if user_parts:
            collapsed, _ = _anthropic_content_to_openai(user_parts)
            messages.append({"role": "user", "content": collapsed})

    # Build OpenAI body
    oai_body: dict = {"messages": messages}

    if "model" in body:
        oai_body["model"] = body["model"]
    if "max_tokens" in body:
        oai_body["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        oai_body["temperature"] = body["temperature"]
    if "top_p" in body:
        oai_body["top_p"] = body["top_p"]
    if "top_k" in body:
        oai_body["top_k"] = body["top_k"]
    if "stop_sequences" in body:
        oai_body["stop"] = body["stop_sequences"]

    # Tools: Anthropic {name, description, input_schema} -> OpenAI function
    if body.get("tools"):
        oai_tools: list[dict] = []
        for tool in body["tools"]:
            if not isinstance(tool, dict):
                continue
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })
        oai_body["tools"] = oai_tools

    # Tool choice: Anthropic shape -> OpenAI shape
    tc = body.get("tool_choice")
    if tc is not None:
        if isinstance(tc, dict):
            tc_type = tc.get("type", "auto")
            if tc_type == "auto":
                oai_body["tool_choice"] = "auto"
            elif tc_type == "any":
                oai_body["tool_choice"] = "required"
            elif tc_type == "tool":
                oai_body["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tc.get("name", "")},
                }
            elif tc_type == "none":
                oai_body["tool_choice"] = "none"

    # Anthropic thinking config -> v3 reasoning label.
    # Claude Code sends {"thinking": {"type": "enabled", "budget_tokens": N}}.
    # The v3 gateway accepts "reasoning" as an opaque string label (fx CLI
    # passes model-catalog-defined values like "low"/"high"/"max" through
    # verbatim). We route every thinking config upstream so the provider
    # decides what to do with it.
    has_explicit_reasoning = "reasoning" in body or "reasoning_effort" in body
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        if not has_explicit_reasoning:
            oai_body["reasoning"] = "enabled"
    elif isinstance(thinking, dict) and thinking.get("type") == "disabled":
        pass  # explicitly disabled — don't send reasoning upstream

    # Explicit reasoning/reasoning_effort from the client always wins over
    # the thinking-derived default.
    if "reasoning" in body:
        oai_body["reasoning"] = body["reasoning"]
    if "reasoning_effort" in body:
        oai_body["reasoning_effort"] = body["reasoning_effort"]

    # Stream flag
    if "stream" in body:
        oai_body["stream"] = body["stream"]

    return oai_body


# --------------------------------------------------------------------------- #
# OpenAI -> Anthropic (non-streaming)
# --------------------------------------------------------------------------- #


def _openai_finish_to_anthropic(reason: str | None) -> str:
    """Map OpenAI finish_reason -> Anthropic stop_reason."""
    if reason is None:
        return "end_turn"
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }.get(reason, "end_turn")


def _openai_usage_to_anthropic(usage: dict) -> dict:
    """Map OpenAI usage -> Anthropic usage shape."""
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    cache_read = 0
    pd = usage.get("prompt_tokens_details") or {}
    if isinstance(pd, dict):
        cache_read = pd.get("cached_tokens", 0)
    result: dict = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cache_read,
    }
    return result


def openai_to_anthropic(openai_resp: dict, model: str = "") -> dict:
    """Convert an OpenAI chat.completion response to the Anthropic Message shape."""
    choices = openai_resp.get("choices", [])
    choice = choices[0] if choices else {}
    msg = choice.get("message", {})
    finish_reason = choice.get("finish_reason")
    usage = openai_resp.get("usage", {})

    content_blocks: list[dict] = []
    reasoning = msg.get("reasoning_content")
    if reasoning:
        content_blocks.append({"type": "thinking", "thinking": reasoning})
    text = msg.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        args_raw = fn.get("arguments", "{}")
        try:
            input_obj = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except (json.JSONDecodeError, TypeError):
            input_obj = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
            "name": fn.get("name", ""),
            "input": input_obj,
        })

    if not content_blocks:
        content_blocks = [{"type": "text", "text": ""}]

    return {
        "id": openai_resp.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model or openai_resp.get("model", ""),
        "stop_reason": _openai_finish_to_anthropic(finish_reason),
        "stop_sequence": None,
        "usage": _openai_usage_to_anthropic(usage),
    }


# --------------------------------------------------------------------------- #
# OpenAI SSE stream -> Anthropic SSE stream
# --------------------------------------------------------------------------- #


class _AnthropicStreamState:
    """Stateful translator: OpenAI chat.completion.chunk stream -> Anthropic SSE."""

    def __init__(self, model: str):
        self.msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.model = model
        self.text_started = False
        self.text_block_index: int | None = None
        self.thinking_started = False
        self.thinking_block_index: int | None = None
        self.next_block_index = 0  # tracks the next available content_block index
        # tool call tracking: index -> {id, name, arguments_buffer, block_index}
        self.tools: dict[int, dict] = {}
        self.started = False
        self.finish_reason: str | None = None
        self.usage: dict = {}

    def _sse(self, obj: dict) -> str:
        return f"event: {obj['type']}\ndata: {json.dumps(obj)}\n\n"

    def start(self) -> str:
        self.started = True
        return self._sse({
            "type": "message_start",
            "message": {
                "id": self.msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            },
        })

    def thinking_delta(self, delta: str) -> str:
        out = ""
        if not self.thinking_started:
            self.thinking_started = True
            self.thinking_block_index = self.next_block_index
            self.next_block_index += 1
            out += self._sse({
                "type": "content_block_start",
                "index": self.thinking_block_index,
                "content_block": {"type": "thinking", "thinking": ""},
            })
        out += self._sse({
            "type": "content_block_delta",
            "index": self.thinking_block_index,
            "delta": {"type": "thinking_delta", "thinking": delta},
        })
        return out

    def text_delta(self, delta: str) -> str:
        out = ""
        # If a thinking block is open, close it before starting the text block
        # (Anthropic protocol: content blocks are sequential, not interleaved).
        if self.thinking_started:
            out += self._sse({
                "type": "content_block_stop",
                "index": self.thinking_block_index,
            })
            self.thinking_started = False
        if not self.text_started:
            self.text_started = True
            self.text_block_index = self.next_block_index
            self.next_block_index += 1
            out += self._sse({
                "type": "content_block_start",
                "index": self.text_block_index,
                "content_block": {"type": "text", "text": ""},
            })
        out += self._sse({
            "type": "content_block_delta",
            "index": self.text_block_index,
            "delta": {"type": "text_delta", "text": delta},
        })
        return out

    def tool_call(self, tc: dict) -> str:
        """Handle a streaming OpenAI tool_call delta.

        OpenAI sends incremental arguments across multiple deltas for the same
        index; Anthropic expects content_block_start, input_json_delta, then
        content_block_stop at the end.
        """
        idx = tc.get("index", 0)
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args_delta = fn.get("arguments", "")
        call_id = tc.get("id", "")

        out = ""
        # If a thinking block is open, close it before starting the tool block.
        if self.thinking_started:
            out += self._sse({
                "type": "content_block_stop",
                "index": self.thinking_block_index,
            })
            self.thinking_started = False
        if idx not in self.tools:
            tool_index = self.next_block_index
            self.next_block_index += 1
            self.tools[idx] = {
                "id": call_id or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": name,
                "block_index": tool_index,
                "args_buffer": "",
            }
            out += self._sse({
                "type": "content_block_start",
                "index": tool_index,
                "content_block": {
                    "type": "tool_use",
                    "id": self.tools[idx]["id"],
                    "name": name,
                    "input": {},
                },
            })
        else:
            if name:
                self.tools[idx]["name"] = name
            if call_id:
                self.tools[idx]["id"] = call_id

        if args_delta:
            self.tools[idx]["args_buffer"] += args_delta
            out += self._sse({
                "type": "content_block_delta",
                "index": self.tools[idx]["block_index"],
                "delta": {"type": "input_json_delta", "partial_json": args_delta},
            })
        return out

    def finish(self, finish_reason: str | None, usage: dict | None) -> str:
        out = ""
        # Close open thinking block
        if self.thinking_started:
            out += self._sse({
                "type": "content_block_stop",
                "index": self.thinking_block_index,
            })
        # Close open text block
        if self.text_started:
            out += self._sse({
                "type": "content_block_stop",
                "index": self.text_block_index,
            })
        # Close all tool blocks
        for entry in self.tools.values():
            out += self._sse({
                "type": "content_block_stop",
                "index": entry["block_index"],
            })

        stop_reason = _openai_finish_to_anthropic(finish_reason)
        if usage:
            self.usage = _openai_usage_to_anthropic(usage)

        out += self._sse({
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None,
            },
            "usage": self.usage if self.usage else {
                "input_tokens": 0,
                "output_tokens": 0,
            },
        })
        out += self._sse({"type": "message_stop"})
        return out

    def failed(self, message: str) -> str:
        return self._sse({
            "type": "error",
            "error": {"type": "api_error", "message": message},
        })


def _parse_openai_sse_chunk(chunk_str: str) -> dict | None:
    """Parse an OpenAI SSE data line to a dict, or None for [DONE]/empty."""
    stripped = chunk_str.strip()
    if not stripped:
        return None
    if not stripped.startswith("data:"):
        return None
    payload = stripped[len("data:"):].strip()
    if not payload or payload in ("[DONE]", "DONE"):
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def openai_chunk_to_anthropic_sse(
    chunk_str: str,
    state: _AnthropicStreamState,
) -> str | None:
    """Translate one OpenAI SSE data line to Anthropic SSE events (or None).

    Returns a string of concatenated ``event:\\n data:\\n\\n`` blocks, or None
    when there is nothing to emit (e.g. a non-data line or [DONE]).
    """
    chunk = _parse_openai_sse_chunk(chunk_str)
    if chunk is None:
        return None

    # Stream error from upstream
    if chunk.get("type") == "error":
        err = chunk.get("error", {})
        msg = err.get("message", "upstream error") if isinstance(err, dict) else str(err)
        return state.failed(msg)

    choices = chunk.get("choices") or []
    choice = choices[0] if choices else {}
    delta = choice.get("delta") or {}
    finish_reason = choice.get("finish_reason")

    out = ""
    if not state.started:
        out += state.start()
    if delta.get("role") == "assistant":
        pass  # role delta — no Anthropic equivalent, already in message_start
    if delta.get("content"):
        out += state.text_delta(delta["content"])
    if delta.get("reasoning_content"):
        out += state.thinking_delta(delta["reasoning_content"])
    for tc in delta.get("tool_calls") or []:
        out += state.tool_call(tc)
    if finish_reason is not None:
        out += state.finish(finish_reason, chunk.get("usage"))
    return out or None


async def anthropic_stream_iter(
    openai_sse: AsyncIterator[str],
    model: str = "",
) -> AsyncIterator[str]:
    """Translate an OpenAI SSE stream into Anthropic SSE events.

    Consumes OpenAI chat.completion.chunk SSE strings, yields Anthropic
    message event strings. Used by the server's streaming handler.
    """
    state = _AnthropicStreamState(model)
    async for chunk_str in openai_sse:
        out = openai_chunk_to_anthropic_sse(chunk_str, state)
        if out:
            yield out
    # If the stream ended without a finish chunk, emit a graceful stop
    if not state.started:
        yield state.start()
    if state.finish_reason is None and not state.usage:
        yield state.finish("stop", None)


# --------------------------------------------------------------------------- #
# Token counting
# --------------------------------------------------------------------------- #


def count_anthropic_tokens(body: dict) -> int:
    """Rough token estimate for an Anthropic Messages API request.

    Uses a simple character/4 heuristic (no tokenizer dependency). This is
    the same approach the Anthropic SDK uses when no server-side count is
    available — accurate enough for planning, not for billing.
    """
    total_chars = 0

    system = body.get("system")
    if system:
        if isinstance(system, list):
            for block in system:
                if isinstance(block, dict):
                    total_chars += len(block.get("text", ""))
        else:
            total_chars += len(str(system))

    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    total_chars += len(str(block))
                    continue
                btype = block.get("type", "text")
                if btype == "text":
                    total_chars += len(block.get("text", ""))
                elif btype == "tool_use":
                    inp = block.get("input", {})
                    try:
                        total_chars += len(json.dumps(inp))
                    except (TypeError, ValueError):
                        pass
                elif btype == "tool_result":
                    tc_content = block.get("content")
                    if isinstance(tc_content, str):
                        total_chars += len(tc_content)
                    elif isinstance(tc_content, list):
                        for part in tc_content:
                            if isinstance(part, dict):
                                total_chars += len(part.get("text", ""))

    # Tools contribute tokens too
    for tool in body.get("tools", []):
        if isinstance(tool, dict):
            total_chars += len(tool.get("description", ""))
            schema = tool.get("input_schema", {})
            try:
                total_chars += len(json.dumps(schema))
            except (TypeError, ValueError):
                pass

    return (total_chars + 3) // 4

"""Responses API (OpenAI new format) <-> v3.

Translates OpenAI Responses API input items into chat-completions messages,
re-shapes OpenAI chat.completion responses into the Responses API shape,
and streams OpenAI chat.completion chunks as Responses SSE events.
"""

from __future__ import annotations

import json
import time
import uuid

from .response import _v3_finish_reason, _v3_usage_to_openai


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
            # Preserve system/developer roles; default anything else to user.
            messages.append({
                "role": role if role in ("system", "developer", "assistant") else "user",
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
        self.msg_started = False  # message item created lazily on first text
        self.tool_ids: dict[str, str] = {}   # call_id -> fc_ item id
        self.tool_index: dict[str, int] = {}  # call_id -> output index
        self.text_parts: list[str] = []
        self.tool_calls: list[dict] = []
        self.started = False
        self.usage: dict = {}

    def _sse(self, obj: dict) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    def _output_index_for_tools(self) -> int:
        """Output index for the next tool item: 0 if no message, 1 if message."""
        return 1 if self.msg_started else 0

    def start(self) -> str:
        """Emit response.created. The message item is created lazily in
        text_delta() so tool-only responses don't emit a dangling empty
        message item."""
        self.started = True
        response = {
            "id": self.response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "in_progress",
            "model": self.model,
            "output": [],
        }
        return self._sse({"type": "response.created", "response": response})

    def _ensure_message_item(self) -> str:
        """Lazily create the message item (index 0) on first text delta.
        Returns the SSE events for item/part creation, or "" if already started."""
        if self.msg_started:
            return ""
        self.msg_started = True
        self.msg_id = f"msg_{uuid.uuid4().hex[:16]}"
        out = self._sse({
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
        out = self._ensure_message_item()
        self.text_parts.append(delta)
        out += self._sse({
            "type": "response.output_text.delta",
            "item_id": self.msg_id,
            "output_index": 0,
            "delta": delta,
        })
        return out

    def tool_call(self, call: dict) -> str:
        """Emit output_item.added + function_call_arguments.delta for a
        single-assignment OpenAI tool-call chunk."""
        call_id = call.get("id", "")
        fn = call.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", "")
        base_index = self._output_index_for_tools()
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
                "output_index": base_index + idx,
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
                "output_index": base_index + idx,
                "delta": args,
            })
        return out

    def finish(self, finish_reason: str | None, usage: dict | None) -> str:
        """Emit output_text.done / output_item.done for each item + response.completed."""
        out = ""
        full_text = "".join(self.text_parts)
        base_index = self._output_index_for_tools()
        # Only close the message item if it was created (text was received).
        if self.msg_started:
            out += self._sse({
                "type": "response.output_text.done",
                "item_id": self.msg_id,
                "output_index": 0,
                "text": full_text,
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
                        "text": full_text,
                        "annotations": [],
                    }],
                },
            })
        for i, tc in enumerate(self.tool_calls):
            item = dict(tc)
            item["status"] = "completed"
            out += self._sse({
                "type": "response.output_item.done",
                "output_index": base_index + i,
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

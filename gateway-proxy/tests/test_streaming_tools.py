"""Tool-call robustness: name backfill from tool-input-start, id synthesis,
duplicate suppression, and delta buffering (fx wire-pattern parity)."""

import json

from converter.streaming import (
    _StreamState,
    _process_stream_event,
    v3_sse_stream_to_openai,
    v3_stream_to_openai,
)


def _chunks(text: str) -> list[dict]:
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if block.startswith("data: ") and block[6:] != "[DONE]":
            out.append(json.loads(block[6:]))
    return out


def _tool_deltas(chunks: list[dict]) -> list[dict]:
    found = []
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            found.extend(choice.get("delta", {}).get("tool_calls", []))
    return found


def test_tool_name_backfilled_from_tool_input_start():
    """fx pattern: tool-call arrives without toolName; the start event has it."""
    events = [
        {"type": "tool-input-start", "id": "c1", "toolName": "read_file"},
        {"type": "tool-input-delta", "id": "c1", "delta": "{\"path\":"},
        {"type": "tool-input-delta", "id": "c1", "delta": "\"x\"}"},
        {"type": "tool-call", "toolCallId": "c1"},  # no toolName, no input
    ]
    calls = _tool_deltas(_chunks(v3_stream_to_openai(events)))
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "x"}


def test_missing_tool_call_id_gets_synthesized():
    events = [
        {"type": "tool-input-start", "id": "c2", "toolName": "search"},
        {"type": "tool-call"},  # nothing at all to correlate
    ]
    state = _StreamState("chatcmpl-t", "m")
    out: list[str] = []
    for ev in events:
        out.extend(_process_stream_event(state, ev))
    calls = _tool_deltas(_chunks("".join(out)))
    assert len(calls) == 1
    assert calls[0]["id"].startswith("call_")


def test_duplicate_tool_call_events_emit_once():
    call = {"type": "tool-call", "toolCallId": "dup1", "toolName": "f",
            "input": {"a": 1}}
    state = _StreamState("chatcmpl-t", "m")
    first = _process_stream_event(state, dict(call))
    second = _process_stream_event(state, dict(call))
    assert len(first) == 1
    assert second == []


def test_empty_arguments_become_valid_json_object():
    events = [{"type": "tool-call", "toolCallId": "e1", "toolName": "ping"}]
    calls = _tool_deltas(_chunks(v3_stream_to_openai(events)))
    assert calls[0]["function"]["arguments"] == "{}"
    # Must be parseable by OpenAI clients.
    json.loads(calls[0]["function"]["arguments"])


def test_non_streaming_collector_merges_names_and_buffered_args():
    events = [
        {"type": "text-delta", "delta": "hello"},
        {"type": "tool-input-start", "id": "c9", "toolName": "get_weather"},
        {"type": "tool-input-delta", "id": "c9", "delta": "{\"city\":\"Par"},
        {"type": "tool-input-delta", "id": "c9", "delta": "is\"}"},
        {"type": "tool-call", "toolCallId": "c9"},  # bare consolidation
        {"type": "finish", "finishReason": {"unified": "tool-calls"}},
    ]
    result = v3_sse_stream_to_openai(iter(events), model="m")
    calls = result["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["id"] == "c9"
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}
    assert result["choices"][0]["finish_reason"] == "tool_calls"


def test_non_streaming_collector_suppresses_duplicates_and_mints_ids():
    events = [
        {"type": "tool-call", "toolName": "f", "input": {"x": 1}},
        {"type": "tool-call", "toolName": "f", "input": {"x": 1}},  # dup, no id
        {"type": "finish", "finishReason": {"unified": "tool-calls"}},
    ]
    result = v3_sse_stream_to_openai(iter(events), model="m")
    calls = result["choices"][0]["message"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["id"].startswith("call_")
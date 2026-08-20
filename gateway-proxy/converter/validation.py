"""Client-side tool-history validation (modeled on fx's validateToolMessageHistory)."""

from __future__ import annotations

import json


# --------------------------------------------------------------------------- #
# Client-side tool-history validation (modeled on fx's validateToolMessageHistory)
# --------------------------------------------------------------------------- #


def _reject_duplicate_keys(pairs: list[tuple]) -> dict:
    """object_pairs_hook that raises ValueError on duplicate keys (fx parity)."""
    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError(f"duplicate key: {key}")
        seen.add(key)
    return dict(pairs)


def _parse_tool_args(args) -> str | None:
    """Return an error message if tool args are not valid JSON, else None."""
    if isinstance(args, dict):
        return None
    if isinstance(args, str):
        if not args.strip():
            return "tool call arguments are empty"
        try:
            json.loads(args, object_pairs_hook=_reject_duplicate_keys)
        except ValueError:
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

        call_names: dict[str, str] = {
            call.get("id", ""): call.get("function", {}).get("name", "")
            for call in calls
        }

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
            result_name = result.get("name")
            expected_name = call_names.get(rid, "")
            if result_name and expected_name and result_name != expected_name:
                return (f"tool result for {rid} names tool {result_name!r} "
                        f"but call {rid} is {expected_name!r}")
            if result.get("content") is None:
                return f"tool result for {rid} has no content"

        if len(matched_ids) != len(calls):
            return (f"assistant tool calls at index {i} are not all paired "
                    f"({len(matched_ids)}/{len(calls)} results)")

        i = j  # skip the consumed result block, continue after it

    return None

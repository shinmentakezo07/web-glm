"""Unit tests for converter.validation — client-side tool-history validation."""

from converter.validation import validate_tool_history


# =====================================================================
# validate_tool_history
# =====================================================================


class TestValidateToolHistory:
    def _call(self, call_id="call_1", name="calc", args="{}"):
        return {"id": call_id, "type": "function",
                "function": {"name": name, "arguments": args}}

    def test_valid_single_round(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [self._call()]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
            {"role": "user", "content": "thanks"},
        ]
        assert validate_tool_history(messages) is None

    def test_valid_multi_call_parallel(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                self._call("call_1", "calc"),
                self._call("call_2", "search"),
            ]},
            {"role": "tool", "tool_call_id": "call_2", "content": "b"},
            {"role": "tool", "tool_call_id": "call_1", "content": "a"},
        ]
        assert validate_tool_history(messages) is None

    def test_valid_two_rounds(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [self._call("call_1")]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
            {"role": "assistant", "content": None, "tool_calls": [self._call("call_2", "step2")]},
            {"role": "tool", "tool_call_id": "call_2", "content": "x"},
        ]
        assert validate_tool_history(messages) is None

    def test_no_tools_is_valid(self):
        assert validate_tool_history([{"role": "user", "content": "hi"}]) is None

    def test_orphan_tool_result_rejected(self):
        messages = [{"role": "user", "content": "go"}, {"role": "tool", "tool_call_id": "x", "content": "y"}]
        err = validate_tool_history(messages)
        assert err is not None and "no preceding assistant" in err

    def test_missing_results_rejected(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [self._call()]},
        ]
        err = validate_tool_history(messages)
        assert err is not None and "no matching tool results" in err

    def test_unmatched_result_id_rejected(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [self._call("call_1")]},
            {"role": "tool", "tool_call_id": "other", "content": "y"},
        ]
        err = validate_tool_history(messages)
        assert err is not None and "unknown tool call" in err

    def test_duplicate_tool_call_id_rejected(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                self._call("call_1"), self._call("call_1"),
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "a"},
            {"role": "tool", "tool_call_id": "call_1", "content": "b"},
        ]
        err = validate_tool_history(messages)
        assert err is not None and "duplicate tool call id" in err

    def test_invalid_json_args_rejected(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                self._call("call_1", args="not json"),
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "y"},
        ]
        err = validate_tool_history(messages)
        assert err is not None and "not valid JSON" in err

    def test_empty_arguments_rejected(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                self._call("call_1", args=""),
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "y"},
        ]
        err = validate_tool_history(messages)
        assert err is not None and "empty" in err

    def test_missing_tool_name_rejected(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [self._call(name="")]},
            {"role": "tool", "tool_call_id": "call_1", "content": "y"},
        ]
        err = validate_tool_history(messages)
        assert err is not None and "missing a function name" in err


class TestValidationFixes:
    def _call(self, call_id="call_1", name="calc", args="{}"):
        return {"id": call_id, "type": "function",
                "function": {"name": name, "arguments": args}}

    def test_duplicate_key_args_rejected(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                self._call(args='{"a":1,"a":2}'),
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "y"},
        ]
        err = validate_tool_history(messages)
        assert err is not None and "not valid JSON" in err

    def test_nested_duplicate_key_args_rejected(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                self._call(args='{"a":{"b":1,"b":2}}'),
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "y"},
        ]
        err = validate_tool_history(messages)
        assert err is not None and "not valid JSON" in err

    def test_result_name_mismatch_rejected(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [self._call()]},
            {"role": "tool", "tool_call_id": "call_1", "name": "wrong_tool", "content": "y"},
        ]
        err = validate_tool_history(messages)
        assert err is not None and "wrong_tool" in err

    def test_result_name_match_accepted(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [self._call()]},
            {"role": "tool", "tool_call_id": "call_1", "name": "calc", "content": "y"},
        ]
        assert validate_tool_history(messages) is None

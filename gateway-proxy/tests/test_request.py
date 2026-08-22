"""Unit tests for converter.request — full OpenAI chat-completions -> v3."""

from converter.request import openai_to_v3


# =====================================================================
# openai_to_v3 full conversion
# =====================================================================


class TestOpenAIToV3:
    def test_basic_user_message(self):
        body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        result = openai_to_v3(body)
        assert result["prompt"][0] == {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        assert result["tools"] == []
        assert result["toolChoice"] == {"type": "auto"}

    def test_system_message_flattened(self):
        body = {"messages": [{"role": "system", "content": "be nice"}]}
        result = openai_to_v3(body)
        assert result["prompt"][0] == {"role": "system", "content": "be nice"}

    def test_system_multi_part(self):
        body = {"messages": [{"role": "system", "content": [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]}]}
        result = openai_to_v3(body)
        assert result["prompt"][0]["content"] == "ab"

    def test_tools_flattened_with_inputSchema(self):
        body = {
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{
                "type": "function",
                "function": {"name": "read_file", "description": "reads", "parameters": {"type": "object"}},
            }],
        }
        result = openai_to_v3(body)
        assert result["tools"] == [{
            "type": "function",
            "name": "read_file",
            "description": "reads",
            "inputSchema": {"type": "object"},
        }]

    def test_assistant_tool_calls_are_content_parts(self):
        body = {
            "messages": [
                {"role": "user", "content": "run"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "calc", "arguments": '{"a":1}'},
                    }],
                },
            ],
        }
        result = openai_to_v3(body)
        assistant_msg = result["prompt"][1]
        assert "toolCalls" not in assistant_msg
        assert assistant_msg["content"] == [
            {"type": "text", "text": ""},
            {"type": "tool-call", "toolCallId": "call_1", "toolName": "calc", "input": {"a": 1}},
        ]

    def test_assistant_tool_call_with_text_content(self):
        body = {
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "let me check",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }],
                },
            ],
        }
        result = openai_to_v3(body)
        parts = result["prompt"][1]["content"]
        assert parts[0] == {"type": "text", "text": "let me check"}
        assert parts[1]["type"] == "tool-call"

    def test_tool_result_name_backfilled_from_prior_assistant(self):
        body = {
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "my_tool", "arguments": "{}"},
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "result",
                },
            ],
        }
        result = openai_to_v3(body)
        tool_msg = result["prompt"][2]
        part = tool_msg["content"][0]
        assert part["type"] == "tool-result"
        assert part["toolName"] == "my_tool"
        assert part["output"] == {"type": "text", "value": "result"}

    def test_tool_result_without_prior_call_defaults_unknown(self):
        body = {"messages": [{"role": "tool", "tool_call_id": "x", "content": "y"}]}
        result = openai_to_v3(body)
        assert result["prompt"][0]["content"][0]["toolName"] == "unknown"

    def test_param_passthrough(self):
        body = {"messages": [{"role": "user", "content": "x"}], "temperature": 0.7, "max_tokens": 50, "top_p": 0.9, "stop": "end"}
        result = openai_to_v3(body)
        assert result["temperature"] == 0.7
        assert result["maxOutputTokens"] == 50
        assert result["topP"] == 0.9
        assert result["stopSequences"] == ["end"]

    def test_top_k_passthrough(self):
        body = {"messages": [{"role": "user", "content": "x"}], "top_k": 40}
        result = openai_to_v3(body)
        assert result["topK"] == 40

    def test_stop_string_to_list(self):
        body = {"messages": [{"role": "user", "content": "x"}], "stop": "stop_word"}
        result = openai_to_v3(body)
        assert result["stopSequences"] == ["stop_word"]

    def test_maxOutputTokens_passed_through(self):
        body = {"messages": [{"role": "user", "content": "x"}], "maxOutputTokens": 100}
        result = openai_to_v3(body)
        assert result["maxOutputTokens"] == 100

    def test_tool_choice_normalization_via_body(self):
        for raw, expected in [
            ("auto", {"type": "auto"}),
            ("none", {"type": "none"}),
            ("required", {"type": "required"}),
            ({"type": "function", "function": {"name": "t"}}, {"type": "tool", "toolName": "t"}),
        ]:
            body = {"messages": [{"role": "user", "content": "x"}], "tool_choice": raw}
            assert openai_to_v3(body)["toolChoice"] == expected

    def test_response_format_json_object(self):
        body = {"messages": [{"role": "user", "content": "x"}], "response_format": {"type": "json_object"}}
        assert openai_to_v3(body)["responseFormat"] == {
            "type": "json", "name": "", "description": "", "schema": {},
        }

    def test_response_format_json_schema(self):
        body = {"messages": [{"role": "user", "content": "x"}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "book", "description": "a book", "schema": {"type": "object"}}}}
        assert openai_to_v3(body)["responseFormat"] == {
            "type": "json", "name": "book", "description": "a book", "schema": {"type": "object"},
        }

    def test_response_format_absent(self):
        body = {"messages": [{"role": "user", "content": "x"}]}
        assert "responseFormat" not in openai_to_v3(body)

    def test_reasoning_passthrough(self):
        body = {"messages": [{"role": "user", "content": "x"}], "reasoning": {"effort": "high"}}
        assert openai_to_v3(body)["reasoning"] == "high"

    def test_reasoning_effort_mapping(self):
        body = {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "minimal"}
        assert openai_to_v3(body)["reasoning"] == "minimal"

    def test_provider_options_passthrough(self):
        body = {"messages": [{"role": "user", "content": "x"}],
                "providerOptions": {"gateway": {"cache": "auto"}}}
        assert openai_to_v3(body)["providerOptions"] == {"gateway": {"cache": "auto"}}

    def test_provider_options_empty_omitted(self):
        body = {"messages": [{"role": "user", "content": "x"}], "providerOptions": {}}
        assert "providerOptions" not in openai_to_v3(body)

    def test_input_body_not_mutated(self):
        import copy
        body = {"messages": [{"role": "user", "content": "x"}], "tool_choice": "auto"}
        snapshot = copy.deepcopy(body)
        openai_to_v3(body)
        assert body == snapshot


# =====================================================================
# Body user-agent scoping (fx only sends it for zai/glm-5.2)
# =====================================================================


class TestBodyUserAgentScoping:
    def test_glm52_includes_user_agent(self):
        body = {"model": "zai/glm-5.2", "messages": [{"role": "user", "content": "x"}]}
        assert openai_to_v3(body)["headers"] == {"user-agent": "fx/0.0.5"}

    def test_other_model_omits_user_agent(self):
        body = {"model": "anthropic/claude", "messages": [{"role": "user", "content": "x"}]}
        assert "headers" not in openai_to_v3(body)

    def test_override_model_set(self):
        body = {"model": "anthropic/claude", "messages": [{"role": "user", "content": "x"}]}
        result = openai_to_v3(body, product_user_agent_models=frozenset({"anthropic/claude"}))
        assert result["headers"] == {"user-agent": "fx/0.0.5"}

    def test_all_models_when_none(self):
        body = {"model": "anthropic/claude", "messages": [{"role": "user", "content": "x"}]}
        result = openai_to_v3(body, product_user_agent_models=None)
        assert result["headers"] == {"user-agent": "fx/0.0.5"}

    def test_no_models_when_empty(self):
        body = {"model": "zai/glm-5.2", "messages": [{"role": "user", "content": "x"}]}
        result = openai_to_v3(body, product_user_agent_models=frozenset())
        assert "headers" not in result

    def test_custom_user_agent(self):
        body = {"model": "zai/glm-5.2", "messages": [{"role": "user", "content": "x"}]}
        result = openai_to_v3(body, product_user_agent="fx/0.0.9")
        assert result["headers"] == {"user-agent": "fx/0.0.9"}


# =====================================================================
# Reasoning normalization (v3 uses a string label)
# =====================================================================


class TestReasoningNormalization:
    def test_reasoning_string_passthrough(self):
        body = {"messages": [{"role": "user", "content": "x"}], "reasoning": "high"}
        assert openai_to_v3(body)["reasoning"] == "high"

    def test_reasoning_effort_dict_normalized_to_string(self):
        body = {"messages": [{"role": "user", "content": "x"}], "reasoning": {"effort": "high"}}
        assert openai_to_v3(body)["reasoning"] == "high"

    def test_reasoning_effort_param_unchanged(self):
        body = {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "minimal"}
        assert openai_to_v3(body)["reasoning"] == "minimal"

"""Unit tests for converter.parts — low-level OpenAI -> v3 part translation.

Wire-format assertions ensure converter and server always agree on the
exact shape sent to / received from the Vercel AI Gateway (fx wire format:
tool calls as content parts with raw-JSON `input`).
"""

from converter.parts import (
    _image_url_to_v3_part,
    _normalize_tool_choice,
    _openai_content_to_v3_parts,
    _openai_tool_call_to_v3,
    _openai_tool_msg_to_v3,
)


# =====================================================================
# Content-part conversion
# =====================================================================


class TestContentParts:
    def test_none_becomes_empty_text(self):
        assert _openai_content_to_v3_parts(None) == [{"type": "text", "text": ""}]

    def test_empty_string(self):
        assert _openai_content_to_v3_parts("") == [{"type": "text", "text": ""}]

    def test_plain_string(self):
        assert _openai_content_to_v3_parts("hello") == [{"type": "text", "text": "hello"}]

    def test_list_of_strings(self):
        assert _openai_content_to_v3_parts(["a", "b"]) == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]

    def test_mixed_list_with_image(self):
        parts = _openai_content_to_v3_parts([
            {"type": "text", "text": "see image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ])
        assert parts == [
            {"type": "text", "text": "see image"},
            {"type": "file", "mediaType": "image/png", "data": "abc"},
        ]


class TestImageParts:
    def test_data_url_becomes_file_part(self):
        part = _image_url_to_v3_part("data:image/png;base64,iVBORw0KGgo=")
        assert part == {"type": "file", "mediaType": "image/png", "data": "iVBORw0KGgo="}

    def test_data_url_without_mime_defaults_octet_stream(self):
        part = _image_url_to_v3_part("data:;base64,AAAA")
        assert part == {"type": "file", "mediaType": "application/octet-stream", "data": "AAAA"}

    def test_remote_url_stays_image_part(self):
        part = _image_url_to_v3_part("https://example.com/a.png")
        assert part == {"type": "image", "image": "https://example.com/a.png"}

    def test_content_to_v3_parts_maps_image_url_to_file_part(self):
        parts = _openai_content_to_v3_parts([
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
        ])
        assert parts == [{"type": "file", "mediaType": "image/jpeg", "data": "AAAA"}]


# =====================================================================
# Tool-call conversion (fx wire format: content parts, raw JSON input)
# =====================================================================


class TestToolCallConversion:
    def test_string_args_parsed_to_raw_object(self):
        tc = {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"path":"x"}'}}
        result = _openai_tool_call_to_v3(tc)
        assert result == {
            "type": "tool-call",
            "toolCallId": "call_1",
            "toolName": "read",
            "input": {"path": "x"},
        }

    def test_object_args_passed_as_object(self):
        tc = {"id": "call_2", "type": "function", "function": {"name": "write", "arguments": {"path": "x", "content": "y"}}}
        result = _openai_tool_call_to_v3(tc)
        assert result["input"] == {"path": "x", "content": "y"}
        assert isinstance(result["input"], dict)

    def test_invalid_args_defaults_to_empty_object(self):
        tc = {"id": "call_3", "type": "function", "function": {"name": "x", "arguments": None}}
        result = _openai_tool_call_to_v3(tc)
        assert result["input"] == {}

    def test_missing_type_treated_as_function(self):
        tc = {"id": "call_4", "function": {"name": "f", "arguments": "{}"}}
        result = _openai_tool_call_to_v3(tc)
        assert result["type"] == "tool-call"
        assert result["toolName"] == "f"


class TestToolMsgConversion:
    def test_string_output(self):
        msg = {"role": "tool", "tool_call_id": "call_1", "content": "result text"}
        result = _openai_tool_msg_to_v3(msg)
        assert result["role"] == "tool"
        assert result["content"] == [{
            "type": "tool-result",
            "toolCallId": "call_1",
            "toolName": "unknown",
            "output": {"type": "text", "value": "result text"},
        }]

    def test_default_tool_name_unknown(self):
        msg = {"role": "tool", "tool_call_id": "call_1", "content": "x"}
        assert _openai_tool_msg_to_v3(msg)["content"][0]["toolName"] == "unknown"

    def test_name_from_message(self):
        msg = {"role": "tool", "tool_call_id": "call_1", "name": "my_tool", "content": "x"}
        assert _openai_tool_msg_to_v3(msg)["content"][0]["toolName"] == "my_tool"

    def test_text_part_list(self):
        msg = {"role": "tool", "tool_call_id": "call_1",
               "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
        result = _openai_tool_msg_to_v3(msg)
        assert result["content"][0]["output"]["value"] == "ab"


class TestNormalizeToolChoice:
    def test_string_auto(self):
        assert _normalize_tool_choice("auto") == {"type": "auto"}

    def test_string_none(self):
        assert _normalize_tool_choice("none") == {"type": "none"}

    def test_string_required(self):
        assert _normalize_tool_choice("required") == {"type": "required"}

    def test_unknown_string_defaults_auto(self):
        assert _normalize_tool_choice("bogus") == {"type": "auto"}

    def test_function_shape_to_tool(self):
        assert _normalize_tool_choice({"type": "function", "function": {"name": "my_tool"}}) == {
            "type": "tool", "toolName": "my_tool",
        }

    def test_v3_object_passthrough(self):
        assert _normalize_tool_choice({"type": "auto"}) == {"type": "auto"}
        assert _normalize_tool_choice({"type": "tool", "toolName": "x"}) == {"type": "tool", "toolName": "x"}

    def test_none_defaults_auto(self):
        assert _normalize_tool_choice(None) == {"type": "auto"}

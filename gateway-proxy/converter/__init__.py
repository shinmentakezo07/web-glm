"""OpenAI <-> AI SDK v3 format converter (package).

Re-exports the public API so the documented library pattern keeps working:

    from converter import openai_to_v3, v3_to_openai, validate_tool_history
"""

from .parts import (
    _image_url_to_v3_part,
    _normalize_tool_choice,
    _openai_content_to_v3_parts,
    _openai_response_format_to_v3,
    _openai_tool_call_to_v3,
    _openai_tool_msg_to_v3,
)
from .request import openai_to_v3
from .validation import validate_tool_history
from .response import _FINISH_REASON_MAP, _v3_finish_reason, _v3_usage_to_openai, v3_to_openai
from .streaming import (
    _process_stream_event,
    _sse_chunk,
    _StreamState,
    v3_sse_stream_to_openai,
    v3_stream_iter,
    v3_stream_to_openai,
)
from .responses import (
    _ResponsesStreamState,
    openai_chunk_to_responses_sse,
    openai_to_responses,
    responses_input_to_messages,
    v3_stream_to_responses_sse,
)

__all__ = [
    "_image_url_to_v3_part", "_normalize_tool_choice", "_openai_content_to_v3_parts",
    "_openai_response_format_to_v3", "_openai_tool_call_to_v3",
    "_openai_tool_msg_to_v3", "openai_to_v3", "validate_tool_history",
    "_FINISH_REASON_MAP", "_v3_finish_reason", "_v3_usage_to_openai",
    "v3_to_openai", "_process_stream_event", "_sse_chunk", "_StreamState",
    "v3_sse_stream_to_openai", "v3_stream_iter", "v3_stream_to_openai",
    "_ResponsesStreamState", "openai_chunk_to_responses_sse",
    "openai_to_responses", "responses_input_to_messages",
    "v3_stream_to_responses_sse",
]

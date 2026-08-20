# Converter Package Refactor + fx Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `gateway-proxy/converter.py` into a `converter/` package, add new conversion logic matching the current `vercel-labs/fx` wire format, and update docs.

**Architecture:** Pure conversion logic moves into a `converter/` package with one module per responsibility (`parts`, `request`, `validation`, `response`, `streaming`, `responses`), re-exported through `__init__.py` so the documented `from converter import ...` pattern keeps working. `server.py` stays the HTTP transport and imports from the explicit submodules. Behavior changes are layered on one task at a time, each TDD with tests split to mirror the package.

**Tech Stack:** Python 3.12, pytest, FastAPI + httpx (unchanged), uv.

## Global Constraints

- All commands run from `gateway-proxy/`: `uv sync`, `uv run pytest tests/`.
- `converter/` modules are pure: no network I/O, no env reads, no side effects. Env-derived configuration lives in `server.py` and is passed in as parameters.
- `test_proxy.py` is a live-server smoke test (hardcodes `http://localhost:8799`). Never run it with pytest; do not modify it.
- Never commit `.env` or any `AI_GATEWAY_API_KEY` / `PROXY_API_KEY` value.
- Do not modify `baicode_install.sh` (repo root, unrelated third-party script).
- Do not commit or alter untracked user files (`CLAUDE.md`, `baicode_install.sh`).
- Protocol invariants (from CLAUDE.md) must never be violated: `ai-language-model-streaming: true` always sent upstream; body always contains `tools: []` and `toolChoice`; `system` content is a string, `user`/`assistant`/`tool` content is an array of parts; `_v3_headers()` in `server.py` stays the single source of truth for fx headers; HTTP/2 stays enabled.
- Commit message style follows repo history: `feat:`, `fix:`, `docs:`, `refactor:` prefixes, lowercase subject.

---

### Task 1: Split converter.py into the converter/ package (mechanical move, zero behavior change)

**Files:**
- Create: `gateway-proxy/converter/__init__.py`
- Create: `gateway-proxy/converter/__main__.py`
- Create: `gateway-proxy/converter/parts.py`
- Create: `gateway-proxy/converter/request.py`
- Create: `gateway-proxy/converter/validation.py`
- Create: `gateway-proxy/converter/response.py`
- Create: `gateway-proxy/converter/streaming.py`
- Create: `gateway-proxy/converter/responses.py`
- Modify: `gateway-proxy/server.py` (import block only)
- Delete: `gateway-proxy/converter.py`
- Create: `gateway-proxy/tests/test_parts.py`, `tests/test_request.py`, `tests/test_validation.py`, `tests/test_response.py`, `tests/test_streaming.py`, `tests/test_responses.py`, `tests/test_cli.py`
- Delete: `gateway-proxy/tests/test_converter.py`

**Interfaces:**
- Produces (consumed by every later task): module packages with these symbols relocated verbatim:
  - `converter/parts.py`: `_openai_content_to_v3_parts(content) -> list[dict]`, `_openai_tool_call_to_v3(tool_call: dict) -> dict`, `_openai_tool_msg_to_v3(msg: dict) -> dict`, `_normalize_tool_choice(tool_choice) -> dict`, `_openai_response_format_to_v3(response_format) -> dict | None`
  - `converter/request.py`: `openai_to_v3(body: dict) -> dict` (signature will be extended in Task 3)
  - `converter/validation.py`: `_parse_tool_args(args) -> str | None`, `validate_tool_history(messages: list[dict]) -> str | None`
  - `converter/response.py`: `_FINISH_REASON_MAP`, `_v3_finish_reason(v3_reason) -> str`, `_v3_usage_to_openai(usage_data: dict) -> dict`, `v3_to_openai(v3_data: dict, model: str = "") -> dict`
  - `converter/streaming.py`: `_sse_chunk(...) -> str`, `_StreamState`, `_process_stream_event(state, event) -> list[str]`, `v3_stream_to_openai(events: list[dict], model="", include_usage=True) -> str`, `v3_stream_iter(events: AsyncIterator[dict], model="", include_usage=True) -> AsyncIterator[str]`, `v3_sse_stream_to_openai(events: Iterator[dict], model="") -> dict`
  - `converter/responses.py`: `responses_input_to_messages(input_items) -> list[dict]`, `openai_to_responses(openai_resp: dict, model: str = "") -> dict`, `_ResponsesStreamState`, `openai_chunk_to_responses_sse(chunk_str: str, state) -> str | None`, `v3_stream_to_responses_sse(events: list[dict], model="") -> str`
  - `converter/__init__.py` re-exports all of the above so `from converter import openai_to_v3, validate_tool_history, ...` keeps working

- [ ] **Step 1: Create the package skeleton and move code verbatim**

Create each module by cutting the corresponding section out of `converter.py` unchanged (same docstrings, same code, same imports of `json`/`time`/`uuid` where used). Module-local imports needed:

- `converter/request.py` imports:
  ```python
  from __future__ import annotations
  import json
  from .parts import (
      _normalize_tool_choice,
      _openai_content_to_v3_parts,
      _openai_response_format_to_v3,
      _openai_tool_call_to_v3,
      _openai_tool_msg_to_v3,
  )
  ```
- `converter/validation.py` imports: `from __future__ import annotations` + `import json`
- `converter/response.py` imports: `from __future__ import annotations` + `import json, time, uuid`
- `converter/streaming.py` imports:
  ```python
  from __future__ import annotations
  import json
  import time
  import uuid
  from collections.abc import AsyncIterator, Iterator
  from .response import _v3_finish_reason, _v3_usage_to_openai
  ```
- `converter/responses.py` imports:
  ```python
  from __future__ import annotations
  import json
  import time
  import uuid
  from .response import _v3_finish_reason, _v3_usage_to_openai
  ```

`converter/__init__.py`:
```python
"""OpenAI <-> AI SDK v3 format converter (package).

Re-exports the public API so the documented library pattern keeps working:

    from converter import openai_to_v3, v3_to_openai, validate_tool_history
"""

from .parts import (
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
    "_normalize_tool_choice", "_openai_content_to_v3_parts",
    "_openai_response_format_to_v3", "_openai_tool_call_to_v3",
    "_openai_tool_msg_to_v3", "openai_to_v3", "validate_tool_history",
    "_FINISH_REASON_MAP", "_v3_finish_reason", "_v3_usage_to_openai",
    "v3_to_openai", "_process_stream_event", "_sse_chunk", "_StreamState",
    "v3_sse_stream_to_openai", "v3_stream_iter", "v3_stream_to_openai",
    "_ResponsesStreamState", "openai_chunk_to_responses_sse",
    "openai_to_responses", "responses_input_to_messages",
    "v3_stream_to_responses_sse",
]
```

`converter/__main__.py` (CLI, previously the `if __name__ == "__main__":` block in converter.py):
```python
"""CLI: python -m converter <input.json> [--stream] [--reverse]"""
from __future__ import annotations

import json
import sys

from .request import openai_to_v3
from .response import v3_to_openai
from .streaming import v3_stream_to_openai


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m converter <input.json> [--stream] [--reverse]")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    is_stream = "--stream" in sys.argv
    is_reverse = "--reverse" in sys.argv

    if is_stream:
        events = data if isinstance(data, list) else [data]
        print(v3_stream_to_openai(events))
    elif is_reverse:
        print(json.dumps(v3_to_openai(data), indent=2))
    elif "prompt" in data:
        print(json.dumps(v3_to_openai(data), indent=2))
    else:
        print(json.dumps(openai_to_v3(data), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Update server.py imports**

Replace the `from converter import (...) ` block with:
```python
from converter.request import openai_to_v3
from converter.validation import validate_tool_history
from converter.streaming import v3_stream_iter, v3_sse_stream_to_openai
from converter.responses import (
    responses_input_to_messages,
    openai_to_responses,
    openai_chunk_to_responses_sse,
    _ResponsesStreamState,
)
```

- [ ] **Step 3: Delete converter.py**

`git rm gateway-proxy/converter.py`

- [ ] **Step 4: Split the tests**

Split `tests/test_converter.py` into these files (move the class bodies verbatim; keep `async_iter`/`run_stream`/`_collect` helpers only in `test_streaming.py`, which is the only file that needs them):

- `tests/test_parts.py` — `TestContentParts`, `TestToolCallConversion`, `TestToolMsgConversion`, `TestNormalizeToolChoice`
- `tests/test_request.py` — `TestOpenAIToV3`
- `tests/test_validation.py` — `TestValidateToolHistory`
- `tests/test_response.py` — `TestV3ToOpenAI`
- `tests/test_streaming.py` — `TestStreaming`, `TestNonStreamingCollection` + helpers
- `tests/test_responses.py` — `TestResponsesInputToMessages`, `TestOpenAIToResponses`, `TestResponsesStreaming`

For `test_parts.py` the import becomes:
```python
from converter.parts import (
    _normalize_tool_choice,
    _openai_content_to_v3_parts,
    _openai_tool_call_to_v3,
    _openai_tool_msg_to_v3,
)
```
For `test_request.py`:
```python
from converter.request import openai_to_v3
```
For `test_validation.py`:
```python
from converter.validation import validate_tool_history
```
For `test_response.py`:
```python
from converter.response import _v3_finish_reason, _v3_usage_to_openai, v3_to_openai
```
For `test_streaming.py`:
```python
from converter.streaming import (
    v3_sse_stream_to_openai,
    v3_stream_iter,
    v3_stream_to_openai,
)
```
For `test_responses.py`:
```python
from converter.responses import (
    _ResponsesStreamState,
    openai_chunk_to_responses_sse,
    openai_to_responses,
    responses_input_to_messages,
    v3_stream_to_responses_sse,
)
```
Then `git rm tests/test_converter.py`.

- [ ] **Step 5: Add the CLI test**

Create `tests/test_cli.py`:
```python
"""CLI smoke test: python -m converter <input> [--stream] [--reverse]."""
import json
import subprocess
import sys


def test_cli_forward(tmp_path):
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}))
    out = subprocess.run(
        [sys.executable, "-m", "converter", str(inp)],
        capture_output=True, text=True, check=True,
    )
    parsed = json.loads(out.stdout)
    assert parsed["prompt"][0]["content"] == [{"type": "text", "text": "hi"}]


def test_cli_reverse(tmp_path):
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps({"content": [{"type": "text", "text": "hello"}], "finishReason": "stop"}))
    out = subprocess.run(
        [sys.executable, "-m", "converter", str(inp), "--reverse"],
        capture_output=True, text=True, check=True,
    )
    parsed = json.loads(out.stdout)
    assert parsed["choices"][0]["message"]["content"] == "hello"
```

- [ ] **Step 6: Run the full suite**

Run: `cd gateway-proxy && uv sync && uv run pytest tests/ -q`
Expected: all tests pass (the move is behavior-neutral).

- [ ] **Step 7: Commit**

```bash
git add gateway-proxy/converter gateway-proxy/server.py gateway-proxy/tests
git rm gateway-proxy/converter.py gateway-proxy/tests/test_converter.py
git commit -m "refactor: split converter.py into converter package"
```

---

### Task 2: Image file parts (fx wire format)

**Files:**
- Modify: `gateway-proxy/converter/parts.py` (add `_image_url_to_v3_part`, use it in `_openai_content_to_v3_parts`)
- Test: `gateway-proxy/tests/test_parts.py`

**Interfaces:**
- Consumes: `_openai_content_to_v3_parts` from Task 1.
- Produces: `_image_url_to_v3_part(url: str) -> dict` — data URLs become `{"type": "file", "mediaType": ..., "data": ...}`, remote URLs stay `{"type": "image", "image": url}`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_parts.py`:
```python
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
```
Update the import line in `test_parts.py` to also import `_image_url_to_v3_part`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_parts.py::TestImageParts -v`
Expected: FAIL with `ImportError: cannot import name '_image_url_to_v3_part'`.

- [ ] **Step 3: Implement**

In `converter/parts.py` add:
```python
def _image_url_to_v3_part(url: str) -> dict:
    """Map an OpenAI image_url value to a v3 content part.

    Data URLs use the current fx wire shape:
        {"type": "file", "mediaType": "<mime>", "data": "<base64>"}
    Remote http(s) URLs can't be fetched by this pure converter, so they
    keep the AI SDK image part shape: {"type": "image", "image": "<url>"}.
    """
    if isinstance(url, str) and url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = "application/octet-stream"
        params = header[len("data:"):].split(";") if len(header) > len("data:") else []
        if params and params[0]:
            media_type = params[0]
        return {"type": "file", "mediaType": media_type, "data": data}
    return {"type": "image", "image": url}
```
Then replace the image_url branch inside `_openai_content_to_v3_parts`:
```python
                elif part.get("type") == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    parts.append(_image_url_to_v3_part(url))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_parts.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest tests/ -q`
Expected: PASS (the old `test_mixed_list_with_image` in `TestContentParts` still passes: it uses `data:image/png;base64,abc` and asserts `{"type": "image", "image": "data:image/png;base64,abc"}` — this assertion must be updated in this task to expect the new file part shape).

Update in `tests/test_parts.py`:
```python
    def test_mixed_list_with_image(self):
        parts = _openai_content_to_v3_parts([
            {"type": "text", "text": "see image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ])
        assert parts == [
            {"type": "text", "text": "see image"},
            {"type": "file", "mediaType": "image/png", "data": "abc"},
        ]
```

- [ ] **Step 6: Commit**

```bash
git add gateway-proxy/converter/parts.py gateway-proxy/tests/test_parts.py
git commit -m "fix: emit fx image file parts for data URLs"
```

---

### Task 3: Body user-agent scoping + reasoning normalization

**Files:**
- Modify: `gateway-proxy/converter/request.py`
- Test: `gateway-proxy/tests/test_request.py`

**Interfaces:**
- Consumes: nothing new from other tasks.
- Produces: `openai_to_v3(body: dict, *, product_user_agent: str = "fx/0.0.4", product_user_agent_models: frozenset[str] | None = frozenset({"zai/glm-5.2"})) -> dict`. When `product_user_agent_models` is `None`, the header is included for every model; when it is an empty frozenset, never. The body-level `headers.user-agent` is included only when the request model is in the set (mirrors fx's "Scope Gateway request identity to GLM 5.2").
- Also: `reasoning` normalization — a dict `{"effort": "..."}` emits the string value; string values pass through.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_request.py`:
```python
class TestBodyUserAgentScoping:
    def test_glm52_includes_user_agent(self):
        body = {"model": "zai/glm-5.2", "messages": [{"role": "user", "content": "x"}]}
        assert openai_to_v3(body)["headers"] == {"user-agent": "fx/0.0.4"}

    def test_other_model_omits_user_agent(self):
        body = {"model": "anthropic/claude", "messages": [{"role": "user", "content": "x"}]}
        assert "headers" not in openai_to_v3(body)

    def test_override_model_set(self):
        body = {"model": "anthropic/claude", "messages": [{"role": "user", "content": "x"}]}
        result = openai_to_v3(body, product_user_agent_models=frozenset({"anthropic/claude"}))
        assert result["headers"] == {"user-agent": "fx/0.0.4"}

    def test_all_models_when_none(self):
        body = {"model": "anthropic/claude", "messages": [{"role": "user", "content": "x"}]}
        result = openai_to_v3(body, product_user_agent_models=None)
        assert result["headers"] == {"user-agent": "fx/0.0.4"}

    def test_no_models_when_empty(self):
        body = {"model": "zai/glm-5.2", "messages": [{"role": "user", "content": "x"}]}
        result = openai_to_v3(body, product_user_agent_models=frozenset())
        assert "headers" not in result

    def test_custom_user_agent(self):
        body = {"model": "zai/glm-5.2", "messages": [{"role": "user", "content": "x"}]}
        result = openai_to_v3(body, product_user_agent="fx/0.0.9")
        assert result["headers"] == {"user-agent": "fx/0.0.9"}


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
```
Update `test_request.py` imports: `openai_to_v3` already imported from Task 1.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_request.py::TestBodyUserAgentScoping tests/test_request.py::TestReasoningNormalization -v`
Expected: FAIL — `openai_to_v3` currently always emits `{"headers": {"user-agent": "fx-converter"}}` and passes `reasoning` objects through.

- [ ] **Step 3: Implement**

In `converter/request.py`:

Change the signature and body-user-agent logic:
```python
def openai_to_v3(
    body: dict,
    *,
    product_user_agent: str = "fx/0.0.4",
    product_user_agent_models: frozenset[str] | None = frozenset({"zai/glm-5.2"}),
) -> dict:
    """Convert an OpenAI chat-completions request to AI SDK v3 format.

    Translates:
    - messages: role/content format conversion (tool calls aligned to fx)
    - tools: OpenAI {"type":"function","function":{...}} -> v3 flat {"type":"function","name","description","inputSchema"}
    - temperature, max_tokens, top_p, stop, response_format, reasoning,
      providerOptions parameters

    `product_user_agent` / `product_user_agent_models` control the body-level
    `headers.user-agent` (fx only sends it for zai/glm-5.2). Pass
    `product_user_agent_models=None` for all models, `frozenset()` for none.

    Does NOT modify the input body.
    """
    model = body.get("model", "")
    ...
    v3_body: dict = {
        "prompt": prompt,
        "tools": v3_tools,
        "toolChoice": _normalize_tool_choice(body.get("tool_choice", {"type": "auto"})),
    }
    if product_user_agent_models is None or model in product_user_agent_models:
        v3_body["headers"] = {"user-agent": product_user_agent}
    ...
```

Replace the reasoning block with normalization:
```python
    # Reasoning effort (OpenAI) / reasoning (v3, fx uses a string label).
    if "reasoning" in body:
        _reasoning = body["reasoning"]
        if isinstance(_reasoning, dict) and isinstance(_reasoning.get("effort"), str):
            v3_body["reasoning"] = _reasoning["effort"]
        else:
            v3_body["reasoning"] = _reasoning
    elif "reasoning_effort" in body:
        v3_body["reasoning"] = body["reasoning_effort"]
```
(Remove the old `_effort_map` identity-map block — the map was identity for the four known values; direct passthrough is equivalent.)

- [ ] **Step 4: Run the new tests**

Run: `uv run pytest tests/test_request.py -q`
Expected: PASS.

- [ ] **Step 5: Fix the pre-existing reasoning passthrough test**

`test_reasoning_passthrough` in `tests/test_request.py` currently asserts:
```python
def test_reasoning_passthrough(self):
    body = {"messages": [{"role": "user", "content": "x"}], "reasoning": {"effort": "high"}}
    assert openai_to_v3(body)["reasoning"] == {"effort": "high"}
```
Update it to expect the normalized string:
```python
def test_reasoning_passthrough(self):
    body = {"messages": [{"role": "user", "content": "x"}], "reasoning": {"effort": "high"}}
    assert openai_to_v3(body)["reasoning"] == "high"
```

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest tests/ -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add gateway-proxy/converter/request.py gateway-proxy/tests/test_request.py
git commit -m "feat: scope body user-agent to glm-5.2 and normalize reasoning"
```

---

### Task 4: Stricter tool-history validation (match fx)

**Files:**
- Modify: `gateway-proxy/converter/validation.py`
- Test: `gateway-proxy/tests/test_validation.py`

**Interfaces:**
- Consumes: `validate_tool_history` from Task 1.
- Produces: same `validate_tool_history(messages) -> str | None` signature, with two new rejections: duplicate-key JSON arguments (treated as malformed) and tool result `name` that mismatches the matched assistant call's name.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_validation.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_validation.py::TestValidationFixes -v`
Expected: FAIL — current code accepts duplicate keys and name mismatches.

- [ ] **Step 3: Implement**

In `converter/validation.py`:

Add a duplicate-key-rejecting object hook and use it in `_parse_tool_args`:
```python
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
```

In the results loop of `validate_tool_history`, add the name-match check. Build a call-name map from the assistant block before the results loop:
```python
        call_names: dict[str, str] = {
            call.get("id", ""): call.get("function", {}).get("name", "")
            for call in calls
        }
```
and inside the loop, after the `rid` checks:
```python
            result_name = result.get("name")
            expected_name = call_names.get(rid, "")
            if result_name and expected_name and result_name != expected_name:
                return (f"tool result for {rid} names tool {result_name!r} "
                        f"but call {rid} is {expected_name!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_validation.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add gateway-proxy/converter/validation.py gateway-proxy/tests/test_validation.py
git commit -m "fix: reject duplicate-key tool args and mismatched result names"
```

---

### Task 5: Richer usage mapping

**Files:**
- Modify: `gateway-proxy/converter/response.py`
- Test: `gateway-proxy/tests/test_response.py`

**Interfaces:**
- Consumes: `_v3_usage_to_openai` from Task 1.
- Produces: `_v3_usage_to_openai(usage_data: dict) -> dict` with identical signature; output gains `prompt_tokens_details.cached_tokens` (from `inputTokens.cacheRead`) and `output_tokens_details.reasoning_tokens` (from `outputTokens.reasoning`), only when the upstream reports those fields.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_response.py`:
```python
class TestUsageDetails:
    def test_cached_tokens_mapped(self):
        usage = _v3_usage_to_openai({
            "inputTokens": {"total": 100, "cacheRead": 80},
            "outputTokens": {"total": 10},
        })
        assert usage["prompt_tokens"] == 100
        assert usage["prompt_tokens_details"] == {"cached_tokens": 80}

    def test_reasoning_tokens_mapped(self):
        usage = _v3_usage_to_openai({
            "inputTokens": {"total": 5},
            "outputTokens": {"total": 20, "reasoning": 15},
        })
        assert usage["completion_tokens"] == 20
        assert usage["output_tokens_details"] == {"reasoning_tokens": 15}

    def test_details_omitted_when_absent(self):
        usage = _v3_usage_to_openai({
            "inputTokens": {"total": 5},
            "outputTokens": {"total": 10},
        })
        assert usage == {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
        assert "prompt_tokens_details" not in usage
        assert "output_tokens_details" not in usage

    def test_flat_input_output_tokens(self):
        usage = _v3_usage_to_openai({"inputTokens": 7, "outputTokens": 3})
        assert usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_response.py::TestUsageDetails -v`
Expected: FAIL — current mapping returns only the three totals.

- [ ] **Step 3: Implement**

In `converter/response.py`, replace `_v3_usage_to_openai`:
```python
def _v3_usage_to_openai(usage_data: dict) -> dict:
    pt = usage_data.get("inputTokens", {})
    prompt_total = pt.get("total", 0) if isinstance(pt, dict) else (pt or 0)
    ct = usage_data.get("outputTokens", {})
    completion_total = ct.get("total", 0) if isinstance(ct, dict) else (ct or 0)
    result: dict = {
        "prompt_tokens": prompt_total,
        "completion_tokens": completion_total,
        "total_tokens": prompt_total + completion_total,
    }
    if isinstance(pt, dict):
        cache_read = pt.get("cacheRead")
        if cache_read is not None:
            result["prompt_tokens_details"] = {"cached_tokens": cache_read}
    if isinstance(ct, dict):
        reasoning = ct.get("reasoning")
        if reasoning is not None:
            result["output_tokens_details"] = {"reasoning_tokens": reasoning}
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_response.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest tests/ -q`
Expected: PASS (existing `test_usage_mapping` in `TestV3ToOpenAI` still passes: no details fields in that input).

- [ ] **Step 6: Commit**

```bash
git add gateway-proxy/converter/response.py gateway-proxy/tests/test_response.py
git commit -m "feat: map cached and reasoning tokens in usage"
```

---

### Task 6: Streaming event handling (reasoning, steps, metadata)

**Files:**
- Modify: `gateway-proxy/converter/streaming.py`
- Test: `gateway-proxy/tests/test_streaming.py`

**Interfaces:**
- Consumes: `_StreamState`, `_process_stream_event`, `_sse_chunk`, `v3_stream_to_openai`, `v3_stream_iter`, `v3_sse_stream_to_openai` from Task 1.
- Produces: `_sse_chunk(..., reasoning_delta: str | None = None)`; `_StreamState` gains `last_step_usage: dict` and uses `response-metadata.modelId` when `model` is empty; `_process_stream_event` handles all fx-known event types via an explicit if/elif dispatch (unknown types ignored).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_streaming.py`:
```python
class TestStreamingNewEvents:
    def _data_chunks(self, sse):
        return [json.loads(l[6:]) for l in sse.split("\n\n") if l.startswith("data: ") and "[DONE]" not in l]

    def test_reasoning_delta_becomes_reasoning_content(self):
        events = [
            {"type": "reasoning-start", "id": "r1"},
            {"type": "reasoning-delta", "id": "r1", "delta": "let me think"},
            {"type": "reasoning-delta", "id": "r1", "delta": " harder"},
            {"type": "reasoning-end", "id": "r1"},
            {"type": "text-delta", "delta": "answer"},
            {"type": "finish", "finishReason": "stop"},
        ]
        sse = v3_stream_to_openai(events, model="gpt-4")
        reasoning = [c["choices"][0]["delta"].get("reasoning_content")
                     for c in self._data_chunks(sse)
                     if c["choices"][0]["delta"].get("reasoning_content")]
        assert reasoning == ["let me think", " harder"]

    def test_finish_step_usage_fallback(self):
        events = [
            {"type": "text-delta", "delta": "x"},
            {"type": "finish-step", "usage": {"inputTokens": {"total": 4}, "outputTokens": {"total": 2}}},
            {"type": "finish", "finishReason": "stop"},
        ]
        sse = v3_stream_to_openai(events)
        chunks = self._data_chunks(sse)
        finish = next(c for c in chunks if c["choices"][0]["finish_reason"] == "stop")
        assert finish["usage"]["total_tokens"] == 6

    def test_finish_usage_wins_over_step_usage(self):
        events = [
            {"type": "finish-step", "usage": {"inputTokens": {"total": 4}, "outputTokens": {"total": 2}}},
            {"type": "finish", "finishReason": "stop", "usage": {"inputTokens": {"total": 9}, "outputTokens": {"total": 1}}},
        ]
        sse = v3_stream_to_openai(events)
        chunks = self._data_chunks(sse)
        finish = next(c for c in chunks if c["choices"][0]["finish_reason"] == "stop")
        assert finish["usage"]["total_tokens"] == 10

    def test_extra_events_do_not_break_stream(self):
        events = [
            {"type": "start"},
            {"type": "start-step", "id": "s1"},
            {"type": "source", "source": {}},
            {"type": "file", "file": {}},
            {"type": "raw", "raw": {}},
            {"type": "tool-result", "toolCallId": "c1", "toolName": "t", "output": {"type": "text", "value": "ok"}},
            {"type": "text-delta", "delta": "hi"},
            {"type": "finish", "finishReason": "stop"},
        ]
        sse = v3_stream_to_openai(events, model="m")
        assert "hi" in sse
        assert sse.rstrip().endswith("data: [DONE]")

    def test_unknown_event_ignored(self):
        events = [
            {"type": "mystery-event", "payload": {}},
            {"type": "text-delta", "delta": "ok"},
            {"type": "finish", "finishReason": "stop"},
        ]
        sse = v3_stream_to_openai(events, model="m")
        assert "ok" in sse
        assert sse.rstrip().endswith("data: [DONE]")

    def test_response_metadata_sets_model_when_empty(self):
        events = [
            {"type": "response-metadata", "id": "x", "modelId": "glm52"},
            {"type": "text-delta", "delta": "hi"},
            {"type": "finish", "finishReason": "stop"},
        ]
        sse = v3_stream_to_openai(events, model="")
        chunks = self._data_chunks(sse)
        assert all(c["model"] == "glm52" for c in chunks)

    def test_non_stream_collector_uses_step_usage_fallback(self):
        events = [
            {"type": "text-delta", "delta": "x"},
            {"type": "finish-step", "usage": {"inputTokens": {"total": 4}, "outputTokens": {"total": 2}}},
            {"type": "finish", "finishReason": "stop"},
        ]
        result = v3_sse_stream_to_openai(iter(events), model="m")
        assert result["usage"] == {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streaming.py::TestStreamingNewEvents -v`
Expected: FAIL — reasoning deltas ignored, no usage fallback, `model` stays empty.

- [ ] **Step 3: Implement**

In `converter/streaming.py`:

Extend `_sse_chunk`:
```python
def _sse_chunk(
    chat_id: str,
    model: str,
    delta_text: str = "",
    finish_reason: str | None = None,
    role: str | None = None,
    usage: dict | None = None,
    reasoning_delta: str | None = None,
) -> str:
    delta: dict = {}
    if role:
        delta["role"] = role
    if delta_text:
        delta["content"] = delta_text
    if reasoning_delta:
        delta["reasoning_content"] = reasoning_delta
    ...
```
(rest unchanged)

Extend `_StreamState.__init__`:
```python
        self.last_step_usage: dict = {}
```

Rewrite `_process_stream_event`'s if/elif chain to add the new branches (existing branches for `text-delta`, `tool-input-delta`, `tool-call`, `finish`, `error` stay; the `finish` branch becomes):
```python
    elif etype == "finish":
        finish_reason = _v3_finish_reason(event.get("finishReason", "stop"))
        usage_data = event.get("usage", {}) or state.last_step_usage or {}
        usage = _v3_usage_to_openai(usage_data) if (usage_data and state.include_usage) else None
        chunks.append(_sse_chunk(state.chat_id, state.model, finish_reason=finish_reason, usage=usage))
        state.finished = True

    elif etype == "reasoning-delta":
        delta = event.get("delta", "")
        if delta:
            chunks.append(_sse_chunk(state.chat_id, state.model, reasoning_delta=delta))

    elif etype == "finish-step":
        usage_data = event.get("usage", {})
        if usage_data:
            state.last_step_usage = usage_data

    elif etype == "response-metadata":
        if not state.model and event.get("modelId"):
            state.model = event["modelId"]

    elif etype in (
        "start", "start-step", "tool-result", "source", "file", "raw",
        "text-start", "text-end", "reasoning-start", "reasoning-end",
        "tool-input-start", "tool-input-end",
    ):
        pass  # explicitly handled no-ops: no OpenAI chunk for these
```
(Any event type not listed still falls through to no output, as before.)

In `v3_sse_stream_to_openai`, add the same fallback. Track `step_usage` before the loop:
```python
    step_usage: dict = {}
```
and in the loop body:
```python
        elif etype == "finish-step":
            if event.get("usage"):
                step_usage = event["usage"]
```
then change the `finish` branch to:
```python
        elif etype == "finish":
            finish_reason = _v3_finish_reason(event.get("finishReason", "stop"))
            usage_data = event.get("usage", {}) or step_usage or {}
            usage_data = usage_data or {}
            break
```
and the usage line at the end stays `_v3_usage_to_openai(usage_data) if usage_data else {}`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_streaming.py -q`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add gateway-proxy/converter/streaming.py gateway-proxy/tests/test_streaming.py
git commit -m "feat: forward reasoning deltas and handle step/metadata events"
```

---

### Task 7: server.py — fx/0.0.4 user agent and session headers

**Files:**
- Modify: `gateway-proxy/server.py`
- Test: `gateway-proxy/tests/test_server_headers.py` (new)

**Interfaces:**
- Consumes: `openai_to_v3` with the new signature from Task 3.
- Produces: module constants `FX_USER_AGENT = os.getenv("FX_USER_AGENT", "fx/0.0.4")`, `PRODUCT_USER_AGENT_MODELS` (a frozenset, `None` for all, empty for none), `GATEWAY_SESSION_ID`, `GATEWAY_SESSION_AFFINITY`; `_v3_headers(model: str, streaming: bool, *, session_id: str | None = None, session_affinity: str | None = None) -> dict[str, str]`; both chat routes read inbound `x-session-id` / `x-session-affinity` headers (falling back to env) and pass them through.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_server_headers.py`:
```python
"""Unit tests for server.py header construction (env-free assertions)."""
from server import _v3_headers


def test_v3_headers_default_identity():
    headers = _v3_headers("zai/glm-5.2", streaming=True)
    assert headers["User-Agent"].startswith("fx/")
    assert headers["ai-language-model-streaming"] == "true"
    assert headers["ai-language-model-id"] == "zai/glm-5.2"
    assert headers["ai-gateway-protocol-version"] == "0.0.1"
    assert headers["ai-language-model-specification-version"] == "4"
    assert "x-session-id" not in headers
    assert "x-session-affinity" not in headers


def test_v3_headers_non_streaming_flag():
    headers = _v3_headers("zai/glm-5.2", streaming=False)
    assert headers["ai-language-model-streaming"] == "false"
    assert "Accept" not in headers


def test_v3_headers_session_params():
    headers = _v3_headers(
        "zai/glm-5.2", streaming=True,
        session_id="sess-1", session_affinity="sess-1",
    )
    assert headers["x-session-id"] == "sess-1"
    assert headers["x-session-affinity"] == "sess-1"
```
Note: `from server import _v3_headers` triggers `load_dotenv()` and reads env at import; the assertions above avoid env-dependent values.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server_headers.py -v`
Expected: FAIL — `_v3_headers` has no session params and no `Accept` handling for non-streaming (the current code always sends `Accept` only when streaming; check the actual failure output: the session-param test fails with `TypeError`).

- [ ] **Step 3: Implement**

In `server.py` config section, after the existing env constants, add:
```python
FX_USER_AGENT = os.getenv("FX_USER_AGENT", "fx/0.0.4")
GATEWAY_SESSION_ID = os.getenv("GATEWAY_SESSION_ID", "")
GATEWAY_SESSION_AFFINITY = os.getenv("GATEWAY_SESSION_AFFINITY", "")

# Models that receive the body-level headers.user-agent (fx scopes it to
# zai/glm-5.2). "*" = all models, "" = none, else comma-separated list.
_raw_pua_models = os.getenv("PRODUCT_USER_AGENT_MODELS", "zai/glm-5.2")
if _raw_pua_models == "*":
    PRODUCT_USER_AGENT_MODELS: frozenset[str] | None = None
elif _raw_pua_models == "":
    PRODUCT_USER_AGENT_MODELS = frozenset()
else:
    PRODUCT_USER_AGENT_MODELS = frozenset(
        m.strip() for m in _raw_pua_models.split(",") if m.strip()
    )
```

Replace `_v3_headers`:
```python
def _v3_headers(
    model: str,
    streaming: bool,
    *,
    session_id: str | None = None,
    session_affinity: str | None = None,
) -> dict[str, str]:
    headers: dict[str, str] = {
        "Authorization": f"Bearer {GATEWAY_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": FX_USER_AGENT,
        "HTTP-Referer": "https://github.com/vercel-labs/fx",
        "X-Title": "fx",
        "ai-gateway-protocol-version": "0.0.1",
        "ai-language-model-specification-version": "4",
        "ai-language-model-id": model,
        "ai-language-model-streaming": "true" if streaming else "false",
    }
    if streaming:
        headers["Accept"] = "text/event-stream"
    if GATEWAY_TEAM:
        headers["x-vercel-ai-gateway-team"] = GATEWAY_TEAM
    sid = session_id or GATEWAY_SESSION_ID
    affinity = session_affinity or GATEWAY_SESSION_AFFINITY
    if sid:
        headers["x-session-id"] = sid
    if affinity:
        headers["x-session-affinity"] = affinity
    return headers
```

Update the two `openai_to_v3(...)` call sites (chat_completions and responses_route) to pass the new params:
```python
    v3_body = openai_to_v3(
        body,
        product_user_agent=FX_USER_AGENT,
        product_user_agent_models=PRODUCT_USER_AGENT_MODELS,
    )
```

Update both routes to forward inbound session headers. In `chat_completions` and `responses_route`, after `model` is resolved:
```python
    session_id = request.headers.get("x-session-id") or GATEWAY_SESSION_ID
    session_affinity = request.headers.get("x-session-affinity") or GATEWAY_SESSION_AFFINITY
```
and pass them to `_v3_headers`:
```python
    headers = _v3_headers(model, streaming=True, session_id=session_id, session_affinity=session_affinity)
```
(in both the streaming and non-streaming branches of both routes — `_v3_headers` is invoked once per route before the branch; the current code calls it once per route at `headers = _v3_headers(model, streaming=True)`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_server_headers.py -q && uv run pytest tests/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gateway-proxy/server.py gateway-proxy/tests/test_server_headers.py
git commit -m "feat: fx/0.0.4 user agent and session header passthrough"
```

---

### Task 8: Docs — CLAUDE.md, SAUCE.md, README.md, .env.example

**Files:**
- Modify: `gateway-proxy/SAUCE.md`, `gateway-proxy/README.md`, `gateway-proxy/.env.example`
- Modify: `CLAUDE.md` (repo root; the two-layer split note and test command line only)

**Interfaces:** none (docs only).

- [ ] **Step 1: Update CLAUDE.md (repo root)**

- Under Tooling, change `uv run pytest tests/` comment and `test_converter.py::TestOpenAIToV3` to the new files:
  ```bash
  uv run pytest tests/                                 # unit tests for the converter package
  uv run pytest tests/test_request.py::TestOpenAIToV3  # single test class
  ```
- In "Two-layer split", replace the converter.py bullet with:
  ```text
  - **`converter/`** — pure functions, no I/O. OpenAI ↔ AI SDK v3 translation
    (`request.py`, `parts.py`, `response.py`, `streaming.py`, `responses.py`)
    + `validation.py` (tool-history validation). Add request/response-shape
    work here, with tests in `tests/test_*.py` mirroring the module layout.
  ```
- In "Protocol invariants", change `User-Agent: fx/0.0.3` to `User-Agent: fx/0.0.4` and update the two `converter.py` mentions to `converter/`.

- [ ] **Step 2: Update SAUCE.md**

Make these specific edits:
- Step 3 header capture: change `User-Agent: fx/0.0.3` to `User-Agent: fx/0.0.4` and add the new headers to the captured list: `x-vercel-ai-gateway-team` (when set), `x-session-id`, `x-session-affinity` (when session pinning is used).
- "The content-type rules" table: add a row for images: `user` (with image) content part can be `{"type":"file","mediaType":"image/png","data":"<base64>"}` (data URLs) or `{"type":"image","image":"<url>"}` (remote URLs).
- Add a new subsection after "Event mapping": "Newer stream events (fx-aligned)" with a table:
  | v3 event type | Proxy handling |
  |---|---|
  | `reasoning-start` / `reasoning-end` | no-op |
  | `reasoning-delta` | chunk with `delta.reasoning_content` |
  | `start` / `start-step` | no-op |
  | `finish-step` | remembers step usage; used if `finish` has none |
  | `tool-result` / `source` / `file` / `raw` | no-op |
  | `response-metadata` | captures `modelId` for the response model |
  | unknown | ignored |
- "Usage token mapping": add cached/reasoning breakdown lines:
  ```text
  v3: {"inputTokens": {"total": 19, "cacheRead": 10560}, "outputTokens": {"total": 6, "reasoning": 0}}
  OpenAI: {"prompt_tokens": 19, "prompt_tokens_details": {"cached_tokens": 10560},
            "completion_tokens": 6, "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 25}
  ```
- The `## 🧩 server.py — function-by-function breakdown` section: the line references are stale (they refer to a pre-refactor file). Replace the intro sentence with a note that `_openai_to_v3`/`_v3_stream_to_openai` now live in the `converter/` package (`request.py` / `streaming.py`) and that `server.py` imports them; update the `_v3_headers()` section (lines 123-140) description to mention `FX_USER_AGENT` (`fx/0.0.4`) and the session headers.
- Add a short "converter/ package map" section:
  ```text
  converter/
  ├── parts.py       # content parts (text/image/tool-call/tool-result), response_format, tool_choice
  ├── request.py     # openai_to_v3 + body user-agent scoping + reasoning normalization
  ├── validation.py  # validate_tool_history (duplicate-key JSON, result-name match)
  ├── response.py    # v3_to_openai + finish-reason + usage mapping (cached/reasoning tokens)
  ├── streaming.py   # stream state, event dispatch, v3_stream_iter / v3_stream_to_openai
  ├── responses.py   # Responses API translation
  └── __main__.py    # CLI: python -m converter
  ```
- In "Gotchas and edge cases", change item 8 to note the key location comment and add: session headers (`x-session-id`/`x-session-affinity`) are forwarded when the client sends them.

- [ ] **Step 3: Update README.md**

- "Key headers the proxy sends": change `User-Agent: fx/0.0.3` to `User-Agent: fx/0.0.4`; add `x-session-id` / `x-session-affinity` lines noting they are sent only when configured.
- In "Notes", add: the body-level `headers.user-agent` is only sent for `zai/glm-5.2` (override with `PRODUCT_USER_AGENT_MODELS`).

- [ ] **Step 4: Update .env.example**

Append:
```bash
# --- fx identity alignment ---
# User-Agent header sent to the gateway (mirrors current fx CLI).
# FX_USER_AGENT=fx/0.0.4

# Models that receive the body-level headers.user-agent.
# "*" = all models, "" = none, else comma-separated model ids.
# PRODUCT_USER_AGENT_MODELS=zai/glm-5.2

# Optional session pinning headers (also forwarded from inbound requests).
# GATEWAY_SESSION_ID=
# GATEWAY_SESSION_AFFINITY=
```

- [ ] **Step 5: Verify no stale references remain**

Run: `cd /teamspace/studios/this_studio/Fionn && rg -n "fx/0\.0\.3|fx-converter" --glob '!docs/superpowers/**'`
Expected: no matches (spec/plan docs may mention them as historical notes).

- [ ] **Step 6: Commit**

```bash
git add gateway-proxy/SAUCE.md gateway-proxy/README.md gateway-proxy/.env.example CLAUDE.md
git commit -m "docs: align proxy docs with current fx wire format"
```

---

### Task 9: Final validation

**Files:** none (verification only).

- [ ] **Step 1: Full test suite**

Run: `cd gateway-proxy && uv sync && uv run pytest tests/ -q`
Expected: all pass.

- [ ] **Step 2: Repo hygiene check**

Run: `git status --short` and `rg -n "from converter import|converter\.py" gateway-proxy --glob '!*.pyc'`
Expected: no `from converter import` left outside tests/docs; no references to `converter.py` as a file.

- [ ] **Step 3: Optional live smoke test (manual, requires key + network)**

Start the proxy and run `python test_proxy.py` per README — only if `AI_GATEWAY_API_KEY` is available. Do not include in automated CI.

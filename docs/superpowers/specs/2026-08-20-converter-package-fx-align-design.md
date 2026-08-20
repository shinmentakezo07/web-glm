# Design: converter package refactor + alignment with current fx wire format

Date: 2026-08-20
Status: approved (user)

## Goal

Refactor `gateway-proxy/converter.py` into a `converter/` package with submodules,
add new conversion logic verified against the current `vercel-labs/fx` source, and
update `SAUCE.md` / `README.md` to match reality.

## Research basis

Findings from cloning `vercel-labs/fx` (main, Aug 2026) and reading
`src/core/gateway/gateway_json.zig`, `src/gateway/client.zig`,
`src/gateway/host_stream_provider.zig`, `src/core/images/image_attachments.zig`,
and the Aug 18 2026 commit "Scope Gateway request identity to GLM 5.2":

- fx's User-Agent is now `fx/0.0.4` (`pub const version = "0.0.4"` in `src/main.zig`).
  Proxy/SAUCE still hardcode `fx/0.0.3`.
- Header set the fx CLI sends: `HTTP-Referer: https://github.com/vercel-labs/fx`,
  `X-Title: fx`, `ai-gateway-protocol-version: 0.0.1`,
  `ai-language-model-specification-version: 4`, `ai-language-model-id: <model>`,
  `ai-language-model-streaming: true|false`, optional `x-vercel-ai-gateway-team`,
  optional `x-session-id` / `x-session-affinity`.
- The body-level `headers.user-agent` is now only emitted for `zai/glm-5.2`
  (newest fx commit). The proxy currently always sends
  `{"headers": {"user-agent": "fx-converter"}}`.
- Image parts: fx emits `{"type":"file","mediaType":"<mime>","data":"<base64>"}`.
  The converter still emits the older `{"type":"image","image":"data:..."}` shape.
- Stream events fx knows: `response-metadata`, `text-start/delta/end`,
  `reasoning-start/delta/end`, `tool-input-start/delta/end`, `tool-call`,
  `tool-result`, `source`, `file`, `raw`, `error`, `start`, `start-step`,
  `finish-step`, `finish`. The converter only handles `text-delta`,
  `tool-input-delta`, `tool-call`, `finish`, `error`.
- `reasoning` in the v3 body is a string (e.g. "minimal", "high"), not an object.
- fx's `validateToolMessageHistory` requires tool results to carry a tool name
  that matches the call's name, and treats duplicate-key JSON args as malformed.
  Our `validate_tool_history` checks neither.
- `toolChoice` in fx is one of `auto`, `none`, `required` (string forms).

## Package layout

```
gateway-proxy/
├── converter/
│   ├── __init__.py      # re-exports public API; `from converter import ...` keeps working
│   ├── __main__.py      # CLI: python -m converter <input.json> [--stream] [--reverse]
│   ├── parts.py         # content-part conversion, image file parts, response_format, tool_choice
│   ├── request.py       # openai_to_v3 (request assembly + fx identity rules)
│   ├── validation.py    # validate_tool_history + new checks
│   ├── response.py      # v3_to_openai, finish-reason map, usage mapping
│   ├── streaming.py     # _StreamState, event dispatch table, stream converters
│   └── responses.py     # Responses API translation (behavior unchanged, moved)
```

- `server.py` and `tests/` import from the explicit submodules.
- `converter/__init__.py` re-exports the full public API (including underscore
  helpers used by tests) so the documented library usage
  (`from converter import openai_to_v3, ...`) still works.
- CLAUDE.md two-layer section updated to reference the package.

## Behavior changes

### 1. fx identity alignment
- `FX_USER_AGENT` module/env constant, default `fx/0.0.4`; `_v3_headers()` uses it.
- Body `headers.user-agent` only included when `model == "zai/glm-5.2"`,
  value `fx/0.0.4`. Overridable via `PRODUCT_USER_AGENT_MODELS` env
  (comma-separated list; empty string disables, "*" enables for all).
- `x-session-id` / `x-session-affinity`: forwarded from inbound request headers
  if the client sent them; otherwise from `GATEWAY_SESSION_ID` /
  `GATEWAY_SESSION_AFFINITY` env vars (empty by default).

### 2. Image parts
- `image_url` part whose url is a data URL (`data:<mime>;base64,<data>`):
  `{"type":"file","mediaType":"<mime>","data":"<base64-data>"}`.
  Default media type `application/octet-stream` when unparsable.
- Remote http(s) URLs: unchanged `{"type":"image","image":"<url>"}` (a pure
  converter cannot fetch them; documented).

### 3. Reasoning streaming
- `reasoning-start` / `reasoning-end`: no-op.
- `reasoning-delta`: OpenAI chunk with `delta: {"reasoning_content": "<delta>"}`.
- Implemented for both `v3_stream_iter` (live) and `v3_stream_to_openai` (offline).
- `openai_to_v3` reasoning normalization: if the client sends `reasoning` as a
  dict with an `effort` key, emit `reasoning["effort"]` (a string, fx shape);
  string values pass through unchanged; `reasoning_effort` mapping unchanged.

### 4. Validation
- If a tool result carries `name` and it differs from the matching assistant
  call's name -> reject. Missing name is back-filled as today (OpenAI format).
- Tool-call arguments parsed with duplicate-key detection; duplicate keys are
  treated as malformed JSON (fx behavior).

### 5. Richer usage
- `inputTokens.cacheRead` -> `prompt_tokens_details.cached_tokens`.
- `outputTokens.reasoning` -> `output_tokens_details.reasoning_tokens`.
- Details keys emitted only when the upstream reports the underlying field.

### 6. Multi-step / extra stream events
- Explicit handler table covering all fx-known event types; unknown types
  remain silently ignored.
- `finish-step`: remember its usage as a fallback for the final `finish` event
  if that one carries no usage.
- `response-metadata`: capture `modelId`; used as the response model when the
  requested model string is empty.
- `tool-result`, `source`, `file`, `raw`, `start`, `start-step`:
  explicit no-ops (no client-visible output).

## Docs

- `SAUCE.md`: fix stale content (fx/0.0.3 -> 0.0.4, old function/line-number
  references -> package module map, image part shape, reasoning events,
  validation rules, usage mapping, session headers, event table expansion).
- `README.md`: env table + header list updates.
- `.env.example`: add `FX_USER_AGENT`, `PRODUCT_USER_AGENT_MODELS`,
  `GATEWAY_SESSION_ID`, `GATEWAY_SESSION_AFFINITY`.

## Tests

Tests split to mirror the package: `tests/test_request.py`, `test_parts.py`,
`test_validation.py`, `test_response.py`, `test_streaming.py`,
`test_responses.py`. Existing assertions preserved (refactor must be proven
non-breaking first), plus new cases:

- data-URL image -> file part; remote URL -> image part
- body user-agent only for `zai/glm-5.2` (+ overrides)
- `reasoning-delta` -> `reasoning_content` chunk
- tool result name mismatch rejected; duplicate-key args rejected
- usage details (cached/reasoning tokens)
- extra events (`start-step`, `finish-step`, `tool-result`, `source`, `file`,
  `raw`, `response-metadata`) do not break the stream; finish-step usage
  fallback; modelId capture
- CLI still works via `python -m converter`

Validation command: `cd gateway-proxy && uv sync && uv run pytest tests/`.

## Out of scope

- Fetching remote image URLs in the converter (stays pure).
- New server endpoints.
- Changing the Responses API event shapes (moved only).

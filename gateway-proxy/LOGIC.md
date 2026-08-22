# Proxy Logic — Complete Technical Documentation

This document explains every piece of the proxy: what it does, how it
translates between formats, why each design choice was made, and how data
flows end-to-end. It is the reference for anyone modifying the codebase.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [The Problem Being Solved](#2-the-problem-being-solved)
3. [Layer Split](#3-layer-split)
4. [Request Pipeline (Inbound)](#4-request-pipeline-inbound)
5. [Response Pipeline (Outbound)](#5-response-pipeline-outbound)
6. [The v3 AI SDK Protocol](#6-the-v3-ai-sdk-protocol)
7. [Converter Modules (Deep Dive)](#7-converter-modules-deep-dive)
8. [Server Modules (Deep Dive)](#8-server-modules-deep-dive)
9. [Reasoning / Thinking Pipeline](#9-reasoning--thinking-pipeline)
10. [Tool Call Translation](#10-tool-call-translation)
11. [Image Handling](#11-image-handling)
12. [Identity Sync](#12-identity-sync)
13. [Key Pool and Failover](#13-key-pool-and-failover)
14. [Usage Tracking](#14-usage-tracking)
15. [Validation](#15-validation)
16. [Protocol Invariants](#16-protocol-invariants)
17. [Test Suite](#17-test-suite)

---

## 1. Architecture Overview

```
                         ┌─────────────────────────────────────────────┐
                         │                  PROXY                       │
                         │                                             │
  OpenAI client ───────► │  /v1/chat/completions                       │
  (curl, CLI, SDK)       │  /v1/responses                              │
                         │  /v1/embeddings                             │
  ─────────────────────  │                                             │  ─────────────────────
                         │                                             │
  Anthropic client ────► │  /v1/messages                               │  Upstream Gateway
  (Claude Code, SDK)     │  /v1/messages/count_tokens                  │  ai-gateway.vercel.sh
                         │                                             │
  ─────────────────────  │  ┌─────────────────────────────────────┐    │  ─────────────────────
                         │  │         converter/ (pure)          │    │
                         │  │                                     │    │  POST /v3/ai/language-model
                         │  │  OpenAI ──► v3   (request.py)       │    │  (AI SDK v3 protocol)
                         │  │  v3 ──► OpenAI   (streaming.py,     │    │
                         │  │                   response.py)      │    │  Headers:
                         │  │  Anthropic ──► OpenAI (anthropic.py)│    │    User-Agent: fx/<ver>
                         │  │  Responses ──► OpenAI (responses.py)│    │    ai-gateway-protocol-version
                         │  │  Validation     (validation.py)    │    │    ai-language-model-specification-version: 4
                         │  │  Parts/shapes   (parts.py)         │    │    ai-language-model-streaming: true
                         │  └─────────────────────────────────────┘    │
                         │                                             │
                         │  server.py (FastAPI, HTTP transport)        │
                         │  identity.py (fx version sync)             │
                         │  keys.py (key pool + failover)              │
                         │  usage.py (per-caller tracking)            │
                         └─────────────────────────────────────────────┘
```

The proxy accepts requests in three different client formats (OpenAI chat
completions, OpenAI Responses API, Anthropic Messages API), translates all
of them to a single canonical intermediate format (OpenAI chat-completions
shape), then translates that into the v3 AI SDK protocol the Vercel AI
Gateway expects. Responses flow back through the same pipeline in reverse.

---

## 2. The Problem Being Solved

The Vercel AI Gateway has two endpoints:

| Endpoint | Protocol | Credit card required |
|---|---|---|
| `/v1/chat/completions` | OpenAI v1 | Yes |
| `/v3/ai/language-model` | AI SDK v3 | No (for free models like `zai/glm-5.2`) |

The fx CLI uses the v3 endpoint with specific headers. This proxy does the
same thing, so you get the same free-tier access without needing the fx CLI
or a credit card.

The v3 endpoint speaks a different protocol than OpenAI:
- Different body shape (`prompt` instead of `messages`, `toolChoice` instead
  of `tool_choice`, `maxOutputTokens` instead of `max_tokens`, etc.)
- Different content part shapes (tool calls are content parts, not top-level
  fields)
- Different SSE event types (`text-delta`, `tool-call`, `reasoning-delta`
  instead of OpenAI's delta chunks)
- Different headers (fx CLI identity headers that must match exactly)

The proxy bridges all of these differences.

---

## 3. Layer Split

The codebase enforces a strict two-layer split:

### `converter/` package — Pure translation (no I/O)

All format translation logic lives here. These modules:
- Have no side effects
- Don't read environment variables
- Don't make network calls
- Don't import `server.py` or each other's internals (only via `__init__.py` re-exports)

This makes them trivially testable with plain dict inputs and assertions.

### `server.py` — HTTP transport (all I/O)

Everything that touches the network, environment, or filesystem stays here:
- FastAPI routes and middleware
- Auth (proxy key verification)
- Upstream HTTP client (shared `httpx.AsyncClient`, HTTP/2)
- Key pool management
- Usage tracking
- Request/response logging
- Image fetching (hydration)

### Why this split?

- **Testability**: converter functions are tested with plain dicts, no
  mocking of HTTP, env, or async fixtures needed.
- **Reusability**: the converter package can be used as a standalone library
  (e.g., `python -m converter body.json` runs the OpenAI-to-v3 conversion
  from the CLI).
- **Clarity**: when a 400/503 comes back from upstream, you know the issue
  is either in the converter (wrong body shape) or server (wrong headers),
  and you know exactly which file to look at.

---

## 4. Request Pipeline (Inbound)

Every request follows the same pattern regardless of which API format the
client uses:

```
Client request (any format)
    │
    ├─ 1. Auth check (verify_proxy_key or verify_anthropic_key)
    │
    ├─ 2. Parse JSON body
    │
    ├─ 3. Convert to OpenAI chat-completions shape
    │     ┌─────────────────────────────────────────────────────┐
    │     │ /v1/chat/completions: body is already OpenAI         │
    │     │ /v1/responses:        responses_input_to_messages()  │
    │     │ /v1/messages:         anthropic_to_openai()          │
    │     └─────────────────────────────────────────────────────┘
    │
    ├─ 4. validate_tool_history() — check tool call/result pairing
    │
    ├─ 5. _hydrate_remote_images() — download http(s) image URLs to data URLs
    │
    ├─ 6. openai_to_v3() — convert OpenAI body to v3 AI SDK format
    │
    ├─ 7. _v3_headers() — build fx-identity headers
    │
    ├─ 8. _upstream_pooled() — send to gateway (with key rotation/failover)
    │
    └─ 9. Response handling (see next section)
```

### Step 3: Format conversion to OpenAI

All three client formats are converted to an OpenAI chat-completions body
first. This is the canonical intermediate shape. The conversion is always
one-way and lossless:

**`/v1/chat/completions`**: No conversion needed — the body is already
OpenAI format. It goes straight to step 4.

**`/v1/responses`**: The Responses API uses `input` items (typed objects
like `{"type": "message", "role": "user", ...}` or
`{"type": "function_call", ...}`) instead of `messages`.
`responses_input_to_messages()` in `converter/responses.py` flattens these
into standard OpenAI messages. Function call items become assistant
`tool_calls`; function call outputs become `role: tool` messages.

**`/v1/messages`**: The Anthropic Messages API uses a different message
shape (content blocks, top-level `system`, `tool_use`/`tool_result` blocks,
`input_schema` on tools). `anthropic_to_openai()` in
`converter/anthropic.py` translates:
- `system` (string or block array) → system message
- Content blocks (`text`, `image`, `tool_use`, `tool_result`) → OpenAI
  message parts
- `tools` with `input_schema` → OpenAI function tools with `parameters`
- `tool_choice` (`auto`/`any`/`tool`/`none`) → OpenAI tool_choice
- `max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences` → OpenAI
  params
- `thinking` config → `reasoning` field (see [Reasoning](#9-reasoning--thinking-pipeline))

### Step 6: `openai_to_v3()` — The core translation

This is where the OpenAI chat-completions body becomes a v3 AI SDK body.
The function lives in `converter/request.py` and does:

| OpenAI field | v3 field | Notes |
|---|---|---|
| `messages` | `prompt` | Role-dependent content shape (see below) |
| `tools` | `tools` | Flat shape: `{type, name, description, inputSchema}` |
| `tool_choice` | `toolChoice` | Object shape: `{type: "auto"}` etc. |
| `temperature` | `temperature` | Direct passthrough |
| `max_tokens` | `maxOutputTokens` | Renamed |
| `top_p` | `topP` | Renamed |
| `top_k` | `topK` | Renamed |
| `stop` | `stopSequences` | Always a list |
| `response_format` | `responseFormat` | JSON mode / JSON schema |
| `reasoning` / `reasoning_effort` | `reasoning` | String label (see [Reasoning](#9-reasoning--thinking-pipeline)) |
| `providerOptions` | `providerOptions` | Direct passthrough |

**Critical: `tools: []` and `toolChoice: {type: auto}` are always injected**
even when the client sends no tools. Missing either causes a 503 from the
gateway.

**Content shape is role-dependent**:
- `system` content MUST be a plain string (not an array)
- `user`/`assistant`/`tool` content MUST be an array of parts (not a string)

These are enforced by `_openai_content_to_v3_parts()` in `parts.py`, which
wraps bare strings into `[{type: "text", text: "..."}]` and normalizes
image URLs into v3 file/image parts.

---

## 5. Response Pipeline (Outbound)

The upstream gateway always responds with a v3 SSE stream, even for
non-streaming client requests. The proxy handles both cases:

### Streaming response

```
Upstream v3 SSE stream
    │
    ├─ _v3_lines_to_events() — parse raw SSE lines to event dicts
    │
    ├─ v3_stream_iter() — translate v3 events to OpenAI chat.completion.chunk SSE
    │     ┌─────────────────────────────────────────────────────────┐
    │     │  text-delta       → delta.content                       │
    │     │  reasoning-delta   → delta.reasoning_content             │
    │     │  tool-call         → delta.tool_calls                    │
    │     │  tool-input-delta  → buffered into tool-call             │
    │     │  finish            → finish_reason + usage                │
    │     │  error             → error chunk + stop                  │
    │     └─────────────────────────────────────────────────────────┘
    │
    ├─ If /v1/chat/completions: yield OpenAI chunks directly
    │
    ├─ If /v1/responses: openai_chunk_to_responses_sse() re-translates to Responses SSE
    │
    └─ If /v1/messages: anthropic_stream_iter() re-translates to Anthropic SSE
```

### Non-streaming response

```
Upstream v3 SSE stream
    │
    ├─ _collect_response() — consume entire stream, collect events
    │
    ├─ v3_sse_stream_to_openai() — aggregate events into one OpenAI chat.completion dict
    │     ┌─────────────────────────────────────────────────────────┐
    │     │  text-delta       → message.content                      │
    │     │  reasoning-delta  → message.reasoning_content            │
    │     │  tool-call         → message.tool_calls                  │
    │     │  finish            → finish_reason + usage                │
    │     └─────────────────────────────────────────────────────────┘
    │
    ├─ If /v1/chat/completions: return OpenAI response directly
    │
    ├─ If /v1/responses: openai_to_responses() reshapes to Responses API
    │
    └─ If /v1/messages: openai_to_anthropic() reshapes to Anthropic Message
```

The key insight: **OpenAI chat-completions is the canonical intermediate
shape for both requests and responses.** All client formats convert to
OpenAI on the way in, and from OpenAI on the way out. The v3 protocol is
only spoken to the upstream gateway.

---

## 6. The v3 AI SDK Protocol

The v3 protocol is what the fx CLI speaks to the Vercel AI Gateway. It is
documented in `SAUCE.md` (the reverse-engineering writeup). Here is what
makes it different from OpenAI:

### Body shape

```json
{
  "prompt": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
  ],
  "tools": [],
  "toolChoice": {"type": "auto"},
  "temperature": 0.7,
  "maxOutputTokens": 1024,
  "topP": 0.9,
  "topK": 40,
  "stopSequences": ["END"],
  "responseFormat": {"type": "json", "name": "my_schema", "schema": {...}},
  "reasoning": "high",
  "headers": {"user-agent": "fx/0.0.4"},
  "providerOptions": {}
}
```

### SSE event types

| v3 event | Meaning | OpenAI equivalent |
|---|---|---|
| `start` | Stream begin | First chunk (role: assistant) |
| `start-step` | Model step begin | (no-op) |
| `response-metadata` | Model ID, response metadata | (used for model resolution) |
| `text-delta` | Text token | `delta.content` |
| `text-start` / `text-end` | Text block boundaries | (no-op) |
| `reasoning-start` | Reasoning block begin | (no-op) |
| `reasoning-delta` | Reasoning token | `delta.reasoning_content` |
| `reasoning-end` | Reasoning block end | (no-op) |
| `tool-input-start` | Tool call begin (with name) | (buffered) |
| `tool-input-delta` | Tool argument fragment | (buffered) |
| `tool-input-end` | Tool input complete | (no-op) |
| `tool-call` | Consolidated tool call | `delta.tool_calls` |
| `tool-result` | Tool result | (no-op) |
| `finish-step` | Step end (with usage) | (usage captured) |
| `finish` | Stream end | `finish_reason` + `usage` |
| `error` | Upstream error | error chunk |

### Headers

The gateway requires specific headers that match the fx CLI exactly:

```
Authorization: Bearer <gateway_key>
User-Agent: fx/<version>
HTTP-Referer: https://github.com/vercel-labs/fx
X-Title: fx
ai-gateway-protocol-version: 0.0.1
ai-language-model-specification-version: 4
ai-language-model-id: <model>
ai-language-model-streaming: true
```

These are built by `_v3_headers()` in `server.py`, with values sourced
live from `identity.state` (which is kept fresh by the background sync
in `identity.py`).

---

## 7. Converter Modules (Deep Dive)

### `converter/parts.py` — Low-level part shapes

Pure helpers that convert individual pieces of an OpenAI request to v3
shapes:

- **`_openai_content_to_v3_parts(content)`**: Converts message content
  (string, list, None) to v3 array-of-parts. Strings become
  `[{type: "text", text: "..."}]`. Image URLs become v3 file parts (for
  data URLs) or image parts (for remote URLs). Never returns null or empty
  (always at least one text part).

- **`_openai_tool_call_to_v3(tool_call)`**: Converts an OpenAI tool call
  `{id, type: "function", function: {name, arguments}}` to a v3 content
  part `{type: "tool-call", toolCallId, toolName, input}`. The `input`
  field is a raw JSON object (not a string), which is the fx wire format.

- **`_openai_tool_msg_to_v3(msg)`**: Converts an OpenAI `role: tool`
  message to a v3 tool-result content part. The `toolName` is back-filled
  by the caller from the preceding assistant message (defaulting to
  `"unknown"`, matching fx behavior).

- **`_normalize_tool_choice(tc)`**: Normalizes OpenAI's tool_choice (string
  or object) to the v3 object shape. `"auto"` → `{type: "auto"}`,
  `"required"` → `{type: "required"}`, `{"type": "function", ...}` →
  `{type: "tool", toolName: ...}`.

- **`_openai_response_format_to_v3(rf)`**: Maps OpenAI response_format
  to v3 responseFormat. `json_object` → `{type: "json", schema: {}}`.
  `json_schema` → `{type: "json", name, description, schema}`.

- **`_image_url_to_v3_part(url)`**: Splits data URLs into
  `{type: "file", mediaType, data}` and keeps remote URLs as
  `{type: "image", image: url}`.

### `converter/request.py` — Full request assembly

The `openai_to_v3()` function is the main entry point. It:

1. Builds a `tool_call_name_map` from assistant messages so tool results
   can be back-filled with the correct `toolName`.
2. Iterates messages, converting each to v3 prompt format:
   - `system` → `{role: "system", content: "<string>"}`
   - `user`/`assistant` → `{role: ..., content: [parts]}`
   - `tool` → `{role: "tool", content: [{type: "tool-result", ...}]}`
   - Assistant with `tool_calls` → content parts include both text and
     tool-call parts
3. Builds the `tools` array in flat v3 format (not nested under
   `function`).
4. Injects `tools: []` and `toolChoice: {type: auto}` (always present).
5. Maps all optional parameters (temperature, maxOutputTokens, topP,
   topK, stopSequences, responseFormat, reasoning, providerOptions).
6. Adds body-level `headers.user-agent` for specific models (fx only
   sends it for `zai/glm-5.2`; controlled by
   `product_user_agent_models`).

### `converter/streaming.py` — SSE stream translation

Two modes:

**Streaming (`v3_stream_iter`)**: An async generator that yields OpenAI
SSE chunk strings as v3 events arrive. Uses `_StreamState` to track tool
call indices, deduplicate events, and buffer tool input deltas. The state
machine in `_process_stream_event()` handles every v3 event type:

- `text-delta` → emits an OpenAI chunk with `delta.content`
- `reasoning-delta` → emits an OpenAI chunk with `delta.reasoning_content`
- `tool-call` → emits an OpenAI chunk with `delta.tool_calls`
- `tool-input-start` → records the tool name (for back-fill)
- `tool-input-delta` → buffers arguments (used if tool-call omits input)
- `finish` → emits the final chunk with `finish_reason` and `usage`
- `error` → emits an error chunk and a stop

Tool call robustness (fx parity): duplicate consolidated `tool-call` events
are deduplicated by `toolCallId`. Anonymous calls (no id) are deduplicated
by `(name, args)` signature. Missing ids are minted as `call_<uuid>` since
OpenAI clients can't reply without an id.

**Non-streaming (`v3_sse_stream_to_openai`)**: Consumes the entire v3
event stream and returns a single OpenAI chat.completion dict. Collects
text parts, reasoning parts, and tool calls. Handles the same tool-call
robustness as the streaming path.

### `converter/response.py` — Non-streaming v3 response

The `v3_to_openai()` function converts a v3 gateway response (which has
content parts in the v3 shape) to an OpenAI chat.completion response.
Handles:
- Text parts → `message.content`
- Tool-call parts → `message.tool_calls`
- Finish reason mapping (`stop` → `stop`, `tool-calls` → `tool_calls`, etc.)
- Usage mapping (inputTokens/outputTokens → prompt_tokens/completion_tokens)

Also exports `_v3_finish_reason()` and `_v3_usage_to_openai()` which are
shared with the streaming converter.

Usage detail mapping includes:
- `prompt_tokens_details.cached_tokens` (from v3 `cacheRead`)
- `output_tokens_details.reasoning_tokens` (from v3 `reasoning`)

### `converter/responses.py` — Responses API translation

Translates the OpenAI Responses API format (used by the newer OpenAI SDKs):

- **`responses_input_to_messages(input)`**: Converts Responses API input
  items (typed objects) to standard OpenAI messages. `function_call`
  items become assistant tool_calls; `function_call_output` items become
  tool messages. Ordering invariant is preserved (tool calls flushed
  before their outputs).

- **`openai_to_responses(resp, model)`**: Converts an OpenAI
  chat.completion response to the Responses API shape with `output` items
  (message items with `output_text` content, function_call items).

- **`_ResponsesStreamState` + `openai_chunk_to_responses_sse()`**: A
  stateful translator that converts OpenAI streaming chunks to Responses
  SSE events (`response.created`, `response.output_text.delta`,
  `response.function_call_arguments.delta`, `response.completed`, etc.).

### `converter/anthropic.py` — Anthropic Messages API translation

The largest converter module. Translates the Anthropic Messages API wire
format (used by Claude Code, Anthropic SDKs) to/from OpenAI shape.

**Request side (`anthropic_to_openai`)**:
- `system` (string or block array) → system message
- Content blocks: `text` → text parts, `image` → image_url parts,
  `tool_use` → tool_calls, `tool_result` → tool messages
- `tools` with `input_schema` → OpenAI function tools with `parameters`
- `tool_choice` (`auto`/`any`/`tool`/`none`) → OpenAI tool_choice
  (`auto`/`required`/`{type: function, ...}`/`none`)
- `max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences` → OpenAI
  params
- `thinking` config → `reasoning` field (see [Reasoning](#9-reasoning--thinking-pipeline))

**Non-streaming response (`openai_to_anthropic`)**:
- `message.content` → `text` content block
- `message.reasoning_content` → `thinking` content block (placed before
  text blocks)
- `message.tool_calls` → `tool_use` content blocks
- `finish_reason` mapping: `stop` → `end_turn`, `length` → `max_tokens`,
  `tool_calls` → `tool_use`
- `usage` mapping: `prompt_tokens` → `input_tokens`,
  `completion_tokens` → `output_tokens`, plus cache fields

**Streaming response (`_AnthropicStreamState` + `anthropic_stream_iter`)**:

A stateful translator that converts OpenAI streaming chunks to Anthropic
SSE events. The state machine tracks:
- `message_start` — emitted once at the beginning
- `content_block_start` — emitted when a new content block (thinking,
  text, or tool_use) begins
- `content_block_delta` — emitted for each delta (thinking_delta,
  text_delta, input_json_delta)
- `content_block_stop` — emitted when a content block ends
- `message_delta` — emitted at the end with stop_reason and usage
- `message_stop` — final event

Content blocks are **sequential, not interleaved** (matching the Anthropic
protocol). A thinking block is closed before a text or tool block starts.
This is enforced by `text_delta()` and `tool_call()` both checking for an
open thinking block and closing it first.

**Token counting (`count_anthropic_tokens`)**:
A character-based heuristic (total chars / 4, rounded up). No tokenizer
dependency. Accurate enough for planning, not for billing.

### `converter/validation.py` — Tool history validation

`validate_tool_history(messages)` checks that assistant tool calls are
properly paired with tool results, modeled on fx's
`validateToolMessageHistory`:

- Tool role messages must be preceded by an assistant tool call block
- Every tool call must have a unique id, non-empty name, and valid JSON
  arguments (duplicate keys in JSON are rejected)
- The messages immediately following a tool-calling assistant message must
  be tool results covering every call (in any order)
- Tool result ids must match call ids
- Tool result names (if present) must match call names

Returns an error message string or None if valid. This runs client-side so
the proxy returns a clear 400 instead of the gateway's opaque "Invalid
input".

---

## 8. Server Modules (Deep Dive)

### `server.py` — FastAPI app

**Lifecycle (`lifespan`)**:
- Creates a shared `httpx.AsyncClient` (HTTP/2, connection pooling,
  configurable timeouts)
- Starts the identity sync background task
- Logs key pool status
- On shutdown: cancels identity task, closes HTTP client

**Middleware**:
- Request logging (method, path, status code, duration)

**Auth**:
- `verify_proxy_key()`: For OpenAI routes. Checks `Authorization: Bearer`
  against `PROXY_API_KEY`. When no key is set, proxy is open.
- `verify_anthropic_key()`: For Anthropic routes. Accepts `x-api-key`
  header (Claude Code style) or `Authorization: Bearer`. Raises
  `AnthropicError` (not `HTTPException`) so error bodies match the
  Anthropic shape.

**Routes**:

| Route | Auth | Converter path |
|---|---|---|
| `POST /v1/chat/completions` | Bearer | OpenAI → v3 directly |
| `POST /v1/responses` | Bearer | Responses → OpenAI → v3 |
| `POST /v1/messages` | x-api-key or Bearer | Anthropic → OpenAI → v3 |
| `POST /v1/messages/count_tokens` | x-api-key or Bearer | count_anthropic_tokens() |
| `GET /v1/models` | Bearer | Cached gateway model list |
| `POST /v1/embeddings` | Bearer | Direct v1 passthrough |
| `GET /v1/usage` | Bearer | UsageTracker snapshot |
| `GET /healthz` | None | Health + key pool + usage |

**Shared helpers**:

- `_v3_headers(model, streaming, api_key, ...)` — Builds the fx-identity
  headers. Values come from `identity.state` (live-synced). Always sets
  `ai-language-model-streaming: true`.

- `_send_to_v3(client, request, chat_body, model)` — Shared upstream send
  used by the Anthropic route. Hydrates images, builds v3 body, pooled send.
  Returns `(response, used_key)`.

- `_upstream_pooled(build)` — Key pool wrapper. Tries keys in round-robin
  order, retries on key-attributable failures (401/402/403/408/429/5xx),
  applies cooldowns. Returns `(response, used_key)`.

- `_chat_stream(resp, model, include_usage)` — Streaming generator that
  consumes the upstream v3 stream and yields OpenAI SSE chunks. Closes
  the upstream response in a `finally` block (client disconnect cancels
  the upstream request).

- `_anthropic_stream(resp, model)` — Streaming generator that wraps
  `_chat_stream` and re-translates OpenAI chunks to Anthropic SSE via
  `anthropic_stream_iter()`.

- `_responses_stream(resp, model)` — Streaming generator that wraps
  `_chat_stream` and re-translates OpenAI chunks to Responses SSE via
  `openai_chunk_to_responses_sse()`.

- `_collect_response(resp, model)` — Non-streaming helper that consumes
  the entire upstream stream and returns a single OpenAI chat.completion
  dict via `v3_sse_stream_to_openai()`.

- `_tracked_stream(aiter, caller, model)` — Pass-through wrapper that
  extracts usage from the final chunk and records it in the
  `UsageTracker`.

- `_hydrate_remote_images(client, body)` — Downloads http(s) image URLs
  and converts them to data URLs before conversion (fx parity: fx fetches
  attachments locally because the gateway can't fetch URLs).

- `_client_error(resp)` — Normalizes upstream errors to OpenAI error
  shape `{error: {message, type}}`.

- `_anthropic_error(resp)` — Normalizes upstream errors to Anthropic
  error shape `{type: "error", error: {type, message}}`.

**Error handling**:
- `AnthropicError` exception class with a custom handler that renders the
  Anthropic error JSON shape. Raised by Anthropic route auth/validation.
- OpenAI routes use `HTTPException` and `_invalid_request()` /
  `_client_error()` helpers.

### `identity.py` — Live fx identity sync

Keeps the proxy's wire identity fresh by syncing from GitHub:

1. **Latest release tag**: Fetches
   `https://api.github.com/repos/vercel-labs/fx/releases/latest` and
   extracts the version from `tag_name` (strips the `v` prefix).
   User-Agent becomes `fx/<version>`.

2. **Protocol/spec versions**: Fetches the raw fx source
   (`client.zig`) and regexes out `ai-gateway-protocol-version` and
   `ai-language-model-specification-version`.

3. **Local binary fallback**: If GitHub sync is disabled, checks for a
   locally installed `fx` binary and uses its version.

State is stored in `identity.state` (a mutable dict), read on every
request by `_v3_headers()` and `openai_to_v3()`.

Env knobs:
- `FX_AUTO_UPDATE=0` — disable background sync
- `FX_USER_AGENT=fx/1.2.3` — pin the User-Agent manually
- `FX_REFRESH_SECS=3600` — refresh interval

### `keys.py` — Key pool with failover

`KeyPool` manages multiple gateway API keys:

- **Round-robin**: Requests alternate across keys
  (`KEY_ROTATION=1`, default on).
- **Failover**: On a key-attributable error (401/402/403/408/429/5xx) or
  network error, transparently retries the next key
  (`KEY_FAILOVER=1`, default on). Request faults like 400 are not retried.
- **Cooldown**: A failing key sits out of rotation for `KEY_COOLDOWN`
  seconds (default 30). If every key is cooling, the coolest is still
  handed out so the proxy degrades instead of dying.

Thread-safe (uses a `threading.Lock`).

Keys are loaded from env:
```
AI_GATEWAY_API_KEY=vck_key1           # legacy single key (or comma-separated)
AI_GATEWAY_API_KEY_1=vck_key2         # numbered keys
AI_GATEWAY_API_KEY_2=vck_key3         # up to _20
```

### `usage.py` — Per-caller usage tracking

`UsageTracker` is an in-memory, thread-safe counter:

- Records requests, errors, prompt tokens, and completion tokens per
  calling client (keyed by IP address).
- Token counts come from the upstream's usage field.
- Streams that end without a usage chunk still count as requests.
- Exposed at `GET /v1/usage` (full per-caller breakdown) and summarized
  in `/healthz`.
- In-memory only; resets on restart.
- `USAGE_TRACKING=0` disables (record becomes a no-op).

---

## 9. Reasoning / Thinking Pipeline

Reasoning (also called "thinking" or "extended thinking") flows through all
three API surfaces. Here is the complete path:

### Request side (client → upstream)

The v3 gateway accepts `reasoning` as a plain string label in the body.
The fx CLI passes model-catalog-defined values like `"low"`, `"high"`,
`"max"` through verbatim. The value is opaque to the proxy.

**From OpenAI clients** (`/v1/chat/completions`):
- `reasoning_effort: "high"` → `reasoning: "high"` in v3 body
- `reasoning: "high"` → `reasoning: "high"` in v3 body
- `reasoning: {"effort": "high"}` → `reasoning: "high"` in v3 body
- Omitted → `reasoning` field omitted entirely (matching fx CLI behavior
  with `auto`)

**From Anthropic clients** (`/v1/messages`):
- `thinking: {"type": "enabled", "budget_tokens": N}` →
  `reasoning: "enabled"` in v3 body
- `thinking: {"type": "disabled"}` → `reasoning` field omitted
- Explicit `reasoning` or `reasoning_effort` in the body always takes
  precedence over the thinking-derived default

**From Responses API** (`/v1/responses`):
- Passes through the OpenAI body's `reasoning` / `reasoning_effort` the
  same way as chat completions

The proxy always forwards the reasoning field upstream regardless of the
client-side level or label, so the model uses its highest available
reasoning level.

### Streaming response (upstream → client)

The v3 SSE stream emits reasoning as separate events:
- `reasoning-start` — reasoning block begins (no-op in OpenAI translation)
- `reasoning-delta` — reasoning text token
- `reasoning-end` — reasoning block ends (no-op)

**`v3_stream_iter()`** (streaming) and **`v3_sse_stream_to_openai()`**
(non-streaming) in `streaming.py` collect `reasoning-delta` events into
`reasoning_content` on the message/delta:

- Streaming: each `reasoning-delta` emits an OpenAI chunk with
  `delta.reasoning_content`
- Non-streaming: all `reasoning-delta` events are concatenated into
  `message.reasoning_content`

**For Anthropic clients**, `_AnthropicStreamState.thinking_delta()` in
`anthropic.py` converts `reasoning_content` to Anthropic `thinking`
content blocks:

```
event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me think..."}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: content_block_start
data: {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "The answer is..."}}
```

### Non-streaming response

`openai_to_anthropic()` in `anthropic.py` checks for `reasoning_content`
on the OpenAI message and adds a `thinking` content block before the
text block:

```json
{
  "content": [
    {"type": "thinking", "thinking": "Let me think about this..."},
    {"type": "text", "text": "The answer is 42."}
  ]
}
```

### Content block ordering (Anthropic)

Anthropic's protocol requires content blocks to be **sequential, not
interleaved**. A thinking block must be closed (`content_block_stop`)
before a text or tool block starts. This is enforced in:

- `_AnthropicStreamState.text_delta()` — closes any open thinking block
  before starting a text block
- `_AnthropicStreamState.tool_call()` — closes any open thinking block
  before starting a tool block
- `_AnthropicStreamState.finish()` — closes any remaining open block
  (thinking, text, or tool)

The `next_block_index` counter ensures each block gets a unique, sequential
index.

### Usage tracking

The v3 `finish` event's usage includes `outputTokens.reasoning` (the
number of reasoning tokens). This is mapped to
`output_tokens_details.reasoning_tokens` in the OpenAI usage shape, and
included in the Anthropic usage as part of `output_tokens`.

---

## 10. Tool Call Translation

Tool calls have different wire formats across the three APIs. Here is how
each is translated:

### OpenAI → v3 (request)

OpenAI tool calls are top-level on the message:
```json
{"role": "assistant", "tool_calls": [
  {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}}
]}
```

v3 tool calls are content parts (fx wire format):
```json
{"role": "assistant", "content": [
  {"type": "tool-call", "toolCallId": "call_1", "toolName": "get_weather", "input": {"city": "SF"}}
]}
```

The `input` field is a raw JSON object (not a string). The proxy parses
the OpenAI `arguments` string to an object.

### v3 → OpenAI (response)

The v3 stream emits tool calls as events:
- `tool-input-start` (with `toolCallId` and `toolName`)
- `tool-input-delta` (argument fragments)
- `tool-call` (consolidated, with full `input`)

The proxy buffers `tool-input-delta` fragments and emits the consolidated
OpenAI tool call when `tool-call` arrives. If `tool-call` omits the name,
it's back-filled from `tool-input-start`.

### Anthropic → OpenAI (request)

Anthropic tool calls are `tool_use` content blocks:
```json
{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}
```

Converted to OpenAI tool calls:
```json
{"id": "toolu_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}}
```

Anthropic tool results are `tool_result` blocks in user messages:
```json
{"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny, 72°F"}
```

Converted to OpenAI tool messages:
```json
{"role": "tool", "tool_call_id": "toolu_1", "content": "Sunny, 72°F"}
```

### OpenAI → Anthropic (response)

OpenAI tool calls are converted back to Anthropic `tool_use` blocks:
```json
{"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "SF"}}
```

In streaming, the Anthropic state machine emits:
1. `content_block_start` with `{type: "tool_use", id, name, input: {}}`
2. `content_block_delta` with `{type: "input_json_delta", partial_json: ...}`
3. `content_block_stop`

Tool call arguments are streamed incrementally via `input_json_delta`
events, matching the Anthropic SDK's expectations.

---

## 11. Image Handling

### Inbound (client → proxy)

Images arrive in different formats:

**OpenAI format**: `{"type": "image_url", "image_url": {"url": "..."}}`
where the URL is either a data URL (`data:image/png;base64,...`) or a
remote URL (`https://...`).

**Anthropic format**: `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}`

### Conversion to v3

`_image_url_to_v3_part()` in `parts.py`:
- Data URLs → `{type: "file", mediaType: "<mime>", data: "<base64>"}`
- Remote URLs → `{type: "image", image: "<url>"}`

Anthropic image blocks are converted to OpenAI `image_url` parts (with
data URLs) by `anthropic_to_openai()`, then go through the same v3
conversion.

### Remote image hydration

`_hydrate_remote_images()` in `server.py` downloads remote http(s) image
URLs and converts them to data URLs before conversion. This matches fx
behavior (fx fetches attachments locally because the gateway cannot fetch
URLs). Failures leave the original URL unchanged.

Config:
- `IMAGE_FETCH=1` (default on) — enable/disable
- `IMAGE_FETCH_MAX_BYTES` — size limit (default 5MB)
- `IMAGE_FETCH_TIMEOUT` — download timeout (default 10s)

---

## 12. Identity Sync

The proxy must send headers that match the fx CLI exactly. If Vercel
ships a new fx release with a new version number or bumped protocol
headers, a stale hardcoded identity may get rejected.

`identity.py` solves this by syncing from GitHub on a background loop:

1. **Fetch latest release tag** from the GitHub API → User-Agent becomes
   `fx/<version>`.
2. **Fetch raw fx source** (`client.zig`) → regex out
   `ai-gateway-protocol-version` and
   `ai-language-model-specification-version`.
3. **Local binary fallback** → if GitHub sync is disabled, use the
   locally installed fx CLI's version (via `fx --version`).
4. **Hot-swap** → state is stored in a mutable dict, read on every
   request by `_v3_headers()` and `openai_to_v3()`.

The sync never raises — if it fails, the proxy keeps using the current
state. The refresh interval is configurable (`FX_REFRESH_SECS=3600`).

---

## 13. Key Pool and Failover

Multiple gateway keys can be configured for load distribution and
redundancy:

**Configuration** (in `.env`):
```
AI_GATEWAY_API_KEY=vck_primary
AI_GATEWAY_API_KEY_1=vck_backup_1
AI_GATEWAY_API_KEY_2=vck_backup_2   # up to _20
```

**Round-robin** (`KEY_ROTATION=1`, default on):
Requests alternate across keys. The internal index advances after each
`next()` call.

**Failover** (`KEY_FAILOVER=1`, default on):
On a key-attributable error (401/402/403/408/429/5xx) or network error,
the request is transparently retried on the next key. Request faults
(400) are not retried — they indicate a problem with the request body,
not the key.

**Cooldown** (`KEY_COOLDOWN=30`, default 30 seconds):
A failing key sits out of rotation for the cooldown period. If every key
is cooling, the coolest one is still handed out so the proxy degrades
gracefully instead of refusing all requests.

The `_upstream_pooled()` function in `server.py` wraps the send logic:
it tries keys in order, retries on failures, and returns the first
successful response (or the last failing one if all keys are exhausted).

---

## 14. Usage Tracking

`UsageTracker` in `usage.py` is a thread-safe, in-memory counter that
tracks per-caller (by IP address) metrics:

- **Requests**: total count
- **Errors**: count of non-200 responses
- **Prompt tokens**: from upstream usage
- **Completion tokens**: from upstream usage
- **Models**: per-model request count

Data is collected by:
- `_tracked_stream()` — wraps streaming responses, extracts usage from the
  final chunk
- Direct `USAGE.record()` calls on non-streaming responses and error paths

Exposed at:
- `GET /v1/usage` — full per-caller breakdown
- `GET /healthz` — aggregated totals

In-memory only; resets on restart. `USAGE_TRACKING=0` disables.

---

## 15. Validation

`validate_tool_history()` in `converter/validation.py` checks the
OpenAI-shaped messages for proper tool call / result pairing before
sending to the gateway. It runs on **all three routes** (after the
`→openai` step for Anthropic/Responses).

Checks:
1. Every assistant `tool_calls` block must have a unique id, non-empty
   name, and valid JSON arguments
2. Duplicate keys in JSON arguments are rejected (fx parity — the gateway
   rejects these)
3. The messages immediately following a tool-calling assistant message
   must be `role: tool` results covering every call id
4. Tool result ids must match call ids
5. Tool result names (if present) must match call names
6. All calls must have results (no missing, no extra)

Returns an error message string or None if valid. The server returns a
clear 400 with the error message instead of the gateway's opaque
"Invalid input".

---

## 16. Protocol Invariants

These are hard rules that the gateway enforces. Violating any of them
causes a 400 or 503. They are NOT linter-enforced — treat them as manual
rules when editing converter or server code:

1. **Always send `ai-language-model-streaming: true`** upstream.
   Non-streaming client requests still open a streaming upstream
   connection and collect deltas internally.

2. **v3 body must include `tools: []` and `toolChoice: {type: auto}`**
   even when the client sends no tools. Missing either causes 503.
   `openai_to_v3()` injects them.

3. **Content shape is role-dependent**: `system` content MUST be a plain
   string; `user`/`assistant`/`tool` content MUST be an array of parts.
   Wrong shape causes 400.

4. **fx headers must match the fx CLI exactly**. `_v3_headers()` in
   `server.py` is the single source of truth; values come from
   `identity.state` (live-synced from GitHub).

5. **HTTP/2 must stay enabled** on the shared `httpx.AsyncClient`.

6. **Tool calls use fx wire format**: `{type: "tool-call", toolCallId,
   toolName, input}` with raw JSON `input` (not a string).

7. **Tool history ordering**: every assistant `tool_calls` block MUST be
   immediately followed by matching `role: tool` results covering all
   call ids. `validate_tool_history()` enforces this client-side.

---

## 17. Test Suite

266 tests across 15 files, all using `httpx.MockTransport` (no real
network traffic):

| Test file | Tests | What it covers |
|---|---|---|
| `test_anthropic.py` | 48 | Anthropic ↔ OpenAI conversion (request, response, streaming, thinking blocks, tool indices, async streaming, token counting, reasoning routing) |
| `test_cli.py` | 2 | CLI module entry point |
| `test_identity.py` | 5 | fx identity sync (version parsing, source parsing, apply) |
| `test_keys.py` | 34 | Key pool (round-robin, failover, cooldown, masking, env loading) |
| `test_parts.py` | 18 | Low-level part shapes (content, tool calls, tool messages, tool_choice, response_format, images) |
| `test_request.py` | 27 | openai_to_v3() full request assembly (messages, tools, params, reasoning, top_k, providerOptions) |
| `test_response.py` | 13 | v3_to_openai() non-streaming response (text, tools, usage, reasoning content, reasoning tokens) |
| `test_responses.py` | 16 | Responses API translation (input items, response shape, streaming SSE) |
| `test_server.py` | 53 | Server routes (auth, chat completions, responses, anthropic messages, count_tokens, reasoning, thinking, top_k, embeddings, models, usage, healthz) |
| `test_server_headers.py` | 3 | Header construction (fx identity, team, session) |
| `test_server_images.py` | 3 | Remote image hydration |
| `test_streaming.py` | 31 | v3 SSE → OpenAI SSE (text, tools, reasoning, usage, errors, non-streaming collection) |
| `test_streaming_tools.py` | 6 | Tool call streaming robustness (dedup, back-fill, anonymous calls) |
| `test_usage.py` | 5 | UsageTracker (recording, totals, concurrency, disabled) |
| `test_validation.py` | 16 | Tool history validation (pairing, duplicates, missing results, name mismatch) |

`test_proxy.py` is a **live-server smoke test** (not a pytest) that hits
`http://localhost:8799` — run it manually with the proxy listening.

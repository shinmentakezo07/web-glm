# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Fionn** — an OpenAI-compatible FastAPI proxy that forwards to the Vercel AI Gateway using the same v3 AI SDK protocol the fx CLI uses. This gives OpenAI-compatible CLIs free-tier access to models like `zai/glm-5.2` without a credit card. Full reverse-engineering writeup is in `gateway-proxy/SAUCE.md`; user-facing setup is in `gateway-proxy/README.md`. All meaningful code lives in `gateway-proxy/`.

## Tooling

The project uses **`uv`** (lockfile is `gateway-proxy/uv.lock`); do not use `pip` directly. Python 3.12 is pinned in `gateway-proxy/.python-version`. Run all commands from `gateway-proxy/`:

```bash
uv sync                                              # install deps from lockfile
uv run pytest tests/                                 # unit tests
uv run pytest tests/test_request.py::TestOpenAIToV3  # single test class
uv run server.py                                     # start proxy (see README for full setup)
```

Test files mirror source modules (`tests/test_request.py` ↔ `converter/request.py`, etc.).

For day-to-day run/test setup see `gateway-proxy/README.md`; this file documents only what the README doesn't.

## Layer split (preserve it)

- **`converter/` package** — pure functions, no I/O. OpenAI ↔ AI SDK v3 translation + `validate_tool_history()`. Modules: `request.py` (`openai_to_v3`, includes `top_k` → `topK` mapping), `parts.py` (content/tool-call part shapes), `streaming.py` (v3 SSE → OpenAI chunks, collects `reasoning-delta` events into `reasoning_content`), `response.py` (non-streaming aggregation, surfaces `reasoning_content`), `responses.py` (Responses API shim), `anthropic.py` (Anthropic Messages API ↔ OpenAI translation, `thinking` config routing, `thinking` content blocks), `validation.py`. Add request/response-shape work here, with tests in the matching `tests/test_*.py`.
- **`server.py`** — FastAPI + HTTP transport (shared `httpx.AsyncClient` in `lifespan`, HTTP/2 on). Routes, auth, key pooling, upstream send. Add routes or transport changes here.
- **`identity.py`** — background loop syncing the fx identity (User-Agent + protocol/spec header versions) from the vercel-labs/fx GitHub repo into `identity.state`, hot-swapped in memory. Read on every request by `_v3_headers()` and `openai_to_v3()`. `FX_AUTO_UPDATE=0` disables; `FX_USER_AGENT=fx/x.y.z` pins manually.
- **`keys.py`** — `KeyPool`: multi-key round-robin + failover + cooldown for upstream gateway keys (`AI_GATEWAY_API_KEY`, `AI_GATEWAY_API_KEY_1..20`). Used via `_upstream_pooled()` in `server.py`.
- **`usage.py`** — `UsageTracker`: in-memory per-caller request/error/token counters (thread-safe, reset on restart), fed by `_tracked_stream()` and direct `USAGE.record()` calls on every route; exposed at `/v1/usage` and summarized in `/healthz`.
- **`main.py`** is a 7-line stub that delegates to `server.app` / `server.main` so both `uv run server.py` and `uv run main.py` work — don't duplicate the entrypoint.

## Translation flow (v3 is the canonical shape)

All client formats funnel through one converter: every request reaches `converter.openai_to_v3()` before going upstream, and every response/stream is produced from an OpenAI chat-completion as the intermediate shape.

- **Request side:** `/v1/chat/completions` calls `openai_to_v3()` directly. `/v1/responses` first runs `responses_input_to_messages()`, and `/v1/messages` first runs `anthropic_to_openai()` — both produce an OpenAI chat body, then hand it to the same `openai_to_v3()`. A new client format is a new `→openai` converter, not a new v3 path.
- **Response side:** the upstream v3 SSE stream is always translated to OpenAI chunks first (`streaming.v3_stream_iter` / non-streaming `v3_sse_stream_to_openai`). `/v1/messages` then re-translates those OpenAI chunks into Anthropic SSE (`anthropic_stream_iter`); `/v1/responses` re-translates into Responses SSE (`openai_chunk_to_responses_sse`). Two-stage on the way out mirrors two-stage on the way in.
- The Anthropic route reuses the OpenAI pipeline end-to-end via `_send_to_v3()`; the chat and responses routes inline the same hydrate → `openai_to_v3` → `_upstream_pooled` sequence.
- `validate_tool_history()` runs on the OpenAI-shaped messages for **all three** routes (after the `→openai` step for Anthropic/Responses).

## Reasoning / thinking pipeline

Reasoning flows through all three API surfaces and is translated at each boundary:

- **Request side (client → upstream):** `reasoning_effort` and `reasoning` in
  OpenAI requests are forwarded as the `reasoning` string label in the v3 body.
  Anthropic `thinking: {"type": "enabled"}` is routed to `reasoning: "enabled"`
  in `anthropic_to_openai()`; explicit `reasoning`/`reasoning_effort` takes
  precedence. `auto` or omitted → the `reasoning` field is omitted entirely
  (matching the fx CLI behavior).
- **Streaming response (upstream → client):** the v3 SSE stream emits
  `reasoning-start` / `reasoning-delta` / `reasoning-end` events.
  `v3_sse_stream_to_openai()` collects these into `reasoning_content` on the
  message dict. For Anthropic clients, `_AnthropicStreamState.thinking_delta()`
  emits `thinking` content blocks via `thinking_delta` SSE events.
- **Non-streaming response:** `reasoning_content` is surfaced as a
  `thinking` content block in `openai_to_anthropic()`.
- **Content block ordering (Anthropic):** thinking blocks are sequential, not
  interleaved. `text_delta()` and `tool_call()` close any open thinking block
  before starting a new text or tool block. `finish()` closes any remaining
  open block.

## Protocol invariants (load-bearing)

The upstream Gateway returns 400/503 if these are violated. They are NOT linter-enforceable; treat them as hard rules when editing `converter/` or `server.py`:

- The proxy **MUST always send `ai-language-model-streaming: true`** to the upstream v3 endpoint. `_v3_headers()` is always called with `streaming=True` from `chat_completions`; non-streaming client requests still open a streaming upstream connection and collect deltas internally.
- The upstream v3 body **MUST always include `tools: []` and `toolChoice: {type: auto}`**, even when the client sends no tools. Missing either → 503. (`openai_to_v3()` injects them.)
- Content shape is **role-dependent**: `system` content MUST be a plain string; `user`/`assistant`/`tool` content MUST be an array of parts. Wrong shape → 400.
- The fx headers (`User-Agent: fx/<version>`, `ai-gateway-protocol-version`, `ai-language-model-specification-version: 4`, `ai-language-model-id: <model>`, `HTTP-Referer`, `X-Title`) **MUST match** what the fx CLI sends. `_v3_headers()` in `server.py` is the single source of truth; values come live from `identity.state` — never hardcode versions outside it.
- HTTP/2 **MUST** stay enabled on the shared `httpx.AsyncClient` (set in `lifespan`).
- Tool-call wire format (fx): assistant tool calls are content parts `{type: "tool-call", toolCallId, toolName, input}` where `input` is the **raw JSON object**, not a string. Top-level `toolCalls` is NOT the fx shape (but `converter` accepts it for robustness).
- Tool history ordering: every assistant `tool_calls` block MUST be immediately followed by matching `role: tool` results covering all call ids. `validate_tool_history()` enforces this client-side; the upstream only returns opaque "Invalid input".

## Where to look

- 503/400 from upstream → check `SAUCE.md` §"How the v3 protocol was discovered" — most are caused by missing `tools: []`, wrong content shape, or `streaming: false`.
- New AI SDK / OpenAI parameter → add the helper in the right `converter/` module and pass it through in `openai_to_v3()`; add a test case asserting the field appears in the v3 body.
- `test_proxy.py` is a **live-server smoke test, not a pytest** — it hardcodes `http://localhost:8799`. Run it with the proxy listening on that port; don't `pytest` it.
- `.env` is gitignored; never commit `AI_GATEWAY_API_KEY` or `PROXY_API_KEY`.

# Fionn — OpenAI-compatible AI Gateway Proxy

An OpenAI-compatible **FastAPI** proxy that forwards to the Vercel AI Gateway
using the **same v3 AI SDK protocol** the [fx CLI](https://github.com/vercel-labs/fx)
uses — including the exact headers that bypass credit-card requirements for
free-tier models like `zai/glm-5.2`.

Point any OpenAI-compatible CLI/tool at it and get free model access — no credit
card needed.

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141+-009688.svg)](https://fastapi.tiangolo.com/)
[![uv](https://img.shields.io/badge/uv-managed-de25a1.svg)](https://docs.astral.sh/uv/)

---

## Table of Contents

- [Why?](#why)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Architecture](#architecture)
- [Protocol invariants](#protocol-invariants)
- [Testing](#testing)
- [Development](#development)
- [Documentation](#documentation)
- [Disclaimer](#disclaimer)

---

## Why?

The Vercel AI Gateway exposes two protocols:

| Endpoint | Protocol | Credit card required |
|---|---|---|
| `/v1/chat/completions` | OpenAI v1 | ✅ Yes |
| `/v3/ai/language-model` | AI SDK v3 | ❌ No (for free models) |

The fx CLI talks to the **v3** endpoint with a specific set of headers.
**Fionn** does the same thing, but exposes a standard **OpenAI v1 API** so any
OpenAI-compatible tool can use it — giving you the same free-tier behavior
the fx CLI enjoys.

```
Your CLI (OpenAI format)
    ↓
Fionn (FastAPI on :8787)
    ├─ translates OpenAI → AI SDK v3 format
    ├─ adds fx headers (User-Agent: fx/0.0.4, ai-gateway-* ...)
    └─ forwards to https://ai-gateway.vercel.sh/v3/ai/language-model
    ↓
Gateway (v3 SSE response)
    ↓
Fionn translates back → OpenAI format
    ↓
Your CLI receives a standard OpenAI response
```

---

## How it works

1. A client sends a standard OpenAI `POST /v1/chat/completions` request.
2. The proxy validates the request (including tool-history pairing) and
   translates the OpenAI body into AI SDK v3 format (`converter.openai_to_v3`).
3. It adds the exact fx headers via `_v3_headers()` and forwards to the
   Gateway's v3 endpoint over HTTP/2, **always streaming** (even for
   non-streaming client requests — the v3 endpoint requires it).
4. The v3 SSE stream is translated back to OpenAI chunks in real time
   (`converter.v3_stream_iter`) and streamed to the client, or assembled into
   a single response for non-streaming requests.

The full reverse-engineering story (how the v3 protocol was discovered, what's
required vs. optional, the 503/400 failure modes) is in
[`gateway-proxy/SAUCE.md`](gateway-proxy/SAUCE.md).

---

## Quick start

```bash
cd gateway-proxy
uv sync                         # install deps from the lockfile
cp .env.example .env            # create config
```

Get your AI Gateway API key:

```bash
# Option A: if you have the fx CLI installed, it's already here:
cat ~/.fx/api-key

# Option B: create one at
# https://vercel.com/d?to=%2F%5Bteam%5D%2F~%2Fai-gateway%2Fapi-keys
```

Edit `.env`:

```env
AI_GATEWAY_API_KEY=vck_your_key_here
PROXY_API_KEY=your_proxy_key_here   # callers must send this as a Bearer token
```

Run:

```bash
uv run server.py                 # starts on http://0.0.0.0:8787
```

---

## Usage

### curl

```bash
# Non-streaming
curl http://localhost:8787/v1/chat/completions \
  -H "Authorization: Bearer your_proxy_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai/glm-5.2",
    "messages": [{"role": "user", "content": "Say hi"}]
  }'

# Streaming
curl -N http://localhost:8787/v1/chat/completions \
  -H "Authorization: Bearer your_proxy_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai/glm-5.2",
    "messages": [{"role": "user", "content": "Say hi"}],
    "stream": true
  }'
```

### Any OpenAI-compatible CLI

Point any tool that supports `OPENAI_API_BASE` / `OPENAI_BASE_URL` at the proxy:

```bash
export OPENAI_API_BASE=http://localhost:8787/v1
export OPENAI_API_KEY=your_proxy_key_here
```

### List models

```bash
curl http://localhost:8787/v1/models \
  -H "Authorization: Bearer your_proxy_key_here"
```

---

## Endpoints

| Route | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completions (OpenAI format) |
| `/v1/responses` | POST | OpenAI Responses API (input items → v3) |
| `/v1/models` | GET | List available models (cached, TTL configurable) |
| `/v1/embeddings` | POST | Embeddings (forwards to the v1 endpoint) |
| `/healthz` | GET | Health check |

### Key headers the proxy sends upstream (matching fx CLI)

```
Authorization: Bearer <gateway_key>
User-Agent: fx/0.0.4
HTTP-Referer: https://github.com/vercel-labs/fx
X-Title: fx
ai-gateway-protocol-version: 0.0.1
ai-language-model-specification-version: 4
ai-language-model-id: <model>
ai-language-model-streaming: true
x-vercel-ai-gateway-team: <team>      # only when GATEWAY_TEAM is set
x-session-id: <sid>                    # only when configured (session pinning)
x-session-affinity: <affinity>         # only when configured (session pinning)
```


---

## Configuration

All configuration is via environment variables (loaded from `.env`). See
[`gateway-proxy/.env.example`](gateway-proxy/.env.example) for the full list.

| Variable | Default | Description |
|---|---|---|
| `AI_GATEWAY_API_KEY` | *(empty)* | Your AI Gateway API key (required). At `~/.fx/api-key` if you use the fx CLI. |
| `PROXY_API_KEY` | *(empty)* | Key callers must send as a Bearer token. Empty = open proxy. |
| `GATEWAY_BASE_URL` | `https://ai-gateway.vercel.sh` | Gateway base URL. |
| `GATEWAY_TEAM` | *(empty)* | Pin a Vercel team (sends `x-vercel-ai-gateway-team`). |
| `DEFAULT_MODEL` | `zai/glm-5.2` | Model used when the client doesn't specify one. |
| `HOST` / `PORT` | `0.0.0.0` / `8787` | Server bind address. |
| `GATEWAY_TIMEOUT_CONNECT` | `15` | Upstream connect timeout (s). |
| `GATEWAY_TIMEOUT_READ` | `300` | Upstream read timeout (s). |
| `GATEWAY_TIMEOUT_WRITE` | `60` | Upstream write timeout (s). |
| `GATEWAY_TIMEOUT_POOL` | `60` | Upstream pool timeout (s). |
| `GATEWAY_HTTP2` | `1` | Use HTTP/2 to the gateway (`1`/`0`). |
| `MODELS_CACHE_TTL` | `300` | How long to cache `/v1/models` (s). |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`/`INFO`/`WARNING`). |
| `FX_USER_AGENT` | `fx/0.0.4` | User-Agent sent to the gateway (mirrors fx CLI). |
| `PRODUCT_USER_AGENT_MODELS` | `zai/glm-5.2` | Models that get the body-level `headers.user-agent`. `*` = all, empty = none, else comma-separated. |
| `GATEWAY_SESSION_ID` | *(empty)* | Optional session pinning header. |
| `GATEWAY_SESSION_AFFINITY` | *(empty)* | Optional session affinity header. |
| `RELOAD` | *(unset)* | Enable uvicorn reload when set. |

> **Security:** `.env` is gitignored. Never commit `AI_GATEWAY_API_KEY` or
> `PROXY_API_KEY`.


---

## Project structure

```
vercela/
├── CLAUDE.md                          # Guidance for AI coding agents
├── README.md                          # This file
├── .gitignore
├── docs/
│   └── superpowers/
│       ├── plans/                     # Implementation plans
│       └── specs/                     # Design specs
└── gateway-proxy/                     # All meaningful code lives here
    ├── README.md                      # Sub-project setup (quick reference)
    ├── SAUCE.md                       # Full reverse-engineering writeup
    ├── pyproject.toml                 # Project metadata + deps (uv)
    ├── uv.lock                        # Locked dependencies
    ├── .python-version                # 3.12
    ├── .env.example                   # Config template
    ├── main.py                        # Entrypoint stub (delegates to server)
    ├── server.py                      # FastAPI app + HTTP transport layer
    ├── test_proxy.py                  # Live smoke test (NOT a pytest)
    ├── converter/                     # Pure OpenAI ↔ v3 conversion package
    │   ├── __init__.py                #   re-exports public API
    │   ├── __main__.py                #   CLI: python -m converter <file>
    │   ├── parts.py                   #   low-level content-part translation
    │   ├── request.py                 #   openai_to_v3() request assembly
    │   ├── response.py                #   v3_to_openai() non-stream response
    │   ├── streaming.py               #   v3 SSE → OpenAI SSE (live + offline)
    │   ├── responses.py               #   Responses API translation
    │   └── validation.py              #   client-side tool-history validation
    └── tests/                         # pytest unit tests
        ├── test_cli.py
        ├── test_parts.py
        ├── test_request.py
        ├── test_response.py
        ├── test_responses.py
        ├── test_server.py
        ├── test_server_headers.py
        ├── test_streaming.py
        └── test_validation.py
```


---

## Architecture

The codebase follows a strict **two-layer split**:

### `converter/` — pure functions, no I/O

All OpenAI ↔ AI SDK v3 translation logic. No network calls, no env reads, no
side effects. Env-derived config is passed in as parameters from `server.py`.

| Module | Responsibility |
|---|---|
| `parts.py` | Low-level helpers: content parts, image URLs → v3 file parts, tool calls, tool messages, `tool_choice` normalization, `response_format` mapping |
| `request.py` | `openai_to_v3()` — assembles a full v3 request body (messages, tools, params, fx identity rules) |
| `response.py` | `v3_to_openai()` — non-streaming v3 response → OpenAI `chat.completion`; finish-reason map; usage mapping (incl. cache/reasoning token details) |
| `streaming.py` | `_StreamState`, `_process_stream_event()` event dispatch, `v3_stream_iter()` (live async), `v3_stream_to_openai()` (offline), `v3_sse_stream_to_openai()` (collect-to-one) |
| `responses.py` | OpenAI Responses API (`/v1/responses`): input items → messages, response reshaping, SSE translation |
| `validation.py` | `validate_tool_history()` — client-side tool-history pairing checks (clear 400s instead of opaque gateway errors) |

### `server.py` — FastAPI + HTTP transport

The shared `httpx.AsyncClient` (HTTP/2 on) is created in the `lifespan`
context manager. Routes translate requests via the `converter` package, forward
upstream, and translate responses back. Also handles auth, request logging,
upstream error normalization, and client-disconnect cancellation.

### `main.py` — entrypoint stub

A 6-line stub that delegates to `server.app` / `server.main`, so both
`uv run server.py` and `uv run main.py` work.

### Features beyond plain forwarding

- **Client-side tool-history validation** — clear `400` errors instead of
  opaque "Invalid input" gateway rejections
- **Upstream error normalization** — gateway errors are reshaped to the
  OpenAI error format
- **Client-disconnect handling** — the upstream stream is cancelled when the
  client disconnects
- **Usage reporting** — optional usage on the final streaming chunk
  (cache-read & reasoning-token details included)
- **Reasoning streaming** — `reasoning-delta` → OpenAI `reasoning_content`
- **Image support** — data-URL images become v3 `file` parts (fx wire format)
- **Session pinning** — `x-session-id` / `x-session-affinity` forwarded
- **Model-list caching** — configurable TTL on `/v1/models`
- **Configurable timeouts & HTTP/2**


---

## Protocol invariants

> These are **load-bearing**. The upstream Gateway returns `400`/`503` if they
> are violated. They are not linter-enforced — treat them as hard rules.

1. **Always send `ai-language-model-streaming: true`** to the v3 endpoint.
   Non-streaming client requests still open a streaming upstream connection and
   collect deltas internally.
2. **The v3 body must always include `tools: []` and `toolChoice: {type: auto}`**,
   even when the client sends no tools. Missing either → `503`.
   (`openai_to_v3()` injects them.)
3. **Content shape is role-dependent**: `system` content MUST be a plain string;
   `user`/`assistant`/`tool` content MUST be an array of parts. Wrong shape → `400`.
4. **The fx headers must match exactly** what the fx CLI sends.
   `_v3_headers()` in `server.py` is the single source of truth.
5. **HTTP/2 must stay enabled** on the shared `httpx.AsyncClient`.
6. **Tool-call wire format (fx)**: assistant tool calls are content parts
   `{type: "tool-call", toolCallId, toolName, input}` where `input` is the raw
   JSON object, not a string.
7. **Tool history ordering**: every assistant `tool_calls` block MUST be
   immediately followed by matching `role: tool` results covering all call ids.
   `validate_tool_history()` enforces this client-side.

---

## Testing

### Unit tests (pytest)

The `converter/` package is fully unit-tested with a mocked upstream gateway
(deterministic `httpx.MockTransport`, no real network traffic):

```bash
cd gateway-proxy
uv run pytest tests/                       # full suite
uv run pytest tests/test_request.py -v     # single file
uv run pytest tests/test_streaming.py -k "reasoning"   # filtered
```

Test files mirror the package layout (135 tests across 9 files):
`test_cli.py`, `test_parts.py`, `test_request.py`, `test_response.py`,
`test_responses.py`, `test_server.py`, `test_server_headers.py`,
`test_streaming.py`, `test_validation.py`.

### Live smoke test

`test_proxy.py` is a **live-server smoke test, not a pytest** — it hardcodes
`http://localhost:8799`. Start the proxy on that port first, then:

```bash
cd gateway-proxy
PROXY_API_KEY=your_key python test_proxy.py
```

Do **not** run it through pytest.


---

## Development

This project uses **`uv`** (lockfile: `gateway-proxy/uv.lock`). Python 3.12 is
pinned in `gateway-proxy/.python-version`. Do not use `pip` directly.

```bash
cd gateway-proxy
uv sync                                   # install deps from lockfile
uv run pytest tests/ -q                   # run the unit suite
uv run server.py                          # start the proxy
uv run python -m converter body.json      # convert an OpenAI request to v3 (CLI)
uv run python -m converter events.json --stream   # convert v3 SSE events to OpenAI SSE
```

### Conventions

- Keep the two-layer split: conversion logic in `converter/`, transport in
  `server.py`.
- `converter/` modules stay pure — no I/O, no env reads, no side effects.
- Add new AI SDK / OpenAI parameters as helpers in `converter/` and pass them
  through in `openai_to_v3()`; add a test asserting the field appears in the
  v3 body.
- Commit messages follow repo history: `feat:`, `fix:`, `docs:`, `refactor:`
  prefixes, lowercase subject.
- Never commit `.env` or any API key value.
- `baicode_install.sh` at the repo root (if present) is an unrelated
  third-party installer — do not modify it.

---

## Documentation

| File | What it covers |
|---|---|
| [`gateway-proxy/README.md`](gateway-proxy/README.md) | Sub-project quick-start & setup |
| [`gateway-proxy/SAUCE.md`](gateway-proxy/SAUCE.md) | Full reverse-engineering writeup — how the v3 protocol was discovered, required vs. optional fields, 503/400 failure modes, request flow diagram |
| [`CLAUDE.md`](CLAUDE.md) | Guidance for AI coding agents working in this repo |
| [`docs/superpowers/specs/`](docs/superpowers/specs/) | Design specs |
| [`docs/superpowers/plans/`](docs/superpowers/plans/) | Implementation plans |

---

## Disclaimer

This is a reverse-engineering project. The fx CLI and Vercel AI Gateway are
products of Vercel Inc. This proxy is not affiliated with or endorsed by Vercel.
It simply forwards requests using the same publicly-observable protocol the fx
CLI uses.

The free model access (`zai/glm-5.2` without a credit card) is a feature of the
Vercel AI Gateway's v3 protocol, not a hack. The proxy just makes it accessible
from OpenAI-compatible tools.


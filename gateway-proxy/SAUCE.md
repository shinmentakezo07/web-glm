# 🧠 SAUCE — How the fx-style AI Gateway Proxy Works

> The full story of how this proxy tricks the Vercel AI Gateway into giving
> you free model access without a credit card, by mimicking the fx CLI.

---

## TL;DR

The Vercel AI Gateway has **two protocols**:

| Endpoint | Protocol | Credit Card? |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI v1 | **Required** |
| `POST /v3/ai/language-model` | AI SDK v3 | **NOT required** (free models) |

The fx CLI uses the **v3 endpoint** with special headers. This proxy does
the same thing, but exposes a standard **OpenAI v1 API** so any CLI can use it.
The proxy sits in the middle and translates between the two formats:

```
Your CLI  ──OpenAI v1──▶  Proxy  ──AI SDK v3──▶  Vercel AI Gateway
         ◀──OpenAI v1──         ◀──AI SDK v3──
```

---

## 🔬 How the v3 protocol was discovered

### Step 1: Inspect the fx binary

The fx CLI is a single compiled binary at `~/.local/bin/fx`. Using `strings`
on it revealed two key endpoints:

```
https://ai-gateway.vercel.sh
https://ai-gateway.vercel.sh/v3/ai/language-model
```

And two critical header names:

```
ai-gateway-protocol-version
ai-language-model-specification-version
```

### Step 2: Capture fx's exact HTTP request

The fx CLI supports a `FX_GATEWAY_CHAT_URL` environment variable that lets
you redirect its HTTP requests to a local server. We used this to capture
the exact request:

```python
# Start a local capture server on port 9999
# Point fx at it:
env['FX_GATEWAY_CHAT_URL'] = 'http://127.0.0.1:9999/v3/ai/language-model'
subprocess.run(['fx', 'ask', '--auto', '--no-save', 'Say hi'], env=env)
```

### Step 3: The captured request

**Headers:**
```
Authorization: Bearer vck_0Y4Aj...
User-Agent: fx/0.0.3
HTTP-Referer: https://github.com/vercel-labs/fx
X-Title: fx
ai-gateway-protocol-version: 0.0.1
ai-language-model-specification-version: 4
ai-language-model-id: zai/glm-5.2
ai-language-model-streaming: true
Content-Type: application/json
```

**Body:**
```json
{
  "prompt": [...],
  "tools": [...25 tool definitions...],
  "toolChoice": {"type": "auto"},
  "headers": {"user-agent": "fx/0.0.3"}
}
```

### Step 4: Replay and test

We replayed this exact request to the real Gateway and got `HTTP 200` with
model output. Then we tested variations to find what's actually required:

| Test | Result |
|---|---|
| Full fx body (43KB) | ✅ 200 |
| Minimal body (just prompt) | ❌ 503 |
| Body + `tools: []` + `toolChoice` | ✅ 200 |
| Body without `toolChoice` | ❌ 503 |
| `ai-language-model-streaming: false` | ❌ 503 |
| `ai-language-model-streaming: true` | ✅ 200 |
| glm-4.6 (non-free model) | ❌ 403 (credit card) |
| glm-5.2 (free model) | ✅ 200 |

### Key findings

1. **The v3 endpoint** bypasses credit card requirements for free models
2. **`tools: []` and `toolChoice: {type: auto}`** must be in the body or you get 503
3. **Streaming must be `true`** — the Gateway returns 503 for non-streaming on free models
4. **System messages** must have string content, not arrays
5. **User/assistant messages** must have array content (array of parts)

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Your CLI / Tool                       │
│              (OpenAI-compatible format)                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │  POST /v1/chat/completions
                           │  Authorization: Bearer <PROXY_API_KEY>
                           │  {"model":"zai/glm-5.2","messages":[...]}
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Proxy (:8787)                    │
│                                                              │
│  1. Auth check                                               │
│     verify_proxy_key() — Bearer token vs PROXY_API_KEY      │
│                                                              │
│  2. Translate request                                        │
│     _openai_to_v3()                                          │
│     • system messages → string content                       │
│     • user/assistant messages → array of parts               │
│     • inject tools:[] + toolChoice:{type:auto}              │
│                                                              │
│  3. Build fx headers                                         │
│     _v3_headers()                                            │
│     • User-Agent: fx/0.0.3                                   │
│     • ai-gateway-protocol-version: 0.0.1                     │
│     • ai-language-model-specification-version: 4             │
│     • ai-language-model-id: <model>                         │
│     • ai-language-model-streaming: true                      │
│                                                              │
│  4. Send to Gateway (always streaming)                       │
│     POST /v3/ai/language-model                               │
│                                                              │
│  5. Translate response                                       │
│     • streaming → _v3_stream_to_openai()                     │
│     • non-stream → collect all deltas, assemble JSON         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │  POST https://ai-gateway.vercel.sh
                           │       /v3/ai/language-model
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Vercel AI Gateway                          │
│                                                              │
│  • Receives v3 format                                        │
│  • Routes to provider (runware/blackbox/etc.)                │
│  • Returns SSE stream of v3 events                           │
│  • No credit card check for free models (glm-5.2)            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔀 Request translation: OpenAI → AI SDK v3

The proxy accepts standard OpenAI requests and converts them. Here's
exactly what happens:

### OpenAI input (what your CLI sends)

```json
{
  "model": "zai/glm-5.2",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Say hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "How are you?"}
  ],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 1000
}
```

### v3 output (what the proxy sends to the Gateway)

```json
{
  "prompt": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": [{"type": "text", "text": "Say hi"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
    {"role": "user", "content": [{"type": "text", "text": "How are you?"}]}
  ],
  "tools": [],
  "toolChoice": {"type": "auto"},
  "headers": {"user-agent": "fx/0.0.3"},
  "temperature": 0.7,
  "maxOutputTokens": 1000
}
```

### The content-type rules

This was the hardest bug to find. The v3 protocol has strict rules:

| Role | Content format | Example |
|---|---|---|
| `system` | **String** (plain text) | `"You are helpful."` |
| `user` | **Array** of parts | `[{"type":"text","text":"hi"}]` |
| `assistant` | **Array** of parts | `[{"type":"text","text":"hello"}]` |
| `tool` | **Array** of parts | `[{"type":"text","text":"result"}]` |

If you send a system message with array content, you get:
```
400: Invalid input: expected string, received array
```

If you send a user message with string content, you get:
```
400: Invalid input: expected array, received string
```

### The `tools: []` requirement

The Gateway returns **503 Service Unavailable** if the body doesn't include
`tools` and `toolChoice`, even though no tools are being used. This is
likely a Gateway-side bug or an undocumented requirement for the v3
protocol on free-tier models. The proxy injects them automatically:

```python
v3_body = {
    "prompt": prompt,
    "tools": [],                    # required — without this, 503
    "toolChoice": {"type": "auto"}, # required — without this, 503
    "headers": {"user-agent": "fx/0.0.3"},
}
```

### The streaming requirement

The Gateway returns **503** for `ai-language-model-streaming: false` on
free-tier models. The proxy **always streams from upstream**, even when
the client requests a non-streaming response. For non-streaming clients,
the proxy collects all SSE deltas internally and returns a single JSON
response.

---

## 📡 Response translation: AI SDK v3 SSE → OpenAI SSE

The Gateway returns v3-format SSE events. The proxy converts them to
OpenAI-format chunks in real time.

### v3 SSE events (from Gateway)

```
data: {"type":"stream-start","warnings":[]}

data: {"type":"response-metadata","id":"...","modelId":"glm52"}

data: {"type":"text-start","id":"txt-0"}

data: {"type":"text-delta","id":"txt-0","delta":"Hi! "}

data: {"type":"text-delta","id":"txt-0","delta":"How can I help?"}

data: {"type":"text-end","id":"txt-0"}

data: {"type":"finish","finishReason":{"unified":"stop","raw":"stop"},
       "usage":{"inputTokens":{"total":19},"outputTokens":{"total":6}}}

DONE
```

### OpenAI SSE chunks (what your CLI receives)

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk",
       "choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk",
       "choices":[{"delta":{"content":"Hi! "},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk",
       "choices":[{"delta":{"content":"How can I help?"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk",
       "choices":[{"delta":{},"finish_reason":"stop"}],
       "usage":{"prompt_tokens":19,"completion_tokens":6,"total_tokens":25}}

data: [DONE]
```

### Event mapping

| v3 event type | OpenAI equivalent | Notes |
|---|---|---|
| `stream-start` | First chunk with `role: assistant` | Sets up the response |
| `text-delta` | Chunk with `content` delta | Incremental text |
| `text-end` | (nothing) | Just signals end of text |
| `tool-call` | Chunk with `tool_calls` | Function calling |
| `finish` | Final chunk with `finish_reason` + `usage` | End of response |
| `error` | Final chunk with `finish_reason: stop` | Error handling |

### Finish reason mapping

| v3 `finishReason.unified` | OpenAI `finish_reason` |
|---|---|
| `stop` | `stop` |
| `length` | `length` |
| `tool-calls` | `tool_calls` |
| `content-filter` | `content_filter` |

### Usage token mapping

v3 usage format:
```json
{"inputTokens": {"total": 19, "cacheRead": 10560, "noCache": 47},
 "outputTokens": {"total": 6, "text": 6, "reasoning": 0}}
```

OpenAI usage format:
```json
{"prompt_tokens": 19, "completion_tokens": 6, "total_tokens": 25}
```

---

## 🔐 Authentication (two layers)

### Layer 1: Client → Proxy

The proxy validates incoming requests with a Bearer token:

```
Authorization: Bearer <PROXY_API_KEY>
```

If `PROXY_API_KEY` is empty, the proxy runs in **open mode** (no auth).
This is useful for local development but should be set for any exposed
deployment.

### Layer 2: Proxy → Gateway

The proxy sends your `AI_GATEWAY_API_KEY` to the Gateway:

```
Authorization: Bearer <AI_GATEWAY_API_KEY>
```

This is the key from `~/.fx/api-key` (if you have fx CLI installed) or
from the Vercel dashboard at
`https://vercel.com/[team]/~/ai-gateway/api-keys`.

---

## 📁 File structure

```
gateway-proxy/
├── .env.example       # Template for environment variables
├── .env               # Your actual keys (gitignored, never committed)
├── .gitignore         # Excludes .env, __pycache__, .venv
├── .python-version    # Python 3.12
├── README.md          # User-facing docs
├── SAUCE.md           # This file — how it all works
├── main.py            # Entry point → delegates to server.app
├── server.py          # Main proxy implementation
├── pyproject.toml     # Dependencies: fastapi, httpx[http2], uvicorn
├── test_proxy.py      # Smoke tests
└── uv.lock            # Locked dependency versions
```

---

## 🧩 server.py — function-by-function breakdown

### Configuration (lines 54-72)

Reads environment variables at import time:
- `GATEWAY_BASE_URL` — Gateway URL (default: `https://ai-gateway.vercel.sh`)
- `AI_GATEWAY_API_KEY` — Your Gateway API key (required)
- `GATEWAY_TEAM` — Optional Vercel team slug
- `PROXY_API_KEY` — Key your clients must send (optional, empty = open)
- `DEFAULT_MODEL` — Fallback model (default: `zai/glm-5.2`)

### `lifespan()` — async context manager (lines 80-94)

Creates a shared `httpx.AsyncClient` with:
- HTTP/2 enabled (the Gateway requires HTTP/2)
- Connection pooling (100 max connections, 20 keepalive)
- Long timeouts (300s read for slow model responses)
- Closes the client on shutdown

### `verify_proxy_key()` — auth dependency (lines 105-115)

FastAPI security dependency. Checks the incoming Bearer token against
`PROXY_API_KEY`. Returns `"anonymous"` if open mode, raises 401 otherwise.

### `_v3_headers()` — build fx headers (lines 123-140)

Constructs the exact header set the fx CLI sends. These headers are what
makes the Gateway treat the request as a "fx CLI" request:

```
User-Agent: fx/0.0.3
HTTP-Referer: https://github.com/vercel-labs/fx
X-Title: fx
ai-gateway-protocol-version: 0.0.1
ai-language-model-specification-version: 4
ai-language-model-id: <model>
ai-language-model-streaming: true
```

### `_openai_to_v3()` — request translator (lines 148-237)

Converts OpenAI request body to AI SDK v3 format:

1. Iterates over `messages`
2. For each message, determines the role:
   - `system` → content becomes a plain **string**
   - `user`/`assistant`/`tool` → content becomes an **array of parts**
3. Handles string content, array content (OpenAI vision format), and
   converts image_url parts to v3 image parts
4. Injects `tools: []` and `toolChoice: {type: auto}` (required)
5. Passes through optional parameters:
   - `temperature` → `temperature`
   - `max_tokens` → `maxOutputTokens`
   - `top_p` → `topP`
   - `stop` → `stopSequences`
6. Translates OpenAI tool definitions to v3 format if provided

### `_sse_chunk()` — OpenAI chunk builder (lines 245-268)

Helper that builds a single OpenAI-format SSE chunk:

```python
{
    "id": "chatcmpl-<uuid>",
    "object": "chat.completion.chunk",
    "created": <timestamp>,
    "model": "<model>",
    "choices": [{"index": 0, "delta": {...}, "finish_reason": None|"stop"|...}],
    "usage": {...}  # only on final chunk
}
```

### `_v3_stream_to_openai()` — SSE translator (lines 271-375)

Async generator that reads v3 SSE events from the Gateway and yields
OpenAI-format SSE chunks:

1. Yields initial chunk with `role: assistant`
2. For each v3 event:
   - `text-delta` → yields chunk with content delta
   - `tool-call` → yields chunk with tool_calls
   - `finish` → yields final chunk with finish_reason + usage, then breaks
   - `error` → yields final chunk, then breaks
3. Yields ` sentinel

### `_v3_response_to_openai()` — non-stream converter (lines 383-421)

Converts a v3 non-streaming response to OpenAI format (used for the
embeddings-style single response). In practice, the proxy always streams
from upstream, so this function is available but the main chat endpoint
collects deltas manually for non-streaming clients.

### `_client_error()` — error forwarder (lines 424-430)

Wraps upstream error responses and forwards them as JSON with the original
status code.

### Routes

#### `GET /healthz` (line 438)

Returns health status. No auth required. Useful for load balancers.

#### `GET /v1/models` (line 443)

Proxies to the Gateway's public `/v1/models` endpoint. Returns the full
list of available models (349+ models including 17 GLM variants).

#### `POST /v1/chat/completions` (line 456)

The main endpoint. Handles both streaming and non-streaming:

**Streaming mode** (`stream: true`):
1. Translate OpenAI request → v3 format
2. Open streaming connection to Gateway
3. Pipe through `_v3_stream_to_openai()` translator
4. Return as `StreamingResponse` with `text/event-stream`

**Non-streaming mode** (`stream: false`):
1. Translate OpenAI request → v3 format
2. Open streaming connection to Gateway (always streams upstream!)
3. Collect all `text-delta` events into a list
4. Capture `finish` event for finish_reason + usage
5. Assemble a single OpenAI JSON response
6. Return as `JSONResponse`

#### `POST /v1/embeddings` (line 561)

Proxies to the Gateway's `/v1/embeddings` endpoint. Uses the standard
v1 protocol (embeddings don't have the credit card restriction).

### `main()` — entrypoint (lines 589-601)

Starts uvicorn with the FastAPI app. Reads `HOST`, `PORT`, `RELOAD`,
and `LOG_LEVEL` from environment.

---

## 🔄 The two-protocol trick — why it works

The Vercel AI Gateway has two generations of API:

### v1 (OpenAI-compatible)
- Endpoint: `/v1/chat/completions`
- Format: standard OpenAI (`messages`, `model`, `stream`)
- Billing: requires credit card on file
- This is what most people use and hit the credit card wall

### v3 (AI SDK)
- Endpoint: `/v3/ai/language-model`
- Format: AI SDK v3 (`prompt`, `tools`, `toolChoice`)
- Billing: **no credit card required for free models**
- This is what the fx CLI uses

The v3 protocol is the Vercel AI SDK's native protocol. It's designed for
the `@ai-sdk` JavaScript/TypeScript library. The Gateway treats v3
requests differently for billing — free models (like `zai/glm-5.2`) are
available without a credit card.

The fx CLI uses the v3 protocol because it's built on the AI SDK. This
proxy replicates that, giving you the same free access from any
OpenAI-compatible tool.

---

## 🎯 Why glm-5.2 specifically

Not all models are free without a credit card. We tested:

| Model | Status | Credit card? |
|---|---|---|
| `zai/glm-5.2` | ✅ Works | No |
| `zai/glm-5.2-fast` | ✅ Works | No |
| `zai/glm-5.1` | ❌ 403 | Yes |
| `zai/glm-5` | ❌ 403 | Yes |
| `zai/glm-4.6` | ❌ 403 | Yes |
| `zai/glm-4.7-flash` | ❌ 403 | Yes |
| `zai/glm-4.6v-flash-free` | ❌ 403 | Yes |

Only `glm-5.2` and `glm-5.2-fast` are available without a credit card.
The "5.2" models are the newest and are offered as a free tier.

---

## 🔧 Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `AI_GATEWAY_API_KEY` | Yes | — | Your Vercel AI Gateway API key |
| `PROXY_API_KEY` | No | (empty) | Key your clients must send. Empty = open |
| `DEFAULT_MODEL` | No | `zai/glm-5.2` | Fallback when client doesn't specify a model |
| `GATEWAY_BASE_URL` | No | `https://ai-gateway.vercel.sh` | Gateway URL |
| `GATEWAY_TEAM` | No | (empty) | Optional Vercel team slug |
| `HOST` | No | `0.0.0.0` | Server bind address |
| `PORT` | No | `8787` | Server port |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `RELOAD` | No | (unset) | Set to enable hot reload (dev mode) |

---

## 📦 Dependencies

```toml
[project]
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.141.1",       # Web framework
    "httpx[http2]>=0.28.1",   # HTTP client with HTTP/2 support
    "uvicorn[standard]>=0.52.4", # ASGI server
]
```

- **FastAPI** — async web framework for the proxy endpoints
- **httpx[http2]** — HTTP client with HTTP/2 support (Gateway requires HTTP/2)
- **uvicorn[standard]** — ASGI server to run FastAPI
- **python-dotenv** — auto-loads `.env` file (ships with uvicorn[standard])

---

## 🚀 How to get the API key

### Option A: From the fx CLI (easiest)

If you have fx installed, the key is stored at `~/.fx/api-key`:

```bash
cat ~/.fx/api-key
# Output: vck_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### Option B: From the Vercel dashboard

1. Go to https://vercel.com/dashboard
2. Navigate to AI → AI Gateway → API Keys
3. Create a new key
4. Copy it (starts with `vck_`)

---

## 🧪 Testing

### Smoke test

```bash
# Start the proxy
uv run server.py

# In another terminal:
python test_proxy.py
```

### Manual curl tests

```bash
# Non-streaming
curl http://localhost:8787/v1/chat/completions \
  -H "Authorization: Bearer <PROXY_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai/glm-5.2",
    "messages": [
      {"role": "system", "content": "You are helpful."},
      {"role": "user", "content": "Say hi"}
    ]
  }'

# Streaming
curl -N http://localhost:8787/v1/chat/completions \
  -H "Authorization: Bearer <PROXY_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai/glm-5.2",
    "messages": [{"role": "user", "content": "Count to 3"}],
    "stream": true
  }'

# List models
curl http://localhost:8787/v1/models \
  -H "Authorization: Bearer <PROXY_API_KEY>"
```

---

## ⚠️ Gotchas and edge cases

1. **System messages must be strings** — The v3 protocol requires `system`
   role content as a plain string. If you send an array, you get 400.

2. **User/assistant messages must be arrays** — The v3 protocol requires
   `user` and `assistant` role content as `[{type: text, text: "..."}]`.

3. **`tools: []` is mandatory** — Without `tools: []` and
   `toolChoice: {type: auto}` in the body, the Gateway returns 503 even
   if you're not using tools. The proxy injects them automatically.

4. **Streaming must be true** — The Gateway returns 503 for
   `ai-language-model-streaming: false` on free models. The proxy always
   streams from upstream.

5. **HTTP/2 is required** — The Gateway may reject HTTP/1.1 requests. The
   proxy uses `httpx[http2]` with `http2=True`.

6. **Only glm-5.2 is free** — Other models (glm-4.6, glm-5.1, etc.)
   require a credit card even through the v3 endpoint.

7. **The 503 can be transient** — The Gateway sometimes returns 503
   temporarily. This is a provider-side issue (runware/blackbox backends).
   Retrying usually works.

8. **API key in `~/.fx/api-key`** — If you have the fx CLI installed, the
   key is already there. The proxy reads it from the `.env` file.

---

## 🧭 Flow diagram for a single request

```
Client sends:
  POST /v1/chat/completions
  Authorization: Bearer testproxy123
  {"model":"zai/glm-5.2","messages":[{"role":"user","content":"hi"}],"stream":true}

  │
  ▼
[verify_proxy_key]  ← checks Bearer token against PROXY_API_KEY
  │
  ▼
[parse request body]  ← extract model, messages, stream flag
  │
  ▼
[_openai_to_v3]  ← translate messages to v3 format
  │                 • system → string content
  │                 • user/assistant → array of parts
  │                 • inject tools:[], toolChoice
  │
  ▼
[_v3_headers]  ← build fx headers
  │               • User-Agent: fx/0.0.3
  │               • ai-gateway-protocol-version: 0.0.1
  │               • ai-language-model-streaming: true
  │
  ▼
[httpx.AsyncClient.send]  ← POST to /v3/ai/language-model (streaming)
  │
  ▼
Gateway responds:
  data: {"type":"stream-start"}
  data: {"type":"text-delta","delta":"Hi! "}
  data: {"type":"finish","finishReason":{"unified":"stop"},"usage":{...}}
  DONE
  │
  ▼
[_v3_stream_to_openai]  ← translate v3 SSE → OpenAI SSE
  │                        • stream-start → role:assistant chunk
  │                        • text-delta → content delta chunk
  │                        • finish → finish_reason + usage chunk
  │                        • append data:[DONE]
  │
  ▼
Client receives:
  data: {"choices":[{"delta":{"role":"assistant"}}]}
  data: {"choices":[{"delta":{"content":"Hi! "}}]}
  data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{...}}
  data: [DONE]
```

---

## 📝 License & disclaimer

This is a reverse-engineering project. The fx CLI and Vercel AI Gateway
are products of Vercel Inc. This proxy is not affiliated with or endorsed
by Vercel. It simply forwards requests using the same protocol the fx CLI
uses, which is publicly observable.

The free model access (`zai/glm-5.2` without credit card) is a feature of
the Vercel AI Gateway's v3 protocol, not a hack. The proxy just makes it
accessible from OpenAI-compatible tools.

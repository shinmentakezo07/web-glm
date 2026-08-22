# fx-style AI Gateway Proxy

An OpenAI-compatible proxy that forwards to the Vercel AI Gateway using the
**same v3 AI SDK protocol** the fx CLI uses — including the exact headers that
bypass credit-card requirements for free-tier models like `zai/glm-5.2`.

## Why?

The Vercel AI Gateway has two endpoints:

| Endpoint | Protocol | Credit card required |
|---|---|---|
| `/v1/chat/completions` | OpenAI v1 | ✅ Yes |
| `/v3/ai/language-model` | AI SDK v3 | ❌ No (for free models) |

The fx CLI uses the v3 endpoint with specific headers. This proxy does the same,
so you get the same behavior — **no credit card needed** for free models like
`zai/glm-5.2`.

## How it works

```
Your CLI (OpenAI format)
    ↓
Proxy (FastAPI on :8787)
    ├─ translates OpenAI → AI SDK v3 format
    ├─ adds fx headers:
    │    User-Agent: fx/0.0.4
    │    ai-gateway-protocol-version: 0.0.1
    │    ai-language-model-specification-version: 4
    │    HTTP-Referer: https://github.com/vercel-labs/fx
    │    X-Title: fx
    └─ forwards to https://ai-gateway.vercel.sh/v3/ai/language-model
    ↓
Gateway (v3 SSE response)
    ↓
Proxy translates back → OpenAI format
    ↓
Your CLI receives standard OpenAI response
```

## Setup

```bash
cd gateway-proxy
uv sync
cp .env.example .env
```

Edit `.env` and set your API key:

```bash
# Option A: if you have fx CLI installed, the key is at ~/.fx/api-key
cat ~/.fx/api-key  # copy this value

# Option B: create one at https://vercel.com/d?to=%2F%5Bteam%5D%2F~%2Fai-gateway%2Fapi-keys
```

```env
AI_GATEWAY_API_KEY=vck_your_key_here
PROXY_API_KEY=your_proxy_key_here
```

### Multiple gateway keys (round-robin + failover)

The proxy supports several upstream gateway keys. List them in `.env`:

```env
AI_GATEWAY_API_KEY_1=vck_first_key
AI_GATEWAY_API_KEY_2=vck_second_key
```

With more than one key configured:

- **Round-robin** — requests alternate across keys (`KEY_ROTATION=1`, default on).
- **Automatic failover** — if a key fails with `401/402/403/408/429`, any `5xx`,
  or a network error, the request is transparently retried on the next key
  (`KEY_FAILOVER=1`, default on). Request faults like `400` are not retried.
- **Cooldown** — a failing key sits out of rotation for `KEY_COOLDOWN` seconds
  (default 30, `0` disables) so a dead key stops costing latency.

All three switches are plain `.env` flags — set `0` to disable. `/healthz`
reports the pool state (`keys.count`, `rotation`, `failover`, `cooling`) and
startup logs each key's masked tail.

## Run

```bash
uv run server.py
```

Server starts on `http://0.0.0.0:8787`.

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

## Endpoints

| Route | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completions (OpenAI format) |
| `/v1/models` | GET | List available models |
| `/v1/embeddings` | POST | Embeddings (uses v1 endpoint) |
| `/healthz` | GET | Health check |
| `/v1/usage` | GET | Per-caller usage counters since process start |

## Key headers the proxy sends (matching fx CLI)

```
Authorization: Bearer <gateway_key>
User-Agent: fx/0.0.4
HTTP-Referer: https://github.com/vercel-labs/fx
X-Title: fx
ai-gateway-protocol-version: 0.0.1
ai-language-model-specification-version: 4
ai-language-model-id: <model>
ai-language-model-streaming: true
x-session-id: <sid>            # only sent when configured (session pinning)
x-session-affinity: <affinity> # only sent when configured (session pinning)
```

## Notes

- The proxy always streams from the upstream v3 endpoint (even for non-streaming
  client requests) and assembles the response — this is required because the v3
  endpoint only works in streaming mode for free-tier models.
- The body must include `"tools": []` and `"toolChoice": {"type": "auto"}` — without
  these, the Gateway returns 503 for free-tier models. The proxy adds them automatically.
- The body-level `headers.user-agent` is only sent for `zai/glm-5.2` (mirrors the fx
  CLI); override with `PRODUCT_USER_AGENT_MODELS` (`*` = all models, empty = none).
- `zai/glm-5.2` is confirmed to work without a credit card. Other models may require one.

## Docker

```bash
cd gateway-proxy
cp .env.example .env   # set AI_GATEWAY_API_KEY / PROXY_API_KEY first
docker compose up --build -d
```

The container listens on `${PORT:-8787}` and reuses your `.env` via `env_file`.

## Remote images & usage tracking

- **Remote images** (`IMAGE_FETCH=1`, default on): `http(s)` image URLs in
  messages are downloaded and inlined as data URLs before conversion — same
  behaviour as fx, which fetches attachments locally. Failures leave the URL
  unchanged; cap size with `IMAGE_FETCH_MAX_BYTES`.
- **Usage tracking** (`USAGE_TRACKING=1`, default on): request/error/token
  counters per calling client, exposed at `GET /v1/usage` and summarised in
  `/healthz`. In-memory only; resets on restart.
- **Identity fallback**: if the GitHub fx sync is disabled or unreachable,
  the proxy uses the version of any locally installed `fx` binary before
  falling back to the hardcoded default User-Agent.

# Repository Guidelines

## Project Structure & Module Organization

All meaningful code lives in `gateway-proxy/`:

```
gateway-proxy/
├── converter/          # Pure OpenAI ↔ AI SDK v3 translation (no I/O)
│   ├── parts.py         #   content parts, tool calls, tool_choice, response_format
│   ├── request.py       #   openai_to_v3() — full v3 request assembly
│   ├── response.py      #   v3_to_openai() — non-streaming response
│   ├── streaming.py     #   v3 SSE → OpenAI SSE (live + offline)
│   ├── responses.py     #   Responses API translation
│   ├── anthropic.py    #   Anthropic Messages API ↔ OpenAI translation
│   └── validation.py    #   tool-history pairing checks
├── server.py           # FastAPI app + HTTP transport (routes, auth, key pool)
├── identity.py         # Background fx identity sync from GitHub
├── keys.py             # KeyPool: round-robin + failover + cooldown
├── usage.py            # In-memory per-caller usage tracking
├── main.py             # Entrypoint stub (delegates to server.app)
├── tests/              # pytest unit tests (mirror converter/ layout)
└── test_proxy.py       # Live smoke test (NOT a pytest — hits localhost:8799)
```

## Build, Test, and Development Commands

Uses `uv` (not `pip`). Run everything from `gateway-proxy/`:

```bash
uv sync                                    # install deps from lockfile
uv run pytest tests/                       # full unit suite
uv run pytest tests/test_request.py -v     # single file
uv run server.py                           # start proxy on :8787
uv run python -m converter body.json       # CLI: convert OpenAI request to v3
```

## Coding Style & Naming Conventions

- Python 3.12, type hints with `from __future__ import annotations`.
- **Two-layer split is mandatory**: `converter/` is pure (no I/O, no env reads, no side effects). All transport, auth, and env access stays in `server.py`.
- Env-derived config is passed into converter functions as parameters, never read directly.
- Private helpers prefixed with `_`; public API re-exported in `converter/__init__.py`.
- No formatting/linting tool configured; match existing style (4-space indent, double quotes for strings).

## Testing Guidelines

- Framework: `pytest` with `httpx.MockTransport` (no real network traffic).
- Test files mirror source: `tests/test_request.py` ↔ `converter/request.py`.
- 135+ tests across 12 files. Run the full suite before submitting changes.
- `test_proxy.py` is a **live smoke test** — do not run it through pytest. It requires the proxy running on port 8799.
- When adding a converter parameter, add a test asserting the field appears in the v3 body.

## Commit & Pull Request Guidelines

- Commit prefixes from history: `feat:`, `fix:`, `docs:`, `refactor:` — lowercase subject.
- Never commit `.env` or any API key value.
- Keep the two-layer split in PRs: request/response-shape work goes in `converter/` with matching tests; routes and transport go in `server.py`.

## Protocol Invariants

The upstream Gateway returns 400/503 if violated. These are **not** linter-enforced:

1. Always send `ai-language-model-streaming: true` upstream.
2. v3 body must include `tools: []` and `toolChoice: {type: auto}` even with no tools.
3. Content shape is role-dependent: `system` = plain string; `user`/`assistant`/`tool` = array of parts.
4. fx headers must match the fx CLI exactly — `_v3_headers()` in `server.py` is the single source of truth.
5. HTTP/2 must stay enabled on the shared `httpx.AsyncClient`.
6. Tool calls use fx wire format: `{type: "tool-call", toolCallId, toolName, input}` with raw JSON `input`.
7. Every assistant `tool_calls` block must be immediately followed by matching `role: tool` results.

See `gateway-proxy/SAUCE.md` for the full reverse-engineering writeup and `CLAUDE.md` for detailed guidance.

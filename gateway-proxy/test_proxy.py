#!/usr/bin/env python3
"""Quick smoke test for the gateway proxy."""
import os
import httpx

BASE = "http://localhost:8799"
PROXY_KEY = os.getenv("PROXY_API_KEY", "testproxy123")

headers = {"Authorization": f"Bearer {PROXY_KEY}", "Content-Type": "application/json"}

# 1. healthz
r = httpx.get(f"{BASE}/healthz")
print(f"[1] healthz: {r.status_code} {r.json()}")

# 2. no auth -> 401
r = httpx.get(f"{BASE}/v1/models")
print(f"[2] no auth: {r.status_code} {r.text[:80]}")

# 3. bad key -> 401
r = httpx.get(f"{BASE}/v1/models", headers={"Authorization": "Bearer wrong"})
print(f"[3] bad key: {r.status_code} {r.text[:80]}")

# 4. models list
r = httpx.get(f"{BASE}/v1/models", headers=headers)
print(f"[4] models: {r.status_code}", end=" ")
if r.status_code == 200:
    data = r.json()
    models = data.get("data", [])
    print(f"({len(models)} models)")
    for m in models[:5]:
        print(f"      - {m['id']}")
else:
    print(r.text[:120])

# 5. chat completion with zai/glm-5.2 (non-stream)
r = httpx.post(
    f"{BASE}/v1/chat/completions",
    headers=headers,
    json={
        "model": "zai/glm-5.2",
        "messages": [{"role": "user", "content": "Say hi in exactly 3 words"}],
        "stream": False,
    },
    timeout=60,
)
print(f"[5] chat (glm-5.2, non-stream): {r.status_code}")
if r.status_code == 200:
    d = r.json()
    print(f"      model: {d.get('model')}")
    print(f"      usage: {d.get('usage')}")
    choice = d.get("choices", [{}])[0]
    print(f"      content: {choice.get('message', {}).get('content', '')!r}")
    print(f"      finish: {choice.get('finish_reason')}")
else:
    print(f"      ERROR: {r.text[:200]}")

# 6. chat completion streaming
print("[6] chat (glm-5.2, stream):")
with httpx.stream(
    "POST",
    f"{BASE}/v1/chat/completions",
    headers=headers,
    json={
        "model": "zai/glm-5.2",
        "messages": [{"role": "user", "content": "Count from 1 to 3"}],
        "stream": True,
    },
    timeout=60,
) as r:
    print(f"      status: {r.status_code}")
    if r.status_code == 200:
        line_count = 0
        for line in r.iter_lines():
            if line:
                print(f"      {line}")
                line_count += 1
            if line_count >= 15:
                print("      ... (truncated)")
                break
    else:
        for line in r.iter_lines():
            if line:
                print(f"      {line}")

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

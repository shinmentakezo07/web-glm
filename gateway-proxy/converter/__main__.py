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

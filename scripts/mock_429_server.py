"""Tiny server that returns 429 a configurable number of times, then proxies to the real API.

Usage:
    python scripts/mock_429_server.py [--fail-count 3] [--port 8765]

Then run inspect with:
    OPENAI_BASE_URL=http://localhost:8765/v1 inspect eval <task> --model openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

_lock = threading.Lock()
_request_counts: dict[str, int] = {}


class Handler(BaseHTTPRequestHandler):
    fail_count: int = 3

    def do_POST(self) -> None:
        key = f"{threading.current_thread().name}:{self.path}"
        with _lock:
            _request_counts[key] = _request_counts.get(key, 0) + 1
            count = _request_counts[key]

        if count <= self.fail_count:
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "1")
            self.end_headers()
            body = json.dumps(
                {
                    "error": {
                        "message": f"Rate limit exceeded (mock failure {count}/{self.fail_count})",
                        "type": "requests",
                        "code": "rate_limit_exceeded",
                    }
                }
            )
            self.wfile.write(body.encode())
            self.log_message("-> 429 (failure %d/%d)", count, self.fail_count)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        self.headers.get("Content-Type", "")
        self.rfile.read(content_length) if content_length else b""

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        body = json.dumps(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Mock response after retries",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )
        self.wfile.write(body.encode())
        self.log_message("-> 200 (success after %d failures)", self.fail_count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fail-count", type=int, default=3)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    Handler.fail_count = args.fail_count
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Mock 429 server on http://127.0.0.1:{args.port}")
    print(f"Will return 429 for first {args.fail_count} requests per path, then 200")
    print(
        f"\nRun: OPENAI_BASE_URL=http://localhost:{args.port}/v1 inspect eval <task> --model openai/gpt-4o-mini -T log_level=http"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

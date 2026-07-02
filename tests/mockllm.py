"""A tiny in-process fake Ollama server so brain tests need no real model."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

DEFAULT_DECISION = {
    "thought": "I'm parched and someone is talking to me.",
    "action": "drink",
    "say": "Oh! You're here. What's your name, out there?",
    "importance": 7,
}


def start_mock_ollama(decision: dict | None = None,
                      insights: list[str] | None = None,
                      delay: float = 0.0) -> tuple[HTTPServer, str]:
    """Serve Ollama-shaped /api/chat and /api/tags on an ephemeral port.

    Decision requests get `decision`; reflection requests (system prompt
    containing "reflective mind") get `insights`. `delay` adds thinking
    latency to chat responses (availability probes stay fast). Returns
    (server, base_url); caller should srv.stop() when done.
    """
    decision = decision or DEFAULT_DECISION
    insights = insights if insights is not None else []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep test output clean
            pass

        def do_GET(self):  # /api/tags availability probe
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self):
            if delay:
                time.sleep(delay)
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            system = body["messages"][0]["content"]
            if "reflective mind" in system:
                content = json.dumps({"insights": insights})
            else:
                content = json.dumps(decision)
            resp = json.dumps({"message": {"role": "assistant", "content": content}})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp.encode())

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    srv.stop = lambda: (srv.shutdown(), srv.server_close())
    host, port = srv.server_address
    return srv, f"http://{host}:{port}"

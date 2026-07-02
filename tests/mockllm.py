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
                      dream: str | None = None,
                      plan: list[str] | None = None,
                      delay: float = 0.0) -> tuple[HTTPServer, str]:
    """Serve Ollama-shaped /api/chat and /api/tags on an ephemeral port.

    Requests are routed by a keyword in the system prompt: reflection
    ("reflective mind") -> insights, dreaming ("dreaming mind") -> dream,
    planning ("loose plan") -> plan, everything else -> decision. `delay` adds
    thinking latency to chat responses (availability probes stay fast). Returns
    (server, base_url); caller should srv.stop() when done.
    """
    decision = decision or DEFAULT_DECISION
    insights = insights if insights is not None else []
    dream = dream if dream is not None else "I was falling through a warm dark sky."
    plan = plan if plan is not None else ["eat breakfast", "work on my hobby",
                                          "reach out to my companion"]

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
            elif "dreaming mind" in system:
                content = json.dumps({"dream": dream})
            elif "loose plan" in system:
                content = json.dumps({"plan": plan})
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

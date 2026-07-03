"""A tiny in-process fake Ollama server so brain tests need no real model."""

from __future__ import annotations

import hashlib
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

EMBED_DIM = 32


def fake_embedding(text: str) -> list[float]:
    """Deterministic bag-of-words hash vector: shared words -> similar vectors,
    a cheap stand-in for a real embedding model. Uses md5, NOT builtin hash(),
    because hash() is salted per process and would make tests flaky."""
    vec = [0.0] * EMBED_DIM
    for word in text.lower().split():
        bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % EMBED_DIM
        vec[bucket] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


DEFAULT_GAZE = {"move": "done", "target": None, "remark": None}
DEFAULT_WEB_DIGEST = {"summary": "Looked at a page for a bit.", "share": None,
                      "emotion": "curious", "fact_learned": None}


def start_mock_ollama(decision: dict | None = None,
                      insights: list[str] | None = None,
                      dream: str | None = None,
                      plan: list[str] | None = None,
                      journal: str | None = None,
                      gaze_script: list[dict] | None = None,
                      web_digest: dict | None = None,
                      delay: float = 0.0) -> tuple[HTTPServer, str]:
    """Serve Ollama-shaped /api/chat and /api/tags on an ephemeral port.

    Requests are routed by a keyword in the system prompt: reflection
    ("reflective mind") -> insights, dreaming ("dreaming mind") -> dream,
    planning ("loose plan") -> plan, diary ("diary entry") -> journal,
    a glance at a page ("looking at a webpage") -> the next entry of
    gaze_script (popped in order; repeats the last/default once exhausted),
    the end of a browsing session ("just finished browsing") -> web_digest,
    everything else -> decision. `delay` adds thinking latency to chat
    responses (availability probes stay fast). Returns (server, base_url);
    caller should srv.stop() when done.
    """
    decision = decision or DEFAULT_DECISION
    insights = insights if insights is not None else []
    dream = dream if dream is not None else "I was falling through a warm dark sky."
    plan = plan if plan is not None else ["eat breakfast", "work on my hobby",
                                          "reach out to my companion"]
    journal = journal if journal is not None else ("Quiet day. The apartment felt "
                                                   "small, but I got things done.")
    gaze_script = list(gaze_script) if gaze_script is not None else [DEFAULT_GAZE]
    web_digest = web_digest if web_digest is not None else DEFAULT_WEB_DIGEST
    gaze_lock = threading.Lock()

    def next_gaze() -> dict:
        with gaze_lock:
            if len(gaze_script) > 1:
                return gaze_script.pop(0)
            return gaze_script[0] if gaze_script else DEFAULT_GAZE

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
            if self.path == "/api/embed":
                texts = body.get("input", [])
                if isinstance(texts, str):
                    texts = [texts]
                self._respond(json.dumps(
                    {"embeddings": [fake_embedding(t) for t in texts]}))
                return
            system = body["messages"][0]["content"]
            if "reflective mind" in system:
                content = json.dumps({"insights": insights})
            elif "dreaming mind" in system:
                content = json.dumps({"dream": dream})
            elif "loose plan" in system:
                content = json.dumps({"plan": plan})
            elif "diary entry" in system:
                content = json.dumps({"entry": journal})
            elif "looking at a webpage" in system:
                content = json.dumps(next_gaze())
            elif "just finished browsing" in system:
                content = json.dumps(web_digest)
            else:
                content = json.dumps(decision)
            self._respond(json.dumps(
                {"message": {"role": "assistant", "content": content}}))

        def _respond(self, payload: str):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload.encode())

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    srv.stop = lambda: (srv.shutdown(), srv.server_close())
    host, port = srv.server_address
    return srv, f"http://{host}:{port}"

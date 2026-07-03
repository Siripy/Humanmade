"""A tiny local website for fully-offline browser tests (see internet.py).

No real network needed: WebSenses points at http://127.0.0.1:<port> and gets
real rendered pages, real links, real scrolling, real navigation history —
exercising the actual Playwright code path without ever touching the real
internet. Mirrors tests/mockllm.py's start/stop convenience.
"""

from __future__ import annotations

import socketserver
import threading
from http.server import BaseHTTPRequestHandler

# a long paragraph forces real scrolling in a small test viewport
_FILLER = " ".join(f"line-{i}" for i in range(300))

PAGES = {
    "/": """<html><head><title>Home Base</title></head><body>
        <h1>Welcome Home</h1>
        <p><a href="/volcanoes">Learn about volcanoes</a></p>
        <p><a href="/deep-sea">Learn about deep-sea life</a></p>
        <p><a href="http://example.invalid/nope">A suspicious offsite link</a></p>
        <p>{filler}</p>
        </body></html>""",
    "/volcanoes": """<html><head><title>Volcanoes</title></head><body>
        <h1>Volcanoes of the World</h1>
        <p>Mount Etna glows at night over Sicily.</p>
        <p><a href="/deep-sea">Learn about deep-sea life</a></p>
        <p><a href="/">Back home</a></p>
        </body></html>""",
    "/deep-sea": """<html><head><title>Deep Sea Life</title></head><body>
        <h1>Creatures of the Deep</h1>
        <p>Anglerfish light up the endless dark.</p>
        <p><a href="/">Back home</a></p>
        </body></html>""",
    "/search": """<html><head><title>Search Results</title></head><body>
        <h1>Results</h1>
        <p><a href="/volcanoes">Learn about volcanoes</a></p>
        </body></html>""",
}


def start_mock_website(pages: dict | None = None) -> tuple[socketserver.TCPServer, str]:
    """Serve `pages` (path -> HTML, `{filler}` optionally interpolated) on an
    ephemeral local port. Returns (server, base_url); caller should
    srv.stop() when done."""
    served = pages if pages is not None else PAGES

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            body = served.get(path)
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            html = body.format(filler=_FILLER).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    srv.stop = lambda: (srv.shutdown(), srv.server_close())
    host, port = srv.server_address
    return srv, f"http://{host}:{port}"

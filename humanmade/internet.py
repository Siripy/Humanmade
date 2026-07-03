"""The human's window onto the wider world: a real, rendered browser tab.

Sight, not scraping. The human perceives only what is drawn inside a fixed
viewport, exactly as a person looking at a screen does — foveated vision, not
a DOM dump. Nothing below the fold exists until it scrolls there; nothing
behind a link exists until it's clicked. A session is one sitting at the
computer: an ephemeral browser tab with no saved cookies or history, closed
when the human is done, so it never "remembers" being logged in anywhere
because it never was.

Safety is enforced at the browser, not by asking the model nicely: every
request but a plain GET is aborted, downloads are cancelled, popups are
closed unopened, JS dialogs are dismissed, and navigation outside the
allowlist never loads. The human can look; it structurally cannot act on
what it sees — there is no "type" or "submit" in its vocabulary here.
"""

from __future__ import annotations

import fnmatch
import os
import random
import time
from dataclasses import dataclass
from urllib.parse import quote_plus, urljoin, urlparse

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:                                    # pragma: no cover
    PLAYWRIGHT_AVAILABLE = False
    PlaywrightError = Exception

DEFAULT_ALLOWLIST = ("en.wikipedia.org", "*.wikipedia.org")
DEFAULT_SEARCH_TEMPLATE = ("https://en.wikipedia.org/w/index.php?search={query}"
                           "&title=Special:Search&fulltext=1")
DEFAULT_RANDOM_URL = "https://en.wikipedia.org/wiki/Special:Random"
MAX_VISIBLE_CHARS = 4000
MAX_LINKS = 40
_SANDBOX_CHROMIUM = "/opt/pw-browsers/chromium"  # convenience fallback, see WebSenses

# Foveated vision: only text that actually renders inside the viewport is
# read. A block-level bounding-box check alone isn't enough — a paragraph
# taller than the viewport still "intersects" it while most of its lines
# are off-screen, so any element taller than the viewport gets a slower,
# word-level check (via Range.getClientRects, which reports each wrapped
# line's real position) instead of being swallowed whole.
_VIEWPORT_SCRIPT = """
() => {
  const vh = window.innerHeight, vw = window.innerWidth;
  function rectVisible(r) {
    return r.bottom > 0 && r.top < vh && r.right > 0 && r.left < vw
           && r.width > 0 && r.height > 0;
  }
  let text = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    const el = node.parentElement;
    if (!el) continue;
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') continue;
    const full = node.textContent;
    if (!full.trim()) continue;
    const elRect = el.getBoundingClientRect();
    if (!rectVisible(elRect)) continue;                 // fully off-screen
    if (elRect.top >= 0 && elRect.bottom <= vh) {        // wholly on-screen
      text.push(full.replace(/\\s+/g, ' ').trim());
      continue;
    }
    // partially on-screen and taller than the viewport (e.g. a long
    // wrapped paragraph): only keep the words that actually render here
    const words = full.split(/(\\s+)/);
    let offset = 0, buf = [];
    for (const w of words) {
      const start = offset;
      offset += w.length;
      if (!w.trim()) continue;
      const range = document.createRange();
      range.setStart(node, start);
      range.setEnd(node, offset);
      const rects = range.getClientRects();
      for (let i = 0; i < rects.length; i++) {
        if (rectVisible(rects[i])) { buf.push(w); break; }
      }
    }
    const t = buf.join(' ').replace(/\\s+/g, ' ').trim();
    if (t) text.push(t);
  }
  const links = Array.from(document.querySelectorAll('a[href]'))
    .filter(a => rectVisible(a.getBoundingClientRect()))
    .map(a => [a.innerText.replace(/\\s+/g, ' ').trim(), a.href])
    .filter(pair => pair[0].length > 0);
  const scrollMax = Math.max(1, document.body.scrollHeight - vh);
  return {
    title: document.title,
    text: text.join(' ').slice(0, 8000),
    links: links.slice(0, 60),
    scrollFraction: Math.min(1, window.scrollY / scrollMax),
    atBottom: (window.scrollY + vh) >= (document.body.scrollHeight - 4),
  };
}
"""


class BrowserUnavailable(Exception):
    """Playwright (or its browser binary) isn't installed."""


def domain_allowed(url: str, allowlist: list[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    return any(fnmatch.fnmatch(host, pattern.lower()) for pattern in allowlist)


def resolve_start_url(query: str | None, config: dict) -> str:
    """Turn a motive (or none — idle curiosity) into somewhere to look.

    A specific query becomes a Wikipedia search (real search, no API key,
    stays inside the default allowlist). No query at all means wandering:
    one of the configured start_urls if there are any, else a random
    article. Both templates are config-overridable so tests can point this
    at a local mock site instead of the real internet.
    """
    start_urls = [u for u in (config.get("start_urls") or []) if u]
    query = (query or "").strip()
    if not query:
        if start_urls:
            return random.choice(start_urls)
        return config.get("random_url", DEFAULT_RANDOM_URL)
    template = config.get("search_url_template", DEFAULT_SEARCH_TEMPLATE)
    return template.format(query=quote_plus(query))


@dataclass
class Glimpse:
    """What was actually seen in one look at the screen — not the whole page,
    only what rendered inside the viewport at that moment."""
    url: str
    title: str
    text: str
    links: list[tuple[str, str]]
    scroll_fraction: float
    at_bottom: bool
    blocked: str | None = None


def _best_link_match(query: str, links: list[tuple[str, str]]) -> str | None:
    """A human clicks what they read, not a CSS selector: fuzzy match on the
    visible link text the mind was actually shown."""
    q = query.strip().lower()
    if not q:
        return None
    for text, href in links:                  # exact first
        if text.strip().lower() == q:
            return href
    for text, href in links:                  # then substring, either way
        t = text.strip().lower()
        if q in t or t in q:
            return href
    return None


class WebSenses:
    """One sitting at the computer. Use as a context manager; the browser tab
    (and any cookies it picked up) is discarded on exit — every session
    starts as if it had never been online before."""

    def __init__(self, allowlist: list[str] | None = None,
                 viewport: tuple[int, int] = (1280, 800),
                 nav_timeout_ms: int = 10_000,
                 chromium_path: str | None = None):
        if not PLAYWRIGHT_AVAILABLE:
            raise BrowserUnavailable(
                "playwright is not installed — `pip install playwright` "
                "(the browser downloads automatically) gives the human eyes "
                "on the web.")
        self.allowlist = list(allowlist or DEFAULT_ALLOWLIST)
        self.viewport = {"width": viewport[0], "height": viewport[1]}
        self.nav_timeout_ms = nav_timeout_ms
        self._chromium_path = chromium_path
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None

    def __enter__(self) -> "WebSenses":
        self._pw = sync_playwright().start()
        launch_kwargs = {"headless": True}
        if self._chromium_path:
            launch_kwargs["executable_path"] = self._chromium_path
        try:
            self._browser = self._pw.chromium.launch(**launch_kwargs)
        except PlaywrightError:
            # Sandboxed dev environments may only ship the full Chromium
            # build under a stable symlink rather than the default headless
            # shell Playwright expects; fall back to it if present. Harmless
            # elsewhere — the path simply won't exist on a normal machine.
            if not self._chromium_path and os.path.exists(_SANDBOX_CHROMIUM):
                self._browser = self._pw.chromium.launch(
                    headless=True, executable_path=_SANDBOX_CHROMIUM)
            else:
                self._pw.stop()
                raise
        self._context = self._browser.new_context(viewport=self.viewport)
        self._context.set_default_navigation_timeout(self.nav_timeout_ms)
        self._context.set_default_timeout(self.nav_timeout_ms)
        # --- read-only enforcement: the human can look, never act ---
        self._page = None  # guards _close_stray_page against self-closing below
        self._context.route("**/*", self._guard_request)
        self._context.on("page", self._close_stray_page)   # no popups/new tabs
        self._context.add_init_script("window.open = () => null;")
        self._page = self._context.new_page()
        self._page.on("dialog", lambda d: d.dismiss())      # no alert/confirm
        self._page.on("download", lambda d: d.cancel())     # no downloads
        return self

    def _close_stray_page(self, page) -> None:
        # Fires for every page created in the context, including our own
        # main page — only close ones that AREN'T it (self._page is still
        # None while the main page is being created, so it's never caught
        # by this).
        if self._page is not None and page is not self._page:
            try:
                page.close()
            except PlaywrightError:
                pass

    def __exit__(self, *exc_info) -> None:
        try:
            if self._context:
                self._context.close()
        finally:
            try:
                if self._browser:
                    self._browser.close()
            finally:
                if self._pw:
                    self._pw.stop()

    def _guard_request(self, route) -> None:
        request = route.request
        if request.method != "GET":
            route.abort()
            return
        if request.resource_type == "document" and not domain_allowed(
                request.url, self.allowlist):
            route.abort()
            return
        route.continue_()

    # -- glances ----------------------------------------------------------

    def open(self, url: str) -> Glimpse:
        if not domain_allowed(url, self.allowlist):
            return Glimpse(url=url, title="", text="", links=[],
                           scroll_fraction=0.0, at_bottom=True,
                           blocked=f"{urlparse(url).hostname or url} isn't "
                                   "somewhere you know how to reach")
        try:
            self._page.goto(url, wait_until="domcontentloaded")
        except PlaywrightError as e:
            return Glimpse(url=url, title="", text="", links=[],
                           scroll_fraction=0.0, at_bottom=True,
                           blocked=f"the page wouldn't load ({e.message if hasattr(e, 'message') else e})")
        return self._look()

    def scroll(self) -> Glimpse:
        self._page.evaluate("window.scrollBy(0, window.innerHeight * 0.9)")
        self._page.wait_for_timeout(80)
        return self._look()

    def click(self, link_text: str) -> Glimpse:
        glimpse = self._look()
        href = _best_link_match(link_text, glimpse.links)
        if href is None:
            return Glimpse(url=glimpse.url, title=glimpse.title, text=glimpse.text,
                           links=glimpse.links, scroll_fraction=glimpse.scroll_fraction,
                           at_bottom=glimpse.at_bottom,
                           blocked=f'no visible link matches "{link_text}"')
        return self.open(href)

    def go_back(self) -> Glimpse:
        try:
            self._page.go_back(wait_until="domcontentloaded")
        except PlaywrightError:
            pass
        return self._look()

    def screenshot(self) -> bytes:
        """Raw pixels of the current viewport, for a future vision model.
        Never persisted by this module — callers must discard after use."""
        return self._page.screenshot()

    def _look(self) -> Glimpse:
        data = self._page.evaluate(_VIEWPORT_SCRIPT)
        seen_hrefs: set[str] = set()
        links: list[tuple[str, str]] = []
        for text, href in data.get("links", []):
            if not text.strip() or href in seen_hrefs:
                continue
            seen_hrefs.add(href)
            links.append((text[:80], urljoin(self._page.url, href)))
            if len(links) >= MAX_LINKS:
                break
        return Glimpse(
            url=self._page.url,
            title=(data.get("title") or "")[:200],
            text=(data.get("text") or "")[:MAX_VISIBLE_CHARS],
            links=links,
            scroll_fraction=float(data.get("scrollFraction", 0.0)),
            at_bottom=bool(data.get("atBottom", True)),
        )


def run_browse_session(brain, persona: dict, query: str | None, config: dict) -> dict:
    """One full sitting at the computer, start to finish: opens a page,
    glances, decides move by move whether to scroll/click/back/stop
    (budget scaled by curiosity), then reflects on the whole session.

    `brain` is duck-typed — anything with .gaze(persona, glimpse_text,
    glances_left) and .web_digest(persona, transcript) works (LLMBrain).
    Returns a dict: summary, share, emotion, fact_learned, sites_visited,
    or {"error": ...} if the session couldn't be completed.
    """
    dims = persona.get("disposition", {}).get("dimensions", {})
    openness = dims.get("openness", 0.5)
    max_glances = int(config.get("max_glances", 8))
    budget = max(2, round(max_glances * (0.6 + openness)))
    timeout_s = float(config.get("session_timeout_seconds", 45))
    allowlist = config.get("allowlist") or list(DEFAULT_ALLOWLIST)
    viewport = config.get("viewport") or {"width": 1280, "height": 800}
    nav_timeout_ms = int(float(config.get("nav_timeout_seconds", 10)) * 1000)
    chromium_path = config.get("chromium_path")
    start_url = resolve_start_url(query, config)

    transcript: list[str] = []
    sites_visited: set[str] = set()
    deadline = time.monotonic() + timeout_s

    with WebSenses(allowlist=allowlist,
                  viewport=(viewport.get("width", 1280), viewport.get("height", 800)),
                  nav_timeout_ms=nav_timeout_ms, chromium_path=chromium_path) as senses:
        glimpse = senses.open(start_url)
        glances_used = 0
        while glances_used < budget and time.monotonic() < deadline:
            if glimpse.blocked:
                transcript.append(f"Tried to reach {glimpse.url}, but "
                                  f"{glimpse.blocked}.")
                break
            host = urlparse(glimpse.url).hostname or ""
            if host:
                sites_visited.add(host)
            entry = (f'On "{glimpse.title}" ({host}): '
                     f'{glimpse.text[:600] or "(a mostly blank page)"}')
            if glimpse.at_bottom:
                entry += "\n(this is the bottom of the page — nothing more below)"
            if glimpse.links:
                entry += ("\nVisible links here: "
                         + "; ".join(t for t, _ in glimpse.links[:8]))
            transcript.append(entry)
            glances_used += 1

            gaze = brain.gaze(persona, entry, budget - glances_used)
            if gaze.get("remark"):
                transcript.append(f"(their thought: {gaze['remark']})")
            move = gaze.get("move", "done")
            if move == "done":
                break
            elif move == "scroll":
                if not glimpse.at_bottom:
                    glimpse = senses.scroll()
                # already at the bottom: nothing to scroll to, so the same
                # glimpse repeats and the mind gets another (budget-bounded)
                # chance to click/back/done instead of the session vanishing
            elif move == "click" and gaze.get("target"):
                glimpse = senses.click(gaze["target"])
            elif move == "back":
                glimpse = senses.go_back()
            else:
                break   # click-without-target, or an unrecognized move

    if not transcript:
        return {"summary": None, "share": None, "emotion": None,
                "fact_learned": None, "sites_visited": []}
    digest = brain.web_digest(persona, "\n\n".join(transcript))
    digest["sites_visited"] = sorted(sites_visited)
    return digest

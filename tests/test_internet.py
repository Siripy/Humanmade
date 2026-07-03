"""Tests for internet.py: the browser substrate itself, entirely offline
against a local mock website (tests/mockweb.py) — no real network, but a
real rendered Chromium tab doing real navigation, scrolling, and clicking."""

from __future__ import annotations

import unittest

from humanmade import internet
from humanmade.internet import (BrowserUnavailable, WebSenses,
                                domain_allowed, resolve_start_url,
                                run_browse_session)
from tests.mockweb import start_mock_website

VIEWPORT = (800, 400)

_browser_probe: bool | None = None


def real_browser_available() -> bool:
    """True when Playwright AND a launchable Chromium are actually present.
    Import alone isn't enough — `pip install playwright` without
    `playwright install chromium` has the module but no binary."""
    global _browser_probe
    if _browser_probe is None:
        if not internet.PLAYWRIGHT_AVAILABLE:
            _browser_probe = False
        else:
            try:
                with WebSenses(allowlist=["127.0.0.1"], viewport=VIEWPORT,
                              nav_timeout_ms=8000):
                    pass
                _browser_probe = True
            except Exception:
                _browser_probe = False
    return _browser_probe


needs_browser = unittest.skipUnless(
    real_browser_available(),
    "needs Playwright + Chromium (pip install playwright && "
    "playwright install chromium)")


class TestDomainAllowlist(unittest.TestCase):
    def test_exact_host_allowed(self):
        self.assertTrue(domain_allowed("https://en.wikipedia.org/wiki/Owl",
                                       ["en.wikipedia.org"]))

    def test_other_host_blocked(self):
        self.assertFalse(domain_allowed("https://evil.example/",
                                        ["en.wikipedia.org"]))

    def test_wildcard_subdomain(self):
        self.assertTrue(domain_allowed("https://commons.wikipedia.org/x",
                                       ["*.wikipedia.org"]))
        self.assertFalse(domain_allowed("https://wikipedia.org/x",
                                        ["*.wikipedia.org"]))  # bare apex, no leading label

    def test_no_hostname_is_never_allowed(self):
        self.assertFalse(domain_allowed("not a url", ["*"]))


class TestResolveStartUrl(unittest.TestCase):
    def test_query_becomes_a_search(self):
        url = resolve_start_url("deep sea creatures", {})
        self.assertTrue(url.startswith("https://en.wikipedia.org"))
        self.assertIn("deep+sea+creatures", url)

    def test_no_query_no_start_urls_is_random(self):
        self.assertEqual(resolve_start_url(None, {}), internet.DEFAULT_RANDOM_URL)

    def test_no_query_picks_a_start_url(self):
        cfg = {"start_urls": ["http://x.test/a", "http://x.test/b"]}
        self.assertIn(resolve_start_url("", cfg), cfg["start_urls"])

    def test_templates_are_overridable_for_tests(self):
        cfg = {"search_url_template": "http://127.0.0.1:1/s?q={query}"}
        self.assertEqual(resolve_start_url("owls", cfg),
                         "http://127.0.0.1:1/s?q=owls")


class TestBrowserUnavailable(unittest.TestCase):
    def test_raises_when_playwright_missing(self):
        original = internet.PLAYWRIGHT_AVAILABLE
        internet.PLAYWRIGHT_AVAILABLE = False
        try:
            with self.assertRaises(BrowserUnavailable):
                WebSenses()
        finally:
            internet.PLAYWRIGHT_AVAILABLE = original


@needs_browser
class TestWebSensesAgainstMockSite(unittest.TestCase):
    """Real Playwright, real rendering, real navigation — just pointed at a
    local server instead of the internet."""

    @classmethod
    def setUpClass(cls):
        cls.srv, cls.base = start_mock_website()

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def senses(self, allowlist=("127.0.0.1",)):
        return WebSenses(allowlist=list(allowlist), viewport=VIEWPORT,
                         nav_timeout_ms=8000)

    def test_open_renders_a_real_page(self):
        with self.senses() as s:
            g = s.open(self.base + "/")
            self.assertEqual(g.title, "Home Base")
            self.assertIsNone(g.blocked)
            self.assertIn("Welcome Home", g.text)

    def test_only_viewport_is_seen_at_first_not_the_whole_page(self):
        """Foveation: the long filler paragraph is far below the fold in an
        800x400 viewport, so the first glimpse must not contain it, but a
        page-full of scrolling eventually reveals it."""
        with self.senses() as s:
            g = s.open(self.base + "/")
            self.assertNotIn("line-299", g.text)  # last filler word, off-screen
            self.assertFalse(g.at_bottom)
            seen_filler = False
            for _ in range(20):
                if g.at_bottom:
                    break
                g = s.scroll()
                if "line-0" in g.text or "line-1 " in g.text:
                    seen_filler = True
            self.assertTrue(seen_filler, "scrolling should eventually reveal "
                            "content that was originally off-screen")

    def test_click_follows_a_real_link_by_visible_text(self):
        with self.senses() as s:
            s.open(self.base + "/")
            g = s.click("Learn about volcanoes")
            self.assertEqual(g.title, "Volcanoes")
            self.assertIn("Etna", g.text)

    def test_click_is_fuzzy_case_insensitive_substring(self):
        with self.senses() as s:
            s.open(self.base + "/")
            g = s.click("volcanoes")   # lowercase, partial
            self.assertEqual(g.title, "Volcanoes")

    def test_click_with_no_match_is_reported_not_crashed(self):
        with self.senses() as s:
            s.open(self.base + "/")
            g = s.click("a link that does not exist anywhere")
            self.assertIsNotNone(g.blocked)

    def test_back_returns_to_real_prior_page(self):
        with self.senses() as s:
            s.open(self.base + "/")
            s.click("Learn about volcanoes")
            g = s.go_back()
            self.assertEqual(g.title, "Home Base")

    def test_offsite_link_is_blocked_not_followed(self):
        """The read-only enforcement: an off-allowlist navigation must never
        actually load, even though the link is right there on the page."""
        with self.senses() as s:
            s.open(self.base + "/")
            g = s.click("A suspicious offsite link")
            self.assertIsNotNone(g.blocked)

    def test_direct_open_of_offsite_url_is_blocked(self):
        with self.senses() as s:
            g = s.open("http://example.invalid/nope")
            self.assertIsNotNone(g.blocked)
            self.assertEqual(g.text, "")

    def test_screenshot_returns_real_pixels(self):
        with self.senses() as s:
            s.open(self.base + "/")
            shot = s.screenshot()
            self.assertIsInstance(shot, bytes)
            self.assertGreater(len(shot), 100)

    def test_no_form_post_ever_reaches_the_server(self):
        """Even if a page tried to submit a form, only GET survives the
        request guard — this is the structural safety property, not a
        prompt-level instruction."""
        post_page = {"/": """<html><body>
            <form method="POST" action="/submit"><input name="x"></form>
            <script>fetch('/submit', {{method:'POST'}});</script>
            </body></html>""",
                     "/submit": "<html><body>should never be reached</body></html>"}
        srv, base = start_mock_website(post_page)
        try:
            with self.senses() as s:
                g = s.open(base + "/")
                self.assertIsNone(g.blocked)
                # give the page's own fetch() a moment to (fail to) fire
                s._page.wait_for_timeout(200)
        finally:
            srv.stop()


@needs_browser
class TestRunBrowseSession(unittest.TestCase):
    """The full gaze loop against the mock site, driven by a fake brain
    (no LLM needed to test the orchestration logic itself)."""

    @classmethod
    def setUpClass(cls):
        cls.srv, cls.base = start_mock_website()

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def _config(self, **overrides):
        cfg = {"allowlist": ["127.0.0.1"], "max_glances": 6,
              "session_timeout_seconds": 25, "nav_timeout_seconds": 8,
              "viewport": {"width": VIEWPORT[0], "height": VIEWPORT[1]},
              "random_url": self.base + "/"}
        cfg.update(overrides)
        return cfg

    def _persona(self, openness=0.5):
        return {"disposition": {"dimensions": {"openness": openness}}}

    def test_scripted_session_clicks_then_stops(self):
        class ScriptedBrain:
            def __init__(self):
                self.moves = [
                    {"move": "click", "target": "Learn about volcanoes", "remark": "curious"},
                    {"move": "done", "target": None, "remark": "enough for now"},
                ]
            def gaze(self, persona, glimpse_text, glances_left):
                return self.moves.pop(0) if self.moves else {"move": "done"}
            def web_digest(self, persona, transcript):
                self.transcript = transcript
                return {"summary": "Read about volcanoes.", "share": None,
                       "emotion": "curious", "fact_learned": "Etna glows at night"}

        brain = ScriptedBrain()
        result = run_browse_session(brain, self._persona(), None, self._config())
        self.assertEqual(result["summary"], "Read about volcanoes.")
        self.assertEqual(result["fact_learned"], "Etna glows at night")
        self.assertIn("127.0.0.1", result["sites_visited"])
        self.assertIn("Volcanoes", brain.transcript)   # actually saw the second page

    def test_done_on_first_glance_still_produces_a_digest(self):
        class ImmediatelyDoneBrain:
            def gaze(self, persona, glimpse_text, glances_left):
                return {"move": "done", "target": None, "remark": None}
            def web_digest(self, persona, transcript):
                return {"summary": "Just a glance.", "share": None,
                       "emotion": None, "fact_learned": None}

        result = run_browse_session(ImmediatelyDoneBrain(), self._persona(),
                                    None, self._config())
        self.assertEqual(result["summary"], "Just a glance.")

    def test_glance_budget_is_respected(self):
        class InfiniteScroller:
            def __init__(self):
                self.calls = 0
            def gaze(self, persona, glimpse_text, glances_left):
                self.calls += 1
                return {"move": "scroll", "target": None, "remark": None}
            def web_digest(self, persona, transcript):
                return {"summary": "Scrolled a lot.", "share": None,
                       "emotion": None, "fact_learned": None}

        brain = InfiniteScroller()
        cfg = self._config(max_glances=3)
        run_browse_session(brain, self._persona(openness=0.5), None, cfg)
        # budget = round(3 * (0.6 + 0.5)) = 3; must not run away forever
        self.assertLessEqual(brain.calls, 4)

    def test_curious_persona_gets_a_bigger_budget(self):
        class Counter:
            def __init__(self):
                self.calls = 0
            def gaze(self, persona, glimpse_text, glances_left):
                self.calls += 1
                return {"move": "scroll", "target": None, "remark": None}
            def web_digest(self, persona, transcript):
                return {"summary": "ok", "share": None, "emotion": None,
                       "fact_learned": None}

        incurious, curious = Counter(), Counter()
        cfg = self._config(max_glances=10)
        run_browse_session(incurious, self._persona(openness=0.1), None, cfg)
        run_browse_session(curious, self._persona(openness=0.9), None, cfg)
        self.assertGreater(curious.calls, incurious.calls)

    def test_blocked_start_url_yields_a_graceful_digest(self):
        class AnyBrain:
            def gaze(self, *a, **k):
                return {"move": "done"}
            def web_digest(self, persona, transcript):
                self.saw = transcript
                return {"summary": "Couldn't get anywhere.", "share": None,
                       "emotion": None, "fact_learned": None}

        brain = AnyBrain()
        cfg = self._config(allowlist=["127.0.0.1"])
        result = run_browse_session(brain, self._persona(),
                                    "site:example.invalid nope", cfg)
        # query resolves to a Wikipedia search URL, off this test's allowlist
        self.assertEqual(result["sites_visited"], [])
        self.assertTrue("reach" in brain.saw or "load" in brain.saw)

    def test_playwright_missing_propagates_as_browser_unavailable(self):
        original = internet.PLAYWRIGHT_AVAILABLE
        internet.PLAYWRIGHT_AVAILABLE = False
        try:
            with self.assertRaises(BrowserUnavailable):
                run_browse_session(object(), self._persona(), None, self._config())
        finally:
            internet.PLAYWRIGHT_AVAILABLE = original


if __name__ == "__main__":
    unittest.main()

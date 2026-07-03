"""Agent-level integration tests for the browse action: gating, cooldowns,
result application, habit/news wiring, and full end-to-end sessions driven
by a real (local, mock) LLM against a real (local, mock) website."""

from __future__ import annotations

import tempfile
import time
import unittest
from unittest.mock import patch

from humanmade import internet
from humanmade.agent import Human
from tests.mockllm import start_mock_ollama
from tests.mockweb import start_mock_website
from tests.test_agent import THINK_TIMEOUT, drive_one_thought, make_human
from tests.test_internet import needs_browser

FAKE_SESSION_RESULT = {"summary": None, "share": None, "emotion": None,
                       "fact_learned": None, "sites_visited": []}


def no_real_browsing(*args, **kwargs):
    """Stand-in for internet.run_browse_session: instant, no Playwright, no
    network — for tests that exercise the agent's gating/bookkeeping around
    a browse session, not a real session's content (see test_internet.py and
    TestFullBrowsingSession for those)."""
    return dict(FAKE_SESSION_RESULT)


def wait_for_web_memory(human, timeout=THINK_TIMEOUT):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with human.lock:
            if human.memory.recent(5, kinds=("web",)):
                return
        time.sleep(0.05)


def wait_for_browse_unavailable(human, timeout=THINK_TIMEOUT):
    deadline = time.time() + timeout
    while not human._browse_unavailable and time.time() < deadline:
        time.sleep(0.05)


def wait_for_calls(mock_obj, n, timeout=5.0):
    """Wait for an async worker thread to actually reach a (patched, no
    real I/O) call — avoids racing the background thread's own scheduling,
    and makes sure it's done before the test tears the human down."""
    deadline = time.time() + timeout
    while mock_obj.call_count < n and time.time() < deadline:
        time.sleep(0.02)
    time.sleep(0.05)  # let the worker finish applying its result under lock


class TestActionGating(unittest.TestCase):
    def test_browse_absent_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)  # no "internet" key at all
            self.assertFalse(human.internet_enabled)
            self.assertNotIn("browse", human._available_actions())
            human.memory.close()

    def test_browse_present_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, {"internet": {"enabled": True}})
            self.assertTrue(human.internet_enabled)
            self.assertIn("browse", human._available_actions())
            human.memory.close()

    def test_browse_absent_once_marked_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, {"internet": {"enabled": True}})
            human._browse_unavailable = True
            self.assertNotIn("browse", human._available_actions())
            human.memory.close()

    def test_model_choosing_browse_while_disabled_falls_back_to_idle(self):
        srv, base_url = start_mock_ollama(
            decision={"thought": "let's look something up", "action": "browse",
                      "say": None, "browse_query": "owls", "importance": 4})
        self.addCleanup(srv.stop)
        cfg = {"llm": {"provider": "ollama", "base_url": base_url,
                       "model": "test", "timeout_seconds": 5}}
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, cfg)   # internet not enabled
            drive_one_thought(human)
            self.assertEqual(human.current_action, "idle")
            human.memory.close()


class TestPerformGuards(unittest.TestCase):
    """_perform("browse", ...) must only actually start a session when
    everything lines up, and must honestly fall back to idle otherwise."""

    def test_no_session_when_internet_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = make_human(tmp)
            human.llm_online = True  # so only "internet_enabled" is the gate
            human._perform("browse", browse_query="owls")
            self.assertEqual(human.current_action, "idle")
            self.assertFalse(any("settles in at the computer" in e for e in events))
            human.memory.close()

    def test_no_session_when_llm_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = make_human(tmp, {"internet": {"enabled": True}})
            human.llm_online = False
            human._perform("browse", browse_query="owls")
            self.assertEqual(human.current_action, "idle")
            self.assertFalse(any("settles in" in e for e in events))
            human.memory.close()

    @patch("humanmade.agent.internet.run_browse_session", side_effect=no_real_browsing)
    def test_cooldown_blocks_a_second_session(self, mock_run):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = make_human(
                tmp, {"internet": {"enabled": True, "cooldown_minutes": 90}})
            human.llm_online = True
            human._perform("browse", browse_query="owls")
            self.assertEqual(human.current_action, "browse")
            wait_for_calls(mock_run, 1)
            events.clear()
            human._perform("browse", browse_query="owls again")
            self.assertEqual(human.current_action, "idle")
            self.assertFalse(any("settles in" in e for e in events))
            self.assertEqual(mock_run.call_count, 1)  # the second never ran
            human.memory.close()

    @patch("humanmade.agent.internet.run_browse_session", side_effect=no_real_browsing)
    def test_cooldown_expires(self, mock_run):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = make_human(
                tmp, {"internet": {"enabled": True, "cooldown_minutes": 10}})
            human.llm_online = True
            human._perform("browse", browse_query="owls")
            wait_for_calls(mock_run, 1)
            human.body.sim_minutes += 11
            events.clear()
            human._perform("browse", browse_query="owls again")
            self.assertEqual(human.current_action, "browse")
            wait_for_calls(mock_run, 2)
            self.assertEqual(mock_run.call_count, 2)
            human.memory.close()


class TestApplyBrowseResult(unittest.TestCase):
    def _human(self, tmp):
        human, speech, events = make_human(tmp, {"internet": {"enabled": True}})
        return human, speech, events

    def test_unavailable_marks_the_flag_and_narrates_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = self._human(tmp)
            human._apply_browse_result({"error": "no playwright", "unavailable": True})
            self.assertTrue(human._browse_unavailable)
            self.assertTrue(any("no working browser" in e for e in events))
            human.memory.close()

    def test_error_is_narrated_and_remembered_lightly(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = self._human(tmp)
            human._apply_browse_result({"error": "the page wouldn't load"})
            self.assertTrue(any("gives up" in e for e in events))
            obs = [m for m in human.memory.recent(10) if m.kind == "observation"]
            self.assertTrue(any("couldn't connect" in m.text for m in obs))
            human.memory.close()

    def test_summary_becomes_a_web_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human(tmp)
            human._apply_browse_result({"summary": "Read about owls.",
                                        "emotion": None, "fact_learned": None,
                                        "share": None, "sites_visited": []})
            web = human.memory.recent(5, kinds=("web",))
            self.assertTrue(web)
            self.assertIn("owls", web[0].text)
            human.memory.close()

    def test_emotion_is_felt_and_raises_importance(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human(tmp)
            human._apply_browse_result({"summary": "Unsettling article.",
                                        "emotion": "unsettled", "fact_learned": None,
                                        "share": None, "sites_visited": []})
            self.assertEqual(human.emotion, "unease")
            web = human.memory.recent(5, kinds=("web",))
            self.assertEqual(web[0].importance, 7.0)
            human.memory.close()

    def test_fact_learned_becomes_a_deduped_lesson(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human(tmp)
            result = {"summary": "Learned something.", "emotion": None,
                     "fact_learned": "Owls can rotate their heads 270 degrees",
                     "share": None, "sites_visited": []}
            human._apply_browse_result(dict(result))
            human._apply_browse_result(dict(result))  # repeat: must not duplicate
            lessons = human.memory.recent(10, kinds=("lesson",))
            self.assertEqual(len(lessons), 1)
            human.memory.close()

    def test_share_is_queued_as_news_only_while_away(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human(tmp)
            result = {"summary": "Neat find.", "emotion": None, "fact_learned": None,
                     "share": "Owls can rotate their heads a lot!",
                     "sites_visited": []}
            human._apply_browse_result(dict(result))
            self.assertEqual(human.pending_news, [])  # companion present: no need
            human.set_away()
            human._apply_browse_result(dict(result))
            self.assertTrue(any("rotate their heads" in n for n in human.pending_news))
            human.memory.close()

    def test_fun_gain_scales_with_openness(self):
        with tempfile.TemporaryDirectory() as tmp:
            incurious, _, _ = self._human(tmp)
        with tempfile.TemporaryDirectory() as tmp2:
            curious, _, _ = self._human(tmp2)
        incurious.persona["disposition"]["dimensions"]["openness"] = 0.1
        curious.persona["disposition"]["dimensions"]["openness"] = 0.9
        incurious.body.fun = curious.body.fun = 20.0
        result = {"summary": "ok", "emotion": None, "fact_learned": None,
                 "share": None, "sites_visited": []}
        incurious._apply_browse_result(dict(result))
        curious._apply_browse_result(dict(result))
        self.assertGreater(curious.body.fun, incurious.body.fun)
        incurious.memory.close(); curious.memory.close()

    def test_empty_result_is_a_quiet_non_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = self._human(tmp)
            human._apply_browse_result({"summary": None, "emotion": None,
                                        "fact_learned": None, "share": None,
                                        "sites_visited": []})
            self.assertFalse(human.memory.recent(5, kinds=("web",)))
            self.assertTrue(any("nothing much to show" in e for e in events))
            human.memory.close()


class TestHabitAndNewLife(unittest.TestCase):
    @patch("humanmade.agent.internet.run_browse_session", side_effect=no_real_browsing)
    def test_browsing_is_recorded_into_the_routine(self, mock_run):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, {"internet": {"enabled": True}})
            human.llm_online = True
            human.body.sim_minutes = (human.body.sim_minutes // 1440) * 1440 + 20 * 60
            for i in range(6):
                human._last_browse_sim = -10 ** 9  # bypass cooldown for the test
                human._perform("browse", browse_query="owls")
                wait_for_calls(mock_run, i + 1)
                human.body.sim_minutes += 1440
            self.assertIn(20, human.habitual_hours("browse", min_count=3.0))
            human.memory.close()

    def test_new_life_resets_cooldown_but_not_unavailable_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, {"internet": {"enabled": True}})
            human._browse_unavailable = True
            human._last_browse_sim = 12345.0
            human.new_life()
            self.assertEqual(human._last_browse_sim, -10 ** 9)
            # reflects "this computer has no browser", a fact about the
            # environment, not the person -- carries over between lives
            self.assertTrue(human._browse_unavailable)
            human.memory.close()


@needs_browser
class TestFullBrowsingSession(unittest.TestCase):
    """The whole pipeline for real: a mock LLM chooses to browse, a real
    Playwright session runs against a real local website, and the result
    lands back in the human's memory, mood, and relationship with news."""

    @classmethod
    def setUpClass(cls):
        cls.web_srv, cls.web_base = start_mock_website()

    @classmethod
    def tearDownClass(cls):
        cls.web_srv.stop()

    def _config(self, ollama_base, **internet_overrides):
        internet_cfg = {"enabled": True, "allowlist": ["127.0.0.1"],
                        "start_urls": [self.web_base + "/"],
                        "max_glances": 5, "session_timeout_seconds": 20,
                        "nav_timeout_seconds": 8,
                        "viewport": {"width": 800, "height": 400}}
        internet_cfg.update(internet_overrides)
        return {"llm": {"provider": "ollama", "base_url": ollama_base,
                        "model": "test", "timeout_seconds": 8},
               "internet": internet_cfg}

    def test_end_to_end_browse_forms_a_memory_and_a_feeling(self):
        srv, ollama_base = start_mock_ollama(
            decision={"thought": "curious", "action": "browse", "say": None,
                      "browse_query": None, "importance": 4},
            gaze_script=[{"move": "click", "target": "Learn about volcanoes",
                         "remark": "ooh lava"},
                        {"move": "done", "target": None, "remark": None}],
            web_digest={"summary": "Read about volcanoes, thought it was neat.",
                       "share": "Did you know Etna glows at night?",
                       "emotion": "curious", "fact_learned": "Etna glows at night"})
        self.addCleanup(srv.stop)
        with tempfile.TemporaryDirectory() as tmp:
            human, speech, events = make_human(tmp, self._config(ollama_base))
            self.assertTrue(human.llm_online)
            drive_one_thought(human)
            self.assertEqual(human.current_action, "browse")
            self.assertTrue(any("settles in at the computer" in e for e in events))

            wait_for_web_memory(human)
            web = human.memory.recent(5, kinds=("web",))
            self.assertTrue(web, "a web memory should have formed")
            self.assertIn("volcano", web[0].text.lower())
            self.assertEqual(human.emotion, "curiosity")
            lessons = human.memory.recent(5, kinds=("lesson",))
            self.assertTrue(any("Etna" in m.text for m in lessons))
            human.memory.close()

    def test_offsite_link_never_actually_loads_during_a_real_session(self):
        """The gaze model is tempted by the offsite link on the home page;
        the safety layer must refuse it regardless of what the mind wants."""
        srv, ollama_base = start_mock_ollama(
            decision={"thought": "curious", "action": "browse", "say": None,
                      "browse_query": None, "importance": 4},
            gaze_script=[{"move": "click", "target": "A suspicious offsite link",
                         "remark": None}],
            web_digest={"summary": "Tried to follow a link but it didn't work out.",
                       "share": None, "emotion": None, "fact_learned": None})
        self.addCleanup(srv.stop)
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, self._config(ollama_base))
            drive_one_thought(human)
            wait_for_web_memory(human)
            web = human.memory.recent(5, kinds=("web",))
            self.assertTrue(web)
            self.assertNotIn("example.invalid", web[0].text)
            human.memory.close()

    def test_share_reaches_pending_news_while_away(self):
        srv, ollama_base = start_mock_ollama(
            decision={"thought": "curious", "action": "browse", "say": None,
                      "browse_query": None, "importance": 4},
            gaze_script=[{"move": "done", "target": None, "remark": None}],
            web_digest={"summary": "Read a bit.", "share": "You won't believe this!",
                       "emotion": None, "fact_learned": None})
        self.addCleanup(srv.stop)
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, self._config(ollama_base))
            human.set_away()
            drive_one_thought(human)
            wait_for_web_memory(human)
            self.assertTrue(any("won't believe this" in n for n in human.pending_news))
            human.memory.close()

    def test_playwright_missing_disables_browse_action_after_one_try(self):
        srv, ollama_base = start_mock_ollama(
            decision={"thought": "curious", "action": "browse", "say": None,
                      "browse_query": None, "importance": 4})
        self.addCleanup(srv.stop)
        original = internet.PLAYWRIGHT_AVAILABLE
        internet.PLAYWRIGHT_AVAILABLE = False
        try:
            with tempfile.TemporaryDirectory() as tmp:
                human, _, events = make_human(tmp, self._config(ollama_base))
                drive_one_thought(human)
                wait_for_browse_unavailable(human)
                self.assertTrue(human._browse_unavailable)
                self.assertTrue(any("no working browser" in e for e in events))
                self.assertNotIn("browse", human._available_actions())
                human.memory.close()
        finally:
            internet.PLAYWRIGHT_AVAILABLE = original

    def test_stale_browse_result_after_new_life_is_discarded(self):
        srv, ollama_base = start_mock_ollama(
            decision={"thought": "curious", "action": "browse", "say": None,
                      "browse_query": None, "importance": 4},
            delay=0.3)  # give new_life() time to run mid-session
        self.addCleanup(srv.stop)
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = make_human(tmp, self._config(ollama_base))
            drive_one_thought(human)
            self.assertEqual(human.current_action, "browse")
            human.body.hydration = 0.0  # force death imminently
            human.body.satiety = 0.0
            human.body.health = 0.5
            for _ in range(400):
                human.tick(5.0)
                if not human.body.alive:
                    break
            self.assertFalse(human.body.alive)
            old_name = human.persona["name"]
            human.new_life()
            time.sleep(1.0)  # let any in-flight worker try (and fail) to land
            self.assertFalse(any("closes the browser" in e and old_name in e
                                 for e in events[-3:]))
            human.memory.close()


if __name__ == "__main__":
    unittest.main()

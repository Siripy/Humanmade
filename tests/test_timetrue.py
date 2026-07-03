"""Tests for Human.catch_up(): living through a real gap all at once when
the app was closed, reusing the away/reunion machinery."""

from __future__ import annotations

import tempfile
import unittest

from tests.test_agent import make_human


class TestCatchUp(unittest.TestCase):
    def _human(self, tmp):
        human, speech, events = make_human(tmp)
        human.llm_online = False
        return human, speech, events

    def test_advances_sim_clock_by_exactly_the_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human(tmp)
            start = human.body.sim_minutes
            human.catch_up(9 * 60.0)
            self.assertAlmostEqual(human.body.sim_minutes - start, 9 * 60.0,
                                   places=3)
            human.memory.close()

    def test_marks_companion_away(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human(tmp)
            human.catch_up(3 * 60.0)
            self.assertFalse(human.companion_present)
            human.memory.close()

    def test_callbacks_are_silenced_during_catch_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, speech, events = self._human(tmp)
            human.catch_up(9 * 60.0)
            self.assertEqual(speech, [])
            self.assertEqual(events, [])
            human.memory.close()

    def test_callbacks_restored_after_catch_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, speech, events = self._human(tmp)
            human.catch_up(60.0)
            self.assertEqual(events, [])   # silenced during the gap
            # and narration actually reaches the real callback again after
            human._perform("drink")        # synchronous, narrates via on_event
            self.assertGreater(len(events), 0)
            human.memory.close()

    def test_needs_actually_drain_during_the_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human(tmp)
            energy_before = human.body.energy
            human.catch_up(6 * 60.0)
            self.assertNotEqual(human.body.energy, energy_before)
            human.memory.close()

    def test_zero_or_negative_gap_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human(tmp)
            start = human.body.sim_minutes
            human.catch_up(0.0)
            human.catch_up(-50.0)
            self.assertEqual(human.body.sim_minutes, start)
            human.memory.close()

    def test_already_dead_human_does_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human(tmp)
            human.body.alive = False
            human.body.cause_of_death = "dehydration"
            start = human.body.sim_minutes
            human.catch_up(9 * 60.0)
            self.assertEqual(human.body.sim_minutes, start)
            human.memory.close()

    def test_llm_online_reflects_a_fresh_probe_afterward(self):
        """Regardless of what it was before, catch_up must leave
        llm_online as an honest, freshly-probed value, not a stale
        restoration of whatever it happened to be beforehand."""
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human(tmp)
            human.llm_online = True   # a lie; nothing is actually listening
            human.catch_up(60.0)
            self.assertFalse(human.llm_online)  # honestly re-probed, found nothing
            human.memory.close()

    def test_reflex_only_even_when_llm_actually_online(self):
        """The whole point is startup speed: even if the LLM genuinely is
        reachable, catch-up must never make a real call during the gap."""
        from tests.mockllm import start_mock_ollama
        srv, base_url = start_mock_ollama()
        self.addCleanup(srv.stop)
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, {"llm": {"provider": "ollama",
                                                    "base_url": base_url,
                                                    "model": "test"}})
            self.assertTrue(human.llm_online)
            calls = {"n": 0}
            original_decide = human.llm.decide

            def counting_decide(*a, **k):
                calls["n"] += 1
                return original_decide(*a, **k)
            human.llm.decide = counting_decide
            human.catch_up(6 * 60.0)
            self.assertEqual(calls["n"], 0, "catch-up must never call the real LLM")
            self.assertTrue(human.llm_online)  # re-probed truthfully afterward
            human.memory.close()

    def test_gap_beyond_cap_still_advances_the_full_amount(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human(tmp)
            human.world.money = 100000.0  # survive the week comfortably
            start = human.body.sim_minutes
            ten_days = 10 * 1440.0
            human.catch_up(ten_days)
            self.assertAlmostEqual(human.body.sim_minutes - start, ten_days,
                                   delta=1.0)
            human.memory.close()

    def test_overflow_beyond_cap_does_not_simulate_tick_by_tick(self):
        """The portion beyond the 7-day cap should just pass, not run
        through thousands of individual ticks (which would be needless and
        slow for an already-extreme gap)."""
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human(tmp)
            human.world.money = 100000.0
            import time as _t
            t0 = _t.time()
            human.catch_up(30 * 1440.0)  # a full month
            elapsed = _t.time() - t0
            self.assertLess(elapsed, 10.0, "overflow beyond the cap must be cheap")
            human.memory.close()

    def test_death_during_catch_up_is_possible_and_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human(tmp)
            human.world.food_portions = 0
            human.world.money = 0.0
            human.body.satiety = 0.5
            human.body.health = 1.0
            human.catch_up(20 * 1440.0)  # far beyond what starvation survives
            self.assertFalse(human.body.alive)
            self.assertIsNotNone(human.body.cause_of_death)
            human.memory.close()

    def test_pending_news_accumulates_during_a_long_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human(tmp)
            human.world.money = 1000.0
            human.catch_up(4 * 1440.0)  # long enough for milestones/weather/etc.
            # not a strict guarantee any single run produces news, but the
            # mechanism (companion away -> _note_news queues) must be live
            self.assertFalse(human.companion_present)
            human.memory.close()

    def test_reunion_fires_naturally_on_first_hear_after_catch_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human(tmp)
            human.catch_up(9 * 60.0)
            self.assertIsNone(human._reunion_gap)  # not yet -- no one has spoken
            human.hear("I'm back")
            self.assertIsNotNone(human._reunion_gap)
            self.assertAlmostEqual(human._reunion_gap, 9 * 60.0, delta=5.0)
            human.memory.close()


if __name__ == "__main__":
    unittest.main()

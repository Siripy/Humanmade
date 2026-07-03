"""Tests for identity drift: self-view, trait drift, self-esteem, and the
biography that stitches a life into a first-person story."""

from __future__ import annotations

import os
import tempfile
import time
import unittest

from humanmade.memory import MemoryStream
from humanmade.personality import (DRIFT_ORIGIN_CAP, DRIFT_RATE_PER_DAY,
                                   derive_disposition, drift)
from tests.mockllm import start_mock_ollama
from tests.test_agent import THINK_TIMEOUT, drive_one_thought, make_human

EMPTY_STATS = {"valence_sum": 0.0, "valence_n": 0, "conversations": 0,
              "pride": 0, "shame": 0}


class TestDriftFunction(unittest.TestCase):
    def _dims(self):
        return derive_disposition("gentle")["dimensions"]

    def test_no_signal_means_essentially_no_drift(self):
        dims = self._dims()
        new = drift(dims, dims, EMPTY_STATS)
        # extraversion still moves (silence itself is a signal), but
        # neuroticism/conscientiousness/openness shouldn't with zero data
        self.assertAlmostEqual(new["neuroticism"], dims["neuroticism"], places=6)
        self.assertAlmostEqual(new["conscientiousness"], dims["conscientiousness"],
                               places=6)
        self.assertEqual(new["openness"], dims["openness"])

    def test_bad_day_raises_neuroticism(self):
        dims = self._dims()
        bad_day = {**EMPTY_STATS, "valence_sum": -50.0, "valence_n": 100.0}
        new = drift(dims, dims, bad_day)
        self.assertGreater(new["neuroticism"], dims["neuroticism"])

    def test_good_day_lowers_neuroticism(self):
        dims = self._dims()
        good_day = {**EMPTY_STATS, "valence_sum": 50.0, "valence_n": 100.0}
        new = drift(dims, dims, good_day)
        self.assertLess(new["neuroticism"], dims["neuroticism"])

    def test_conversation_raises_extraversion(self):
        dims = self._dims()
        social = {**EMPTY_STATS, "conversations": 5}
        quiet = {**EMPTY_STATS, "conversations": 0}
        social_new = drift(dims, dims, social)
        quiet_new = drift(dims, dims, quiet)
        self.assertGreater(social_new["extraversion"], quiet_new["extraversion"])

    def test_pride_over_shame_raises_conscientiousness(self):
        dims = self._dims()
        proud_day = {**EMPTY_STATS, "pride": 4, "shame": 0}
        ashamed_day = {**EMPTY_STATS, "pride": 0, "shame": 4}
        proud_new = drift(dims, dims, proud_day)
        ashamed_new = drift(dims, dims, ashamed_day)
        self.assertGreater(proud_new["conscientiousness"],
                           ashamed_new["conscientiousness"])

    def test_single_day_never_exceeds_the_rate(self):
        dims = self._dims()
        extreme_day = {"valence_sum": -1000.0, "valence_n": 1.0,
                       "conversations": 999, "pride": 999, "shame": 0}
        new = drift(dims, dims, extreme_day)
        for key in dims:
            self.assertLessEqual(abs(new[key] - dims[key]),
                                 DRIFT_RATE_PER_DAY + 1e-9, key)

    def test_drift_never_exceeds_origin_cap_over_many_days(self):
        dims = self._dims()
        origin = dict(dims)
        bad_day = {**EMPTY_STATS, "valence_sum": -100.0, "valence_n": 100.0,
                  "conversations": 0, "pride": 0, "shame": 10}
        for _ in range(2000):  # far more days than needed to hit the cap
            dims = drift(dims, origin, bad_day)
        self.assertLessEqual(dims["neuroticism"],
                             origin["neuroticism"] + DRIFT_ORIGIN_CAP + 1e-9)
        self.assertLessEqual(dims["conscientiousness"],
                             origin["conscientiousness"] + DRIFT_ORIGIN_CAP + 1e-9)

    def test_dims_stay_in_valid_range(self):
        dims = self._dims()
        origin = dict(dims)
        extreme = {**EMPTY_STATS, "valence_sum": -100.0, "valence_n": 100.0}
        for _ in range(5000):
            dims = drift(dims, origin, extreme)
            for v in dims.values():
                self.assertGreaterEqual(v, 0.05)
                self.assertLessEqual(v, 0.95)

    def test_drift_does_not_mutate_input(self):
        dims = self._dims()
        original = dict(dims)
        drift(dims, dims, {**EMPTY_STATS, "valence_sum": -50.0, "valence_n": 100.0})
        self.assertEqual(dims, original)


class TestMemoryHighlights(unittest.TestCase):
    def test_highlights_are_most_important_first_then_chronological(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = MemoryStream(os.path.join(tmp, "m.sqlite3"))
            mem.add("event", "trivial early thing", 10, importance=2)
            mem.add("event", "important early thing", 20, importance=9)
            mem.add("event", "important later thing", 30, importance=9)
            mem.add("event", "trivial later thing", 40, importance=2)
            top = mem.highlights(2)
            self.assertEqual([m.text for m in top],
                             ["important early thing", "important later thing"])
            mem.close()

    def test_highlights_empty_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = MemoryStream(os.path.join(tmp, "m.sqlite3"))
            self.assertEqual(mem.highlights(10), [])
            mem.close()


class TestDayStatsAccumulation(unittest.TestCase):
    def _human(self, tmp):
        human, _, _ = make_human(tmp)
        human.llm_online = False
        return human

    def test_tick_accumulates_valence(self):
        with tempfile.TemporaryDirectory() as tmp:
            human = self._human(tmp)
            human.tick(30.0)
            self.assertGreater(human._day_stats["valence_n"], 0)
            human.memory.close()

    def test_hear_increments_conversations(self):
        with tempfile.TemporaryDirectory() as tmp:
            human = self._human(tmp)
            human.hear("hello")
            human.hear("how are you")
            self.assertEqual(human._day_stats["conversations"], 2)
            human.memory.close()

    def test_feel_increments_pride_and_shame(self):
        with tempfile.TemporaryDirectory() as tmp:
            human = self._human(tmp)
            human._feel("pride", 0.8)
            human._feel("shame", 0.5)
            human._feel("joy", 0.5)  # unrelated emotion, no counter bump
            self.assertEqual(human._day_stats["pride"], 1)
            self.assertEqual(human._day_stats["shame"], 1)
            human.memory.close()


class TestSelfEsteem(unittest.TestCase):
    def test_pride_raises_self_esteem(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            before = human.persona["self_esteem"]
            human._feel("pride", 1.0)
            self.assertGreater(human.persona["self_esteem"], before)
            human.memory.close()

    def test_shame_lowers_self_esteem(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            before = human.persona["self_esteem"]
            human._feel("shame", 1.0)
            self.assertLess(human.persona["self_esteem"], before)
            human.memory.close()

    def test_self_esteem_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            for _ in range(500):
                human._feel("pride", 1.0)
            self.assertLessEqual(human.persona["self_esteem"], 0.95)
            for _ in range(500):
                human._feel("shame", 1.0)
            self.assertGreaterEqual(human.persona["self_esteem"], 0.05)
            human.memory.close()

    def test_low_self_esteem_masks_at_higher_closeness(self):
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            low, _, _ = make_human(tmp1)
            high, _, _ = make_human(tmp2)
            low.persona["self_esteem"] = 0.1
            high.persona["self_esteem"] = 0.9
            low.bond["closeness"] = 40.0   # above the baseline 35 threshold
            high.bond["closeness"] = 40.0
            low.body.valence_smoothed = -0.5
            high.body.valence_smoothed = -0.5
            low_masks = any("say you're fine" in line
                            for line in low._social_context())
            high_masks = any("say you're fine" in line
                             for line in high._social_context())
            self.assertTrue(low_masks, "low self-esteem should still mask here")
            self.assertFalse(high_masks, "high self-esteem should open up here")
            low.memory.close(); high.memory.close()

    def test_self_esteem_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.persona["self_esteem"] = 0.73
            human.save()
            human.memory.close()
            reborn, _, _ = make_human(tmp)
            self.assertAlmostEqual(reborn.persona["self_esteem"], 0.73)
            reborn.memory.close()


class TestDailyDriftIntegration(unittest.TestCase):
    def test_apply_daily_drift_changes_dimensions_and_resets_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            human._day_stats = {"valence_sum": -100.0, "valence_n": 100.0,
                                "conversations": 0, "pride": 0, "shame": 5}
            before = dict(human.persona["disposition"]["dimensions"])
            human._apply_daily_drift()
            after = human.persona["disposition"]["dimensions"]
            self.assertNotEqual(before, after)
            self.assertEqual(human._day_stats["valence_n"], 0)
            human.memory.close()

    def test_drift_actually_changes_body_knobs(self):
        """The point isn't just that the number moves — it must change how
        the human actually behaves."""
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            for _ in range(60):  # enough days to produce a measurable shift
                human._day_stats = {"valence_sum": -100.0, "valence_n": 100.0,
                                    "conversations": 0, "pride": 0, "shame": 3}
                human._apply_daily_drift()
            self.assertNotEqual(human.body.neg_reactivity,
                                body_knobs_neg_reactivity_for(0.5))
            human.memory.close()

    def test_daily_checks_triggers_drift_on_day_turnover(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            human._day_stats["valence_sum"] = -50.0
            human._day_stats["valence_n"] = 50.0
            before = dict(human.persona["disposition"]["dimensions"])
            for _ in range(288 + 5):   # cross a day boundary
                human.tick(5.0)
                if not human.body.alive:
                    break
            after = human.persona["disposition"]["dimensions"]
            self.assertNotEqual(before, after)
            human.memory.close()

    def test_new_life_resets_identity_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.persona["self_esteem"] = 0.9
            human.persona["self_view"] = "I have changed."
            human._biography = "A whole life story."
            human._day_stats["pride"] = 7
            human.new_life()
            self.assertEqual(human.persona["self_esteem"], 0.5)
            self.assertIsNone(human.persona["self_view"])
            self.assertIsNone(human._biography)
            self.assertEqual(human._day_stats["pride"], 0)
            human.memory.close()


def body_knobs_neg_reactivity_for(neuroticism: float) -> float:
    return 0.6 + 1.1 * neuroticism


class TestPersonaBackfill(unittest.TestCase):
    def test_old_save_without_identity_fields_is_backfilled(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            del human.persona["origin_dimensions"]
            del human.persona["self_esteem"]
            del human.persona["self_view"]
            human._apply_disposition()
            self.assertIn("origin_dimensions", human.persona)
            self.assertEqual(human.persona["self_esteem"], 0.5)
            self.assertIsNone(human.persona["self_view"])
            human.memory.close()

    def test_backfilled_origin_matches_current_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            del human.persona["origin_dimensions"]
            human._apply_disposition()
            self.assertEqual(human.persona["origin_dimensions"],
                             human.persona["disposition"]["dimensions"])
            human.memory.close()


class TestSelfViewReflection(unittest.TestCase):
    def test_reflection_stores_self_view(self):
        srv, base_url = start_mock_ollama(
            insights=["I keep coming back to the fridge."],
            self_view="I am someone who worries more than I used to.")
        self.addCleanup(srv.stop)
        cfg = {"llm": {"provider": "ollama", "base_url": base_url,
                       "model": "test", "timeout_seconds": 5}}
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = make_human(tmp, cfg)
            human.memory._importance_since_reflection = 999
            human.tick(0.001)
            deadline = time.time() + THINK_TIMEOUT
            while human._reflecting and time.time() < deadline:
                time.sleep(0.01)
            self.assertEqual(human.persona["self_view"],
                             "I am someone who worries more than I used to.")
            self.assertTrue(any("sense of themself shifts" in e for e in events))
            self.assertIn("Who you are, as you see it", human._perception())
            human.memory.close()

    def test_no_self_view_change_is_quiet(self):
        srv, base_url = start_mock_ollama(insights=["a normal insight"])
        self.addCleanup(srv.stop)
        cfg = {"llm": {"provider": "ollama", "base_url": base_url,
                       "model": "test", "timeout_seconds": 5}}
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = make_human(tmp, cfg)
            human.memory._importance_since_reflection = 999
            human.tick(0.001)
            deadline = time.time() + THINK_TIMEOUT
            while human._reflecting and time.time() < deadline:
                time.sleep(0.01)
            self.assertIsNone(human.persona["self_view"])
            self.assertFalse(any("shifts" in e for e in events))
            human.memory.close()


class TestBiography(unittest.TestCase):
    def test_request_biography_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            self.assertEqual(human.request_biography(), "offline")
            human.memory.close()

    def test_request_biography_writes_then_caches(self):
        srv, base_url = start_mock_ollama(
            biography="It began quietly and grew louder.")
        self.addCleanup(srv.stop)
        cfg = {"llm": {"provider": "ollama", "base_url": base_url,
                       "model": "test", "timeout_seconds": 5}}
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, cfg)
            first = human.request_biography()
            self.assertEqual(first, "writing")
            deadline = time.time() + THINK_TIMEOUT
            while human._writing_biography and time.time() < deadline:
                time.sleep(0.02)
            self.assertEqual(human.biography_text(), "It began quietly and grew louder.")
            second = human.request_biography()
            self.assertEqual(second, "ready")  # cached, no second write
            human.memory.close()

    def test_biography_persists_across_restart(self):
        srv, base_url = start_mock_ollama(biography="A story worth keeping.")
        self.addCleanup(srv.stop)
        cfg = {"llm": {"provider": "ollama", "base_url": base_url,
                       "model": "test", "timeout_seconds": 5}}
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, cfg)
            human.request_biography()
            deadline = time.time() + THINK_TIMEOUT
            while human._writing_biography and time.time() < deadline:
                time.sleep(0.02)
            human.save()
            human.memory.close()
            reborn, _, _ = make_human(tmp, cfg)
            self.assertEqual(reborn.biography_text(), "A story worth keeping.")
            reborn.memory.close()

    def test_biography_includes_finished_work_titles(self):
        srv, base_url = start_mock_ollama(biography="...")
        self.addCleanup(srv.stop)
        cfg = {"llm": {"provider": "ollama", "base_url": base_url,
                       "model": "test", "timeout_seconds": 5}}
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, cfg)
            from humanmade.creation import Work
            work = Work(hobby=human.persona["hobby"], kind="novel",
                       title="The Long Way Home", synopsis="", status="finished",
                       started_sim=0.0, finished_sim=10.0, sale_price=40.0)
            human.works.save(work, "s1")
            captured = {}
            original_biography = human.llm.biography

            def spy(persona, origin_desc, current_desc, days, titles, text):
                captured["titles"] = titles
                return original_biography(persona, origin_desc, current_desc,
                                          days, titles, text)
            human.llm.biography = spy
            human.request_biography()
            deadline = time.time() + THINK_TIMEOUT
            while human._writing_biography and time.time() < deadline:
                time.sleep(0.02)
            self.assertIn("The Long Way Home", captured.get("titles", []))
            human.memory.close()

    def test_stale_biography_after_new_life_is_discarded(self):
        srv, base_url = start_mock_ollama(biography="Ghost story.", delay=0.3)
        self.addCleanup(srv.stop)
        cfg = {"llm": {"provider": "ollama", "base_url": base_url,
                       "model": "test", "timeout_seconds": 5}}
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, cfg)
            human.request_biography()  # in flight, delayed 0.3s
            human.new_life()
            time.sleep(1.0)
            self.assertIsNone(human.biography_text())
            human.memory.close()


if __name__ == "__main__":
    unittest.main()

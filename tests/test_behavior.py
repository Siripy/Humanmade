"""Tests for the research-grounded behavior layer: chronotype, emotional
inertia, dreams + overnight consolidation, planning, and reunion."""

import os
import tempfile
import time
import unittest

from humanmade.agent import CHRONOTYPES, Human, make_persona
from humanmade.body import Body
from humanmade.memory import MemoryStream
from tests.mockllm import start_mock_ollama
from tests.test_agent import drive_one_thought, make_human


class TestChronotype(unittest.TestCase):
    def test_persona_has_chronotype(self):
        self.assertIn(make_persona()["chronotype"], CHRONOTYPES)

    def test_lark_and_owl_shift_the_circadian_curve(self):
        lark, owl = Body(), Body()
        lark.chronotype_offset_hours = CHRONOTYPES["lark"][0]
        owl.chronotype_offset_hours = CHRONOTYPES["owl"][0]
        # at 23:00 the lark should feel far more sleep pressure than the owl
        lark.sim_minutes = owl.sim_minutes = 23 * 60
        self.assertGreater(lark.circadian_sleep_drive(),
                           owl.circadian_sleep_drive())


class TestEmotionalInertia(unittest.TestCase):
    def test_mood_lags_a_sudden_need_change(self):
        b = Body(energy=90, hydration=90, satiety=90, hygiene=90,
                 fun=90, social=90, bladder=5, bowel=5)
        good = b.mood()[0]
        self.assertGreater(good, 0.4)          # starts content
        # everything collapses at once
        b.energy = b.hydration = b.satiety = b.fun = b.social = 5
        b.bladder = b.bowel = 95
        b.tick(1.0)                            # only a moment passes
        self.assertGreater(b.mood()[0], -0.1,  # mood hasn't caught up yet
                           "mood should lag, not snap, after a sudden change")
        b.tick(400.0)                          # hours later it has sunk in
        self.assertLess(b.mood()[0], good)

    def test_nudge_is_immediate(self):
        b = Body()
        before = b.valence_smoothed
        b.nudge_mood(0.3)
        self.assertAlmostEqual(b.valence_smoothed, min(1.0, before + 0.3), places=5)

    def test_smoothed_valence_survives_persistence(self):
        b = Body(energy=5, hydration=5, satiety=5)
        b.tick(500)
        low = b.valence_smoothed
        b2 = Body.from_dict(b.to_dict())
        self.assertAlmostEqual(b2.valence_smoothed, low, places=5)
        self.assertTrue(b2._valence_ready)


class TestMemoryConsolidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mem = MemoryStream(os.path.join(self.tmp.name, "m.sqlite3"))

    def tearDown(self):
        self.mem.close()
        self.tmp.cleanup()

    def test_overnight_therapy_softens_charged_memories(self):
        self.mem.add("event", "I was so lonely and afraid today.", 100)  # charged
        self.mem.add("observation", "I read a book.", 100)               # neutral
        charged_before = next(m for m in self.mem.recent(2)
                              if "lonely" in m.text).importance
        softened = self.mem.soften_emotional_charge(before_sim_minutes=200)
        self.assertEqual(softened, 1)
        charged_after = next(m for m in self.mem.recent(2)
                             if "lonely" in m.text).importance
        self.assertLess(charged_after, charged_before)

    def test_only_pre_waking_memories_are_softened(self):
        self.mem.add("event", "an old fear from yesterday", 100)
        self.mem.add("event", "a fresh fear after waking", 300)
        self.mem.soften_emotional_charge(before_sim_minutes=200)
        old = next(m for m in self.mem.recent(2) if "old" in m.text)
        fresh = next(m for m in self.mem.recent(2) if "fresh" in m.text)
        self.assertLess(old.importance, fresh.importance)

    def test_emotional_material_prefers_charged_recent(self):
        self.mem.add("observation", "trivial thing", 100, importance=2)
        self.mem.add("event", "something that scared me", 110, importance=9)
        material = self.mem.emotional_material(since_sim_minutes=50, limit=1)
        self.assertIn("scared", material[0].text)


class TestDreamAndPlan(unittest.TestCase):
    def _human(self, tmp, **mock):
        srv, base_url = start_mock_ollama(**mock)
        self.addCleanup(srv.stop)
        cfg = {"llm": {"provider": "ollama", "base_url": base_url,
                       "model": "test", "timeout_seconds": 5}}
        return make_human(tmp, cfg)

    def _wait(self, pred, timeout=20.0):
        end = time.time() + timeout
        while not pred() and time.time() < end:
            time.sleep(0.01)

    def test_dreams_during_sleep(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human(
                tmp, decision={"thought": "sleepy", "action": "sleep",
                               "say": None, "importance": 3},
                insights=[])
            # give it a charged memory to dream about, then send it to sleep
            human.memory.add("event", "I felt so alone today.",
                             human.body.sim_minutes, importance=8)
            human.body.energy = 10.0
            drive_one_thought(human)
            self.assertTrue(human.body.asleep)
            human.tick(1.0)   # register the fall-asleep transition
            human.tick(65.0)  # cross the 60-min "deeper sleep" dream threshold
            self._wait(lambda: human.last_dream is not None)
            self.assertIsNotNone(human.last_dream)
            dreams = [m for m in human.memory.recent(50) if m.kind == "dream"]
            self.assertTrue(dreams)
            human.memory.close()

    def test_plan_forms_on_waking(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human(
                tmp, decision={"thought": "up", "action": "wake",
                               "say": None, "importance": 2})
            # pretend a full night has already been slept
            human.body.fall_asleep()
            human._was_asleep = True
            human._sleep_started_sim = human.body.sim_minutes - 400
            human.body.wake_up()
            human.tick(1.0)  # transition detected -> planning worker spawned
            self._wait(lambda: bool(human.today_plan))
            self.assertTrue(human.today_plan)
            self.assertEqual(human.plan_day_index, human.body.day)
            human.memory.close()


class TestReunion(unittest.TestCase):
    def test_away_then_return_triggers_reunion_greeting(self):
        srv, base_url = start_mock_ollama(
            decision={"thought": "they're back!", "action": "idle",
                      "say": "You're home! I missed you.", "importance": 8})
        self.addCleanup(srv.stop)
        cfg = {"llm": {"provider": "ollama", "base_url": base_url,
                       "model": "test", "timeout_seconds": 5}}
        with tempfile.TemporaryDirectory() as tmp:
            human, speech, _ = make_human(tmp, cfg)
            human.set_away()
            self.assertFalse(human.companion_present)
            # time passes while alone; something notable happens
            human.body.sim_minutes += 180
            human._note_news("the fridge ran empty")
            self.assertIn("the fridge ran empty", human.pending_news)
            # companion returns
            human.hear("hey, I'm back")
            self.assertTrue(human.companion_present)
            self.assertIsNotNone(human._reunion_gap)
            drive_one_thought(human)
            self.assertTrue(speech)
            self.assertIn("missed", speech[0])
            self.assertEqual(human.pending_news, [])  # news delivered
            human.memory.close()

    def test_speech_while_away_is_not_spoken(self):
        srv, base_url = start_mock_ollama(
            decision={"thought": "lonely", "action": "idle",
                      "say": "Are you there?", "importance": 5})
        self.addCleanup(srv.stop)
        cfg = {"llm": {"provider": "ollama", "base_url": base_url,
                       "model": "test", "timeout_seconds": 5}}
        with tempfile.TemporaryDirectory() as tmp:
            human, speech, _ = make_human(tmp, cfg)
            human.set_away()
            drive_one_thought(human)
            self.assertEqual(speech, [], "no one is home to hear it")
            # but the wish to speak is saved for later
            self.assertTrue(any("Are you there" in n for n in human.pending_news))
            human.memory.close()

    def test_reunion_is_a_mood_lift(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            human.set_away()
            human.body.sim_minutes += 600
            before = human.body.valence_smoothed
            human.hear("I'm home!")
            self.assertGreater(human.body.valence_smoothed, before)
            human.memory.close()


if __name__ == "__main__":
    unittest.main()

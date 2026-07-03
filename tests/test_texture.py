"""Tests: follow-ups on unanswered questions, hedonic adaptation, weather
preferences, memory consolidation, and birthdays."""

import os
import tempfile
import unittest

from humanmade.memory import MemoryStream
from tests.mockllm import start_mock_ollama
from tests.test_agent import drive_one_thought, make_human


def _human(tmp):
    human, speech, events = make_human(tmp)
    human.llm_online = False
    return human, speech, events


class TestOpenQuestions(unittest.TestCase):
    def test_question_tracked_and_followed_up(self):
        srv, base_url = start_mock_ollama(
            decision={"thought": "curious", "action": "idle",
                      "say": "What's it like where you live?", "importance": 5})
        self.addCleanup(srv.stop)
        cfg = {"llm": {"provider": "ollama", "base_url": base_url,
                       "model": "test", "timeout_seconds": 5}}
        with tempfile.TemporaryDirectory() as tmp:
            human, speech, _ = make_human(tmp, cfg)
            human.hear("hello!")
            drive_one_thought(human)
            self.assertTrue(speech)
            self.assertIsNotNone(human.pending_question)
            # nine silent hours later, the perception nudges a follow-up
            human.body.sim_minutes += 9 * 60
            context = " ".join(human._social_context())
            self.assertIn("never answered", context)
            self.assertEqual(human.emotion, "hurt")
            human.memory.close()

    def test_any_reply_clears_the_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human.pending_question = {"text": "who are you?", "sim": 0.0,
                                      "hurt_felt": False}
            human.hear("I'm Siri!")
            self.assertIsNone(human.pending_question)
            human.memory.close()

    def test_pending_question_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human.pending_question = {"text": "still there?", "sim": 5.0,
                                      "hurt_felt": False}
            human.save()
            human.memory.close()
            reborn, _, _ = make_human(tmp)
            self.assertEqual(reborn.pending_question["text"], "still there?")
            reborn.memory.close()


class TestHedonicAdaptation(unittest.TestCase):
    def test_repeats_restore_less_fun(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human.body.fun = 10.0
            human._perform("relax")
            first_gain = human.body.fun - 10.0
            for _ in range(3):
                human.body.fun = 10.0
                human._perform("relax")
            fourth_gain = human.body.fun - 10.0
            self.assertLess(fourth_gain, first_gain)
            human.memory.close()

    def test_variety_resets_the_shine(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            for _ in range(4):
                human._perform("relax")
            for _ in range(4):
                human._perform("exercise")   # pushes relax out of the window
            self.assertNotIn("relax", human._recent_actions)
            human.body.fun = 10.0
            human._perform("relax")
            self.assertGreater(human.body.fun - 10.0, 20.0)  # full gain again
            human.memory.close()


class TestPreferences(unittest.TestCase):
    def test_persona_has_consistent_weather_taste(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            self.assertIn(human.persona["loves_weather"],
                          ("sunny", "cloudy", "rainy", "stormy"))
            self.assertNotEqual(human.persona["loves_weather"],
                                human.persona["hates_weather"])
            # favorite sky lifts ambient valence beyond the base weather effect
            human.world.weather = human.persona["loves_weather"]
            human.tick(1.0)
            loved_ambient = human.body.ambient_valence
            human.world.weather = human.persona["hates_weather"]
            human.tick(1.0)
            self.assertGreater(loved_ambient, human.body.ambient_valence)
            # and it shows up in what the mind perceives
            human.world.weather = human.persona["loves_weather"]
            self.assertIn("favorite kind of sky", human._perception())
            human.memory.close()


class TestConsolidation(unittest.TestCase):
    def test_old_trivia_blurs_important_stays(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = MemoryStream(os.path.join(tmp, "m.sqlite3"))
            for i in range(50):
                mem.add("observation", f"small moment {i}", 100.0, importance=2)
            mem.add("event", "the day everything changed", 100.0, importance=9)
            summary = mem.consolidate(before_sim_minutes=10_000)
            self.assertIsNotNone(summary)
            texts = [m.text for m in mem.recent(100)]
            self.assertFalse(any("small moment" in t for t in texts))
            self.assertTrue(any("everything changed" in t for t in texts))
            self.assertTrue(any("blurs together" in t for t in texts))
            mem.close()

    def test_too_few_memories_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = MemoryStream(os.path.join(tmp, "m.sqlite3"))
            for i in range(5):
                mem.add("observation", f"moment {i}", 100.0, importance=2)
            self.assertIsNone(mem.consolidate(before_sim_minutes=10_000))
            self.assertEqual(mem.count(), 5)
            mem.close()


class TestBirthday(unittest.TestCase):
    def test_birthday_ages_and_fires_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = _human(tmp)
            human.persona["birthday_day"] = 2   # the day after tomorrow
            age = human.persona["age"]
            for _ in range(4 * 288):            # four days
                human.tick(5.0)
                if not human.body.alive:
                    break
            self.assertTrue(human.body.alive)
            self.assertEqual(human.persona["age"], age + 1)
            self.assertEqual(
                sum(1 for e in events if "birthday" in e), 1)
            human.memory.close()


if __name__ == "__main__":
    unittest.main()

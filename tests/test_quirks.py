"""Tests: discrete emotions (appraisal) and trait-consistent imperfections."""

import tempfile
import unittest
from unittest.mock import patch

from tests.test_agent import make_human


def _human(tmp):
    human, speech, events = make_human(tmp)
    human.llm_online = False
    return human, speech, events


class TestEmotions(unittest.TestCase):
    def test_gratitude_after_medicine(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human.body.sickness = 60.0
            human.medicine()
            self.assertEqual(human.emotion, "gratitude")
            self.assertGreater(human.emotion_intensity, 0.5)
            human.memory.close()

    def test_shame_after_accident(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human.body.bladder = 99.5
            human.tick(30.0)
            self.assertEqual(human.emotion, "shame")
            human.memory.close()

    def test_pride_at_milestone(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human.persona["hobby_progress"] = 24.5
            human._perform("work")
            self.assertEqual(human.emotion, "pride")
            human.memory.close()

    def test_joy_at_reunion(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human.set_away()
            human.body.sim_minutes += 300
            human.hear("I'm back!")
            self.assertEqual(human.emotion, "joy")
            human.memory.close()

    def test_emotion_decays(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human._feel("frustration", 0.8)
            human.tick(600.0)   # ten hours
            self.assertIsNone(human.emotion)
            human.memory.close()

    def test_emotion_shows_in_perception(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human._feel("worry", 0.8)
            self.assertIn("feeling of worry", human._perception())
            human.memory.close()

    def test_stronger_emotion_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human._feel("worry", 0.4)
            human._feel("pride", 0.9)
            self.assertEqual(human.emotion, "pride")
            human._feel("hurt", 0.2)     # weaker: mood nudge only
            self.assertEqual(human.emotion, "pride")
            human.memory.close()


class TestImperfections(unittest.TestCase):
    def test_bedtime_procrastination_once_per_night(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = _human(tmp)
            human.plan_adherence = 0.4          # a low-C person
            human.body.energy = 30.0
            human.body.fun = 20.0
            decision = {"thought": "", "action": "sleep", "say": None,
                        "note_about_companion": None, "importance": 2}
            with patch("humanmade.agent.random.random", return_value=0.0):
                first = human._temptation(dict(decision))
                second = human._temptation(dict(decision))
            self.assertEqual(first["action"], "relax")   # one more chapter
            self.assertEqual(second["action"], "sleep")  # only once a night
            self.assertTrue(any("one more chapter" in e for e in events))
            human.memory.close()

    def test_conscientious_person_just_sleeps(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human.plan_adherence = 1.0
            human.body.energy = 30.0
            human.body.fun = 20.0
            decision = {"thought": "", "action": "sleep", "say": None,
                        "note_about_companion": None, "importance": 2}
            with patch("humanmade.agent.random.random", return_value=0.0):
                self.assertEqual(human._temptation(decision)["action"], "sleep")
            human.memory.close()

    def test_work_procrastination(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human.plan_adherence = 0.4
            decision = {"thought": "", "action": "work", "say": None,
                        "note_about_companion": None, "importance": 2}
            with patch("humanmade.agent.random.random", return_value=0.0):
                self.assertEqual(human._temptation(decision)["action"], "relax")
            human.memory.close()

    def test_forced_decisions_are_never_tempted(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human.plan_adherence = 0.0
            human.body.hydration = 5.0          # dying: override fires
            decision = {"thought": "", "action": "work", "say": None,
                        "note_about_companion": None, "importance": 2}
            with patch("humanmade.agent.random.random", return_value=0.0):
                human._apply_decision(decision, "perception", False)
            self.assertEqual(human.current_action, "drink")
            human.memory.close()

    def test_comfort_eating_when_low(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = _human(tmp)
            human.body.satiety = 65.0
            human.body.valence_smoothed = -0.5
            human._perform("eat")
            self.assertTrue(any("comfort food" in e for e in events))
            human.memory.close()

    def test_comfort_food_line_in_perception(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human.body.satiety = 60.0
            human.body.valence_smoothed = -0.5
            self.assertIn("Comfort food", human._perception())
            human.memory.close()

    def test_masking_when_low_and_distant(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human.bond["closeness"] = 10.0
            human.body.valence_smoothed = -0.5
            self.assertIn("say you're fine", " ".join(human._social_context()))
            human.bond["closeness"] = 90.0
            self.assertNotIn("say you're fine", " ".join(human._social_context()))
            human.memory.close()


if __name__ == "__main__":
    unittest.main()

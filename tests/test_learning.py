"""Tests for the learning side: practice curves, skill effects, and lessons
(advice it can actually take)."""

import tempfile
import unittest
from unittest.mock import patch

from humanmade.brain import _parse_decision
from tests.mockllm import start_mock_ollama
from tests.test_agent import drive_one_thought, make_human


def _human(tmp):
    human, speech, events = make_human(tmp)
    human.llm_online = False
    return human, speech, events


class TestPracticeCurve(unittest.TestCase):
    def test_practice_raises_skill_with_diminishing_returns(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            level0 = human.skill("craft")
            human._practice("craft", 0.9)
            early_gain = human.skill("craft") - level0
            human.persona["skills"]["craft"] = 90.0
            human._practice("craft", 0.9)
            late_gain = human.skill("craft") - 90.0
            self.assertGreater(early_gain, late_gain * 3)
            human.memory.close()

    def test_skill_milestone_is_felt_and_remembered(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = _human(tmp)
            human.persona["skills"]["cooking"] = 24.8
            human._practice("cooking", 0.9)
            self.assertEqual(human.emotion, "pride")
            self.assertTrue(any("real progress" in e for e in events))
            self.assertTrue(any("getting better at cooking" in m.text
                                for m in human.memory.recent(5)))
            human.memory.close()

    def test_unpracticed_skills_rust(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human.persona["skills"]["fitness"] = 50.0
            for _ in range(3 * 288):    # three days of not exercising
                human.tick(5.0)
            self.assertLess(human.skill("fitness"), 50.0)
            human.memory.close()

    def test_skills_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human._perform("work")
            human.save()
            craft = human.skill("craft")
            self.assertGreater(craft, 5.0)
            human.memory.close()
            reborn, _, _ = make_human(tmp)
            self.assertAlmostEqual(reborn.skill("craft"), craft, places=5)
            reborn.memory.close()


class TestSkillEffects(unittest.TestCase):
    def test_better_cook_gets_more_from_a_meal(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human.body.satiety = 30.0
            human.persona["skills"]["cooking"] = 5.0
            human._perform("eat")
            novice_meal = human.body.satiety - 30.0
            human.body.satiety = 30.0
            human.persona["skills"]["cooking"] = 90.0
            human._perform("eat")
            chef_meal = human.body.satiety - 30.0
            self.assertGreater(chef_meal, novice_meal * 1.2)
            human.memory.close()

    def test_mastery_pays_better(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human.persona["skills"]["craft"] = 5.0
            m0 = human.world.money
            human._perform("work")
            novice_wage = human.world.money - m0
            human.persona["skills"]["craft"] = 90.0
            m1 = human.world.money
            human._perform("work")
            master_wage = human.world.money - m1
            self.assertGreater(master_wage, novice_wage * 1.5)
            human.memory.close()

    def test_fitness_resists_the_chill(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human.world.weather = "stormy"
            human.persona["skills"]["fitness"] = 5.0
            human.tick(1.0)
            unfit_chill = human.body.chill
            human.persona["skills"]["fitness"] = 95.0
            human.tick(1.0)
            self.assertLess(human.body.chill, unfit_chill)
            human.memory.close()

    def test_skilled_hands_finish_projects_faster(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            with patch("humanmade.agent.random.uniform", return_value=1.5):
                human.persona["skills"]["craft"] = 5.0
                human.persona["hobby_progress"] = 0.0
                human._perform("work")
                novice_progress = float(human.persona["hobby_progress"])
                human.persona["skills"]["craft"] = 90.0
                human.persona["hobby_progress"] = 0.0
                human._perform("work")
                master_progress = float(human.persona["hobby_progress"])
            self.assertGreater(master_progress, novice_progress * 1.5)
            human.memory.close()


class TestLessons(unittest.TestCase):
    def test_lesson_parsed(self):
        d = _parse_decision('{"action": "idle", '
                            '"lesson_learned": "drink water before bed"}')
        self.assertEqual(d["lesson_learned"], "drink water before bed")
        d = _parse_decision('{"action": "idle", "lesson_learned": null}')
        self.assertIsNone(d["lesson_learned"])

    def test_advice_becomes_a_rule_it_lives_by(self):
        srv, base_url = start_mock_ollama(
            decision={"thought": "that's good advice", "action": "idle",
                      "say": "You're right, I'll try that.",
                      "lesson_learned": "Shower before bed to sleep better.",
                      "importance": 6})
        self.addCleanup(srv.stop)
        cfg = {"llm": {"provider": "ollama", "base_url": base_url,
                       "model": "test", "timeout_seconds": 5}}
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = make_human(tmp, cfg)
            human.hear("try showering before bed — you'll sleep better")
            drive_one_thought(human)
            drive_one_thought(human)    # repeated lesson -> deduped
            lessons = human.memory.recent(10, kinds=("lesson",))
            self.assertEqual(len(lessons), 1)
            self.assertIn("Shower before bed", lessons[0].text)
            self.assertTrue(any("takes something to heart" in e for e in events))
            # and from now on the rule is in front of the mind
            self.assertIn("Rules you live by", human._perception())
            human.memory.close()

    def test_lessons_persist_across_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = _human(tmp)
            human.memory.add("lesson", "never skip breakfast", 10.0)
            human.save()
            human.memory.close()
            reborn, _, _ = make_human(tmp)
            self.assertIn("Rules you live by", reborn._perception())
            reborn.memory.close()


if __name__ == "__main__":
    unittest.main()

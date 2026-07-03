"""Tests: the relationship develops — trust, closeness, a model of you,
anniversaries — instead of being a constant."""

import tempfile
import unittest

from humanmade.brain import _parse_decision
from tests.mockllm import start_mock_ollama
from tests.test_agent import drive_one_thought, make_human


class TestBondDynamics(unittest.TestCase):
    def test_care_raises_trust(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            t0 = human.bond["trust"]
            human.restock()
            self.assertGreater(human.bond["trust"], t0)
            human.body.sickness = 60.0
            t1 = human.bond["trust"]
            human.medicine()
            self.assertGreater(human.bond["trust"], t1)
            human.memory.close()

    def test_conversation_builds_closeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            c0 = human.bond["closeness"]
            for i in range(5):
                human.hear(f"message {i}")
            self.assertGreater(human.bond["closeness"], c0)
            human.memory.close()

    def test_long_abandonment_erodes_trust(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            human.set_away()
            human.body.sim_minutes += 3 * 1440   # gone three days
            t0 = human.bond["trust"]
            human.hear("sorry, I'm back")
            self.assertLess(human.bond["trust"], t0)
            human.memory.close()

    def test_bond_persists_across_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            human.restock()
            human.save()
            trust = human.bond["trust"]
            human.memory.close()
            reborn, _, _ = make_human(tmp)
            self.assertEqual(reborn.bond["trust"], trust)
            reborn.memory.close()

    def test_bond_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.bond["closeness"] = 10
            self.assertEqual(human.bond_level(), "strangers, really")
            human.bond["closeness"] = 80
            self.assertEqual(human.bond_level(), "genuinely close")
            human.memory.close()

    def test_new_life_resets_the_relationship(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            human.bond["closeness"] = 90.0
            human.body.hydration = 0.0
            human.body.satiety = 0.0
            human.body.health = 0.5
            human.world.food_portions = 0
            human.world.money = 0
            for _ in range(400):
                human.tick(5.0)
                if not human.body.alive:
                    break
            human.new_life()
            self.assertEqual(human.bond["closeness"], 20.0)
            human.memory.close()


class TestAnniversaries(unittest.TestCase):
    def test_seven_day_anniversary_fires_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            human.set_away()   # queue news while alone
            for _ in range(8 * 288):    # 8 days
                human.tick(5.0)
                if not human.body.alive:
                    break
            self.assertTrue(human.body.alive)
            anniversaries = [m for m in human.memory.recent(3000)
                             if "known each other" in m.text]
            self.assertEqual(len(anniversaries), 1)
            self.assertIn("7 days", anniversaries[0].text)
            human.memory.close()


class TestCompanionModel(unittest.TestCase):
    def test_note_parsed_from_decision(self):
        d = _parse_decision('{"action": "idle", '
                            '"note_about_companion": "their name is Siri"}')
        self.assertEqual(d["note_about_companion"], "their name is Siri")
        d = _parse_decision('{"action": "idle", "note_about_companion": null}')
        self.assertIsNone(d["note_about_companion"])

    def test_note_stored_once_and_fed_back(self):
        srv, base_url = start_mock_ollama(
            decision={"thought": "learning them", "action": "idle", "say": None,
                      "note_about_companion": "Their name is Siri.",
                      "importance": 5})
        self.addCleanup(srv.stop)
        cfg = {"llm": {"provider": "ollama", "base_url": base_url,
                       "model": "test", "timeout_seconds": 5}}
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, cfg)
            human.hear("hi, I'm Siri")
            drive_one_thought(human)
            drive_one_thought(human)   # same note again -> deduped
            notes = human.memory.recent(10, kinds=("companion",))
            self.assertEqual(len(notes), 1)
            self.assertIn("Siri", notes[0].text)
            # and the perception now carries what it knows about you
            self.assertIn("What you know about them", human._perception())
            human.memory.close()

    def test_disclosure_guidance_tracks_closeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.bond["closeness"] = 5.0
            self.assertIn("guard", " ".join(human._social_context()))
            human.bond["closeness"] = 90.0
            self.assertIn("really on your mind", " ".join(human._social_context()))
            human.memory.close()


if __name__ == "__main__":
    unittest.main()

"""Tests for the richer-life layer: weather, sickness & medicine, hobby
milestones, and the evening journal."""

import tempfile
import time
import unittest
from unittest.mock import patch

from humanmade.body import Body
from humanmade.brain import ReflexBrain
from humanmade.world import MEDICINE_PRICE, WEATHERS, World
from tests.mockllm import start_mock_ollama
from tests.test_agent import drive_one_thought, make_human


class TestWeather(unittest.TestCase):
    def test_advance_changes_weather_eventually(self):
        w = World()
        seen = set()
        for _ in range(60):          # ~30 sim-days of weather
            w.advance(720)
            seen.add(w.weather)
            self.assertIn(w.weather, WEATHERS)
        self.assertGreater(len(seen), 1, "weather should vary over a month")

    def test_valence_and_chill_mappings(self):
        w = World()
        w.weather = "sunny"
        self.assertGreater(w.weather_valence(), 0)
        self.assertEqual(w.chill_factor(), 1.0)
        w.weather = "stormy"
        self.assertLess(w.weather_valence(), 0)
        self.assertGreater(w.chill_factor(), 1.5)

    def test_weather_shifts_mood(self):
        sunny, stormy = Body(), Body()
        sunny.ambient_valence = 0.06
        stormy.ambient_valence = -0.12
        self.assertGreater(sunny._instant_valence(), stormy._instant_valence())

    def test_set_weather_changes_and_reports_event(self):
        w = World()
        w.weather = "sunny"
        events = w.set_weather("stormy")
        self.assertEqual(w.weather, "stormy")
        self.assertEqual(events, ["outside, the weather turns stormy"])

    def test_set_weather_no_op_when_unchanged(self):
        w = World()
        w.weather = "cloudy"
        self.assertEqual(w.set_weather("cloudy"), [])

    def test_set_weather_rejects_unknown_weather(self):
        w = World()
        w.weather = "sunny"
        self.assertEqual(w.set_weather("blizzard"), [])
        self.assertEqual(w.weather, "sunny")

    def test_weather_locked_blocks_random_cycling(self):
        w = World()
        w.weather_locked = True
        w.weather = "sunny"
        for _ in range(60):
            events = w.advance(720)
            self.assertEqual(events, [])
        self.assertEqual(w.weather, "sunny")


class TestSickness(unittest.TestCase):
    def test_risk_rises_with_neglect_and_chill(self):
        healthy = Body()
        neglected = Body(hygiene=10.0, energy=10.0)
        neglected.chill = 2.0
        self.assertGreater(neglected.infection_risk_per_minute(),
                           healthy.infection_risk_per_minute() * 5)

    def test_catches_illness_when_roll_hits(self):
        b = Body()
        with patch("humanmade.body.random.random", return_value=0.0):
            events = b.tick(5.0)
        self.assertGreater(b.sickness, 0)
        self.assertTrue(any("coming down" in e for e in events))

    def test_pushing_through_worsens_rest_heals(self):
        sick_worker = Body(sickness=30.0)
        sick_worker.tick(120, "work")
        self.assertGreater(sick_worker.sickness, 30.0)
        sick_sleeper = Body(sickness=30.0)
        sick_sleeper.fall_asleep()
        sick_sleeper.tick(120)
        self.assertLess(sick_sleeper.sickness, 30.0)

    def test_recovery_announces_itself(self):
        b = Body(sickness=3.0)
        b.fall_asleep()
        events = b.tick(120)
        self.assertEqual(b.sickness, 0.0)
        self.assertTrue(any("fever has broken" in e for e in events))

    def test_fever_shows_in_vitals(self):
        well, sick = Body(), Body(sickness=90.0)
        well.tick(1.0, "idle")
        sick.tick(1.0, "work")   # keep it sick while vitals update
        self.assertGreater(sick.body_temp_c, well.body_temp_c + 1.0)

    def test_untreated_severe_illness_can_kill(self):
        b = Body(sickness=95.0, health=2.0)
        b.tick(600, "work")      # refusing to rest
        self.assertFalse(b.alive)
        self.assertEqual(b.cause_of_death, "an untreated illness")

    def test_medicine_knocks_it_down(self):
        b = Body(sickness=80.0)
        b.take_medicine()
        self.assertEqual(b.sickness, 25.0)

    def test_reflex_rests_when_sick(self):
        body = Body(sickness=60.0)
        self.assertEqual(ReflexBrain().decide(body, World())["action"], "sleep")

    def test_sickness_persists(self):
        b = Body(sickness=42.0)
        self.assertEqual(Body.from_dict(b.to_dict()).sickness, 42.0)


class TestMedicineOrder(unittest.TestCase):
    def test_order_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            # not sick -> nothing to treat
            self.assertFalse(human.medicine()["ok"])
            # sick but broke -> declined, remembered
            human.body.sickness = 70.0
            human.world.money = 2.0
            r = human.medicine()
            self.assertFalse(r["ok"])
            self.assertEqual(r["reason"], "broke")
            # sick and funded -> treated, paid
            human.world.money = 20.0
            r = human.medicine()
            self.assertTrue(r["ok"])
            self.assertEqual(human.body.sickness, 15.0)
            self.assertAlmostEqual(human.world.money, 20.0 - MEDICINE_PRICE)
            self.assertTrue(any("medicine" in m.text.lower()
                                for m in human.memory.recent(5)))
            human.memory.close()


class TestRealWeatherOverride(unittest.TestCase):
    def test_set_real_weather_locks_and_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = make_human(tmp)
            human.llm_online = False
            human.world.weather = "sunny"
            human.set_real_weather("stormy")
            self.assertTrue(human.world.weather_locked)
            self.assertEqual(human.world.weather, "stormy")
            self.assertTrue(any("stormy" in e for e in events))
            human.memory.close()

    def test_locked_weather_survives_a_tick(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            human.set_real_weather("rainy")
            for _ in range(20):
                human.tick(60.0)
            self.assertEqual(human.world.weather, "rainy")
            human.memory.close()

    def test_set_real_weather_is_a_no_op_when_dead(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            human.body.alive = False
            human.world.weather = "sunny"
            human.set_real_weather("stormy")
            self.assertFalse(human.world.weather_locked)
            self.assertEqual(human.world.weather, "sunny")
            human.memory.close()

    def test_unknown_reading_leaves_weather_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            human.world.weather = "sunny"
            human.set_real_weather("blizzard")
            self.assertTrue(human.world.weather_locked)
            self.assertEqual(human.world.weather, "sunny")
            human.memory.close()


class TestHobbyProgress(unittest.TestCase):
    def test_work_advances_and_hits_milestones(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = make_human(tmp)
            human.llm_online = False
            for _ in range(100):             # plenty to cross 100%
                human._perform("work")
            milestones = [e for e in events if "taking shape" in e]
            finished = [e for e in events if "FINISHED" in e]
            self.assertGreaterEqual(len(milestones), 3)
            self.assertGreaterEqual(len(finished), 1)
            self.assertLess(float(human.persona["hobby_progress"]), 100.0)
            # milestones become memories worth telling the companion about
            self.assertTrue(any("taking shape" in m.text
                                for m in human.memory.recent(100)))
            human.memory.close()

    def test_progress_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            human._perform("work")
            human.save()
            progress = float(human.persona["hobby_progress"])
            self.assertGreater(progress, 0)
            human.memory.close()
            reborn, _, _ = make_human(tmp)
            self.assertAlmostEqual(float(reborn.persona["hobby_progress"]),
                                   progress, places=5)
            reborn.memory.close()


class TestJournal(unittest.TestCase):
    def test_diary_written_at_bedtime(self):
        srv, base_url = start_mock_ollama(
            decision={"thought": "tired", "action": "sleep",
                      "say": None, "importance": 3},
            journal="Long day. The rain made me slow, but I wrote a little.")
        self.addCleanup(srv.stop)
        cfg = {"llm": {"provider": "ollama", "base_url": base_url,
                       "model": "test", "timeout_seconds": 5}}
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, cfg)
            human.body.energy = 10.0
            drive_one_thought(human)          # mind chooses sleep
            human.tick(1.0)                   # transition -> journal worker
            deadline = time.time() + 20
            while not human.journal_entries(1) and time.time() < deadline:
                time.sleep(0.05)
            entries = human.journal_entries(3)
            self.assertTrue(entries, "a diary entry should exist after bedtime")
            self.assertIn("rain", entries[0].text)
            human.memory.close()

    def test_no_second_entry_within_twelve_hours(self):
        srv, base_url = start_mock_ollama(
            decision={"thought": "tired", "action": "sleep",
                      "say": None, "importance": 3})
        self.addCleanup(srv.stop)
        cfg = {"llm": {"provider": "ollama", "base_url": base_url,
                       "model": "test", "timeout_seconds": 5}}
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, cfg)
            human.body.energy = 10.0
            drive_one_thought(human)
            human.tick(1.0)
            deadline = time.time() + 20
            while not human.journal_entries(1) and time.time() < deadline:
                time.sleep(0.05)
            self.assertEqual(len(human.journal_entries(10)), 1)
            # a nap two sim-hours later must not spawn another entry
            human.body.wake_up()
            human.tick(1.0)
            human.body.sim_minutes += 120
            human.body.fall_asleep()
            human.tick(1.0)
            time.sleep(0.5)
            self.assertEqual(len(human.journal_entries(10)), 1)
            human.memory.close()


if __name__ == "__main__":
    unittest.main()

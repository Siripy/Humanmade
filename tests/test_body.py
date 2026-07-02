import unittest

from humanmade.body import Body


class TestPhysiology(unittest.TestCase):
    def test_awake_needs_drain(self):
        b = Body()
        b.tick(120, "idle")
        self.assertLess(b.hydration, 75)
        self.assertLess(b.satiety, 70)
        self.assertLess(b.energy, 80)
        self.assertGreater(b.bladder, 20)

    def test_sleep_restores_energy(self):
        b = Body(energy=10.0)
        b.fall_asleep()
        b.tick(480)  # a full night
        self.assertGreater(b.energy, 90)
        self.assertTrue(b.asleep)

    def test_actions_change_state(self):
        b = Body(hydration=40.0, satiety=30.0, bladder=90.0, hygiene=10.0)
        b.drink()
        self.assertGreater(b.hydration, 40)
        b.eat()
        self.assertGreater(b.satiety, 30)
        b.use_toilet()
        self.assertEqual(b.bladder, 0.0)
        b.shower()
        self.assertEqual(b.hygiene, 100.0)

    def test_bladder_overflow_is_an_accident_not_death(self):
        b = Body(bladder=99.0)
        events = b.tick(60)
        self.assertEqual(b.accidents, 1)
        self.assertTrue(b.alive)
        self.assertTrue(any("accident" in e or "wet the bed" in e for e in events))
        self.assertLess(b.bladder, 50)  # relieved, involuntarily

    def test_mood_tracks_need_satisfaction(self):
        happy = Body(energy=95, hydration=95, satiety=95, hygiene=95,
                     fun=95, social=95, bladder=5, bowel=5)
        sad = Body(energy=5, hydration=10, satiety=5, hygiene=5,
                   fun=5, social=5, bladder=95, bowel=90)
        self.assertGreater(happy.mood()[0], sad.mood()[0])


class TestBigTickIntegration(unittest.TestCase):
    """A slow LLM at high sim speed produces huge ticks; physiology must
    subdivide them instead of stepping over thresholds."""

    def test_large_tick_stays_in_bounds(self):
        b = Body()
        b.tick(600, "idle")
        for attr in ("energy", "hydration", "satiety", "hygiene", "fun",
                     "social", "bladder", "bowel", "health"):
            v = getattr(b, attr)
            self.assertGreaterEqual(v, 0.0, attr)
            self.assertLessEqual(v, 100.0, attr)

    def test_large_tick_does_not_skip_overflow(self):
        b = Body(bladder=99.0)
        b.tick(600)
        self.assertGreaterEqual(b.accidents, 1)

    def test_repeated_warnings_are_deduplicated(self):
        b = Body(hydration=0.0)
        events = b.tick(60)
        self.assertEqual(
            len([e for e in events if "dehydrated" in e]), 1)

    def test_total_neglect_eventually_kills(self):
        b = Body()
        events = b.tick(10_000)  # ~a week with no water
        self.assertFalse(b.alive)
        self.assertEqual(b.cause_of_death, "dehydration")
        self.assertTrue(any("DIED" in e for e in events))

    def test_death_stops_the_clock(self):
        b = Body()
        b.tick(10_000)
        t = b.sim_minutes
        b.tick(500)
        self.assertEqual(b.sim_minutes, t)


class TestCircadian(unittest.TestCase):
    def test_sleep_drive_higher_at_night(self):
        night, day = Body(), Body()
        night.sim_minutes = 3 * 60       # 03:00
        day.sim_minutes = 15 * 60        # 15:00
        self.assertGreater(night.circadian_sleep_drive(),
                           day.circadian_sleep_drive())

    def test_persistence_roundtrip(self):
        b = Body(hydration=42.0, bladder=77.0)
        b.tick(100)
        b2 = Body.from_dict(b.to_dict())
        self.assertEqual(b2.hydration, b.hydration)
        self.assertEqual(b2.sim_minutes, b.sim_minutes)


if __name__ == "__main__":
    unittest.main()

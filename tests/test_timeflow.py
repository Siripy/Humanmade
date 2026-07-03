"""Tests for timeflow.py: pure anchor math, no clock mocking needed since
every function takes an explicit `now`."""

from __future__ import annotations

import unittest

from humanmade.timeflow import (Anchor, align_hour_of_day, rebase,
                                target_sim_minutes, wmo_to_weather)


class TestAnchorMath(unittest.TestCase):
    def test_target_advances_with_real_time(self):
        a = Anchor(real_seconds=1000.0, sim_minutes=500.0)
        # 120 real seconds = 2 sim-minutes at 60 real-sec-per-sim-min
        self.assertEqual(target_sim_minutes(a, now=1120.0), 502.0)

    def test_target_at_the_anchor_itself(self):
        a = Anchor(real_seconds=1000.0, sim_minutes=500.0)
        self.assertEqual(target_sim_minutes(a, now=1000.0), 500.0)

    def test_large_gap_computes_correctly(self):
        a = Anchor(real_seconds=0.0, sim_minutes=0.0)
        three_days_later = 3 * 86400.0
        self.assertAlmostEqual(target_sim_minutes(a, now=three_days_later),
                               3 * 1440.0)

    def test_time_never_runs_backward(self):
        a = Anchor(real_seconds=1000.0, sim_minutes=500.0)
        # "now" somehow before the anchor -- must not go negative/backward
        self.assertEqual(target_sim_minutes(a, now=500.0), 500.0)

    def test_roundtrip_to_from_dict(self):
        a = Anchor(real_seconds=123.5, sim_minutes=678.9)
        restored = Anchor.from_dict(a.to_dict())
        self.assertEqual(restored, a)


class TestRebase(unittest.TestCase):
    def test_rebase_ties_new_anchor_to_now(self):
        a = rebase(999.0, now=5000.0)
        self.assertEqual(a.sim_minutes, 999.0)
        self.assertEqual(a.real_seconds, 5000.0)
        self.assertEqual(target_sim_minutes(a, now=5000.0), 999.0)

    def test_rebasing_then_targeting_later_advances_correctly(self):
        a = rebase(100.0, now=0.0)
        self.assertEqual(target_sim_minutes(a, now=600.0), 110.0)  # +10 sim-min


class TestAlignHourOfDay(unittest.TestCase):
    def test_preserves_day_shifts_hour(self):
        # day 3 (3*1440=4320), some arbitrary minute -> day 3, hour 15
        result = align_hour_of_day(4320 + 200, target_hour=15.0)
        self.assertEqual(result, 4320 + 15 * 60)

    def test_day_zero(self):
        self.assertEqual(align_hour_of_day(500.0, target_hour=8.5), 8.5 * 60)

    def test_fractional_hour(self):
        result = align_hour_of_day(0.0, target_hour=8.25)
        self.assertAlmostEqual(result, 8.25 * 60)


class TestWmoToWeather(unittest.TestCase):
    def test_clear_sky_codes_are_sunny(self):
        for code in (0, 1, 2):
            self.assertEqual(wmo_to_weather(code), "sunny")

    def test_overcast_and_fog_are_cloudy(self):
        for code in (3, 45, 48):
            self.assertEqual(wmo_to_weather(code), "cloudy")

    def test_rain_drizzle_and_snow_map_to_rainy(self):
        for code in (51, 61, 65, 71, 80, 86):
            self.assertEqual(wmo_to_weather(code), "rainy")

    def test_thunderstorms_are_stormy(self):
        for code in (95, 96, 99):
            self.assertEqual(wmo_to_weather(code), "stormy")

    def test_unrecognized_code_falls_back_to_cloudy(self):
        self.assertEqual(wmo_to_weather(-1), "cloudy")
        self.assertEqual(wmo_to_weather(1000), "cloudy")


if __name__ == "__main__":
    unittest.main()

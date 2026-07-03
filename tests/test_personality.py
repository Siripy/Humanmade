"""Tests: traits must change how the human lives, not just how it talks."""

import tempfile
import unittest

from humanmade.body import Body
from humanmade.brain import ReflexBrain
from humanmade.personality import (agent_knobs, body_knobs,
                                   derive_disposition)
from humanmade.world import World
from tests.test_agent import make_human


def _dims(personality: str) -> dict:
    return derive_disposition(personality)["dimensions"]


class TestDisposition(unittest.TestCase):
    def test_anxious_raises_neuroticism(self):
        self.assertGreater(_dims("anxious, melancholic")["neuroticism"],
                           _dims("gentle, warm")["neuroticism"])

    def test_meticulous_raises_conscientiousness(self):
        self.assertGreater(_dims("meticulous, stubborn")["conscientiousness"],
                           _dims("dreamy, impatient")["conscientiousness"])

    def test_unknown_traits_are_harmless(self):
        d = _dims("unheard-of, mysterious")
        self.assertTrue(all(v == 0.5 for v in d.values()))

    def test_description_reflects_extremes(self):
        desc = derive_disposition("anxious, melancholic, impatient")["description"]
        self.assertIn("hit you hard", desc)

    def test_dimensions_stay_bounded(self):
        d = _dims("anxious, anxious, anxious")  # duplicates in one string
        for v in d.values():
            self.assertGreaterEqual(v, 0.05)
            self.assertLessEqual(v, 0.95)


class TestKnobsChangeTheBody(unittest.TestCase):
    def _body_with(self, personality: str) -> Body:
        b = Body()
        for k, v in body_knobs(_dims(personality)).items():
            setattr(b, k, v)
        return b

    def test_neurotic_mood_falls_faster_and_recovers_slower(self):
        neurotic = self._body_with("anxious, melancholic, impatient")
        stable = self._body_with("gentle, warm, meticulous")
        for b in (neurotic, stable):     # same shock to both
            b.energy = b.hydration = b.satiety = b.fun = b.social = 10
        neurotic.tick(30); stable.tick(30)
        self.assertLess(neurotic.valence_smoothed, stable.valence_smoothed)
        # now both fully recover their needs; the stable one bounces back faster
        for b in (neurotic, stable):
            b.energy = b.hydration = b.satiety = b.fun = b.social = 95
            b.bladder = b.bowel = 5
        neurotic.tick(60); stable.tick(60)
        self.assertLess(neurotic.valence_smoothed, stable.valence_smoothed)

    def test_extravert_social_drains_faster(self):
        extra = self._body_with("playful, warm, curious")
        intro = self._body_with("philosophical, gentle, melancholic")
        extra.tick(240); intro.tick(240)
        self.assertLess(extra.social, intro.social)

    def test_meticulous_showers_sooner(self):
        tidy = self._body_with("meticulous, stubborn, warm")
        slob = self._body_with("dreamy, impatient, playful")
        tidy.hygiene = slob.hygiene = 38.0
        world = World()
        self.assertEqual(ReflexBrain().decide(tidy, world)["action"], "shower")
        self.assertNotEqual(ReflexBrain().decide(slob, world)["action"], "shower")


class TestAgentIntegration(unittest.TestCase):
    def test_disposition_applied_and_backfilled(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            self.assertIn("disposition", human.persona)
            self.assertNotEqual(human.body.neg_reactivity, 0.0)
            self.assertTrue(0.4 <= human.plan_adherence <= 0.95)
            # simulate a pre-personality save: strip and reload
            del human.persona["disposition"]
            human._apply_disposition()
            self.assertIn("disposition", human.persona)
            human.memory.close()

    def test_agent_knob_ranges(self):
        k = agent_knobs(_dims("meticulous, warm, playful"))
        self.assertGreater(k["plan_adherence"], 0.6)
        k2 = agent_knobs(_dims("dreamy, impatient, playful"))
        self.assertLess(k2["plan_adherence"], k["plan_adherence"])


if __name__ == "__main__":
    unittest.main()

import unittest

from humanmade.body import Body
from humanmade.brain import (BrainError, LLMBrain, ReflexBrain,
                             _parse_decision)
from humanmade.world import World
from tests.mockllm import start_mock_ollama

PERSONA = {"name": "Test", "age": 30, "personality": "curious",
           "backstory": "Test subject."}


class TestDecisionParsing(unittest.TestCase):
    def test_valid_decision(self):
        d = _parse_decision('{"thought": "hm", "action": "eat", '
                            '"say": "hello", "importance": 6}')
        self.assertEqual(d["action"], "eat")
        self.assertEqual(d["say"], "hello")
        self.assertEqual(d["importance"], 6.0)

    def test_unknown_action_falls_back_to_idle(self):
        d = _parse_decision('{"thought": "hm", "action": "fly to the moon"}')
        self.assertEqual(d["action"], "idle")

    def test_null_like_say_becomes_none(self):
        for say in ('"null"', '"none"', '""', "null"):
            d = _parse_decision(f'{{"action": "idle", "say": {say}}}')
            self.assertIsNone(d["say"], say)

    def test_json_extracted_from_chatty_output(self):
        d = _parse_decision('Sure! Here is my decision:\n'
                            '{"thought": "ok", "action": "drink"}\nHope that helps!')
        self.assertEqual(d["action"], "drink")

    def test_garbage_raises_brain_error(self):
        with self.assertRaises(BrainError):
            _parse_decision("I am not JSON at all")

    def test_importance_clamped(self):
        d = _parse_decision('{"action": "idle", "importance": 999}')
        self.assertEqual(d["importance"], 10.0)


class TestReflexBrain(unittest.TestCase):
    """The brainstem reads Body/World state directly — survival priorities."""

    def setUp(self):
        self.reflex = ReflexBrain()
        self.world = World()

    def decide(self, body):
        return self.reflex.decide(body, self.world)

    def test_bursting_bladder_beats_thirst(self):
        body = Body(bladder=90.0, hydration=10.0)
        self.assertEqual(self.decide(body)["action"], "toilet")

    def test_thirst_beats_hunger(self):
        body = Body(hydration=10.0, satiety=10.0)
        self.assertEqual(self.decide(body)["action"], "drink")

    def test_hunger_with_food(self):
        body = Body(satiety=10.0)
        self.assertEqual(self.decide(body)["action"], "eat")

    def test_hunger_with_empty_fridge_drinks_instead(self):
        self.world.food_portions = 0
        body = Body(satiety=10.0)
        self.assertEqual(self.decide(body)["action"], "drink")

    def test_exhaustion_sleeps(self):
        body = Body(energy=10.0)
        self.assertEqual(self.decide(body)["action"], "sleep")

    def test_wakes_when_rested(self):
        body = Body(energy=95.0)
        body.fall_asleep()
        self.assertEqual(self.decide(body)["action"], "wake")

    def test_keeps_sleeping_when_tired(self):
        body = Body(energy=50.0)
        body.fall_asleep()
        self.assertEqual(self.decide(body)["action"], "idle")

    def test_contentment_idles(self):
        body = Body(energy=90, hydration=90, satiety=90, hygiene=90,
                    fun=90, social=90, bladder=5, bowel=5)
        self.assertEqual(self.decide(body)["action"], "idle")


class TestLLMBrain(unittest.TestCase):
    def test_decide_against_mock_ollama(self):
        srv, base_url = start_mock_ollama(
            decision={"thought": "thirsty", "action": "drink",
                      "say": "hello out there", "importance": 5})
        try:
            brain = LLMBrain({"base_url": base_url, "model": "test"})
            self.assertTrue(brain.available())
            d = brain.decide(PERSONA, "vitals...", [])
            self.assertEqual(d["action"], "drink")
            self.assertEqual(d["say"], "hello out there")
        finally:
            srv.stop()

    def test_reflect_against_mock_ollama(self):
        srv, base_url = start_mock_ollama(insights=["I am not alone."])
        try:
            brain = LLMBrain({"base_url": base_url, "model": "test"})
            self.assertEqual(brain.reflect(PERSONA, "- memory one"),
                             ["I am not alone."])
        finally:
            srv.stop()

    def test_unreachable_server(self):
        brain = LLMBrain({"base_url": "http://127.0.0.1:9",  # discard port
                          "model": "test", "timeout_seconds": 2})
        self.assertFalse(brain.available())
        with self.assertRaises(BrainError):
            brain.decide(PERSONA, "vitals...", [])


if __name__ == "__main__":
    unittest.main()

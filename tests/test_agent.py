import os
import tempfile
import time
import unittest

from humanmade.agent import Human
from tests.mockllm import start_mock_ollama

THINK_TIMEOUT = 8.0  # wall-clock seconds to wait for a worker-thread thought


def make_human(state_dir, config=None):
    speech, events = [], []
    human = Human(state_dir, config or {},
                  on_speak=speech.append, on_event=events.append)
    return human, speech, events


def drive_one_thought(human, timeout=THINK_TIMEOUT):
    """Force a decision cycle and pump ticks until the async thought lands."""
    human.minutes_since_decision = 10 ** 6
    human.tick(0.001)
    deadline = time.time() + timeout
    while (human._think_thread is not None or human._think_outcome is not None) \
            and time.time() < deadline:
        time.sleep(0.01)
        human.tick(0.001)


class ReflexLifeTest(unittest.TestCase):
    """Without any LLM the brainstem alone must sustain life."""

    def test_survives_48_hours(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, speech, events = make_human(tmp)
            human.llm_online = False
            for _ in range(48 * 12):          # 48h in 5-minute ticks
                human.tick(5.0)
            self.assertTrue(human.body.alive)
            self.assertEqual(human.body.accidents, 0)
            self.assertLess(human.world.food_portions, 6)  # it ate
            self.assertGreater(human.memory.count(), 20)   # it remembers living
            # it slept at least once and got up again
            acted = " ".join(events)
            self.assertIn("drifts off", acted)
            self.assertIn("out of bed", acted)
            human.memory.close()

    def test_state_persists_across_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            human.hear("remember me")
            for _ in range(50):
                human.tick(5.0)
            human.save()
            name, minutes = human.persona["name"], human.body.sim_minutes
            count = human.memory.count()
            human.memory.close()

            reborn, _, _ = make_human(tmp)
            self.assertEqual(reborn.persona["name"], name)
            self.assertAlmostEqual(reborn.body.sim_minutes, minutes, places=3)
            self.assertGreaterEqual(reborn.memory.count(), count)
            reborn.memory.close()

    def test_death_and_new_life(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = make_human(tmp)
            human.llm_online = False
            # a brain lesion: the reflexes no longer act on anything
            human.reflex.decide = lambda body, world: {
                "thought": "", "action": "idle", "say": None, "importance": 1}
            human.body.hydration = 0.5
            human.body.health = 1.0
            for _ in range(600):
                human.tick(5.0)
                if not human.body.alive:
                    break
            self.assertFalse(human.body.alive)
            self.assertEqual(human.body.cause_of_death, "dehydration")
            self.assertTrue(any("has died" in e for e in events))

            old_name = human.persona["name"]
            human.new_life()
            self.assertTrue(human.body.alive)
            self.assertEqual(human.memory.count(), 1)  # only its birth
            archived = [f for f in os.listdir(tmp) if f.startswith("memory-")]
            self.assertEqual(len(archived), 1, "past life must be archived")
            human.memory.close()


class LLMLifeTest(unittest.TestCase):
    """With a (mock) local model, thought is asynchronous and proactive."""

    def _human_with_mock(self, tmp, **mock_kwargs):
        srv, base_url = start_mock_ollama(**mock_kwargs)
        self.addCleanup(srv.stop)
        config = {"llm": {"provider": "ollama", "base_url": base_url,
                          "model": "test", "timeout_seconds": 5}}
        return make_human(tmp, config)

    def test_speaks_first_and_acts(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, speech, _ = self._human_with_mock(tmp)
            self.assertTrue(human.llm_online)
            human.hear("hello in there!")
            drive_one_thought(human)
            self.assertTrue(speech, "the human should have spoken")
            self.assertIn("name", speech[0])
            self.assertEqual(human.current_action, "drink")
            human.memory.close()

    def test_body_keeps_ticking_while_thinking(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human_with_mock(tmp, delay=0.5)
            human.minutes_since_decision = 10 ** 6
            human.tick(0.001)                 # spawns the think worker
            hydration_before = human.body.hydration
            human.tick(60.0)                  # a full sim hour mid-thought
            self.assertLess(human.body.hydration, hydration_before)
            drive_one_thought(human, timeout=THINK_TIMEOUT)
            human.memory.close()

    def test_reflection_stores_insight(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = self._human_with_mock(
                tmp, insights=["My companion checks on me; I am not alone."])
            human.memory._importance_since_reflection = 999
            human.tick(0.001)                 # spawns the reflection worker
            deadline = time.time() + THINK_TIMEOUT
            while human._reflecting and time.time() < deadline:
                time.sleep(0.01)
            reflections = [m for m in human.memory.recent(80)
                           if m.kind == "reflection"]
            self.assertTrue(reflections)
            self.assertIn("not alone", reflections[0].text)
            human.memory.close()

    def test_falls_back_to_reflex_after_repeated_llm_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = self._human_with_mock(tmp)
            human.llm.chat = lambda *a, **k: "I am not JSON at all"
            for _ in range(3):
                drive_one_thought(human)
            self.assertFalse(human.llm_online)
            self.assertTrue(human.body.alive)
            # reflex still decides: force thirst, next thought must drink
            human.body.hydration = 10.0
            drive_one_thought(human)
            self.assertEqual(human.current_action, "drink")
            human.memory.close()

    def test_stale_thought_from_past_life_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, speech, _ = self._human_with_mock(tmp)
            human.minutes_since_decision = 10 ** 6
            human.tick(0.001)                 # thought now in flight
            human.new_life()                  # reincarnate mid-thought
            deadline = time.time() + THINK_TIMEOUT
            while human._think_outcome is None and time.time() < deadline:
                time.sleep(0.01)
            human.tick(0.001)                 # collect: must be discarded
            self.assertEqual(speech, [], "a past life's words must not be spoken")
            self.assertEqual(human.current_action, "idle")
            human.memory.close()


if __name__ == "__main__":
    unittest.main()

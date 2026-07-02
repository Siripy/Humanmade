"""Tests for the usefulness pass: economy, semantic memory, conversation
compression, and first-run naming."""

import os
import tempfile
import time
import unittest

from humanmade.agent import Human
from humanmade.body import Body
from humanmade.brain import ReflexBrain
from humanmade.memory import MemoryStream
from humanmade.world import PORTION_PRICE, WORK_WAGE, World
from tests.mockllm import fake_embedding, start_mock_ollama
from tests.test_agent import drive_one_thought, make_human


class TestEconomy(unittest.TestCase):
    def test_work_earns_wage(self):
        w = World()
        before = w.money
        earned = w.earn(1.0)
        self.assertEqual(earned, WORK_WAGE)
        self.assertEqual(w.money, before + WORK_WAGE)

    def test_restock_spends_credits(self):
        w = World(money=30.0, food_portions=0)
        bought = w.restock(8)
        self.assertEqual(bought, 8)
        self.assertEqual(w.food_portions, 8)
        self.assertAlmostEqual(w.money, 30.0 - 8 * PORTION_PRICE)

    def test_restock_limited_by_credits(self):
        w = World(money=4.0, food_portions=0)
        bought = w.restock(8)
        self.assertEqual(bought, int(4.0 // PORTION_PRICE))
        self.assertGreaterEqual(w.money, 0.0)

    def test_broke_and_empty_fridge_cannot_buy(self):
        w = World(money=1.0, food_portions=0)
        self.assertEqual(w.restock(8), 0)
        self.assertEqual(w.food_portions, 0)

    def test_reflex_works_when_broke_and_fridge_low(self):
        reflex = ReflexBrain()
        body = Body()  # all needs comfortable
        world = World(money=2.0, food_portions=0)
        self.assertEqual(reflex.decide(body, world)["action"], "work")
        world.money = 100.0
        self.assertNotEqual(reflex.decide(body, world)["action"], "work")

    def test_agent_work_action_earns(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            before = human.world.money
            human._perform("work")
            self.assertGreater(human.world.money, before)
            human.memory.close()

    def test_restock_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            human.world.money = 3.0
            r = human.restock()
            self.assertEqual(r["bought"], 2)
            self.assertLessEqual(r["money"], 0.1)
            human.memory.close()

    def test_money_persists(self):
        w = World(money=7.5)
        self.assertEqual(World.from_dict(w.to_dict()).money, 7.5)


class TestSemanticMemory(unittest.TestCase):
    def test_cosine_relevance_beats_recency(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = MemoryStream(os.path.join(tmp, "m.sqlite3"))
            # older, semantically-matching memory vs. newer unrelated one
            mem.add("conversation", "companion favorite color teal", 0)
            mem.add("observation", "ate porridge for breakfast", 5000)
            rows = mem.unembedded(10)
            for mem_id, text in rows:
                mem.set_embedding(mem_id, fake_embedding(text))
            query = "companion favorite color teal"
            top = mem.retrieve(query, 5000, k=1,
                               query_embedding=fake_embedding(query))
            self.assertIn("teal", top[0].text)
            mem.close()

    def test_unembedded_shrinks_as_vectors_land(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = MemoryStream(os.path.join(tmp, "m.sqlite3"))
            mem.add("event", "one", 0)
            mem.add("event", "two", 1)
            self.assertEqual(len(mem.unembedded(10)), 2)
            mem_id, text = mem.unembedded(1)[0]
            mem.set_embedding(mem_id, [0.1, 0.2])
            self.assertEqual(len(mem.unembedded(10)), 1)
            mem.close()

    def test_hippocampus_thread_embeds_via_mock_server(self):
        srv, base_url = start_mock_ollama()
        self.addCleanup(srv.stop)
        cfg = {"embed_interval_seconds": 0.1,
               "llm": {"provider": "ollama", "base_url": base_url,
                       "model": "test", "timeout_seconds": 5}}
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, cfg)
            self.assertTrue(human.llm_online)
            deadline = time.time() + 20
            while time.time() < deadline:
                with human.lock:
                    if not human.memory.unembedded(5):
                        break
                time.sleep(0.05)
            with human.lock:
                self.assertEqual(human.memory.unembedded(5), [],
                                 "birth memory should have been embedded")
            self.assertTrue(human._embed_ok)
            human.memory.close()

    def test_no_embed_model_degrades_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)  # no server at all
            human.llm_online = False
            self.assertIsNone(human._query_embedding("anything"))
            # token-overlap retrieval still works
            human.memory.add("conversation", "the word is banana", 0)
            top = human.memory.retrieve("banana word", 10, k=1)
            self.assertIn("banana", top[0].text)
            human.memory.close()


class TestConversationCompression(unittest.TestCase):
    def test_history_keeps_moment_and_speech_only(self):
        perception = "\n".join([
            "[08:00, day 0] You are awake, currently: idle. Mood: okay (+0.10). ",
            "Body — energy:80 hydration:75 ...",
            "URGENT: very hungry",
            "The fridge holds 6 portions of food. You have 30 credits...",
            "Relevant memories:",
            "  - (thought) something old",
            'Your companion just said: "hello"',
            "What do you think, do, and (optionally) say?",
        ])
        compact = Human._compact_perception(perception)
        self.assertIn("[08:00, day 0]", compact)
        self.assertIn('Your companion just said: "hello"', compact)
        self.assertIn("URGENT: very hungry", compact)
        self.assertNotIn("Body —", compact)
        self.assertNotIn("Relevant memories", compact)
        self.assertNotIn("fridge", compact)

    def test_applied_decision_stores_compact_history(self):
        srv, base_url = start_mock_ollama()
        self.addCleanup(srv.stop)
        cfg = {"llm": {"provider": "ollama", "base_url": base_url,
                       "model": "test", "timeout_seconds": 5}}
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, cfg)
            human.hear("hi there")
            drive_one_thought(human)
            user_turns = [m for m in human.conversation if m["role"] == "user"]
            self.assertTrue(user_turns)
            self.assertNotIn("Relevant memories", user_turns[-1]["content"])
            self.assertNotIn("Body —", user_turns[-1]["content"])
            human.memory.close()


class TestNaming(unittest.TestCase):
    def test_first_run_uses_given_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            human = Human(tmp, {}, on_speak=lambda t: None,
                          on_event=lambda t: None, name="Siri")
            self.assertEqual(human.persona["name"], "Siri")
            human.save()
            human.memory.close()
            # a later run must keep the existing person, ignoring a new name
            again = Human(tmp, {}, on_speak=lambda t: None,
                          on_event=lambda t: None, name="Someone Else")
            self.assertEqual(again.persona["name"], "Siri")
            again.memory.close()


if __name__ == "__main__":
    unittest.main()

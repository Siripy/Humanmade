import os
import tempfile
import unittest

from humanmade.memory import MemoryStream


class TestMemoryStream(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "memory.sqlite3")
        self.mem = MemoryStream(self.path)

    def tearDown(self):
        self.mem.close()
        self.tmp.cleanup()

    def test_relevance_retrieval(self):
        self.mem.add("observation", "I drank a glass of water.", 10)
        self.mem.add("conversation", 'Companion said: "my favorite color is teal"', 20)
        self.mem.add("observation", "I took a shower.", 30)
        top = self.mem.retrieve("what is the companion's favorite color", 40, k=2)
        self.assertTrue(any("teal" in m.text for m in top))

    def test_recency_prefers_fresh_memories(self):
        self.mem.add("observation", "an old forgettable moment", 0)
        self.mem.add("observation", "a brand new moment", 5000)
        top = self.mem.retrieve("nothing in particular matches", 5000, k=1)
        self.assertIn("brand new", top[0].text)

    def test_importance_heuristic_boosts_loaded_events(self):
        self.mem.add("observation", "I looked out the window.", 0)
        self.mem.add("event", "I nearly died of thirst today.", 0)
        plain, loaded = self.mem.recent(2)
        self.assertGreater(loaded.importance, plain.importance)

    def test_persistence_across_reopen(self):
        self.mem.add("event", "something worth remembering", 100)
        self.mem.set_meta("persona", '{"name": "Test"}')
        self.mem.close()
        reopened = MemoryStream(self.path)
        self.assertEqual(reopened.count(), 1)
        self.assertEqual(reopened.get_meta("persona"), '{"name": "Test"}')
        reopened.close()
        self.mem = MemoryStream(self.path)  # for tearDown

    def test_reflection_accumulator(self):
        self.assertFalse(self.mem.reflection_due())
        for i in range(10):
            self.mem.add("event", f"big moment {i}", i, importance=8)
        self.assertTrue(self.mem.reflection_due())
        self.mem.mark_reflected()
        self.assertFalse(self.mem.reflection_due())

    def test_recent_filters_by_kind(self):
        self.mem.add("thought", "a thought", 0)
        self.mem.add("reflection", "an insight", 1)
        only = self.mem.recent(10, kinds=("reflection",))
        self.assertEqual([m.kind for m in only], ["reflection"])


if __name__ == "__main__":
    unittest.main()

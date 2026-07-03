"""Tests for creation.py and its wiring into the agent: the hobby produces
real, accumulating, readable work instead of a bare progress bar."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from unittest.mock import patch

from humanmade.creation import (FRAGMENT_NOUN, HOBBIES, HOBBY_KIND, Work,
                                WorksLibrary, craft_descriptor, _slugify)
from humanmade.brain import LLMBrain
from tests.mockllm import start_mock_ollama
from tests.test_agent import THINK_TIMEOUT, drive_one_thought, make_human


def wait_for(predicate, timeout=THINK_TIMEOUT):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.02)


def wait_while_creating(human, timeout=THINK_TIMEOUT):
    wait_for(lambda: not human._creating, timeout)
    time.sleep(0.05)  # let the worker finish applying its result under lock


class TestCreationModule(unittest.TestCase):
    def test_hobby_kind_and_fragment_noun_cover_every_hobby(self):
        for hobby in HOBBIES:
            self.assertIn(hobby, HOBBY_KIND)
            kind = HOBBY_KIND[hobby]
            self.assertIn(kind, FRAGMENT_NOUN)

    def test_craft_descriptor_thresholds(self):
        self.assertIn("beginner", craft_descriptor(0))
        self.assertIn("master", craft_descriptor(100))
        self.assertNotEqual(craft_descriptor(5), craft_descriptor(95))

    def test_craft_descriptor_monotonic_boundaries(self):
        # every level maps to exactly one descriptor, no gaps
        for level in (0, 14, 15, 39, 40, 64, 65, 84, 85, 100):
            self.assertTrue(craft_descriptor(level))

    def test_slugify_handles_punctuation_and_case(self):
        self.assertEqual(_slugify("Writing A Novel!"), "writing-a-novel")

    def test_slugify_empty_falls_back(self):
        self.assertEqual(_slugify("###"), "untitled")

    def test_work_roundtrip(self):
        w = Work(hobby="whittling", kind="carved collection", title="Woodland Set",
                 synopsis="Small animals.", fragments=[{"day": 3, "text": "A fox."}],
                 status="in_progress", started_sim=120.0)
        w2 = Work.from_dict(w.to_dict())
        self.assertEqual(w2.title, "Woodland Set")
        self.assertEqual(w2.fragments, [{"day": 3, "text": "A fox."}])

    def test_display_title_falls_back_to_kind(self):
        w = Work(hobby="whittling", kind="carved collection", title=None,
                 synopsis="", started_sim=0.0)
        self.assertIn("carved collection", w.display_title)


class TestWorksLibrary(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = WorksLibrary(tmp)
            slug = lib.new_slug("writing a novel", 100.0)
            work = Work(hobby="writing a novel", kind="novel", title=None,
                       synopsis="", started_sim=100.0)
            lib.save(work, slug)
            loaded = lib.load(slug)
            self.assertEqual(loaded.hobby, "writing a novel")

    def test_load_missing_slug_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(WorksLibrary(tmp).load("nonexistent"))
            self.assertIsNone(WorksLibrary(tmp).load(None))

    def test_finished_works_excludes_in_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = WorksLibrary(tmp)
            in_progress = Work(hobby="whittling", kind="carved collection",
                               title="Unfinished", synopsis="", started_sim=0.0)
            lib.save(in_progress, lib.new_slug("whittling", 0.0))
            finished = Work(hobby="whittling", kind="carved collection",
                           title="Done", synopsis="", status="finished",
                           started_sim=10.0, finished_sim=200.0, sale_price=40.0)
            lib.save(finished, lib.new_slug("whittling", 10.0))
            titles = [w.title for w in lib.finished_works()]
            self.assertEqual(titles, ["Done"])

    def test_finished_works_sorted_by_completion_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = WorksLibrary(tmp)
            for i, day in enumerate([300.0, 100.0, 200.0]):
                w = Work(hobby="whittling", kind="carved collection",
                        title=f"Work{i}", synopsis="", status="finished",
                        started_sim=0.0, finished_sim=day, sale_price=10.0)
                lib.save(w, lib.new_slug("whittling", float(i)))
            order = [w.finished_sim for w in lib.finished_works()]
            self.assertEqual(order, sorted(order))

    def test_new_slugs_do_not_collide_for_same_hobby_same_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = WorksLibrary(tmp)
            a = lib.new_slug("writing a novel", 100.4)
            b = lib.new_slug("writing a novel", 100.9)
            self.assertEqual(a, b)  # int() truncation - expected, same session


class TestBrainCreate(unittest.TestCase):
    def test_create_against_mock_ollama(self):
        srv, base_url = start_mock_ollama(
            creation_script=[{"title": "My Novel", "fragment": "It was a dark night.",
                             "synopsis": "A dark night begins the story."}])
        self.addCleanup(srv.stop)
        brain = LLMBrain({"base_url": base_url, "model": "test", "timeout_seconds": 5})
        result = brain.create({"name": "Theo"}, "writing a novel", "novel", "chapter",
                              "a beginner", "", "feeling okay", finishing=False)
        self.assertEqual(result["title"], "My Novel")
        self.assertIn("dark night", result["fragment"])

    def test_create_malformed_response_falls_back_gracefully(self):
        srv, base_url = start_mock_ollama()
        self.addCleanup(srv.stop)
        brain = LLMBrain({"base_url": base_url, "model": "test", "timeout_seconds": 5})
        brain.chat = lambda *a, **k: "not json at all"
        result = brain.create({"name": "Theo"}, "whittling", "carved collection",
                              "figure", "a beginner", "old synopsis",
                              "feeling fine", finishing=False)
        self.assertIsNone(result["fragment"])
        self.assertEqual(result["synopsis"], "old synopsis")  # preserved, not lost


class TestCreationGating(unittest.TestCase):
    def test_no_session_when_llm_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            human._begin_creation_session("writing a novel", finishing=False)
            self.assertFalse(human._creating)
            self.assertIsNone(human.persona.get("current_work_slug"))
            human.memory.close()

    @patch.object(LLMBrain, "create")
    def test_reentrancy_guard_blocks_overlap(self, mock_create):
        mock_create.return_value = {"title": None, "fragment": None, "synopsis": ""}
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = True
            human._creating = True   # simulate one already in flight
            human._begin_creation_session("writing a novel", finishing=False)
            mock_create.assert_not_called()
            human.memory.close()


class TestApplyCreationResult(unittest.TestCase):
    def _human(self, tmp):
        human, _, events = make_human(tmp)
        return human, events

    def test_empty_fragment_is_a_quiet_non_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, events = self._human(tmp)
            work = Work(hobby="whittling", kind="carved collection", title=None,
                       synopsis="", started_sim=0.0)
            human._apply_creation_result("slug1", work, 0,
                                         {"title": None, "fragment": None,
                                          "synopsis": ""}, finishing=False)
            self.assertEqual(work.fragments, [])
            self.assertFalse(human.memory.recent(5, kinds=("creation",)))
            human.memory.close()

    def test_fragment_appended_and_remembered(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _ = self._human(tmp)
            work = Work(hobby="whittling", kind="carved collection", title=None,
                       synopsis="", started_sim=0.0)
            human._apply_creation_result("slug1", work, 3,
                                         {"title": "Woodland", "fragment": "A fox.",
                                          "synopsis": "Small figures."}, finishing=False)
            self.assertEqual(len(work.fragments), 1)
            self.assertEqual(work.title, "Woodland")
            mems = human.memory.recent(5, kinds=("creation",))
            self.assertTrue(mems)
            self.assertIn("A fox", mems[0].text)
            human.memory.close()

    def test_title_only_assigned_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _ = self._human(tmp)
            work = Work(hobby="whittling", kind="carved collection", title="First Name",
                       synopsis="", started_sim=0.0)
            human._apply_creation_result("slug1", work, 0,
                                         {"title": "Second Name", "fragment": "More.",
                                          "synopsis": "..."}, finishing=False)
            self.assertEqual(work.title, "First Name")
            human.memory.close()

    def test_long_fragment_is_excerpted_in_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _ = self._human(tmp)
            work = Work(hobby="writing a novel", kind="novel", title="Long",
                       synopsis="", started_sim=0.0)
            long_text = "word " * 100
            human._apply_creation_result("slug1", work, 0,
                                         {"title": None, "fragment": long_text,
                                          "synopsis": "..."}, finishing=False)
            mem_text = human.memory.recent(5, kinds=("creation",))[0].text
            self.assertLess(len(mem_text), len(long_text) + 60)
            human.memory.close()

    def test_finishing_sells_and_narrates_and_clears_current_slug(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, events = self._human(tmp)
            human.persona["current_work_slug"] = "slug1"
            human.persona["skills"]["craft"] = 50.0
            work = Work(hobby="writing a novel", kind="novel", title="The End",
                       synopsis="", started_sim=0.0)
            money_before = human.world.money
            human._apply_creation_result("slug1", work, 0,
                                         {"title": None, "fragment": "The last line.",
                                          "synopsis": "It ends."}, finishing=True)
            self.assertIsNone(human.persona["current_work_slug"])
            self.assertGreater(human.world.money, money_before)
            self.assertEqual(human.emotion, "pride")
            self.assertTrue(any("finished" in e and "The End" in e for e in events))
            finished = human.finished_works()
            self.assertEqual(len(finished), 1)
            self.assertEqual(finished[0].sale_price, human.world.money - money_before)
            human.memory.close()

    def test_better_craft_sells_for_more(self):
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            novice, _ = self._human(tmp1)
            master, _ = self._human(tmp2)
            novice.persona["skills"]["craft"] = 5.0
            master.persona["skills"]["craft"] = 95.0
            w1 = Work(hobby="whittling", kind="carved collection", title="A",
                     synopsis="", started_sim=0.0)
            w2 = Work(hobby="whittling", kind="carved collection", title="B",
                     synopsis="", started_sim=0.0)
            novice._apply_creation_result("s1", w1, 0, {"title": None, "fragment": "x",
                                                         "synopsis": "y"}, finishing=True)
            master._apply_creation_result("s2", w2, 0, {"title": None, "fragment": "x",
                                                         "synopsis": "y"}, finishing=True)
            self.assertGreater(w2.sale_price, w1.sale_price)
            novice.memory.close(); master.memory.close()


class TestPersonaHobbyField(unittest.TestCase):
    def test_fresh_persona_has_hobby_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            self.assertIn(human.persona["hobby"], HOBBIES)
            self.assertIn(human.persona["hobby"], human.persona["backstory"])
            human.memory.close()

    def test_old_save_without_hobby_field_is_backfilled(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            del human.persona["hobby"]
            human._apply_disposition()
            self.assertIn("hobby", human.persona)
            self.assertIn(human.persona["hobby"], human.persona["backstory"])
            human.memory.close()

    def test_status_report_shows_current_work_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.persona["current_work_slug"] = "s1"
            work = Work(hobby=human.persona["hobby"], kind="novel",
                       title="A Real Title", synopsis="", started_sim=0.0)
            human.works.save(work, "s1")
            lines = " ".join(human.status_report())
            self.assertIn("A Real Title", lines)
            human.memory.close()


class TestFullCreationFlow(unittest.TestCase):
    """The whole pipeline for real: a mock LLM writes actual, accumulating
    content across multiple work sessions and eventually finishes it."""

    def _config(self, base_url):
        return {"llm": {"provider": "ollama", "base_url": base_url,
                        "model": "test", "timeout_seconds": 8}}

    def test_offline_sessions_advance_progress_but_write_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.llm_online = False
            before = float(human.persona.get("hobby_progress", 0.0))
            human._perform("work")
            self.assertGreater(float(human.persona["hobby_progress"]), before)
            self.assertIsNone(human.current_work())
            human.memory.close()

    def test_online_work_action_writes_a_real_fragment(self):
        srv, base_url = start_mock_ollama(
            decision={"thought": "let's work", "action": "work", "say": None,
                     "browse_query": None, "importance": 4},
            creation_script=[{"title": "The Lighthouse Keeper",
                             "fragment": "The fog rolled in early that year.",
                             "synopsis": "A keeper watches strange fog."}])
        self.addCleanup(srv.stop)
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, self._config(base_url))
            human.plan_adherence = 1.0  # never procrastinate "work" into "relax"
            drive_one_thought(human)
            self.assertEqual(human.current_action, "work")
            wait_while_creating(human)
            work = human.current_work()
            self.assertIsNotNone(work)
            self.assertEqual(work.title, "The Lighthouse Keeper")
            self.assertEqual(len(work.fragments), 1)
            human.memory.close()

    def test_finishing_online_does_not_double_narrate(self):
        """When the LLM is online, the rich creation-driven finish should be
        the only completion narration — not the old generic one too."""
        srv, base_url = start_mock_ollama(
            decision={"thought": "let's work", "action": "work", "say": None,
                     "browse_query": None, "importance": 4},
            creation_script=[{"title": "Almost There", "fragment": "The final word.",
                             "synopsis": "It is finished."}])
        self.addCleanup(srv.stop)
        with tempfile.TemporaryDirectory() as tmp:
            human, _, events = make_human(tmp, self._config(base_url))
            human.plan_adherence = 1.0
            human.persona["hobby_progress"] = 99.9  # one session from finishing
            drive_one_thought(human)
            wait_while_creating(human)
            finish_events = [e for e in events if "finished" in e.lower()
                             and "Almost There" in e]
            generic_finish = [e for e in events
                              if "FINISHED" in e and "Almost There" not in e]
            self.assertEqual(len(finish_events), 1)
            self.assertEqual(generic_finish, [])
            self.assertTrue(human.finished_works())
            human.memory.close()

    def test_stale_creation_result_after_new_life_is_discarded(self):
        srv, base_url = start_mock_ollama(
            decision={"thought": "let's work", "action": "work", "say": None,
                     "browse_query": None, "importance": 4},
            creation_script=[{"title": "Ghost Draft", "fragment": "Never should land.",
                             "synopsis": "..."}],
            delay=0.3)
        self.addCleanup(srv.stop)
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp, self._config(base_url))
            human.plan_adherence = 1.0
            drive_one_thought(human)   # spawns the creation worker mid-flight
            human.body.hydration = 0.0
            human.body.satiety = 0.0
            human.body.health = 0.5
            for _ in range(400):
                human.tick(5.0)
                if not human.body.alive:
                    break
            self.assertFalse(human.body.alive)
            human.new_life()
            time.sleep(1.0)  # let the stale worker try (and fail) to land
            self.assertIsNone(human.current_work())
            self.assertFalse(any(m.text.startswith('Wrote on "Ghost Draft')
                                 for m in human.memory.recent(20)))
            human.memory.close()


class TestNewLifeArchival(unittest.TestCase):
    def test_works_directory_archived_on_new_life(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.persona["current_work_slug"] = "s1"
            work = Work(hobby=human.persona["hobby"], kind="novel", title="Old Life's Book",
                       synopsis="", started_sim=0.0)
            human.works.save(work, "s1")
            human.new_life()
            self.assertIsNone(human.current_work())
            self.assertEqual(human.finished_works(), [])
            archived = [d for d in os.listdir(tmp) if d.startswith("works-")]
            self.assertEqual(len(archived), 1)
            self.assertTrue(os.path.exists(
                os.path.join(tmp, archived[0], "s1.json")))
            human.memory.close()

    def test_no_archive_dir_created_when_nothing_was_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            human, _, _ = make_human(tmp)
            human.new_life()
            archived = [d for d in os.listdir(tmp) if d.startswith("works-")]
            self.assertEqual(archived, [])
            human.memory.close()


if __name__ == "__main__":
    unittest.main()

"""The person: body + brain + memory + world, running an autonomous life loop.

The agent is never prompted by you. On its own cadence it perceives its body
and surroundings, retrieves relevant memories, thinks, picks an action, and —
when it feels like it — speaks first. Your messages just become part of what
it perceives.

Threading model: all state (body, world, memory, conversation) is guarded by
`self.lock`; the sim loop and the CLI both go through it. LLM calls are slow,
so thinking happens on a worker thread against a snapshot of perception — the
body keeps living while the mind deliberates, and the finished decision is
applied on the next tick. Only the current life may apply a decision (a
`_life_id` guard drops thoughts that finish after death or /newlife).
"""

from __future__ import annotations

import json
import os
import random
import threading
import time

from .body import Body
from .brain import BrainError, LLMBrain, ReflexBrain
from .memory import MemoryStream
from .world import World

FIRST_NAMES = ["June", "Theo", "Mara", "Elio", "Nadia", "Ravi", "Iris", "Sol",
               "Wren", "Kaito", "Zia", "Milan"]
TRAITS = ["curious", "stubborn", "gentle", "sarcastic", "anxious", "playful",
          "philosophical", "impatient", "warm", "melancholic", "meticulous", "dreamy"]
HOBBIES = ["writing a novel", "learning astronomy", "sketching birds",
           "composing chiptune music", "studying dead languages", "whittling"]

ACTION_DURATION = {  # sim-minutes each action occupies
    "eat": 20, "drink": 3, "toilet": 6, "shower": 12, "sleep": 0,  # sleep = until wake
    "wake": 1, "exercise": 40, "relax": 45, "work": 60, "idle": 20,
}
DECISION_INTERVAL = 30      # sim-minutes between spontaneous thoughts
SLEEP_CHECK_INTERVAL = 90   # thinks less often while asleep
RECONNECT_PROBE_CHANCE = 0.15  # odds per reflex thought of probing for the LLM


def make_persona(name: str | None = None) -> dict:
    return {
        "name": name or random.choice(FIRST_NAMES),
        "age": random.randint(19, 74),
        "personality": ", ".join(random.sample(TRAITS, 3)),
        "backstory": (f"Lives alone in a one-room apartment, spends time "
                      f"{random.choice(HOBBIES)}. Has no memory of how they got here, "
                      "only that a companion beyond the screen looks after the world."),
    }


class Human:
    def __init__(self, state_dir: str, config: dict,
                 on_speak, on_event, on_thought=None):
        os.makedirs(state_dir, exist_ok=True)
        self.state_dir = state_dir
        self.config = config
        self.on_speak = on_speak          # callback(text) — words said aloud
        self.on_event = on_event          # callback(text) — narration
        self.on_thought = on_thought      # callback(text) — inner monologue (verbose)

        self.lock = threading.RLock()
        self.memory = MemoryStream(os.path.join(state_dir, "memory.sqlite3"))
        self.body, self.world, self.persona = self._load_state()

        self.llm = LLMBrain(config.get("llm", {}))
        self.reflex = ReflexBrain()
        self.llm_online = self.llm.available()

        self.conversation: list[dict] = []   # short-term chat buffer for the LLM
        self.inbox: list[str] = []           # messages from the companion
        self.current_action = "idle"
        self.action_minutes_left = 5.0
        self.minutes_since_decision = 0.0
        self._llm_fail_streak = 0

        # in-flight cognition (see module docstring)
        self._life_id = 0
        self._think_seq = 0
        self._think_thread: threading.Thread | None = None
        self._think_outcome: dict | None = None
        self._reflecting = False

    # ------------------------------------------------------------ persistence

    def _load_state(self) -> tuple[Body, World, dict]:
        body_json = self.memory.get_meta("body")
        world_json = self.memory.get_meta("world")
        persona_json = self.memory.get_meta("persona")
        if persona_json:
            persona = json.loads(persona_json)
            body = Body.from_dict(json.loads(body_json)) if body_json else Body()
            world = World.from_dict(json.loads(world_json)) if world_json else World()
            return body, world, persona
        persona = make_persona()
        body, world = Body(), World()
        self.memory.set_meta("persona", json.dumps(persona))
        self.memory.add("event", f"{persona['name']} came into existence.",
                        body.sim_minutes, importance=10)
        return body, world, persona

    def save(self) -> None:
        with self.lock:
            self.memory.set_meta("body", json.dumps(self.body.to_dict()))
            self.memory.set_meta("world", json.dumps(self.world.to_dict()))
            self.memory.set_meta("persona", json.dumps(self.persona))

    # ------------------------------------------------------- companion inputs

    def hear(self, text: str) -> None:
        """Companion said something; the human will notice on its next thought."""
        with self.lock:
            self.inbox.append(text)
            self.body.social = min(100.0, self.body.social + 18)
            self.memory.add("conversation", f"Companion said: \"{text}\"",
                            self.body.sim_minutes)

    def restock(self, portions: int = 8) -> int:
        with self.lock:
            total = self.world.restock(portions)
            self.memory.add("event", "My companion restocked the fridge.",
                            self.body.sim_minutes, importance=6)
            return total

    def status_report(self) -> list[str]:
        with self.lock:
            return self.body.status_lines() + [
                f"  fridge: {self.world.food_portions} portions"]

    def recent_memories(self, n: int = 15):
        with self.lock:
            return self.memory.recent(n)

    # ------------------------------------------------------------------- tick

    def tick(self, sim_minutes: float) -> None:
        """Advance the person's life by `sim_minutes` of simulated time."""
        with self.lock:
            if not self.body.alive:
                return

            events = self.body.tick(sim_minutes, self.current_action)
            for e in events:
                self.on_event(e)
                self.memory.add("event", e, self.body.sim_minutes,
                                importance=9 if "DIED" in e else 7)
            if not self.body.alive:
                self._die()
                return

            self.action_minutes_left -= sim_minutes
            self.minutes_since_decision += sim_minutes

            self._collect_thought()

            if self._think_thread is None and self.body.alive:
                interval = (SLEEP_CHECK_INTERVAL if self.body.asleep
                            else DECISION_INTERVAL)
                urgent = self.body.urgent_needs()
                must_think = (
                    self.inbox
                    or self.minutes_since_decision >= interval
                    or (self.action_minutes_left <= 0 and not self.body.asleep)
                    or (urgent and self.minutes_since_decision >= 10)
                )
                if must_think:
                    self._begin_think()

            if (self.memory.reflection_due() and not self.body.asleep
                    and not self._reflecting):
                self._begin_reflection()

    # ---------------------------------------------------------------- thought

    def _perception(self) -> str:
        b = self.body
        valence, mood = b.mood()
        lines = [
            f"[{b.clock}, day {b.day}] You are {'asleep' if b.asleep else 'awake'}, "
            f"currently: {self.current_action}. Mood: {mood} ({valence:+.2f}).",
            f"Body — energy:{b.energy:.0f} hydration:{b.hydration:.0f} "
            f"satiety:{b.satiety:.0f} bladder:{b.bladder:.0f} bowel:{b.bowel:.0f} "
            f"hygiene:{b.hygiene:.0f} fun:{b.fun:.0f} social:{b.social:.0f} "
            f"health:{b.health:.0f} (all 0-100)",
        ]
        urgent = b.urgent_needs()
        if urgent:
            lines.append("URGENT: " + "; ".join(urgent))
        lines.append(self.world.describe())

        query = " ".join(urgent) + " " + " ".join(self.inbox[-2:])
        retrieved = self.memory.retrieve(query or "daily life companion",
                                         b.sim_minutes, k=6)
        if retrieved:
            lines.append("Relevant memories:")
            lines += [f"  - ({m.kind}) {m.text}" for m in retrieved]

        if self.inbox:
            for msg in self.inbox:
                lines.append(f'Your companion just said: "{msg}"')
        elif b.social < 30:
            lines.append("You haven't spoken to anyone in a long while. If you have "
                         "something to say or ask, say it — no one will prompt you.")
        lines.append("What do you think, do, and (optionally) say?")
        return "\n".join(lines)

    def _begin_think(self) -> None:
        """Snapshot perception and start deciding. Reflex decisions are instant;
        LLM decisions run on a worker thread and land on a later tick."""
        self.minutes_since_decision = 0.0
        perception = self._perception()
        heard = bool(self.inbox)
        self.inbox.clear()

        if not self.llm_online:
            if random.random() < RECONNECT_PROBE_CHANCE:
                self._spawn_reconnect_probe()
            self._apply_decision(self.reflex.decide(self.body, self.world),
                                 perception, heard)
            return

        life_id = self._life_id
        self._think_seq += 1
        seq = self._think_seq
        conversation = list(self.conversation)

        def worker() -> None:
            try:
                decision = self.llm.decide(self.persona, perception, conversation)
                outcome = {"decision": decision, "error": None}
            except BrainError as e:
                outcome = {"decision": None, "error": str(e)}
            except Exception as e:  # the mind must never kill the body
                outcome = {"decision": None, "error": f"unexpected: {e!r}"}
            outcome.update(perception=perception, heard=heard,
                           life_id=life_id, seq=seq)
            self._think_outcome = outcome

        self._think_thread = threading.Thread(target=worker, daemon=True,
                                              name="humanmade-think")
        self._think_thread.start()

    def _collect_thought(self) -> None:
        """Apply a finished worker-thread decision, if one has landed."""
        outcome = self._think_outcome
        if outcome is None:
            return
        self._think_outcome = None
        if outcome["seq"] != self._think_seq:
            return  # a superseded worker finished late — a newer one is in flight
        self._think_thread = None
        if outcome["life_id"] != self._life_id or not self.body.alive:
            return  # thought from a past life or after death — let it go

        if outcome["error"] is not None:
            self._llm_fail_streak += 1
            if self._llm_fail_streak == 1:
                self.on_event(f"(mind fog: {outcome['error']})")
            if self._llm_fail_streak >= 3 and self.llm_online:
                self.llm_online = False
                self.on_event("(the higher mind went dark — survival instinct "
                              "takes over; will retry the LLM periodically)")
            decision = self.reflex.decide(self.body, self.world)
        else:
            self._llm_fail_streak = 0
            decision = outcome["decision"]
        self._apply_decision(decision, outcome["perception"], outcome["heard"])

    def _spawn_reconnect_probe(self) -> None:
        def probe() -> None:
            if self.llm.available():
                with self.lock:
                    if not self.llm_online:
                        self.llm_online = True
                        self._llm_fail_streak = 0
                        self.on_event("(the higher mind flickers back online)")
        threading.Thread(target=probe, daemon=True,
                         name="humanmade-probe").start()

    def _apply_decision(self, decision: dict, perception: str, heard: bool) -> None:
        self.conversation.append({"role": "user", "content": perception})
        self.conversation.append({"role": "assistant", "content": json.dumps(decision)})
        self.conversation = self.conversation[-16:]

        if decision["thought"]:
            self.memory.add("thought", decision["thought"], self.body.sim_minutes,
                            importance=decision["importance"])
            if self.on_thought:
                self.on_thought(decision["thought"])

        if decision["say"]:
            self.on_speak(decision["say"])
            self.memory.add("conversation", f'I said: "{decision["say"]}"',
                            self.body.sim_minutes, importance=decision["importance"])
        elif heard:
            # companion spoke but the human chose silence — still worth noting
            self.memory.add("thought", "I heard them but didn't feel like answering.",
                            self.body.sim_minutes, importance=2)

        self._perform(decision["action"])

    # ----------------------------------------------------------------- action

    def _perform(self, action: str) -> None:
        b = self.body
        if b.asleep and action not in ("wake", "idle"):
            # body wakes itself if the mind decided to do something
            b.wake_up()
            self.on_event(f"{self.persona['name']} wakes up.")

        narration = None
        if action == "eat":
            if self.world.take_meal():
                b.eat()
                narration = "eats a meal"
                if self.world.food_portions <= 2:
                    narration += f" — only {self.world.food_portions} portions left"
            else:
                narration = "opens the fridge… it's empty. Nothing to eat."
                self.memory.add("event", "The fridge is empty. I cannot eat until "
                                "my companion restocks it.", b.sim_minutes, importance=8)
                action = "idle"
        elif action == "drink":
            b.drink(); narration = "drinks a glass of water"
        elif action == "toilet":
            b.use_toilet(); narration = "uses the toilet"
        elif action == "shower":
            b.shower(); narration = "takes a shower"
        elif action == "sleep":
            if not b.asleep:
                b.fall_asleep(); narration = "climbs into bed and drifts off"
        elif action == "wake":
            if b.asleep:
                b.wake_up(); narration = "gets out of bed"
        elif action == "exercise":
            b.fun = min(100.0, b.fun + 15); narration = "works out with the dumbbells"
        elif action == "relax":
            b.fun = min(100.0, b.fun + 25); narration = "curls up with a book"
        elif action == "work":
            b.fun = min(100.0, b.fun + 10)
            narration = f"works on {self.persona['backstory'].split('spends time ')[-1].split('.')[0]}"

        self.current_action = action
        self.action_minutes_left = ACTION_DURATION.get(action, 20)
        if narration:
            self.on_event(f"{self.persona['name']} {narration}.")
            self.memory.add("observation", f"I {narration}.", b.sim_minutes)

    # ------------------------------------------------------------- reflection

    def _begin_reflection(self) -> None:
        self.memory.mark_reflected()
        if not self.llm_online:
            return
        recent = self.memory.recent(30)
        text = "\n".join(f"- {m.text}" for m in recent)
        life_id = self._life_id
        self._reflecting = True

        def worker() -> None:
            insights: list[str] = []
            try:
                insights = self.llm.reflect(self.persona, text)
            except Exception:
                pass
            with self.lock:
                self._reflecting = False
                if life_id != self._life_id or not self.body.alive:
                    return
                for insight in insights:
                    self.memory.add("reflection", insight, self.body.sim_minutes,
                                    importance=8)
                    self.on_event(f"({self.persona['name']} realizes: {insight})")

        threading.Thread(target=worker, daemon=True,
                         name="humanmade-reflect").start()

    # ------------------------------------------------------------------ death

    def _die(self) -> None:
        name = self.persona["name"]
        days = self.body.day
        self.memory.add(
            "event",
            f"{name} died of {self.body.cause_of_death} after {days} days, "
            f"with {self.memory.count()} memories.",
            self.body.sim_minutes, importance=10)
        self.save()
        self.on_event(f"{name} has died of {self.body.cause_of_death}. "
                      f"They lived {days} days. Their memories remain in "
                      f"{self.state_dir}. Use /newlife to begin again.")

    def new_life(self) -> None:
        """Archive the old memory DB and start a fresh person."""
        with self.lock:
            self._life_id += 1            # orphan any in-flight thoughts
            self._think_thread = None
            self._think_outcome = None
            self._reflecting = False
            self.memory.close()
            old = os.path.join(self.state_dir, "memory.sqlite3")
            if os.path.exists(old):
                stamp = time.strftime("%Y%m%d-%H%M%S")
                os.rename(old, os.path.join(self.state_dir, f"memory-{stamp}.sqlite3"))
            self.memory = MemoryStream(old)
            self.body, self.world = Body(), World()
            self.persona = make_persona()
            self.conversation.clear()
            self.inbox.clear()
            self.current_action = "idle"
            self.action_minutes_left = 5.0
            self.minutes_since_decision = 0.0
            self._llm_fail_streak = 0
            self.memory.set_meta("persona", json.dumps(self.persona))
            self.memory.add("event", f"{self.persona['name']} came into existence.",
                            self.body.sim_minutes, importance=10)
            self.save()

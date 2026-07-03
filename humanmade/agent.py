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
from .personality import agent_knobs, body_knobs, derive_disposition
from .world import World

FIRST_NAMES = ["June", "Theo", "Mara", "Elio", "Nadia", "Ravi", "Iris", "Sol",
               "Wren", "Kaito", "Zia", "Milan"]
TRAITS = ["curious", "stubborn", "gentle", "sarcastic", "anxious", "playful",
          "philosophical", "impatient", "warm", "melancholic", "meticulous", "dreamy"]
HOBBIES = ["writing a novel", "learning astronomy", "sketching birds",
           "composing chiptune music", "studying dead languages", "whittling"]
# chronotype -> (circadian offset hours, human-readable habit)
CHRONOTYPES = {
    "lark": (2.0, "an early bird who wakes with the sun and fades by evening"),
    "intermediate": (0.0, "neither an early bird nor a night owl"),
    "owl": (-2.5, "a night owl who comes alive late and hates mornings"),
}

ACTION_DURATION = {  # sim-minutes each action occupies
    "eat": 20, "drink": 3, "toilet": 6, "shower": 12, "sleep": 0,  # sleep = until wake
    "wake": 1, "exercise": 40, "relax": 45, "work": 60, "idle": 20,
}
DECISION_INTERVAL = 30      # sim-minutes between spontaneous thoughts
SLEEP_CHECK_INTERVAL = 90   # thinks less often while asleep
RECONNECT_PROBE_CHANCE = 0.15  # odds per reflex thought of probing for the LLM
MIN_RESTORATIVE_SLEEP = 180.0  # sim-min of sleep needed to dream / consolidate
REUNION_GAP_MINUTES = 20.0     # sim-min apart that counts as a real separation
EMOTION_HALF_LIFE = 120.0      # sim-min for a discrete emotion to fade by half

# appraisal-derived emotions (OCC-lite) and their immediate mood kick
EMOTION_VALENCE = {
    "pride": +0.15, "gratitude": +0.14, "joy": +0.15, "relief": +0.10,
    "frustration": -0.12, "worry": -0.10, "shame": -0.18, "hurt": -0.12,
    "loneliness": -0.08,
}


def make_persona(name: str | None = None) -> dict:
    chronotype = random.choice(list(CHRONOTYPES))
    personality = ", ".join(random.sample(TRAITS, 3))
    loved, hated = random.sample(["sunny", "cloudy", "rainy", "stormy"], 2)
    return {
        "name": name or random.choice(FIRST_NAMES),
        "age": random.randint(19, 74),
        "personality": personality,
        "disposition": derive_disposition(personality),
        "chronotype": chronotype,
        "loves_weather": loved,
        "hates_weather": hated,
        "birthday_day": random.randint(1, 364),  # sim day-of-year
        "backstory": (f"Lives alone in a one-room apartment, spends time "
                      f"{random.choice(HOBBIES)}. Has no memory of how they got here, "
                      "only that a companion beyond the screen looks after the world."),
    }


class Human:
    def __init__(self, state_dir: str, config: dict,
                 on_speak, on_event, on_thought=None, name: str | None = None):
        os.makedirs(state_dir, exist_ok=True)
        self.state_dir = state_dir
        self.config = config
        self.on_speak = on_speak          # callback(text) — words said aloud
        self.on_event = on_event          # callback(text) — narration
        self.on_thought = on_thought      # callback(text) — inner monologue (verbose)

        self.lock = threading.RLock()
        self.memory = MemoryStream(os.path.join(state_dir, "memory.sqlite3"))
        self.body, self.world, self.persona = self._load_state(name)

        self.llm = LLMBrain(config.get("llm", {}))
        self.reflex = ReflexBrain()
        self.llm_online = self.llm.available()

        self.conversation: list[dict] = []   # short-term chat buffer for the LLM
        self.inbox: list[str] = []           # messages from the companion
        self.current_action = "idle"
        self.action_minutes_left = 5.0
        self.minutes_since_decision = 0.0
        self._llm_fail_streak = 0

        # chronotype -> body circadian offset
        offset, _ = CHRONOTYPES.get(self.persona.get("chronotype", "intermediate"),
                                    (0.0, ""))
        self.body.chronotype_offset_hours = offset
        self._apply_disposition()

        # daily life: plan, dreams, sleep bookkeeping
        self.today_plan: list[str] = json.loads(self.memory.get_meta("plan", "[]"))
        self.plan_day_index = int(self.memory.get_meta("plan_day", "-1"))
        self.last_dream: str | None = self.memory.get_meta("last_dream")
        self._was_asleep = self.body.asleep
        self._sleep_started_sim = self.body.sim_minutes
        self._dreamed_this_sleep = False
        self._planning = False
        self._last_journal_sim = -10 ** 9

        # companion presence & attachment (Bowlby/Ainsworth)
        self.companion_present = True
        self.last_seen_sim = self.body.sim_minutes
        self.pending_news: list[str] = []   # things saved up to tell you on return
        self._reunion_gap: float | None = None  # sim-min apart, set on return

        # current discrete emotion (appraisal of recent events, decays)
        self.emotion: str | None = None
        self.emotion_intensity = 0.0
        self._evening_indulged = False   # bedtime procrastination, once/night

        # a question it asked you that you never answered
        self.pending_question = json.loads(
            self.memory.get_meta("pending_question", "null"))
        # variety-seeking: recent leisure choices lose their shine
        self._recent_actions: list[str] = []

        # the relationship itself develops (social penetration theory)
        self.bond = json.loads(self.memory.get_meta("bond", "null")) or {
            "trust": 40.0, "closeness": 20.0,
            "first_met_sim": self.body.sim_minutes,
            "last_anniversary_days": 0.0,
        }
        self._last_day = self.body.day

        # in-flight cognition (see module docstring)
        self._life_id = 0
        self._think_seq = 0
        self._think_thread: threading.Thread | None = None
        self._think_outcome: dict | None = None
        self._reflecting = False

        # semantic memory ("hippocampus"): a background thread embeds new
        # memories; retrieval then works by meaning instead of word overlap.
        # Needs an embedding model on the server; degrades silently without one.
        self._embed_ok = False        # at least one batch succeeded
        self._embed_disabled = False  # gave up (no embed model available)
        self._embed_fails = 0
        self._embed_interval = float(config.get("embed_interval_seconds", 5.0))
        threading.Thread(target=self._embed_loop, daemon=True,
                         name="humanmade-hippocampus").start()

    def _apply_disposition(self) -> None:
        """Wire the persona's temperament into body and behavior parameters."""
        if "disposition" not in self.persona:  # backfill pre-personality saves
            self.persona["disposition"] = derive_disposition(
                self.persona.get("personality", ""))
        if "loves_weather" not in self.persona:  # backfill preferences/birthday
            loved, hated = random.sample(["sunny", "cloudy", "rainy", "stormy"], 2)
            self.persona["loves_weather"] = loved
            self.persona["hates_weather"] = hated
            self.persona.setdefault("birthday_day", random.randint(1, 364))
        dims = self.persona["disposition"]["dimensions"]
        for knob, value in body_knobs(dims).items():
            setattr(self.body, knob, value)
        knobs = agent_knobs(dims)
        self.plan_adherence = knobs["plan_adherence"]
        self.speak_up_threshold = knobs["speak_up_threshold"]

    # ------------------------------------------------------------ persistence

    def _load_state(self, name: str | None = None) -> tuple[Body, World, dict]:
        body_json = self.memory.get_meta("body")
        world_json = self.memory.get_meta("world")
        persona_json = self.memory.get_meta("persona")
        if persona_json:
            persona = json.loads(persona_json)
            body = Body.from_dict(json.loads(body_json)) if body_json else Body()
            world = World.from_dict(json.loads(world_json)) if world_json else World()
            return body, world, persona
        persona = make_persona(name)
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
            self.memory.set_meta("plan", json.dumps(self.today_plan))
            self.memory.set_meta("plan_day", str(self.plan_day_index))
            self.memory.set_meta("bond", json.dumps(self.bond))
            self.memory.set_meta("pending_question",
                                 json.dumps(self.pending_question))
            if self.last_dream:
                self.memory.set_meta("last_dream", self.last_dream)

    # ------------------------------------------------------- companion inputs

    def _bond_adjust(self, trust: float = 0.0, closeness: float = 0.0) -> None:
        self.bond["trust"] = max(0.0, min(100.0, self.bond["trust"] + trust))
        self.bond["closeness"] = max(0.0, min(100.0, self.bond["closeness"] + closeness))

    def bond_level(self) -> str:
        c = self.bond["closeness"]
        return "strangers, really" if c < 15 else \
               "acquaintances" if c < 35 else \
               "friends" if c < 65 else "genuinely close"

    def hear(self, text: str) -> None:
        """Companion said something; the human will notice on its next thought.

        If they've been apart for a while, this is a reunion (attachment
        theory): the human registers the separation and will greet them,
        colored by how long they were gone and its mood."""
        with self.lock:
            gap = self.body.sim_minutes - self.last_seen_sim
            returning = not self.companion_present or gap >= REUNION_GAP_MINUTES
            if returning and gap >= REUNION_GAP_MINUTES:
                self._reunion_gap = gap
                # the relief of reunion is a genuine emotional event
                self._feel("joy", min(0.9, 0.3 + gap / 3000))
                self._bond_adjust(closeness=1.5)
                if gap > 48 * 60:   # abandoned for days — it noticed
                    self._bond_adjust(trust=-3.0, closeness=-1.0)
            self.companion_present = True
            self.last_seen_sim = self.body.sim_minutes
            self.pending_question = None   # any reply counts as being heard
            self.inbox.append(text)
            self.body.social = min(100.0, self.body.social + 18)
            self._bond_adjust(closeness=0.4)
            self.memory.add("conversation", f"Companion said: \"{text}\"",
                            self.body.sim_minutes)
            self.minutes_since_decision = 10 ** 6  # answer promptly

    def set_away(self) -> None:
        """The companion says they're leaving. The human says goodbye (on its
        next thought) and starts keeping watch for their return."""
        with self.lock:
            if self.companion_present:
                self.companion_present = False
                self.last_seen_sim = self.body.sim_minutes
                self.pending_news.clear()
                self.memory.add("event", "My companion said they're leaving for a "
                                "while. I'm on my own now.", self.body.sim_minutes,
                                importance=6)
                self.minutes_since_decision = 10 ** 6  # let it say goodbye

    def restock(self, portions: int = 8) -> dict:
        """Place a grocery order with the human's own credits. Returns what
        happened so the CLI can report it honestly."""
        with self.lock:
            bought = self.world.restock(portions)
            if bought > 0:
                self._bond_adjust(trust=2.5)
                self._feel("gratitude", 0.6)
                self.memory.add("event",
                                f"My companion ordered groceries — {bought} portions, "
                                f"paid from my credits ({self.world.money:.0f} left).",
                                self.body.sim_minutes, importance=6)
            else:
                self.memory.add("event", "My companion tried to order groceries but "
                                "I couldn't afford any. I need to work.",
                                self.body.sim_minutes, importance=7)
            return {"bought": bought, "total": self.world.food_portions,
                    "money": self.world.money}

    def medicine(self) -> dict:
        """Order medicine for the human (its credits). Returns what happened."""
        with self.lock:
            if self.body.sickness <= 0:
                return {"ok": False, "reason": "not sick",
                        "money": self.world.money}
            if not self.world.buy_medicine():
                self.memory.add("event", "My companion tried to order medicine "
                                "but I can't afford it.", self.body.sim_minutes,
                                importance=7)
                return {"ok": False, "reason": "broke", "money": self.world.money}
            self.body.take_medicine()
            self._bond_adjust(trust=5.0, closeness=2.0)  # they took care of me
            self._feel("gratitude", 0.9)
            self.memory.add("event", "My companion ordered medicine for me. "
                            "I already feel it working.", self.body.sim_minutes,
                            importance=7)
            return {"ok": True, "sickness": self.body.sickness,
                    "money": self.world.money}

    def journal_entries(self, n: int = 5):
        with self.lock:
            return self.memory.recent(n, kinds=("journal",))

    def status_report(self) -> list[str]:
        with self.lock:
            hobby = self.persona["backstory"].split("spends time ")[-1].split(".")[0]
            return self.body.status_lines() + [
                f"  fridge: {self.world.food_portions} portions · "
                f"credits: {self.world.money:.0f} · weather: {self.world.weather}",
                f"  {hobby}: {float(self.persona.get('hobby_progress', 0)):.0f}% done"]

    def recent_memories(self, n: int = 15):
        with self.lock:
            return self.memory.recent(n)

    # ------------------------------------------------------------------- tick

    def tick(self, sim_minutes: float) -> None:
        """Advance the person's life by `sim_minutes` of simulated time."""
        with self.lock:
            if not self.body.alive:
                return

            # the outside world moves, and the body feels it
            self.body.chill = self.world.chill_factor()
            ambient = self.world.weather_valence()
            if self.world.weather == self.persona.get("loves_weather"):
                ambient += 0.06     # their kind of sky
            elif self.world.weather == self.persona.get("hates_weather"):
                ambient -= 0.06
            self.body.ambient_valence = ambient
            for e in self.world.advance(sim_minutes):
                self.on_event(e)
                self.memory.add("observation", e, self.body.sim_minutes,
                                importance=3)
            events = self.body.tick(sim_minutes, self.current_action)
            for e in events:
                self.on_event(e)
                self.memory.add("event", e, self.body.sim_minutes,
                                importance=9 if "DIED" in e else 7)
                self._note_news(e)
                self._appraise_event(e)
            if not self.body.alive:
                self._die()
                return

            self._decay_emotion(sim_minutes)

            self._handle_sleep_transitions()

            if self.body.day != self._last_day:
                self._last_day = self.body.day
                self._daily_checks()

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

    # -------------------------------------------------------- semantic memory

    def _embed_loop(self) -> None:
        while True:
            time.sleep(self._embed_interval)
            if not self.llm_online or self._embed_disabled:
                continue
            try:
                with self.lock:
                    batch = self.memory.unembedded(16)
                if not batch:
                    continue
                vectors = self.llm.embed([text for _, text in batch])
                with self.lock:
                    for (mem_id, _), vec in zip(batch, vectors):
                        self.memory.set_embedding(mem_id, vec)
                self._embed_ok = True
                self._embed_fails = 0
            except BrainError:
                self._embed_fails += 1
                if self._embed_fails >= 3:
                    self._embed_disabled = True  # no embed model — stop asking
            except Exception:
                return  # DB closed mid-shutdown or similar — stand down

    def _query_embedding(self, query: str) -> list[float] | None:
        if not (self._embed_ok and self.llm_online and not self._embed_disabled):
            return None
        try:
            return self.llm.embed([query])[0]
        except BrainError:
            return None

    # --------------------------------------------------------------- emotions

    def _feel(self, label: str, intensity: float) -> None:
        """Appraise an event into a discrete emotion (OCC-lite). The stronger
        feeling wins; every emotion also kicks the slow-moving mood."""
        if intensity >= self.emotion_intensity:
            self.emotion, self.emotion_intensity = label, min(1.0, intensity)
        self.body.nudge_mood(EMOTION_VALENCE.get(label, 0.0) * intensity)

    def _decay_emotion(self, sim_minutes: float) -> None:
        if self.emotion_intensity > 0:
            self.emotion_intensity *= 0.5 ** (sim_minutes / EMOTION_HALF_LIFE)
            if self.emotion_intensity < 0.05:
                self.emotion, self.emotion_intensity = None, 0.0

    def _appraise_event(self, event: str) -> None:
        """Body/world events carry emotional meaning (appraisal theory)."""
        e = event.lower()
        if "accident" in e or "wet the bed" in e:
            self._feel("shame", 0.9)
        elif "coming down with" in e:
            self._feel("worry", 0.6)
        elif "burning with fever" in e:
            self._feel("worry", 0.9)
        elif "fever has broken" in e or "woke up naturally" in e:
            self._feel("relief", 0.5)

    # ------------------------------------------------------------ daily rhythm

    ANNIVERSARIES = (7, 30, 60, 90, 180, 365)

    def _daily_checks(self) -> None:
        """Things a human notices at the turn of a day."""
        days_known = (self.body.sim_minutes - self.bond["first_met_sim"]) / 1440
        last = self.bond.get("last_anniversary_days", 0.0)
        for milestone in self.ANNIVERSARIES:
            if last < milestone <= days_known:
                self.bond["last_anniversary_days"] = float(milestone)
                self.memory.add(
                    "event", f"My companion and I have known each other "
                    f"{milestone} days now. Strange and nice to count it.",
                    self.body.sim_minutes, importance=8)
                self._note_news(f"realized it's been {milestone} days since we met")

        # birthday: a year older, whether you like it or not
        if (self.body.day % 365 == self.persona.get("birthday_day", -1)
                and self.persona.get("last_birthday_on") != self.body.day):
            self.persona["last_birthday_on"] = self.body.day
            self.persona["age"] = int(self.persona.get("age", 30)) + 1
            self._feel("joy", 0.6)
            self.memory.add("event", f"Today is my birthday. I'm "
                            f"{self.persona['age']} now.", self.body.sim_minutes,
                            importance=9)
            self._note_news(f"it's my birthday — I turned {self.persona['age']}")
            self.on_event(f"(it's {self.persona['name']}'s birthday — "
                          f"{self.persona['age']} today)")

        # old trivial days blur together (consolidation keeps memory human-sized)
        if self.body.day >= 8:
            summary = self.memory.consolidate(
                before_sim_minutes=self.body.sim_minutes - 7 * 1440)
            if summary:
                self.on_event(f"({summary})")

    # --------------------------------------------------------- sleep & dreams

    def _handle_sleep_transitions(self) -> None:
        """Detect falling asleep / waking and drive the cognition tied to them:
        dreaming (Walker's REM emotional processing), overnight consolidation,
        and morning planning (Park et al.)."""
        now, was, is_now = self.body.sim_minutes, self._was_asleep, self.body.asleep
        if is_now and not was:                    # just fell asleep
            self._sleep_started_sim = now
            self._dreamed_this_sleep = False
            if now - self._last_journal_sim >= 12 * 60:
                self._last_journal_sim = now
                self._begin_journal()
        elif is_now and not self._dreamed_this_sleep and self.llm_online:
            if now - self._sleep_started_sim >= 60:  # into deeper sleep -> a dream
                self._dreamed_this_sleep = True
                self._begin_dream()
        elif was and not is_now:                  # just woke
            self._evening_indulged = False
            slept = now - self._sleep_started_sim
            if slept >= MIN_RESTORATIVE_SLEEP:
                softened = self.memory.soften_emotional_charge(now)
                if softened:
                    self.on_event(f"({self.persona['name']} wakes lighter — the "
                                  "sharpest edges of yesterday have dulled overnight)")
                self._begin_planning()
        self._was_asleep = is_now

    def _begin_journal(self) -> None:
        """Before sleep, put the day into words (expressive writing)."""
        if not self.llm_online:
            return
        recent = self.memory.recent(25)
        text = "\n".join(f"- {m.text}" for m in recent)
        life_id = self._life_id

        def worker() -> None:
            try:
                entry = self.llm.journal(self.persona, text)
            except Exception:
                entry = None
            if not entry:
                return
            with self.lock:
                if life_id != self._life_id:
                    return
                self.memory.add("journal", entry, self.body.sim_minutes,
                                importance=6)

        threading.Thread(target=worker, daemon=True,
                         name="humanmade-journal").start()

    def _begin_dream(self) -> None:
        material = self.memory.emotional_material(self._sleep_started_sim - 900)
        if not material:
            return
        text = "\n".join(f"- {m.text}" for m in material)
        life_id = self._life_id

        def worker() -> None:
            try:
                dream = self.llm.dream(self.persona, text)
            except Exception:
                dream = None
            if not dream:
                return
            with self.lock:
                if life_id != self._life_id:
                    return
                self.last_dream = dream
                self.memory.add("dream", f"I dreamt: {dream}", self.body.sim_minutes,
                                importance=5)
                if not self.companion_present:
                    self._note_news(f"had a strange dream: {dream}")

        threading.Thread(target=worker, daemon=True, name="humanmade-dream").start()

    def _begin_planning(self) -> None:
        if self._planning or not self.llm_online:
            return
        self._planning = True
        context = self._perception()
        life_id = self._life_id
        day = self.body.day

        def worker() -> None:
            plan: list[str] = []
            try:
                plan = self.llm.plan_day(self.persona, context)
            except Exception:
                pass
            with self.lock:
                self._planning = False
                if life_id != self._life_id or not plan:
                    return
                self.today_plan = plan
                self.plan_day_index = day
                self.memory.add("plan", "Today I plan to: " + "; ".join(plan),
                                self.body.sim_minutes, importance=6)

        threading.Thread(target=worker, daemon=True, name="humanmade-plan").start()

    # ------------------------------------------------------------------- news

    def _note_news(self, text: str) -> None:
        """While the companion is away, save up notable happenings to share
        on their return (a bonded person keeps things to tell you)."""
        if self.companion_present:
            return
        boring = ("uses the toilet", "drinks a glass", "gets out of bed",
                  "climbs into bed", "still sleeping")
        if any(b in text for b in boring):
            return
        self.pending_news.append(text)
        self.pending_news = self.pending_news[-8:]

    # ---------------------------------------------------------------- thought

    def _perception(self) -> str:
        b = self.body
        valence, mood = b.mood()
        _, habit = CHRONOTYPES.get(self.persona.get("chronotype", "intermediate"),
                                   (0.0, ""))
        lines = [
            f"[{b.clock}, day {b.day}] You are {'asleep' if b.asleep else 'awake'}, "
            f"currently: {self.current_action}. Mood: {mood} ({valence:+.2f}). "
            f"You are {habit}.",
            f"Body — energy:{b.energy:.0f} hydration:{b.hydration:.0f} "
            f"satiety:{b.satiety:.0f} bladder:{b.bladder:.0f} bowel:{b.bowel:.0f} "
            f"hygiene:{b.hygiene:.0f} fun:{b.fun:.0f} social:{b.social:.0f} "
            f"health:{b.health:.0f} (all 0-100)",
        ]
        if self.emotion and self.emotion_intensity > 0.15:
            lines.append(f"A feeling of {self.emotion} sits with you "
                         f"({self.emotion_intensity:.1f}/1).")
        urgent = b.urgent_needs()
        if urgent:
            lines.append("URGENT: " + "; ".join(urgent))
        if (b.mood()[0] < -0.25 and b.satiety < 75 and b.satiety > 40
                and self.world.food_portions > 0):
            lines.append("Comfort food is calling. You're not really hungry, "
                         "but it might help.")
        if b.sickness > 0:
            lines.append(f"You are ill (severity {b.sickness:.0f}/100). Rest and "
                         "sleep help; medicine (12 credits) works fast, but only "
                         "your companion can order it — ask them.")
        lines.append(self.world.describe())

        if self.today_plan and self.plan_day_index == b.day:
            lines.append("Today's plan: " + "; ".join(self.today_plan))
        if self.world.weather == self.persona.get("loves_weather"):
            lines.append(f"It's {self.world.weather} — your favorite kind of sky.")
        elif self.world.weather == self.persona.get("hates_weather"):
            lines.append(f"It's {self.world.weather} — you've always hated "
                         f"{self.world.weather} days.")
        if self.last_dream and not b.asleep and b.awake_minutes < 120:
            lines.append(f"You just woke; last night you dreamt: {self.last_dream} "
                         "You might mention it if it feels worth sharing.")

        query = (" ".join(urgent) + " " + " ".join(self.inbox[-2:])).strip()
        query = query or "daily life companion"
        retrieved = self.memory.retrieve(query, b.sim_minutes, k=6,
                                         query_embedding=self._query_embedding(query))
        if retrieved:
            lines.append("Relevant memories:")
            lines += [f"  - ({m.kind}) {m.text}" for m in retrieved]

        lines += self._social_context()
        lines.append("What do you think, do, and (optionally) say?")
        return "\n".join(lines)

    def _social_context(self) -> list[str]:
        """Presence, separation, reunion, and the state of the bond."""
        b = self.body
        lines: list[str] = []
        days_known = (b.sim_minutes - self.bond["first_met_sim"]) / 1440
        level = self.bond_level()
        guidance = {
            "strangers, really": "Keep some guard up — small talk and light questions.",
            "acquaintances": "You can share opinions and everyday feelings.",
            "friends": "You can be honest about how you feel and ask personal things.",
            "genuinely close": "You trust them deeply — share what's really on your mind.",
        }[level]
        lines.append(f"You've known your companion {days_known:.0f} days "
                     f"(trust {self.bond['trust']:.0f}/100, closeness "
                     f"{self.bond['closeness']:.0f}/100 — {level}). {guidance}")
        known = self.memory.recent(5, kinds=("companion",))
        if known:
            lines.append("What you know about them: "
                         + "; ".join(m.text for m in known))
        if self.bond["closeness"] < 35 and b.mood()[0] < -0.3:
            lines.append("You feel low, but you don't know them well enough to "
                         "unload. If they ask how you are, you'd probably just "
                         "say you're fine.")
        if self._reunion_gap is not None:
            gap_h = self._reunion_gap / 60.0
            span = (f"{gap_h:.1f} hours" if gap_h >= 1 else
                    f"{self._reunion_gap:.0f} minutes")
            lines.append(f"YOUR COMPANION JUST CAME BACK after being away about "
                         f"{span}. You missed them. Greet them warmly (or however "
                         f"your mood dictates) and, if you like, tell them what "
                         f"happened while they were gone.")
            if self.pending_news:
                lines.append("While they were away: "
                             + "; ".join(self.pending_news))
            self._reunion_gap = None
            self.pending_news.clear()
        if (self.pending_question and self.companion_present
                and b.sim_minutes - self.pending_question["sim"] > 8 * 60):
            lines.append(f'Earlier you asked them: "{self.pending_question["text"]}" '
                         "— they never answered. You might gently follow up.")
            if not self.pending_question["hurt_felt"]:
                self.pending_question["hurt_felt"] = True
                self._feel("hurt", 0.4)
        if self.inbox:
            for msg in self.inbox:
                lines.append(f'Your companion just said: "{msg}"')
        elif not self.companion_present:
            apart = (b.sim_minutes - self.last_seen_sim) / 60.0
            lines.append(f"You are alone; your companion has been gone about "
                         f"{apart:.1f} hours. You can't speak to them until they "
                         f"return — but you can think, feel their absence, and live.")
        elif b.social < self.speak_up_threshold:
            lines.append("You haven't spoken to anyone in a long while. If you have "
                         "something to say or ask, say it — no one will prompt you.")
        return lines

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
                        self._embed_disabled = False  # maybe embeddings too
                        self._embed_fails = 0
                        self.on_event("(the higher mind flickers back online)")
        threading.Thread(target=probe, daemon=True,
                         name="humanmade-probe").start()

    @staticmethod
    def _compact_perception(perception: str) -> str:
        """History entry for a past thought cycle: keep the moment (time, mood)
        and what the companion said; drop stale vitals, memories, and world
        text. Small local models drown in repeated full-perception dumps."""
        keep = [line for line in perception.splitlines()
                if line.startswith(("[", "Your companion", "YOUR COMPANION",
                                    "URGENT:"))]
        return "\n".join(keep) or perception.split("\n", 1)[0]

    def _novelty(self, action: str) -> float:
        """Hedonic adaptation: the fourth book of the day restores less fun
        than the first. Returns a 0.4-1.0 multiplier and records the choice."""
        repeats = self._recent_actions.count(action)
        self._recent_actions.append(action)
        self._recent_actions = self._recent_actions[-4:]
        return max(0.4, 1.0 - 0.25 * repeats)

    def _survival_override(self, decision: dict) -> dict:
        """The brainstem vetoes a negligent cortex. If a need has reached a
        dying/bursting threshold and the mind chose something else, instinct
        forces the survival action — the mind's words and thoughts still
        stand, but the body acts. Without this, a dumb or distracted LLM can
        relax its way to death while URGENT flags scroll past it."""
        b = self.body
        needed: list[str] = []
        if b.hydration < 15:
            needed.append("drink")
        if b.bladder > 90 or b.bowel > 90:
            needed.append("toilet")
        if b.satiety < 12 and self.world.food_portions > 0:
            needed.append("eat")
        if (b.energy < 8 or b.sickness > 85) and not b.asleep:
            needed.append("sleep")
        if not needed or decision["action"] in needed:
            return decision
        forced = needed[0]
        self.on_event(f"(instinct overrides the mind — the body demands: {forced})")
        return {**decision, "action": forced, "_forced": True}

    def _temptation(self, decision: dict) -> dict:
        """Humans are reliably suboptimal in trait-consistent ways.

        Low conscientiousness shows up as bedtime procrastination ("one more
        chapter") and plan procrastination ("I'll start properly in a bit").
        """
        b = self.body
        give_in = 0.65 * (1.0 - self.plan_adherence)
        if (decision["action"] == "sleep" and not b.asleep and b.energy > 12
                and b.fun < 55 and not self._evening_indulged
                and random.random() < give_in):
            self._evening_indulged = True
            self.on_event(f"({self.persona['name']} means to sleep… "
                          "but, one more chapter)")
            return {**decision, "action": "relax"}
        if decision["action"] == "work" and random.random() < give_in * 0.6:
            self.on_event(f"({self.persona['name']} will start properly "
                          "in a bit. Honest.)")
            return {**decision, "action": "relax"}
        return decision

    def _apply_decision(self, decision: dict, perception: str, heard: bool) -> None:
        decision = self._survival_override(decision)
        if not decision.get("_forced"):
            decision = self._temptation(decision)
        self.conversation.append({"role": "user",
                                  "content": self._compact_perception(perception)})
        self.conversation.append({"role": "assistant", "content": json.dumps(decision)})
        self.conversation = self.conversation[-16:]

        if decision["thought"]:
            self.memory.add("thought", decision["thought"], self.body.sim_minutes,
                            importance=decision["importance"])
            if self.on_thought:
                self.on_thought(decision["thought"])

        note = decision.get("note_about_companion")
        if note:
            existing = {m.text.lower() for m in self.memory.recent(
                12, kinds=("companion",))}
            if note.lower() not in existing:
                self.memory.add("companion", note, self.body.sim_minutes,
                                importance=7)
                self._bond_adjust(closeness=1.0)  # learning someone = closeness

        if decision["say"]:
            if self.companion_present:
                self.on_speak(decision["say"])
                if "?" in decision["say"]:
                    self.pending_question = {"text": decision["say"],
                                             "sim": self.body.sim_minutes,
                                             "hurt_felt": False}
                self.memory.add("conversation", f'I said: "{decision["say"]}"',
                                self.body.sim_minutes, importance=decision["importance"])
            else:
                # no one is here to hear it — it becomes something wished-for
                self.memory.add("thought",
                                f'I wanted to tell them: "{decision["say"]}"',
                                self.body.sim_minutes, importance=decision["importance"])
                self._note_news(f'wanted to tell you: "{decision["say"]}"')
        elif heard:
            # companion spoke but the human chose silence — still worth noting
            self.memory.add("thought", "I heard them but didn't feel like answering.",
                            self.body.sim_minutes, importance=2)

        self._perform(decision["action"], forced=decision.get("_forced", False))

    # ----------------------------------------------------------------- action

    def _perform(self, action: str, forced: bool = False) -> None:
        b = self.body
        if b.asleep and action not in ("wake", "idle", "sleep"):
            if forced:
                # a survival need drags the body out of bed (thirst wakes you)
                b.wake_up()
                self.on_event(f"{self.persona['name']} wakes up.")
            else:
                # sleep inertia: the dreaming mind murmurs, the body sleeps on
                action = "idle"

        narration = None
        if action == "eat":
            if self.world.take_meal():
                comfort = b.satiety > 55 and b.mood()[0] < -0.15
                b.eat()
                if comfort:
                    b.fun = min(100.0, b.fun + 8)
                    narration = "raids the fridge for comfort food — it helps, a little"
                else:
                    narration = "eats a meal"
                if self.world.food_portions <= 2:
                    narration += f" — only {self.world.food_portions} portions left"
            else:
                narration = "opens the fridge… it's empty. Nothing to eat."
                self._feel("frustration", 0.7)
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
            b.fun = min(100.0, b.fun + 15 * self._novelty(action))
            narration = "works out with the dumbbells"
        elif action == "relax":
            gain = 25 * self._novelty(action)
            b.fun = min(100.0, b.fun + gain)
            narration = ("curls up with a book" if gain > 15 else
                         "flips through a book, restless — it's not landing today")
        elif action == "work":
            b.fun = min(100.0, b.fun + 10)
            wage = self.world.earn(1.0)
            hobby = self.persona["backstory"].split("spends time ")[-1].split(".")[0]
            narration = (f"works on {hobby} and earns {wage:.0f} credits "
                         f"({self.world.money:.0f} saved)")
            old = float(self.persona.get("hobby_progress", 0.0))
            new = old + random.uniform(1.0, 2.0)  # a real project takes weeks
            for milestone in (25, 50, 75):
                if old < milestone <= new:
                    self._feel("pride", 0.7)
                    self.on_event(f"({hobby} just passed {milestone}% — "
                                  "it's really taking shape)")
                    self.memory.add("event", f"My {hobby} passed {milestone}%. "
                                    "It's really taking shape.", b.sim_minutes,
                                    importance=7)
                    self._note_news(f"made real progress on {hobby} ({milestone}%)")
            if new >= 100:
                new = 0.0
                self._feel("pride", 1.0)
                self.on_event(f"({self.persona['name']} FINISHED {hobby}! "
                              "Already dreaming up the next one.)")
                self.memory.add("event", f"I FINISHED {hobby} today. I actually "
                                "finished it. Starting the next one soon.",
                                b.sim_minutes, importance=9)
                self._note_news(f"finished {hobby}!")
            self.persona["hobby_progress"] = new

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
            offset, _ = CHRONOTYPES.get(self.persona["chronotype"], (0.0, ""))
            self.body.chronotype_offset_hours = offset
            self._apply_disposition()
            self.conversation.clear()
            self.inbox.clear()
            self.current_action = "idle"
            self.action_minutes_left = 5.0
            self.minutes_since_decision = 0.0
            self._llm_fail_streak = 0
            self.today_plan = []
            self.plan_day_index = -1
            self.last_dream = None
            self._was_asleep = False
            self._sleep_started_sim = self.body.sim_minutes
            self._dreamed_this_sleep = False
            self._planning = False
            self._last_journal_sim = -10 ** 9
            self.companion_present = True
            self.last_seen_sim = self.body.sim_minutes
            self.pending_news.clear()
            self._reunion_gap = None
            self.bond = {"trust": 40.0, "closeness": 20.0,
                         "first_met_sim": self.body.sim_minutes,
                         "last_anniversary_days": 0.0}
            self._last_day = self.body.day
            self.pending_question = None
            self._recent_actions.clear()
            self.emotion, self.emotion_intensity = None, 0.0
            self._evening_indulged = False
            self.memory.set_meta("persona", json.dumps(self.persona))
            self.memory.add("event", f"{self.persona['name']} came into existence.",
                            self.body.sim_minutes, importance=10)
            self.save()

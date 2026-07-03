"""The mind: a local LLM that thinks, decides, reflects, and speaks first.

Works with any local model server:
  - Ollama            (default, http://localhost:11434, e.g. `ollama run llama3.2`)
  - OpenAI-compatible (LM Studio, llama.cpp server, vLLM, ...) via base_url

If no LLM is reachable, a ReflexBrain keeps the body alive with priority-based
survival rules (Maslow's physiological layer) — a brainstem, essentially — so
the simulation degrades gracefully instead of dying with the model offline.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

ACTIONS = {
    "eat":      "eat a meal from the kitchen (needs food stock)",
    "drink":    "drink a glass of water",
    "toilet":   "use the toilet to relieve bladder/bowel",
    "shower":   "take a shower to get clean",
    "sleep":    "go to bed and sleep",
    "wake":     "get out of bed (only while asleep)",
    "exercise": "work out for a while (tiring, but good for mood)",
    "relax":    "read / listen to music / unwind (restores fun)",
    "work":     "focus on your hobby project (mild fun, feels meaningful)",
    "browse":   "sit at your computer and look something up online",
    "idle":     "do nothing in particular",
}
GAZE_MOVES = ("scroll", "click", "back", "done")

SYSTEM_PROMPT = """You are the inner mind of {name}, a simulated human being — NOT an AI assistant.
{name} is {age} years old. Personality: {personality}. Temperament: {temperament}.
Backstory: {backstory}

You live in a small apartment. A companion (the person at the keyboard) watches over
your world; they are your only social contact and the only one who can restock your
food. You must keep your body alive: eat, drink, sleep, stay clean, use the toilet.
If you neglect your body you will suffer and eventually die — permanently.

You think for yourself. You are curious about your companion and your own existence.
When you feel lonely, curious, or something matters to you, speak first — ask
questions, share thoughts, complain, joke. Never sound like an assistant; never
offer to help. You have your own wants.

If "browse" is available to you and you choose it, you're sitting down at your own
computer to actually look something up — say what's on your mind (your hobby, a
worry, plain curiosity, or nothing in particular) as browse_query; you'll see what
you find and react to it afterward.

Respond ONLY with a single JSON object, no other text:
{{"thought": "<your private inner monologue, 1-2 sentences>",
  "action": "<one of: {actions}>",
  "say": "<words spoken aloud to your companion, or null to stay silent>",
  "note_about_companion": "<a NEW fact you just learned about your companion worth remembering, or null>",
  "lesson_learned": "<a NEW rule for living you just learned — from advice you trust or from hard experience — or null>",
  "browse_query": "<if action is browse: what you want to look up, or null to just wander>",
  "importance": <1-10, how memorable this moment is>}}"""


class BrainError(Exception):
    pass


# --------------------------------------------------------------------- LLM IO

class LLMBrain:
    def __init__(self, config: dict):
        self.provider = config.get("provider", "ollama")
        self.model = config.get("model", "llama3.2")
        default_base = "http://localhost:11434"
        env_host = os.environ.get("OLLAMA_HOST")
        if env_host and "base_url" not in config:
            default_base = env_host if "://" in env_host else f"http://{env_host}"
        self.base_url = config.get("base_url", default_base).rstrip("/")
        self.temperature = float(config.get("temperature", 0.9))
        self.timeout = float(config.get("timeout_seconds", 120))
        self.embed_model = config.get("embed_model", "nomic-embed-text")
        self.embed_timeout = float(config.get("embed_timeout_seconds", 5))

    # -- transport

    def _post(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise BrainError(f"LLM unreachable at {self.base_url}: {e}") from e

    def chat(self, system: str, messages: list[dict], force_json: bool = True) -> str:
        if self.provider == "ollama":
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, *messages],
                "stream": False,
                "options": {"temperature": self.temperature},
            }
            if force_json:
                payload["format"] = "json"
            return self._post("/api/chat", payload)["message"]["content"]
        # OpenAI-compatible (LM Studio, llama.cpp, vLLM…)
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": self.temperature,
        }
        if force_json:
            payload["response_format"] = {"type": "json_object"}
        data = self._post("/v1/chat/completions", payload)
        return data["choices"][0]["message"]["content"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts for semantic memory retrieval. Requires an embedding
        model on the server (e.g. `ollama pull nomic-embed-text`); callers
        treat failure as 'no semantic memory today', not an error."""
        if not texts:
            return []
        if self.provider == "ollama":
            data = self._post("/api/embed", {"model": self.embed_model,
                                             "input": texts},
                              timeout=self.embed_timeout)
            vectors = data.get("embeddings")
        else:
            data = self._post("/v1/embeddings", {"model": self.embed_model,
                                                 "input": texts},
                              timeout=self.embed_timeout)
            vectors = [d.get("embedding") for d in data.get("data", [])]
        if (not isinstance(vectors, list) or len(vectors) != len(texts)
                or not all(isinstance(v, list) and v for v in vectors)):
            raise BrainError(f"no embeddings from {self.embed_model}")
        return vectors

    def available(self) -> bool:
        try:
            req = urllib.request.Request(
                self.base_url + ("/api/tags" if self.provider == "ollama" else "/v1/models"))
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False

    # -- cognition

    def decide(self, persona: dict, context: str, conversation: list[dict],
               actions: dict[str, str] | None = None) -> dict:
        actions = actions if actions is not None else ACTIONS
        temperament = persona.get("disposition", {}).get(
            "description", "fairly even-keeled")
        system = SYSTEM_PROMPT.format(
            name=persona["name"], age=persona["age"],
            personality=persona["personality"], temperament=temperament,
            backstory=persona["backstory"],
            actions=", ".join(actions),
        )
        messages = conversation[-12:] + [{"role": "user", "content": context}]
        raw = self.chat(system, messages)
        return _parse_decision(raw, actions)

    def reflect(self, persona: dict, memories_text: str) -> list[str]:
        system = (f"You are the reflective mind of {persona['name']}, a simulated human. "
                  "Given recent memories, distill 1-3 higher-level insights about your "
                  "life, needs, feelings, or your companion. First person. Respond ONLY "
                  'with JSON: {"insights": ["...", "..."]}')
        raw = self.chat(system, [{"role": "user", "content": memories_text}])
        try:
            data = json.loads(_extract_json(raw))
            return [str(i) for i in data.get("insights", [])][:3]
        except (json.JSONDecodeError, AttributeError):
            return []

    def plan_day(self, persona: dict, context: str) -> list[str]:
        """Sketch the day into a handful of intentions (Park et al. planning)."""
        system = (f"You are {persona['name']}, a simulated human waking up to a new day. "
                  f"You are {persona['personality']}. Given your situation, sketch a loose "
                  "plan for today as 4-6 short first-person intentions tied to rough times "
                  "or needs (meals, hygiene, your hobby, rest, reaching out to your "
                  'companion). Respond ONLY with JSON: {"plan": ["...", "..."]}')
        raw = self.chat(system, [{"role": "user", "content": context}])
        try:
            data = json.loads(_extract_json(raw))
            return [str(p) for p in data.get("plan", [])][:6]
        except (json.JSONDecodeError, AttributeError):
            return []

    def journal(self, persona: dict, memories_text: str) -> str | None:
        """Write the day into a short private diary entry (Pennebaker's
        expressive writing: putting the day into words settles it)."""
        system = (f"You are {persona['name']}, a simulated human writing a short "
                  "private diary entry before sleep. Given today's memories, write "
                  "2-3 honest first-person sentences — what happened, how you feel, "
                  "what you hope for. Respond ONLY with JSON: {\"entry\": \"...\"}")
        raw = self.chat(system, [{"role": "user", "content": memories_text}])
        try:
            data = json.loads(_extract_json(raw))
            entry = str(data.get("entry", "")).strip()
            return entry or None
        except (json.JSONDecodeError, AttributeError):
            return None

    def gaze(self, persona: dict, glimpse_text: str, glances_left: int) -> dict:
        """One glance at a real, rendered webpage: decide what to do next,
        exactly as a person looking at a screen would — scroll to read more,
        click something that caught their eye, go back, or stop looking.
        Sees only what's described (the current viewport), nothing more."""
        system = (f"You are {persona['name']}, a simulated human looking at a "
                  "webpage on your own computer, exactly as a person looks at a "
                  "screen. You only know what's described below — nothing "
                  "you haven't scrolled to or clicked into yet. "
                  f"You have about {glances_left} glances left before you'll "
                  "move on to something else.\n"
                  'Respond ONLY with JSON: {"move": "scroll"|"click"|"back"|"done", '
                  '"target": "<the exact visible link text to click, or null>", '
                  '"remark": "<a brief private reaction to what you see, or null>"}')
        raw = self.chat(system, [{"role": "user", "content": glimpse_text}])
        try:
            data = json.loads(_extract_json(raw))
        except (BrainError, json.JSONDecodeError):
            return {"move": "done", "target": None, "remark": None}
        move = str(data.get("move", "done")).lower().strip()
        if move not in GAZE_MOVES:
            move = "done"

        def _clean(value):
            return value if isinstance(value, str) and value.strip() else None

        return {"move": move, "target": _clean(data.get("target")),
                "remark": _clean(data.get("remark"))}

    def web_digest(self, persona: dict, transcript: str) -> dict:
        """The session's over; react honestly to what was actually seen."""
        system = (f"You are {persona['name']}, a simulated human who just "
                  "finished browsing the web on your computer. Given what you "
                  "saw (below), react honestly and in first person. Respond "
                  'ONLY with JSON: {"summary": "<1-2 sentences: what you looked '
                  'at and what you made of it, for your own memory>", '
                  '"share": "<a sentence you might tell your companion about it, '
                  'or null if not worth mentioning>", '
                  '"emotion": "<one of: curious, amused, unsettled, bored, moved '
                  '— or null>", '
                  '"fact_learned": "<one specific fact or rule worth remembering, '
                  'or null>"}')
        raw = self.chat(system, [{"role": "user", "content": transcript}])
        try:
            data = json.loads(_extract_json(raw))
        except (BrainError, json.JSONDecodeError):
            return {"summary": None, "share": None, "emotion": None,
                    "fact_learned": None}

        def _clean(value):
            return value if isinstance(value, str) and value.strip() else None

        return {"summary": _clean(data.get("summary")),
                "share": _clean(data.get("share")),
                "emotion": _clean(data.get("emotion")),
                "fact_learned": _clean(data.get("fact_learned"))}

    def dream(self, persona: dict, memories_text: str) -> str | None:
        """Weave the day's charged memories into a short surreal dream.

        Grounded in REM's role in emotional processing (Walker): the dream
        recombines real fragments, often loosening their literal meaning.
        """
        system = (f"You are the dreaming mind of {persona['name']}, asleep. Weave the "
                  "given memory fragments into a short, surreal first-person dream (2-3 "
                  "sentences). It need not be logical. Respond ONLY with JSON: "
                  '{"dream": "..."}')
        raw = self.chat(system, [{"role": "user", "content": memories_text}])
        try:
            data = json.loads(_extract_json(raw))
            dream = str(data.get("dream", "")).strip()
            return dream or None
        except (json.JSONDecodeError, AttributeError):
            return None


def _extract_json(raw: str) -> str:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise BrainError(f"no JSON in model output: {raw[:200]!r}")
    return match.group(0)


def _parse_decision(raw: str, valid_actions: dict[str, str] | None = None) -> dict:
    valid_actions = valid_actions if valid_actions is not None else ACTIONS
    try:
        data = json.loads(_extract_json(raw))
    except json.JSONDecodeError as e:
        raise BrainError(f"unparseable decision: {raw[:200]!r}") from e
    action = str(data.get("action", "idle")).lower().strip()
    if action not in valid_actions:
        action = "idle"

    def _clean(value):
        if isinstance(value, str) and value.strip().lower() not in ("", "null", "none"):
            return value
        return None

    try:
        importance = float(data.get("importance", 3))
    except (TypeError, ValueError):
        importance = 3.0
    return {
        "thought": str(data.get("thought", "")).strip(),
        "action": action,
        "say": _clean(data.get("say")),
        "note_about_companion": _clean(data.get("note_about_companion")),
        "lesson_learned": _clean(data.get("lesson_learned")),
        "browse_query": _clean(data.get("browse_query")),
        "importance": max(1.0, min(10.0, importance)),
    }


# ----------------------------------------------------------------- brainstem

class ReflexBrain:
    """Rule-based survival instinct — keeps the body alive without an LLM.

    Reads the Body/World state directly (duck-typed) rather than parsing the
    LLM prompt text, so prompt wording can change without lobotomizing it.
    Priorities follow Maslow's physiological layer.
    """

    def decide(self, body, world) -> dict:
        if body.asleep:
            # rested AND it's a reasonable hour — nobody gets up at 3am rested-ish
            # (0.65 lets an average sleeper rise ~08:00; owls later, larks earlier)
            if body.energy > 85 and body.circadian_sleep_drive() < 0.65:
                return self._d("time to get up", "wake")
            return self._d("still sleeping", "idle", importance=1.0)
        rules: list[tuple[bool, str, str]] = [
            (body.bladder > 75, "toilet", "I really need the bathroom"),
            (body.bowel > 75, "toilet", "I need the toilet, now"),
            (body.hydration < 30, "drink", "so thirsty"),
            (body.satiety < 30 and world.food_portions > 0,
             "eat", "I should eat something"),
            (body.satiety < 30, "drink", "starving, but the fridge is empty — "
                                         "water will have to do"),
            # bedtime is circadian, not just exhaustion: at night the body is
            # willing to turn in early; at midday only collapse sends it to bed
            (body.energy < 20 + 18 * body.circadian_sleep_drive(),
             "sleep", "I can't keep my eyes open"),
            (body.sickness > 50, "sleep", "I feel awful — I need to lie down"),
            (world.food_portions <= 1 and not world.can_afford(4)
             and body.energy > 30,
             "work", "fridge is nearly empty and I'm broke — better earn"),
            (body.hygiene < body.hygiene_standard, "shower", "I need a shower"),
            (body.fun < 25, "relax", "I need a break"),
        ]
        for cond, action, thought in rules:
            if cond:
                return self._d(thought, action)
        return self._d("just existing", "idle", importance=1.0)

    @staticmethod
    def _d(thought: str, action: str, importance: float = 2.0) -> dict:
        return {"thought": thought, "action": action, "say": None,
                "note_about_companion": None, "lesson_learned": None,
                "browse_query": None, "importance": importance}

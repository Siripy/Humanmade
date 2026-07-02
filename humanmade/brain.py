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
    "idle":     "do nothing in particular",
}

SYSTEM_PROMPT = """You are the inner mind of {name}, a simulated human being — NOT an AI assistant.
{name} is {age} years old. Personality: {personality}. Backstory: {backstory}

You live in a small apartment. A companion (the person at the keyboard) watches over
your world; they are your only social contact and the only one who can restock your
food. You must keep your body alive: eat, drink, sleep, stay clean, use the toilet.
If you neglect your body you will suffer and eventually die — permanently.

You think for yourself. You are curious about your companion and your own existence.
When you feel lonely, curious, or something matters to you, speak first — ask
questions, share thoughts, complain, joke. Never sound like an assistant; never
offer to help. You have your own wants.

Respond ONLY with a single JSON object, no other text:
{{"thought": "<your private inner monologue, 1-2 sentences>",
  "action": "<one of: {actions}>",
  "say": "<words spoken aloud to your companion, or null to stay silent>",
  "importance": <1-10, how memorable this moment is>}}"""


class BrainError(Exception):
    pass


# --------------------------------------------------------------------- LLM IO

class LLMBrain:
    def __init__(self, config: dict):
        self.provider = config.get("provider", "ollama")
        self.model = config.get("model", "llama3.2")
        self.base_url = config.get("base_url", "http://localhost:11434").rstrip("/")
        self.temperature = float(config.get("temperature", 0.9))
        self.timeout = float(config.get("timeout_seconds", 120))

    # -- transport

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
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

    def available(self) -> bool:
        try:
            req = urllib.request.Request(
                self.base_url + ("/api/tags" if self.provider == "ollama" else "/v1/models"))
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False

    # -- cognition

    def decide(self, persona: dict, context: str,
               conversation: list[dict]) -> dict:
        system = SYSTEM_PROMPT.format(
            name=persona["name"], age=persona["age"],
            personality=persona["personality"], backstory=persona["backstory"],
            actions=", ".join(ACTIONS),
        )
        messages = conversation[-12:] + [{"role": "user", "content": context}]
        raw = self.chat(system, messages)
        return _parse_decision(raw)

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


def _extract_json(raw: str) -> str:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise BrainError(f"no JSON in model output: {raw[:200]!r}")
    return match.group(0)


def _parse_decision(raw: str) -> dict:
    try:
        data = json.loads(_extract_json(raw))
    except json.JSONDecodeError as e:
        raise BrainError(f"unparseable decision: {raw[:200]!r}") from e
    action = str(data.get("action", "idle")).lower().strip()
    if action not in ACTIONS:
        action = "idle"
    say = data.get("say")
    if isinstance(say, str) and say.strip().lower() in ("", "null", "none"):
        say = None
    try:
        importance = float(data.get("importance", 3))
    except (TypeError, ValueError):
        importance = 3.0
    return {
        "thought": str(data.get("thought", "")).strip(),
        "action": action,
        "say": say if isinstance(say, str) else None,
        "importance": max(1.0, min(10.0, importance)),
    }


# ----------------------------------------------------------------- brainstem

class ReflexBrain:
    """Rule-based survival instinct — keeps the body alive without an LLM."""

    def decide(self, persona: dict, context: str,
               conversation: list[dict]) -> dict:
        # crude parse of the vitals baked into context
        def val(name: str, default: float = 50.0) -> float:
            m = re.search(rf"{name}[:= ]+([\d.]+)", context, re.IGNORECASE)
            return float(m.group(1)) if m else default

        asleep = "asleep" in context.lower()
        rules: list[tuple[bool, str, str]] = [
            (asleep and val("energy") > 85, "wake", "time to get up"),
            (asleep, "idle", "still sleeping"),
            (val("bladder", 0) > 75, "toilet", "I really need the bathroom"),
            (val("bowel", 0) > 75, "toilet", "I need the toilet, now"),
            (val("hydration") < 30, "drink", "so thirsty"),
            (val("satiety") < 30, "eat", "I should eat something"),
            (val("energy") < 20, "sleep", "I can't keep my eyes open"),
            (val("hygiene") < 30, "shower", "I need a shower"),
            (val("fun") < 25, "relax", "I need a break"),
        ]
        for cond, action, thought in rules:
            if cond:
                return {"thought": thought, "action": action, "say": None,
                        "importance": 2.0}
        return {"thought": "just existing", "action": "idle", "say": None,
                "importance": 1.0}

    def reflect(self, persona: dict, memories_text: str) -> list[str]:
        return []

# Humanmade

A human simulator whose brain is a **local AI model**. A little person lives in a
small apartment inside your terminal. It must survive — eat, drink, sleep, breathe,
stay clean, use the toilet — and it thinks entirely for itself: it is **never
prompted by you**. On its own clock it perceives its body, remembers its life,
decides what to do, and when it feels lonely or curious it **talks to you first**.

If it neglects its body, it suffers. If it suffers too long, it dies — permanently.
Its memories survive between runs (and after death, as an archive).

```
· June drinks a glass of water.
(That's better. But the fridge is looking thin again.)
June: Hey — are you there? We're down to two portions of food.
      Also I've been meaning to ask: what's it like where you are?
you> _
```

## Quick start

Requires Python 3.10+ and **zero dependencies**. For the full mind, run any local
model server; without one, a rule-based "brainstem" keeps the body alive until the
model comes online.

```bash
# 1. (recommended) start a local model
ollama pull llama3.2 && ollama serve

# 2. run the simulator
python main.py
```

LM Studio / llama.cpp / vLLM work too — set `"provider": "openai"` and the
server's `base_url` in `config.json`.

| Command | Effect |
|---|---|
| *(any text)* | talk to the human |
| `/status` | vitals: health, hydration, bladder, heart rate, breathing… |
| `/memories [n]` | browse its memory stream |
| `/restock` | buy groceries (it depends on you for food!) |
| `/speed <n>` | sim-minutes per real second (default 1) |
| `/thoughts` | toggle the inner monologue |
| `/newlife` | after a death, a new person is born (old memories archived) |
| `/quit` | save and exit — it keeps existing between runs |

## How it works

```
        ┌───────────────────────────── every sim-tick ─────────────────────────────┐
        │                                                                          │
  ┌─────┴─────┐   perceives    ┌───────────┐   retrieves    ┌────────────────────┐ │
  │   BODY    │ ─────────────▶ │   BRAIN   │ ◀───────────── │       MEMORY       │ │
  │ body.py   │                │ brain.py  │                │ memory.py (SQLite) │ │
  │ hunger    │   acts         │ local LLM │   stores       │ observations       │ │
  │ thirst    │ ◀───────────── │ + reflex  │ ─────────────▶ │ thoughts           │ │
  │ sleep     │                │ fallback  │                │ conversations      │ │
  │ bladder…  │                └─────┬─────┘                │ reflections        │ │
  └─────┬─────┘        speaks first  │                      └────────────────────┘ │
        │              & answers     ▼                                             │
  ┌─────┴─────┐                ┌───────────┐                                       │
  │   WORLD   │                │    YOU    │ ──── talk / restock groceries ────────┘
  │ world.py  │                └───────────┘
  └───────────┘
```

- **Body** (`humanmade/body.py`) — a homeostasis engine with rates scaled from real
  physiology: ~3 days to die of thirst, weeks to starve, 16 h of wakefulness fills
  the sleep-pressure tank, bladder/bowel fill from intake, breathing and heart rate
  respond to exertion and stress. Ignore a full bladder and there *will* be an
  accident.
- **Brain** (`humanmade/brain.py`) — the local LLM is the *inner mind*: each thought
  cycle it receives its persona, vitals, mood, surroundings, retrieved memories, and
  anything you said, and returns `{thought, action, say, importance}`. `say` is how
  it initiates conversation — nothing forces it to speak, and nothing stops it.
  If the LLM is unreachable, a priority-rule **reflex brain** (a brainstem) takes
  over survival until the model returns.
- **Memory** (`humanmade/memory.py`) — a persistent memory stream. Retrieval scores
  every memory on **recency + importance + relevance** and feeds the best ones back
  into thought. Periodic **reflection** compresses recent memories into first-person
  insights ("my companion always comes back"), which are themselves memories. Tell
  it your name today; it can bring it up next week.
- **World** (`humanmade/world.py`) — water is on tap, but food is finite and only
  *you* can restock it, so the human has real, need-driven reasons to start
  conversations.

## Research notes

The design borrows from actual human-behavior literature:

- **Homeostatic drive theory** (Hull, 1943): behavior is energized by physiological
  deficits; each need here is a drive whose urgency grows until an action reduces it.
- **Maslow's hierarchy** (1943): the reflex brain and the LLM's "URGENT" channel
  prioritize physiological needs, then safety/hygiene, then belonging (the social
  need is what makes it seek you out).
- **Two-process sleep model** (Borbély, 1982): sleep pressure (Process S) accumulates
  with time awake and interacts with a circadian rhythm (Process C) that peaks at
  ~03:00 and dips mid-afternoon.
- **Generative agents** (Park et al., 2023): the memory-stream architecture —
  importance-weighted episodic records, recency/importance/relevance retrieval, and
  reflection into higher-level beliefs.
- **Basic emotion via appraisal** (simplified PAD): mood valence is derived from
  aggregate need satisfaction and colors every thought the brain has.

## State & persistence

Everything lives in `state/` (gitignored): `memory.sqlite3` holds the memory stream,
persona, body, and world. Delete the folder for a clean slate; past lives are
archived as `memory-<timestamp>.sqlite3` after death.

## Configuration (`config.json`)

```json
{
  "speed": 1.0,
  "llm": {
    "provider": "ollama",
    "base_url": "http://localhost:11434",
    "model": "llama3.2",
    "temperature": 0.9,
    "timeout_seconds": 120
  }
}
```

Set `"provider": "openai"` to use any OpenAI-compatible server (LM Studio,
llama.cpp, vLLM) at its `base_url`.

Small models (3B–8B) work fine — the JSON decision format is deliberately simple.
Better models produce a more interesting inner life.

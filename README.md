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
| `/away` | tell it you're leaving — it says goodbye, then waits |
| `/plan` | see the plan it sketched for today |
| `/dream` | recall what it dreamt last night |
| `/speed <n>` | sim-minutes per real second (default 1) |
| `/thoughts` | toggle the inner monologue |
| `/newlife` | after a death, a new person is born (old memories archived) |
| `/quit` | save and exit — it keeps existing between runs |

### Coming home to it

The intended loop: `/away` when you leave, then just type anything when you're
back. While you're gone the human keeps living on its own — it gets lonely, eats,
sleeps and dreams, and **saves up things to tell you**. When you return it notices
how long you were apart and greets you first, colored by its mood and what
happened while it waited (attachment theory, below). Leave it running overnight at
a low `/speed` and it'll have a day's worth of life — and a dream — to recount.

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
  accident. Each person has a **chronotype** (lark / owl / intermediate) that shifts
  their circadian curve, and a **mood with inertia** that lags behind their needs
  rather than snapping to them — a bad moment lingers, a good one carries.
- **Brain** (`humanmade/brain.py`) — the local LLM is the *inner mind*: each thought
  cycle it receives its persona, vitals, mood, surroundings, retrieved memories, and
  anything you said, and returns `{thought, action, say, importance}`. `say` is how
  it initiates conversation — nothing forces it to speak, and nothing stops it. On
  waking it **plans its day** into a few intentions; while asleep it **dreams**.
  Thinking, planning, dreaming and reflecting all run on their own threads against a
  snapshot of perception, so the body keeps living (in accurate ≤5-minute physiology
  steps) while a slow model deliberates. If the LLM is unreachable, a priority-rule
  **reflex brain** (a brainstem) takes over survival until the model returns.
- **Memory** (`humanmade/memory.py`) — a persistent memory stream. Retrieval scores
  every memory on **recency + importance + relevance** and feeds the best ones back
  into thought. Periodic **reflection** compresses recent memories into first-person
  insights ("my companion always comes back"). After a real night's sleep, **overnight
  consolidation** softens the emotional charge of the previous day's painful memories
  while keeping their content. Tell it your name today; it can bring it up next week.
- **World** (`humanmade/world.py`) — water is on tap, but food is finite and only
  *you* can restock it, so the human has real, need-driven reasons to start
  conversations.
- **Attachment** (`humanmade/agent.py`) — the human tracks whether you're present.
  When you leave it registers the separation, feels your absence, and queues up news;
  when you return it greets you first, its warmth scaled by how long you were gone.

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
- **Chronotype** (Roenneberg; Montaruli et al., 2021): larks and owls sit at
  different circadian phases, so each person's Process C is shifted, changing when
  they naturally wake, eat, and tire.
- **Generative agents** (Park et al., 2023): the memory-stream architecture —
  importance-weighted episodic records, recency/importance/relevance retrieval,
  reflection into higher-level beliefs, and **daily planning** decomposed into
  time/need-anchored intentions. Park found that removing reflection/planning made
  agents "degenerate to repetitive, context-free responses within 48 hours."
- **Basic emotion via appraisal** (simplified PAD): mood valence is derived from
  aggregate need satisfaction and colors every thought the brain has.
- **Emotional inertia** (Koval & Kuppens; Houben et al., 2015): emotions are
  autocorrelated over time, so mood is exponentially smoothed toward its target
  rather than tracking needs instantly — an upset lingers, a good mood carries.
- **REM as "overnight therapy"** (Walker & van der Helm, 2009; Berkeley, 2011): REM
  sleep replays emotional memories in a low-noradrenaline state, dampening their
  affective charge while preserving content. On waking, the day's painful memories
  have their retrieval weight softened — and the human recalls a dream stitched from
  those same fragments.
- **Attachment theory** (Bowlby; Ainsworth's Strange Situation): a bonded individual
  notices separation, keeps vigil for the caregiver's return, and shows a
  reunion-specific greeting. The human's absence-tracking and homecoming greeting
  model exactly this separation → waiting → reunion arc.

Sources: [Walker & van der Helm, "Overnight therapy?"](https://pubmed.ncbi.nlm.nih.gov/19702380/) ·
[Berkeley News on REM and painful memories](https://news.berkeley.edu/2011/11/23/dream-sleep/) ·
[Houben et al., emotion dynamics & well-being](https://ppw.kuleuven.be/okp/_pdf/Houben2015TRBST.pdf) ·
[Park et al., Generative Agents](https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763) ·
[Montaruli et al., chronotype & health](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8063933/) ·
[Ainsworth's Strange Situation](https://www.simplypsychology.org/mary-ainsworth.html)

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

## Development

Tests are stdlib-only (`unittest`) and need no real model — an in-process mock
Ollama server (`tests/mockllm.py`) stands in for the mind:

```bash
python -m unittest -v
```

The suite covers physiology (including death), big-tick integration, memory
retrieval and persistence, decision parsing, reflex priorities, asynchronous
thinking, LLM-failure fallback, reincarnation, and the behavior layer:
chronotype, emotional inertia, dreams, overnight consolidation, daily planning,
and the away/return reunion flow.

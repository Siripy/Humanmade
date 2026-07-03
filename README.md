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

# 2. (optional, for semantic memory) an embedding model
ollama pull nomic-embed-text

# 3. run the simulator — on first run you get to name your human
python main.py
```

LM Studio / llama.cpp / vLLM work too — set `"provider": "openai"` and the
server's `base_url` in `config.json`. `OLLAMA_HOST` is honored when
`base_url` isn't set.

| Command | Effect |
|---|---|
| *(any text)* | talk to the human |
| `/status` | vitals: health, hydration, bladder, heart rate, breathing… |
| `/memories [n]` | browse its memory stream |
| `/restock` | place a grocery order — paid from the *human's* credits |
| `/away` | tell it you're leaving — it says goodbye, then waits |
| `/plan` | see the plan it sketched for today |
| `/dream` | recall what it dreamt last night |
| `/journal [n]` | read its private diary (written at bedtime) |
| `/medicine` | order medicine when it's sick — paid from its credits |
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
  into thought. Relevance is **semantic** when a local embedding model is available
  (`ollama pull nomic-embed-text` — a background "hippocampus" thread embeds new
  memories, and recall then works by meaning, not word overlap), falling back to
  token overlap otherwise. Periodic **reflection** compresses recent memories into
  first-person insights ("my companion always comes back"). After a real night's
  sleep, **overnight consolidation** softens the emotional charge of the previous
  day's painful memories while keeping their content. Tell it your name today; it
  can bring it up next week.
- **World** (`humanmade/world.py`) — water is on tap, but food is finite and costs
  money. The human earns credits with its `work` action; only *you* can place the
  grocery order, and it spends *the human's* credits — so "I'm broke", "the fridge
  is empty", and "could you order food?" are all real conversations it has to start.
  **Weather** drifts between sun, cloud, rain, and storm — nudging mood and raising
  the odds of **catching an illness** when it's foul (worse with poor hygiene or
  exhaustion). Sickness means fever in the vitals, faster fatigue, and danger if
  ignored; rest and sleep heal it slowly, `/medicine` (its credits) works fast.
  Meanwhile its hobby genuinely progresses: `work` advances the novel/sketches/music
  through milestones it's proud to tell you about, and each night before sleep it
  writes a **private diary entry** you can read with `/journal`.
- **Attachment** (`humanmade/agent.py`) — the human tracks whether you're present.
  When you leave it registers the separation, feels your absence, and queues up news;
  when you return it greets you first, its warmth scaled by how long you were gone.
- **Personality** (`humanmade/personality.py`) — traits change the machinery, not
  just the prose: the anxious feel bad news harder and longer, extraverts drain
  their social battery faster, the meticulous shower sooner and stick to their
  plans, the dreamy procrastinate ("one more chapter…"). Each person also has a
  chronotype, a favorite and a hated kind of weather, and a birthday.
- **A relationship that develops** — trust and closeness grow with conversation and
  care (groceries, medicine) and erode with abandonment. Disclosure follows
  closeness: small talk from a stranger, real feelings from a friend. It keeps
  notes on what it learns about *you*, remembers questions you never answered and
  follows up, and marks anniversaries of the day you met. `/bond` shows where you
  stand.
- **Emotions, not just mood** — events are appraised into discrete feelings (pride
  at a milestone, gratitude for medicine, joy at reunion, shame after an accident,
  worry, frustration, hurt) that color its thoughts while they last, on top of the
  slow-moving mood. Repeated pleasures dull (hedonic adaptation), comfort eating
  happens on bad days, and old trivial memories blur together into summaries the
  way real weeks do.
- **It learns** — four ways. *Skills*: every meal, work session, and workout
  practices cooking/craft/fitness on a power-law curve; skill genuinely matters (a
  better cook gets more from a meal, mastery raises the wage and hobby speed,
  fitness resists illness) and rusts with disuse. *Lessons*: give it advice it
  trusts and it becomes a rule it lives by — persistent, deduplicated, and in front
  of the mind at every decision. *Consequences*: it tracks how activities have
  actually been leaving it feeling and notices patterns ("exercise has been leaving
  you feeling worse lately"). *Habits*: repeated behavior at consistent hours
  becomes its routine — a chronotype-true bedtime emerges within days, and `/habits`
  shows what it has settled into.

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
- **Weather and mood** (Denissen et al., 2008): daily weather has a real, modest
  effect on affect — here it shifts mood valence and, with poor self-care, raises
  illness risk.
- **Expressive writing** (Pennebaker): putting the day into words is how people
  process it — the nightly diary entry doubles as high-quality memory material for
  later reflection and retrieval.
- **Big Five trait theory**: personality dimensions modulate concrete parameters —
  neuroticism drives asymmetric mood reactivity (Koval's inertia work), extraversion
  the social drain, conscientiousness the hygiene standard and plan adherence.
- **Social penetration theory** (Altman & Taylor, 1973): relationships deepen
  through gradual, reciprocal self-disclosure — modeled as trust/closeness state
  that gates how much the human opens up.
- **Appraisal theory of emotion** (OCC model): emotions arise from evaluating
  events against goals — implemented as event-appraised discrete feelings with
  decaying intensity, layered over mood.
- **Hedonic adaptation** (Brickman & Campbell): repeated pleasures yield
  diminishing returns, pushing variety-seeking.
- **Power law of practice** (Newell & Rosenbloom, 1981): skill gains are steep for
  novices and slow toward mastery — the shape of every skill curve here.
- **Law of effect** (Thorndike): actions followed by bad outcomes are noticed and
  avoided — implemented as per-activity mood-outcome tracking surfaced to the mind.
- **Habit formation** (Lally et al., 2010): behaviors repeated in stable contexts
  become automatic over weeks — modeled as per-hour routine histograms that decay
  without reinforcement, with circadian-gated sleep so bedtimes phase-lock.

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
  "embed_interval_seconds": 5.0,
  "llm": {
    "provider": "ollama",
    "base_url": "http://localhost:11434",
    "model": "llama3.2",
    "embed_model": "nomic-embed-text",
    "temperature": 0.9,
    "timeout_seconds": 120,
    "embed_timeout_seconds": 5
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
thinking, LLM-failure fallback, reincarnation, the behavior layer (chronotype,
emotional inertia, dreams, overnight consolidation, daily planning, the
away/return reunion flow), the credit economy, semantic memory embeddings,
conversation compression, and first-run naming.

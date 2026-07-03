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

Requires Python 3.10+ and **zero required dependencies**. For the full mind, run
any local model server; without one, a rule-based "brainstem" keeps the body alive
until the model comes online. [Playwright](https://playwright.dev/) is an optional
extra, only needed if you turn on [web browsing](#seeing-the-web-optional-off-by-default).

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
| `/web [n]` | what it's read online recently (needs `internet.enabled`) |
| `/read` | the piece it's currently working on, and how far along it is |
| `/works` | the library of everything it's ever finished and sold |
| `/biography` | its whole life story so far, in its own words |
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

### Seeing the web (optional, off by default)

With `internet.enabled: true` in `config.json` and `pip install playwright &&
playwright install chromium`, the human can sit down at its own computer and
actually look things up — a real, rendered browser tab, not a text scrape. It sees
only what's inside a fixed **viewport**, exactly like looking at a screen: nothing
below the fold exists until it scrolls there, nothing behind a link exists until it
clicks. Each glance it decides, on its own, whether to scroll, click something that
caught its eye, go back, or stop looking — a **gaze loop**, budgeted by curiosity
(the more open its personality, the longer it'll wander). When it's done it reacts
honestly to what it saw: a memory forms, sometimes a fact becomes a **lesson**, an
emotion (curiosity, amusement, unease, boredom) colors its mood, and if something
was worth telling you it's saved as news for your next reunion. `/web` shows what
it's read.

Every session is a fresh, cookie-less tab, closed when it's done — it never
"remembers" being logged in anywhere because it never was. Safety is enforced at
the browser, not by asking the model nicely: every request but a plain `GET` is
aborted, downloads are cancelled, popups never open, and navigation outside the
configured allowlist (Wikipedia by default) never loads — the human can look, but
there is no "type" or "submit" anywhere in its vocabulary. See `humanmade/internet.py`
for the whole thing; `tests/test_internet.py` and `tests/test_browsing.py` exercise
it end to end against a local mock website, no real network required.

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
- **Real, accumulating creative work** (`humanmade/creation.py`) — the hobby isn't
  a progress bar. Each `work` session, the LLM writes an actual fragment — a novel
  chapter, a stargazing-log entry, a sketchbook page, a track, a translation, a
  carved figure — voiced at its *actual* craft skill (rough work reads rough) and
  colored by how it's actually been feeling. `/read` shows the work in progress;
  finishing one titles it, sells it for real credits, and archives it — `/works`
  is the library of everything it's ever finished. Files live in `state/works/`,
  genuinely readable outside the app.
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
- **Sight, optionally** (`humanmade/internet.py`) — with a real (local) browser
  behind it, the human can look things up online: a viewport-limited gaze loop that
  scrolls, clicks, and reads exactly as looking at a screen does, structurally
  read-only (GET requests only, no downloads, no popups, allowlisted navigation).
  See "Seeing the web" above.
- **A person who changes** (`humanmade/personality.py`) — the day-200 human isn't
  just carrying more memories than day 1; its actual temperament has moved. Once a
  day, how the day genuinely went (mean mood, real conversation, pride, shame)
  nudges its Big Five dimensions a hair — a lonely stretch raises neuroticism, rich
  conversation raises extraversion, pride over shame raises conscientiousness — then
  the concrete parameters (mood reactivity, social drain, plan adherence...) are
  re-derived so the shift actually changes behavior, not just the number. It's
  rate-limited per day and capped relative to who it was at birth, so this is
  drift, not a random walk into someone else. A separate **self-esteem** score
  (not a fixed trait — built by finished work and skill milestones, eroded by
  accidents) gates how easily it opens up: low self-esteem masks pain even with
  people it's fairly close to, high self-esteem opens up even to someone new.
  Reflection can also update its **self-view** — a first-person sentence of who it
  currently thinks it is, carried into every thought — and `/biography` asks the
  LLM to stitch the highest-importance memories of its whole life (plus how its
  own temperament has genuinely shifted since day one) into an honest first-person
  life story.

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
- **Foveated vision and eye-movement control** (Rayner, 2009): human reading and
  scene perception work through discrete fixations on a small high-resolution
  region, with the rest of a scene perceived only vaguely until attention moves
  there — the model for the browsing gaze loop's viewport-only sight, one glance
  (scroll, click, or back) at a time rather than reading a whole page at once.
- **Personality trait stability and change** (Bleidorn et al., 2021): traits are a
  stable foundation that nonetheless keeps moving across a life in response to
  real experience — the basis for identity drift being slow, bounded, and shaped
  by how days actually go, rather than either fixed forever or freely random.
- **Sociometer theory** (Leary, 1995): self-esteem functions as a gauge of
  relational/competence value that responds asymmetrically — a knock lowers it
  more than a win raises it — matching why shame here costs more self-esteem
  than an equivalent pride restores.

Sources: [Walker & van der Helm, "Overnight therapy?"](https://pubmed.ncbi.nlm.nih.gov/19702380/) ·
[Berkeley News on REM and painful memories](https://news.berkeley.edu/2011/11/23/dream-sleep/) ·
[Houben et al., emotion dynamics & well-being](https://ppw.kuleuven.be/okp/_pdf/Houben2015TRBST.pdf) ·
[Park et al., Generative Agents](https://dl.acm.org/doi/fullHtml/10.1145/3586183.3606763) ·
[Montaruli et al., chronotype & health](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8063933/) ·
[Ainsworth's Strange Situation](https://www.simplypsychology.org/mary-ainsworth.html) ·
[Rayner, eye movements & attention in reading/scene perception/visual search](https://pubmed.ncbi.nlm.nih.gov/19449261/) ·
[Bleidorn et al., Personality Trait Stability and Change](https://journals.sagepub.com/doi/10.5964/ps.6009) ·
[Leary, Sociometer theory](https://www.tandfonline.com/doi/abs/10.1080/10463280540000007)

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
  },
  "internet": {
    "enabled": false,
    "allowlist": ["en.wikipedia.org", "*.wikipedia.org"],
    "start_urls": [],
    "max_glances": 8,
    "cooldown_minutes": 90,
    "session_timeout_seconds": 45,
    "nav_timeout_seconds": 10,
    "viewport": {"width": 1280, "height": 800},
    "chromium_path": null
  }
}
```

Set `"provider": "openai"` to use any OpenAI-compatible server (LM Studio,
llama.cpp, vLLM) at its `base_url`.

Small models (3B–8B) work fine — the JSON decision format is deliberately simple.
Better models produce a more interesting inner life.

`internet.enabled` is `false` by default — see "Seeing the web" above. When on,
`allowlist` bounds where it can navigate (wildcards like `*.wikipedia.org` work),
`start_urls` are the "usual sites" it can wander to when it isn't looking for
anything in particular (otherwise a query becomes a Wikipedia search), and
`chromium_path` only needs setting if Playwright can't find its browser on its own.

## Development

Tests are stdlib-only (`unittest`) and need no real model or real internet — an
in-process mock Ollama server (`tests/mockllm.py`) stands in for the mind, and a
tiny local website (`tests/mockweb.py`) stands in for the web:

```bash
python -m unittest -v
```

The suite covers physiology (including death), big-tick integration, memory
retrieval and persistence, decision parsing, reflex priorities, asynchronous
thinking, LLM-failure fallback, reincarnation, the behavior layer (chronotype,
emotional inertia, dreams, overnight consolidation, daily planning, the
away/return reunion flow), the credit economy, semantic memory embeddings,
conversation compression, first-run naming, real creative-work sessions
(fragment writing, titling, completion and sale, cross-life archiving), identity
drift (bounded/rate-limited/origin-capped trait change, self-esteem, self-view,
biography synthesis), and — when [Playwright](https://playwright.dev/) is
installed — real browsing (viewport-only sight, link-following, scrolling, the
read-only safety layer, and the full browse-action lifecycle) against the local
mock website, with zero real network access.

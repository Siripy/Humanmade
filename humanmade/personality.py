"""Personality that actually changes the machinery, not just the prose.

Trait words map onto Big Five-ish dimensions (0..1, 0.5 = average), and the
dimensions derive concrete knobs:

  - neuroticism      -> negative feelings hit harder and fade slower
                        (asymmetric mood reactivity; Koval's inertia work)
  - extraversion     -> the social battery drains faster, and the urge to
                        speak up arrives sooner
  - conscientiousness-> personal hygiene standards, and how reliably today's
                        plan survives contact with the sofa
  - openness         -> hunger for novelty (idle boredom bites sooner)

Two humans with different traits now *live* differently, not just talk
differently.
"""

from __future__ import annotations

# deltas applied to a 0.5 baseline per dimension
TRAIT_EFFECTS: dict[str, dict[str, float]] = {
    "anxious":       {"neuroticism": +0.30},
    "melancholic":   {"neuroticism": +0.20, "extraversion": -0.10},
    "warm":          {"extraversion": +0.15, "neuroticism": -0.05},
    "playful":       {"extraversion": +0.20, "openness": +0.10},
    "sarcastic":     {"extraversion": +0.05, "neuroticism": +0.05},
    "curious":       {"openness": +0.25},
    "philosophical": {"openness": +0.20, "extraversion": -0.10},
    "dreamy":        {"openness": +0.15, "conscientiousness": -0.20},
    "meticulous":    {"conscientiousness": +0.30},
    "stubborn":      {"conscientiousness": +0.10, "neuroticism": +0.05},
    "impatient":     {"conscientiousness": -0.15, "neuroticism": +0.10},
    "gentle":        {"neuroticism": -0.10, "extraversion": -0.05},
}

_DIMENSIONS = ("neuroticism", "extraversion", "conscientiousness", "openness")


def derive_disposition(personality: str) -> dict:
    """Turn a comma-separated trait string into dimensions + a description."""
    dims = {d: 0.5 for d in _DIMENSIONS}
    for trait in (t.strip().lower() for t in personality.split(",")):
        for dim, delta in TRAIT_EFFECTS.get(trait, {}).items():
            dims[dim] += delta
    for d in _DIMENSIONS:
        dims[d] = max(0.05, min(0.95, dims[d]))
    return {"dimensions": dims, "description": describe(dims)}


def describe(dims: dict) -> str:
    """A short first-person temperament summary for the prompt."""
    parts = []
    n, e, c, o = (dims["neuroticism"], dims["extraversion"],
                  dims["conscientiousness"], dims["openness"])
    if n > 0.62:
        parts.append("feelings hit you hard and take a long time to fade")
    elif n < 0.38:
        parts.append("you are hard to rattle and quick to recover")
    if e > 0.62:
        parts.append("silence gets to you fast — you need people")
    elif e < 0.38:
        parts.append("you're comfortable with long silences")
    if c > 0.62:
        parts.append("you keep your space and your plans in order")
    elif c < 0.38:
        parts.append("plans are more like suggestions to you")
    if o > 0.62:
        parts.append("routine bores you quickly; you crave the new")
    return "; ".join(parts) if parts else "you are fairly even-keeled"


def body_knobs(dims: dict) -> dict:
    """Concrete physiology/behavior parameters from the dimensions."""
    return {
        # mood asymmetry: neurotic = bad news lands harder, recovery is slower
        "neg_reactivity": 0.6 + 1.1 * dims["neuroticism"],
        "pos_reactivity": 1.4 - 0.9 * dims["neuroticism"],
        # extraverts burn social charge faster
        "social_decay_mult": 0.7 + 0.7 * dims["extraversion"],
        # the hygiene level below which it *wants* a shower
        "hygiene_standard": 20.0 + 25.0 * dims["conscientiousness"],
        # boredom bites sooner in the novelty-hungry
        "fun_decay_mult": 0.85 + 0.4 * dims["openness"],
    }


def agent_knobs(dims: dict) -> dict:
    return {
        # probability a planned/dutiful action survives temptation
        "plan_adherence": 0.40 + 0.55 * dims["conscientiousness"],
        # social level at which "you might speak up" appears in perception
        "speak_up_threshold": 18.0 + 26.0 * dims["extraversion"],
    }

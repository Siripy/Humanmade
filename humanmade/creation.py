"""What a life adds up to: the hobby produces something real.

Every `work` session doesn't just move a progress bar — it writes an actual
fragment (a chapter, a journal entry, a carved figure, a few bars of music,
a translation attempt) into an accumulating Work, voiced by the LLM at the
human's actual craft level and colored by its actual recent mood. Finished
works are archived to disk as real, readable files and can be read back
(`/read`, `/works`) — a real record of what a life produced, not a number.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field

HOBBIES = ["writing a novel", "learning astronomy", "sketching birds",
           "composing chiptune music", "studying dead languages", "whittling"]

# hobby (as stored on the persona) -> what kind of thing comes out of it
HOBBY_KIND = {
    "writing a novel": "novel",
    "learning astronomy": "stargazing log",
    "sketching birds": "sketchbook",
    "composing chiptune music": "chiptune album",
    "studying dead languages": "translation notebook",
    "whittling": "carved collection",
}

# what one session's fragment is called, per kind — feeds the creation prompt
FRAGMENT_NOUN = {
    "novel": "chapter",
    "stargazing log": "log entry",
    "sketchbook": "page",
    "chiptune album": "track",
    "translation notebook": "entry",
    "carved collection": "figure",
}

# craft skill (0-100) -> how the work should actually read, ceiling-ordered
_CRAFT_DESCRIPTORS = (
    (15, "a rank beginner — simple, uneven, a little clumsy"),
    (40, "still learning — earnest but rough around the edges"),
    (65, "practiced — confident, finding a real voice"),
    (85, "skilled — assured, textured, distinctly theirs"),
    (101, "a master of it — effortless, the kind of work people notice"),
)


def craft_descriptor(level: float) -> str:
    for ceiling, text in _CRAFT_DESCRIPTORS:
        if level < ceiling:
            return text
    return _CRAFT_DESCRIPTORS[-1][1]


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"


@dataclass
class Work:
    hobby: str
    kind: str
    title: str | None            # assigned once the LLM names it
    synopsis: str
    fragments: list = field(default_factory=list)   # [{"day": int, "text": str}]
    status: str = "in_progress"  # "in_progress" | "finished"
    started_sim: float = 0.0
    finished_sim: float | None = None
    sale_price: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Work":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def display_title(self) -> str:
        return self.title or f"an untitled {self.kind}"


class WorksLibrary:
    """Reads/writes state/works/*.json — one file per work, in-progress or
    finished. Pure file I/O; the LLM call that fills a Work lives in
    brain.py, and the caller (agent.py) already serializes access via its
    own lock, so nothing here needs locking of its own."""

    def __init__(self, state_dir: str):
        self.dir = os.path.join(state_dir, "works")
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, slug: str) -> str:
        return os.path.join(self.dir, f"{slug}.json")

    def new_slug(self, hobby: str, started_sim: float) -> str:
        return f"{int(started_sim)}-{_slugify(hobby)}"

    def load(self, slug: str | None) -> Work | None:
        if not slug:
            return None
        path = self._path(slug)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return Work.from_dict(json.load(f))

    def save(self, work: Work, slug: str) -> None:
        with open(self._path(slug), "w") as f:
            json.dump(work.to_dict(), f, indent=2)

    def finished_works(self) -> list[Work]:
        works = []
        if not os.path.isdir(self.dir):
            return works
        for name in sorted(os.listdir(self.dir)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(self.dir, name)) as f:
                d = json.load(f)
            if d.get("status") == "finished":
                works.append(Work.from_dict(d))
        works.sort(key=lambda w: w.finished_sim or 0)
        return works

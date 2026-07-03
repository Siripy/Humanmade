"""Persistent memory stream, modeled on Park et al. 2023 ("Generative Agents").

Every experience — observations, thoughts, conversations, reflections — is a
timestamped record in SQLite, so memory genuinely survives across runs.

Retrieval scores each memory on three axes and returns the top-K:
    score = w_r * recency  +  w_i * importance  +  w_v * relevance
  - recency:    exponential decay over sim-hours since last access
  - importance: 1-10, heuristic (or LLM-rated when a brain is available)
  - relevance:  cosine similarity of embeddings when a local embedding model
                is available (semantic recall), token overlap otherwise

Reflection periodically compresses recent memories into higher-level insights
that are themselves stored with high importance — this is what lets the
simulated human form beliefs about its life instead of only recalling events.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass

RECENCY_DECAY = 0.995          # per sim-minute of unuse
W_RECENCY, W_IMPORTANCE, W_RELEVANCE = 1.0, 1.0, 1.4

_STOPWORDS = frozenset(
    "a an the i you it is are was were be been to of and or in on at for with "
    "my your me that this its as so but if then had has have do did not".split()
)


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOPWORDS}


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class Memory:
    id: int
    kind: str          # observation | thought | conversation | reflection | event
    text: str
    importance: float  # 1..10
    sim_minutes: float # sim time when formed
    created_at: float  # wall-clock, for the archaeology of past runs
    last_access: float # sim time when last retrieved


class MemoryStream:
    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")  # crash-safe, better concurrency
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                text TEXT NOT NULL,
                importance REAL NOT NULL,
                sim_minutes REAL NOT NULL,
                created_at REAL NOT NULL,
                last_access REAL NOT NULL,
                embedding TEXT
            )"""
        )
        try:  # migrate pre-embedding databases
            self.db.execute("ALTER TABLE memories ADD COLUMN embedding TEXT")
        except sqlite3.OperationalError:
            pass
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            )"""
        )
        self.db.commit()
        self._importance_since_reflection = 0.0

    # ------------------------------------------------------------------ meta

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.db.commit()

    # words that mark a memory as emotionally charged (importance + dreams +
    # overnight softening) or as a life milestone (importance only)
    _EMOTIONAL = ("lonely", "alone", "afraid", "scared", "fear", "love", "happy",
                  "hurt", "cried", "mortified", "miss", "hope", "empty",
                  "starving", "dehydrated", "died", "death", "accident")
    _MILESTONE = ("first", "never", "promise", "name", "born", "came into existence")

    # ----------------------------------------------------------------- write

    def add(self, kind: str, text: str, sim_minutes: float,
            importance: float | None = None) -> None:
        if importance is None:
            importance = self._heuristic_importance(kind, text)
        self.db.execute(
            "INSERT INTO memories(kind,text,importance,sim_minutes,created_at,last_access)"
            " VALUES(?,?,?,?,?,?)",
            (kind, text.strip(), importance, sim_minutes, time.time(), sim_minutes),
        )
        self.db.commit()
        self._importance_since_reflection += importance

    @classmethod
    def _heuristic_importance(cls, kind: str, text: str) -> float:
        base = {"reflection": 8.0, "conversation": 5.0, "event": 5.0,
                "thought": 3.0, "dream": 5.0, "plan": 6.0, "journal": 6.0,
                "observation": 2.0}.get(kind, 3.0)
        if any(w in text.lower() for w in cls._EMOTIONAL + cls._MILESTONE):
            base = min(10.0, base + 3.0)
        return base

    # -------------------------------------------------------------- retrieve

    def retrieve(self, query: str, now_sim_minutes: float, k: int = 8,
                 query_embedding: list[float] | None = None) -> list[Memory]:
        rows = self.db.execute(
            "SELECT id,kind,text,importance,sim_minutes,created_at,last_access,"
            "embedding FROM memories ORDER BY id DESC LIMIT 600"
        ).fetchall()
        if not rows:
            return []
        qtok = _tokens(query)
        scored: list[tuple[float, Memory]] = []
        for row in rows:
            m = Memory(*row[:7])
            recency = RECENCY_DECAY ** max(0.0, now_sim_minutes - m.last_access)
            emb_json = row[7]
            if query_embedding is not None and emb_json:
                # semantic recall: meaning, not word overlap
                relevance = max(0.0, _cosine(query_embedding, json.loads(emb_json)))
            else:
                mtok = _tokens(m.text)
                relevance = (len(qtok & mtok) / math.sqrt(len(qtok) + 1)
                             if qtok else 0.0)
            score = (W_RECENCY * recency
                     + W_IMPORTANCE * m.importance / 10.0
                     + W_RELEVANCE * min(1.0, relevance))
            scored.append((score, m))
        scored.sort(key=lambda s: s[0], reverse=True)
        top = [m for _, m in scored[:k]]
        ids = [m.id for m in top]
        self.db.execute(
            f"UPDATE memories SET last_access=? WHERE id IN ({','.join('?' * len(ids))})",
            [now_sim_minutes, *ids],
        )
        self.db.commit()
        return top

    def recent(self, n: int = 20, kinds: tuple[str, ...] | None = None) -> list[Memory]:
        q = ("SELECT id,kind,text,importance,sim_minutes,created_at,last_access "
             "FROM memories ")
        args: list = []
        if kinds:
            q += f"WHERE kind IN ({','.join('?' * len(kinds))}) "
            args += list(kinds)
        q += "ORDER BY id DESC LIMIT ?"
        args.append(n)
        rows = self.db.execute(q, args).fetchall()
        return [Memory(*r) for r in reversed(rows)]

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    # ------------------------------------------------------------- embeddings

    def unembedded(self, limit: int = 16) -> list[tuple[int, str]]:
        """Most recent memories that still lack an embedding vector."""
        return self.db.execute(
            "SELECT id,text FROM memories WHERE embedding IS NULL "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    def set_embedding(self, memory_id: int, vector: list[float]) -> None:
        self.db.execute("UPDATE memories SET embedding=? WHERE id=?",
                        (json.dumps(vector), memory_id))
        self.db.commit()

    # ------------------------------------------------------------ reflection

    def reflection_due(self, threshold: float = 60.0) -> bool:
        return self._importance_since_reflection >= threshold

    def mark_reflected(self) -> None:
        self._importance_since_reflection = 0.0

    # -------------------------------------------------------- sleep & dreaming

    def emotional_material(self, since_sim_minutes: float,
                           limit: int = 12) -> list[Memory]:
        """The day's most charged memories — raw material for a dream."""
        rows = self.db.execute(
            "SELECT id,kind,text,importance,sim_minutes,created_at,last_access "
            "FROM memories WHERE sim_minutes>=? ORDER BY importance DESC, id DESC "
            "LIMIT ?", (since_sim_minutes, limit)).fetchall()
        return [Memory(*r) for r in rows]

    def soften_emotional_charge(self, before_sim_minutes: float,
                                factor: float = 0.7) -> int:
        """REM 'overnight therapy' (Walker): after real sleep, dampen the
        importance of charged memories formed before waking, so their sting
        fades while the content is kept. Returns how many were softened."""
        rows = self.db.execute(
            "SELECT id,text,importance FROM memories WHERE sim_minutes<? "
            "AND importance>5", (before_sim_minutes,)).fetchall()
        softened = 0
        for mid, text, importance in rows:
            if any(w in text.lower() for w in self._EMOTIONAL):
                self.db.execute("UPDATE memories SET importance=? WHERE id=?",
                                (max(1.0, importance * factor), mid))
                softened += 1
        if softened:
            self.db.commit()
        return softened

    # ---------------------------------------------------------- consolidation

    def consolidate(self, before_sim_minutes: float,
                    min_batch: int = 40) -> str | None:
        """Human memory is not a tape: old trivial moments blur together.

        Low-importance observations/thoughts older than the cutoff are
        replaced by a single summary memory. Important, emotional, and
        relational memories are never touched. Returns the summary text, or
        None if there wasn't enough to blur.
        """
        rows = self.db.execute(
            "SELECT id, kind FROM memories WHERE sim_minutes < ? "
            "AND importance <= 3.5 AND kind IN ('observation', 'thought')",
            (before_sim_minutes,)).fetchall()
        if len(rows) < min_batch:
            return None
        counts: dict[str, int] = {}
        for _, kind in rows:
            counts[kind] = counts.get(kind, 0) + 1
        ids = [r[0] for r in rows]
        self.db.execute(
            f"DELETE FROM memories WHERE id IN ({','.join('?' * len(ids))})", ids)
        detail = " and ".join(f"{n} small {k}s" for k, n in sorted(counts.items()))
        summary = (f"an older stretch of days blurs together — {detail}, "
                   "mostly routine")
        self.add("summary", summary.capitalize() + ".", before_sim_minutes,
                 importance=4)
        self.db.commit()
        return summary

    def close(self) -> None:
        self.db.close()

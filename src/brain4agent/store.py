"""The store: one SQLite file, an in-memory BM25 index on top.

Why SQLite plus an in-memory index, rather than SQLite FTS5 alone:

FTS5's tokeniser cannot be replaced from Python -- registering a custom tokeniser
needs a C extension -- and the default `unicode61` treats a run of Han characters
as a single token, so 媒体投稿 never matches a query for 投稿. The workaround used
elsewhere is to store pre-spaced text, which then has to be maintained by
triggers and kept in step with every write.

An in-memory index sidesteps all of that: the text is stored verbatim, the index
is built at open time, and the ranking is the same BM25 the benchmark measured.
The trade is that memory grows with corpus size, which is fine for a personal
agent brain and is the point at which this design should be revisited rather
than patched.

Sessions are the unit of retrieval, not turns. An answer is often spread across
an exchange, and judging one turn in isolation produced far more false
abstentions in the benchmark.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from brain4agent.evidence import Evidence, JudgedResult, SearchResult, build_evidence
from brain4agent.tokenize import BM25

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
"""


@dataclass(frozen=True, slots=True)
class Memory:
    """One stored turn."""

    id: str
    session_id: str
    role: str
    content: str
    created_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class MemoryStore:
    """Add memories, search them, get evidence back.

    Typical use::

        store = MemoryStore("brain.db")
        sid = store.new_session()
        store.add("中文查询搜不到记忆", session_id=sid, role="user")
        store.add("根因在 keywords.py 第 443 行", session_id=sid, role="assistant")

        result = store.search("中文查询为什么搜不到")
        for e in result.evidence:
            print(e.memory.content, e.coverage, e.missing_terms)
    """

    def __init__(self, path: str | Path = "brain.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.executescript(SCHEMA)
        self._db.commit()
        self._index = BM25()
        self._by_session: dict[str, list[Memory]] = {}
        self._session_text: dict[str, str] = {}
        self._id_to_session: dict[str, str] = {}
        self._dirty = False
        self._reload()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _reload(self) -> None:
        rows = self._db.execute(
            "SELECT id, session_id, role, content, created_at FROM memories ORDER BY created_at"
        ).fetchall()
        self._by_session = {}
        self._session_text = {}
        self._id_to_session = {}
        items: list[tuple[str, str]] = []
        for mid, sid, role, content, created in rows:
            mem = Memory(id=mid, session_id=sid, role=role, content=content, created_at=created)
            self._by_session.setdefault(sid, []).append(mem)
            self._id_to_session[mid] = sid
            items.append((mid, content))

        # One lowered block per session, joined by newline so a term cannot match
        # across two turns by accident.
        for sid, turns in self._by_session.items():
            self._session_text[sid] = "\n".join(m.content.lower() for m in turns)

        self._index = BM25()
        self._index.index(items)
        self._dirty = False

    def _ensure_fresh(self) -> None:
        """Rebuild the index if writes happened since the last read.

        Rebuilding on every write made importing n turns cost O(n^2); deferring
        to the next read makes it O(n) once. Nothing is rebuilt for a batch of
        writes nobody reads.
        """
        if self._dirty:
            self._reload()

    # -- write -------------------------------------------------------------

    @staticmethod
    def new_session() -> str:
        return uuid.uuid4().hex[:12]

    def add(
        self,
        content: str,
        *,
        session_id: str | None = None,
        role: str = "user",
        timestamp: str | None = None,
    ) -> str:
        """Store one turn. Returns its id.

        A turn with no content is rejected rather than stored: an empty string
        would enter the index, match nothing, and inflate the apparent number of
        memories without ever being retrievable.
        """
        text = content.strip()
        if not text:
            raise ValueError("empty content")

        mid = uuid.uuid4().hex
        sid = session_id or self.new_session()
        ts = timestamp or _now()

        with self._db:
            self._db.execute(
                "INSERT INTO memories (id, session_id, role, content, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (mid, sid, role, text, ts),
            )

        mem = Memory(id=mid, session_id=sid, role=role, content=text, created_at=ts)
        self._by_session.setdefault(sid, []).append(mem)
        self._id_to_session[mid] = sid
        prev = self._session_text.get(sid)
        self._session_text[sid] = text.lower() if prev is None else f"{prev}\n{text.lower()}"
        # Deferred: the next read rebuilds once, so a bulk import does not rebuild
        # the whole index per turn.
        self._dirty = True
        return mid

    def add_exchange(
        self,
        *,
        user: str,
        assistant: str,
        session_id: str | None = None,
        timestamp: str | None = None,
    ) -> str:
        """Store a user/assistant pair as one session. Returns the session id."""
        sid = session_id or self.new_session()
        ts = timestamp or _now()
        self.add(user, session_id=sid, role="user", timestamp=ts)
        self.add(assistant, session_id=sid, role="assistant", timestamp=ts)
        return sid

    def forget_session(self, session_id: str) -> int:
        """Delete a session. Returns how many turns were removed."""
        with self._db:
            cur = self._db.execute("DELETE FROM memories WHERE session_id = ?", (session_id,))
        removed = cur.rowcount or 0
        if removed:
            self._reload()
        return removed

    # -- read --------------------------------------------------------------

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_session.values())

    @property
    def sessions(self) -> int:
        return len(self._by_session)

    def stats(self) -> dict[str, int]:
        self._ensure_fresh()
        return {
            "turns": len(self),
            "sessions": self.sessions,
            "index_terms": len(self._index._df),
        }

    def search(self, query: str, *, top_k: int = 5) -> SearchResult:
        """Retrieve sessions and report the checks that need no model.

        Returns evidence, not a verdict. Whether a session actually answers the
        question is the caller's call -- see docs/decisions.md D-011.
        """
        self._ensure_fresh()
        hits = self._index.search(query, top_k=top_k * 3)
        return build_evidence(
            query=query,
            hits=hits,
            by_session=self._by_session,
            session_text=self._session_text,
            id_to_session=self._id_to_session,
            sessions_searched=self.sessions,
            top_k=top_k,
        )

    def search_judged(
        self,
        query: str,
        judge: object,
        *,
        top_k: int = 5,
        max_checks: int = 3,
    ) -> JudgedResult:
        """Retrieve, then optionally ask a hosted model whether it answers.

        Three outcomes, and the third one is the point:

        - ``found``     -- the judge accepted some session
        - ``not_found`` -- every checked session was rejected
        - ``uncertain`` -- the judge failed for at least one of them

        Collapsing ``uncertain`` into ``not_found`` would turn a network outage
        into a confident "I never recorded this", which is the failure mode this
        whole project exists to prevent. Callers must handle all three.
        """
        result = self.search(query, top_k=top_k)
        if result.empty:
            return JudgedResult(result=result, kept=[], verdict="not_found", checks=0)

        kept: list[Evidence] = []
        failed = 0
        checks = 0
        for item in result.evidence[:max_checks]:
            checks += 1
            verdict = judge.answer(query, self.session_text(item.session_id))
            if verdict is True:
                kept.append(item)
                break
            if verdict is None:
                failed += 1

        if kept:
            outcome = "found"
        elif failed:
            outcome = "uncertain"
        else:
            outcome = "not_found"
        return JudgedResult(result=result, kept=kept, verdict=outcome, checks=checks)

    def session_text(self, session_id: str) -> str:
        """Whole session as one block, in order.

        The judge needs context, not a turn: an answer is often spread across an
        exchange, and judging single turns produced far more false abstentions
        in the benchmark.
        """
        turns = self.session(session_id)
        return "\n".join(f"{m.role}: {m.content}" for m in turns)

    def session(self, session_id: str) -> list[Memory]:
        return sorted(self._by_session.get(session_id, []), key=lambda m: m.created_at)

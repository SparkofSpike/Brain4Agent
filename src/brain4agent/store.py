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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from brain4agent.evidence import Evidence, JudgedResult, SearchResult, build_evidence
from brain4agent.tokenize import BM25, both_tokens, jieba_tokens, load_jieba, tokenize


def _tokenizers() -> tuple[Callable[[str], list[str]], Callable[[str], list[str]]]:
    """Index-side and query-side tokenisers, chosen by what is installed.

    With jieba: index with bigrams *and* words, query with words. Measured at
    hit@1 0.925 / MRR 0.940 against 0.825 / 0.891 for bigrams on both sides
    (bench/zh/indexing_probe.py).

    Without jieba: bigrams on both sides. Worse, but working -- which is the
    whole reason jieba is an extra and not a dependency.
    """
    if load_jieba() is None:
        return tokenize, tokenize
    return both_tokens, jieba_tokens


@lru_cache(maxsize=1)
def _cached_tokenizers() -> tuple[Callable[[str], list[str]], Callable[[str], list[str]]]:
    return _tokenizers()


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    tokens      TEXT NOT NULL DEFAULT ''
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

    def __init__(
        self,
        path: str | Path = "brain.db",
        *,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        self.path = Path(path)
        self._k1 = bm25_k1
        self._b = bm25_b
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.executescript(SCHEMA)
        self._migrate(self._db)
        self._db.commit()
        self._index = BM25()
        self._by_session: dict[str, list[Memory]] = {}
        self._session_text: dict[str, str] = {}
        self._id_to_session: dict[str, str] = {}
        self._dirty = False
        self._defer_commit = False
        self._reload()

    # -- lifecycle ---------------------------------------------------------

    @staticmethod
    def _migrate(db: sqlite3.Connection) -> None:
        """Add columns that older databases predate.

        ``tokens`` caches the index-side tokenisation. Without it every rebuild
        re-segments the whole corpus: 30 ms of jieba for a hundred-odd turns, and
        a rebuild happens after every write. Storing the result makes rebuilding
        a counting exercise instead.
        """
        cols = {row[1] for row in db.execute("PRAGMA table_info(memories)")}
        if "tokens" not in cols:
            db.execute("ALTER TABLE memories ADD COLUMN tokens TEXT NOT NULL DEFAULT ''")

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _reload(self) -> None:
        # ORDER BY rowid, not created_at: timestamps have second precision, so
        # everything written in the same second compares equal and SQLite is free
        # to return it in any order. rowid is the write order, which is the order
        # the caller meant.
        rows = self._db.execute(
            "SELECT id, session_id, role, content, created_at, tokens FROM memories ORDER BY rowid"
        ).fetchall()
        self._by_session = {}
        self._session_text = {}
        self._id_to_session = {}
        index_fn, query_fn = _cached_tokenizers()
        prepared: list[tuple[str, list[str]]] = []

        for mid, sid, role, content, created, tokens in rows:
            mem = Memory(id=mid, session_id=sid, role=role, content=content, created_at=created)
            self._by_session.setdefault(sid, []).append(mem)
            self._id_to_session[mid] = sid
            # Rows written before the tokens column existed get segmented once
            # here, then persisted on the next write of that row.
            prepared.append((mid, tokens.split() if tokens else index_fn(content)))

        # One lowered block per session, joined by newline so a term cannot match
        # across two turns by accident.
        for sid, turns in self._by_session.items():
            self._session_text[sid] = "\n".join(m.content.lower() for m in turns)

        self._index = BM25(k1=self._k1, b=self._b, index_fn=index_fn, query_fn=query_fn)
        self._index.index_tokens(prepared)
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
        index_fn, _ = _cached_tokenizers()
        tokens = " ".join(index_fn(text))

        sql = (
            "INSERT INTO memories (id, session_id, role, content, created_at, tokens)"
            " VALUES (?, ?, ?, ?, ?, ?)"
        )
        params = (mid, sid, role, text, ts, tokens)
        if self._defer_commit:
            # Inside batch(): defer the fsync. A thousand individual commits is
            # most of the cost of a thousand writes.
            self._db.execute(sql, params)
        else:
            with self._db:
                self._db.execute(sql, params)

        mem = Memory(id=mid, session_id=sid, role=role, content=text, created_at=ts)
        self._by_session.setdefault(sid, []).append(mem)
        self._id_to_session[mid] = sid
        prev = self._session_text.get(sid)
        self._session_text[sid] = text.lower() if prev is None else f"{prev}\n{text.lower()}"
        # Deferred: the next read rebuilds once, so a bulk import does not rebuild
        # the whole index per turn.
        self._dirty = True
        return mid

    @contextmanager
    def batch(self) -> Iterator[MemoryStore]:
        """Accumulate writes, commit once on exit.

        Importing 1,016 turns one commit at a time took 3.9 s; the same import
        inside a batch is bounded by tokenisation instead of by fsync.

            with store.batch():
                for turn in turns:
                    store.add(turn.text, session_id=turn.sid)
        """
        self._defer_commit = True
        try:
            yield self
        except BaseException:
            # A half-written batch is worse than none: roll back and rebuild the
            # in-memory state so it matches the database again.
            self._defer_commit = False
            self._db.rollback()
            self._reload()
            raise
        else:
            self._defer_commit = False
            self._db.commit()

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

    def prune(self, *, keep_sessions: int) -> int:
        """Delete the oldest sessions until at most `keep_sessions` remain.

        Forgetting is part of the design, not an apology: a memory store that only
        grows degrades, because every old near-duplicate competes with the answer
        (D-020). Retention is by write order, on the assumption that a user's
        recent work is what they ask about — a heuristic, and the simplest one
        that does not require scoring memories.

        Returns the number of sessions removed.
        """
        if keep_sessions < 1:
            raise ValueError("keep_sessions must be at least 1")
        if len(self._by_session) <= keep_sessions:
            return 0

        # Order sessions by their newest turn. rowid is write order and survives
        # same-second timestamps, which created_at does not.
        rows = self._db.execute(
            "SELECT session_id, MAX(rowid) AS last FROM memories GROUP BY session_id"
            " ORDER BY last DESC"
        ).fetchall()
        drop = [sid for sid, _ in rows[keep_sessions:]]

        with self._db:
            self._db.executemany("DELETE FROM memories WHERE session_id = ?", [(s,) for s in drop])
        self._reload()
        return len(drop)

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
        """Turns in the session, in the order they were written.

        No re-sorting: ``_by_session`` is already in rowid order, and sorting by
        a second-precision timestamp would shuffle turns written in the same
        second.
        """
        return list(self._by_session.get(session_id, []))

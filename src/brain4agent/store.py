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
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from brain4agent.evidence import Evidence, JudgedResult, SearchResult, build_evidence
from brain4agent.extract import Extractor
from brain4agent.simhash import DEFAULT_THRESHOLD, near_duplicate
from brain4agent.simhash import fingerprint as simhash_of
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


# A turn of conversation is not a document. Past this, something upstream is
# wrong (a whole file pasted in, a log dump) and a loud error beats a memory that
# silently costs seconds per write.
MAX_CONTENT_CHARS = 100_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    tokens      TEXT NOT NULL DEFAULT '',
    kind        TEXT NOT NULL DEFAULT 'raw',
    importance  REAL NOT NULL DEFAULT 0.5,
    entities    TEXT NOT NULL DEFAULT '',
    source_text TEXT,
    tags        TEXT NOT NULL DEFAULT '',
    superseded_by TEXT,
    refs        TEXT NOT NULL DEFAULT '',
    fingerprint INTEGER NOT NULL DEFAULT 0,
    accesses    INTEGER NOT NULL DEFAULT 0,
    weight      REAL NOT NULL DEFAULT 0.5
);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
"""


@dataclass(frozen=True, slots=True)
class Memory:
    """One stored item.

    ``content`` is what gets indexed and returned. For a raw turn it is the turn
    itself; for an extracted memory it is the distilled sentence, with the
    original turn kept in ``source_text`` for traceability.
    """

    id: str
    session_id: str
    role: str
    content: str
    created_at: str
    kind: str = "raw"
    importance: float = 0.5
    entities: tuple[str, ...] = ()
    source_text: str | None = None
    # Labels the caller attaches at write time: which scope a memory belongs to,
    # what kind of thing it is. Retrieval can filter on them without reading a
    # single word, which is how a bag of text learns to stay out of the results.
    tags: tuple[str, ...] = ()
    # Set when a later memory replaces this one. A superseded memory must not
    # surface: it is not wrong, it is out of date, and returning it silently is
    # how a memory system makes an agent confidently misinformed.
    superseded_by: str | None = None
    # Other memories this one points at. A relation memory is useless without its
    # participants: asking about a dependency returns the relation, and the thing
    # depended on is exactly what the asker needs next. Expansion is opt-in at
    # search time because it trades precision for completeness.
    refs: tuple[str, ...] = ()
    # SimHash of the content. Two memories with fingerprints a few bits apart are
    # probably saying the same thing; the store reports that, it does not act on
    # it. See simhash.py.
    fingerprint: int = 0
    # How many times a search has returned this memory. Recorded, not yet used for
    # ranking -- reinforcement needs evidence first, and there is none yet.
    accesses: int = 0
    # Reinforcement weight, 0..1. Starts at `importance`, rises a little each time
    # the memory is retrieved, falls when decay() runs. Retrieval does not use it
    # unless a caller passes `weight_blend`; the default path is unchanged.
    weight: float = 0.5


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
        extractor: Extractor | None = None,
    ) -> None:
        self.path = Path(path)
        self._k1 = bm25_k1
        self._b = bm25_b
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        # WAL: access counters are written on every search, and the default
        # rollback journal fsyncs the whole file per commit -- measured 2.5 ms
        # per single-row UPDATE, which dominated a 13 ms query. WAL makes the
        # same write ~0.05 ms. It also allows a reader and a writer at once,
        # which matters for an MCP server serving one agent while a CLI reads.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        self._migrate(self._db)
        self._db.commit()
        self._index = BM25()
        self._by_session: dict[str, list[Memory]] = {}
        self._session_text: dict[str, str] = {}
        self._id_to_session: dict[str, str] = {}
        self._dirty = False
        # A depth, not a flag: nested batch() blocks must not let the inner exit
        # commit what the outer one is still accumulating.
        self._defer_depth = 0
        self._data_version = 0
        self._extractor: Extractor | None = extractor
        self._extractor_resolved = extractor is not None
        self._reload()

    @property
    def _defer_commit(self) -> bool:
        return self._defer_depth > 0

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
        # Memory structure arrived later. Existing rows are raw turns with
        # neutral importance, which is exactly what they are.
        for column, ddl in (
            ("kind", "ALTER TABLE memories ADD COLUMN kind TEXT NOT NULL DEFAULT 'raw'"),
            ("importance", "ALTER TABLE memories ADD COLUMN importance REAL NOT NULL DEFAULT 0.5"),
            ("entities", "ALTER TABLE memories ADD COLUMN entities TEXT NOT NULL DEFAULT ''"),
            ("source_text", "ALTER TABLE memories ADD COLUMN source_text TEXT"),
            # Tags and supersession: a memory that predates them is unlabelled and
            # current, which is what an empty string and a NULL mean here.
            ("tags", "ALTER TABLE memories ADD COLUMN tags TEXT NOT NULL DEFAULT ''"),
            ("superseded_by", "ALTER TABLE memories ADD COLUMN superseded_by TEXT"),
            ("refs", "ALTER TABLE memories ADD COLUMN refs TEXT NOT NULL DEFAULT ''"),
            (
                "fingerprint",
                "ALTER TABLE memories ADD COLUMN fingerprint INTEGER NOT NULL DEFAULT 0",
            ),
            ("accesses", "ALTER TABLE memories ADD COLUMN accesses INTEGER NOT NULL DEFAULT 0"),
            ("weight", "ALTER TABLE memories ADD COLUMN weight REAL NOT NULL DEFAULT 0.5"),
        ):
            if column not in cols:
                db.execute(ddl)

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
            "SELECT id, session_id, role, content, created_at, tokens,"
            " kind, importance, entities, source_text, tags, superseded_by, refs,"
            " fingerprint, accesses, weight"
            " FROM memories ORDER BY rowid"
        ).fetchall()
        self._by_session = {}
        self._session_text = {}
        self._id_to_session = {}
        self._by_id = {}
        index_fn, query_fn = _cached_tokenizers()
        prepared: list[tuple[str, list[str]]] = []

        for (
            mid,
            sid,
            role,
            content,
            created,
            tokens,
            kind,
            importance,
            entities,
            source,
            tags,
            superseded,
            refs,
            fingerprint,
            accesses,
            weight,
        ) in rows:
            mem = Memory(
                id=mid,
                session_id=sid,
                role=role,
                content=content,
                created_at=created,
                kind=kind,
                importance=importance,
                entities=tuple(entities.split()) if entities else (),
                source_text=source,
                tags=tuple(tags.split()) if tags else (),
                superseded_by=superseded,
                refs=tuple(refs.split()) if refs else (),
                fingerprint=fingerprint & ((1 << 64) - 1),
                accesses=accesses,
                weight=weight,
            )
            self._by_session.setdefault(sid, []).append(mem)
            self._id_to_session[mid] = sid
            self._by_id[mid] = mem
            # Rows written before the tokens column existed get segmented once
            # here, then persisted on the next write of that row.
            prepared.append((mid, tokens.split() if tokens else index_fn(content)))

        # One lowered block per session, joined by newline so a term cannot match
        # across two turns by accident.
        for sid, turns in self._by_session.items():
            self._session_text[sid] = "\n".join(m.content.lower() for m in turns)

        self._index = BM25(k1=self._k1, b=self._b, index_fn=index_fn, query_fn=query_fn)
        self._index.index_tokens(prepared)
        self._data_version = self._db.execute("PRAGMA data_version").fetchone()[0]
        self._dirty = False

    def _ensure_fresh(self) -> None:
        """Rebuild the index if anything changed since the last read.

        Two sources of change, and the second was missing: this instance's own
        deferred writes, and writes made by *another* MemoryStore on the same
        file. PRAGMA data_version increments only when a different connection
        commits, which is exactly the signal needed -- an instance that never
        writes would otherwise serve a stale index forever while a sibling
        process updated the store.

        Rebuilding on every write made importing n turns cost O(n^2); deferring
        to the next read makes it O(n) once.
        """
        if self._dirty:
            self._reload()
            return
        version = self._db.execute("PRAGMA data_version").fetchone()[0]
        if version != self._data_version:
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
        kind: str = "raw",
        importance: float = 0.5,
        entities: tuple[str, ...] | list[str] = (),
        source_text: str | None = None,
        index_source: bool = False,
        tags: tuple[str, ...] | list[str] = (),
        superseded_by: str | None = None,
        refs: tuple[str, ...] | list[str] = (),
        index_text: str | None = None,
    ) -> str:
        """Store one turn. Returns its id.

        A turn with no content is rejected rather than stored: an empty string
        would enter the index, match nothing, and inflate the apparent number of
        memories without ever being retrievable.

        ``index_text`` replaces the text that goes into the matching index, while
        ``content`` remains what is stored and returned. This exists because a
        long, explanatory document and a question rarely share their important
        words: the explanation contains every ordinary word of its domain, and
        BM25 rewards those crowded matches over the one distinctive name. Measured
        on PrecisionMemBench: indexing the full belief text scores 0.6104;
        indexing only its canonical name and aliases scores 0.7922. Keep it None
        to index what you store.

        ``tags`` are labels the caller supplies -- scope, domain, whatever the
        caller knows and the text does not say. Retrieval can filter on them, and
        that filter is the only way a text index learns to withhold something that
        is textually similar but contextually wrong. Tags must not contain spaces.

        ``superseded_by`` links this memory to the one that replaces it. Prefer
        ``supersede()``, which keeps both ends consistent.
        """
        text = content.strip()
        if not text:
            raise ValueError("empty content")
        if len(text) > MAX_CONTENT_CHARS:
            raise ValueError(
                f"content too long: {len(text)} chars (max {MAX_CONTENT_CHARS}); "
                "split it into turns rather than storing a document"
            )

        mid = uuid.uuid4().hex
        sid = session_id or self.new_session()
        ts = timestamp or _now()
        index_fn, _ = _cached_tokenizers()
        # index_source keeps the original turn in the matching index as well as
        # in source_text. A distilled memory is easier to read but can wash out
        # the words the question used; indexing both gives the query two ways in.
        base = index_text if index_text is not None else text
        indexed = f"{base} {source_text}" if index_source and source_text else base
        tokens = " ".join(index_fn(indexed))
        tag_text = " ".join(t.strip() for t in tags if t.strip())
        for t in tags:
            if " " in t.strip():
                raise ValueError(f"tag must not contain spaces: {t!r}")

        # Fingerprint the stored text, not the index text. Two memories that say
        # the same thing should collide even if one is indexed more narrowly.
        #
        # SimHash is a 64-bit *unsigned* value and SQLite's INTEGER is signed, so
        # anything with the top bit set overflows on insert. Store it as a two's
        # complement signed value and mask it back on read; the fingerprint itself
        # is unchanged.
        fp = simhash_of(text)
        fp_signed = fp - (1 << 64) if fp >= 1 << 63 else fp
        clamped_importance = max(0.0, min(1.0, importance))

        sql = (
            "INSERT INTO memories"
            " (id, session_id, role, content, created_at, tokens,"
            " kind, importance, entities, source_text, tags, superseded_by, refs,"
            " fingerprint, accesses, weight)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = (
            mid,
            sid,
            role,
            text,
            ts,
            tokens,
            kind,
            max(0.0, min(1.0, importance)),
            " ".join(entities),
            source_text,
            tag_text,
            superseded_by,
            " ".join(refs),
            fp_signed,
            0,
            clamped_importance,
        )
        if self._defer_commit:
            # Inside batch(): defer the fsync. A thousand individual commits is
            # most of the cost of a thousand writes.
            self._db.execute(sql, params)
        else:
            with self._db:
                self._db.execute(sql, params)

        mem = Memory(
            id=mid,
            session_id=sid,
            role=role,
            content=text,
            created_at=ts,
            kind=kind,
            importance=clamped_importance,
            entities=tuple(entities),
            source_text=source_text,
            tags=tuple(t.strip() for t in tags if t.strip()),
            superseded_by=superseded_by,
            refs=tuple(refs),
            fingerprint=fp,
            accesses=0,
            weight=clamped_importance,
        )
        self._by_session.setdefault(sid, []).append(mem)
        self._id_to_session[mid] = sid
        self._by_id[mid] = mem
        prev = self._session_text.get(sid)
        self._session_text[sid] = text.lower() if prev is None else f"{prev}\n{text.lower()}"
        # Deferred: the next read rebuilds once, so a bulk import does not rebuild
        # the whole index per turn.
        self._dirty = True
        return mid

    def _resolve_extractor(self) -> Extractor | None:
        """Look up the extractor once, from the environment.

        Deferred rather than done in __init__ so constructing a store never
        touches the ambient configuration. A caller that wants the model-free
        path gets it by simply never calling remember().
        """
        if not self._extractor_resolved:
            self._extractor = Extractor.from_env()
            self._extractor_resolved = True
        return self._extractor

    def remember_exchange(
        self,
        user: str,
        assistant: str,
        *,
        session_id: str | None = None,
        extractor: Extractor | None = None,
        index_source: bool = False,
        tags: tuple[str, ...] | list[str] = (),
    ) -> list[str]:
        """Distil a question and its answer together.

        Preferred over ``remember(assistant)``: the question holds the vocabulary
        a future search will use. Extracting answers alone was measured to drop
        hit@1 from 0.925 to 0.475, because 63 of 127 turns -- almost every
        question -- were judged "nothing durable" and thrown away (D-023).
        """
        question = user.strip()
        answer = assistant.strip()
        if not answer:
            raise ValueError("empty answer")

        ext = extractor if extractor is not None else self._resolve_extractor()
        if ext is None:
            sid = session_id or self.new_session()
            return [
                self.add(question, session_id=sid, role="user"),
                self.add(answer, session_id=sid, role="assistant"),
            ]

        result = ext.extract(answer, question=question)
        if not result.ok:
            sid = session_id or self.new_session()
            return [
                self.add(question, session_id=sid, role="user"),
                self.add(answer, session_id=sid, role="assistant"),
            ]
        if not result.memories:
            return []

        sid = session_id or self.new_session()
        out: list[str] = []
        for memory in result.memories:
            out.append(
                self.add(
                    memory.content,
                    session_id=sid,
                    role="exchange",
                    kind=memory.kind,
                    importance=memory.importance,
                    entities=memory.entities,
                    source_text=f"Q: {question}\nA: {answer}",
                    index_source=index_source,
                    tags=tags,
                )
            )
        return out

    def remember(
        self,
        content: str,
        *,
        session_id: str | None = None,
        role: str = "user",
        extractor: Extractor | None = None,
        tags: tuple[str, ...] | list[str] = (),
    ) -> list[str]:
        """Store a turn by distilling it into durable memories.

        Three outcomes, and the difference is the point:

        - the extractor produced memories -> those are stored, each with the
          original turn kept as ``source_text``
        - the extractor ran and found nothing durable -> **nothing is stored**;
          acknowledgements and restatements should not occupy the index
        - the extractor failed or is not configured -> the raw turn is stored,
          so an outage degrades quality rather than losing data

        Returns the ids written, which may be empty.
        """
        text = content.strip()
        if not text:
            raise ValueError("empty content")

        ext = extractor if extractor is not None else self._resolve_extractor()
        if ext is None:
            return [self.add(text, session_id=session_id, role=role, tags=tags)]

        result = ext.extract(text)
        if not result.ok:
            return [self.add(text, session_id=session_id, role=role, tags=tags)]
        if not result.memories:
            return []

        sid = session_id or self.new_session()
        ids: list[str] = []
        for memory in result.memories:
            ids.append(
                self.add(
                    memory.content,
                    session_id=sid,
                    role=role,
                    kind=memory.kind,
                    importance=memory.importance,
                    entities=memory.entities,
                    source_text=text,
                    tags=tags,
                )
            )
        return ids

    @contextmanager
    def batch(self) -> Iterator[MemoryStore]:
        """Accumulate writes, commit once on exit.

        Importing 1,016 turns one commit at a time took 3.9 s; the same import
        inside a batch is bounded by tokenisation instead of by fsync.

            with store.batch():
                for turn in turns:
                    store.add(turn.text, session_id=turn.sid)
        """
        self._defer_depth += 1
        try:
            yield self
        except BaseException:
            self._defer_depth -= 1
            if self._defer_depth == 0:
                # A half-written batch is worse than none: roll back and rebuild
                # the in-memory state so it matches the database again.
                self._db.rollback()
                self._reload()
            raise
        else:
            self._defer_depth -= 1
            if self._defer_depth == 0:
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

    def reinforce(self, memory_id: str, *, delta: float = 0.03) -> float | None:
        """Raise a memory's weight a little. Returns the new weight, or None.

        This is what `accesses` is for. Counting retrievals without letting them
        affect anything is bookkeeping; reinforcement is the point of the count,
        and pruning is the point of the weight.

        Capped at 1.0. A memory retrieved a thousand times is not ten times more
        relevant than one retrieved a hundred times, and unbounded growth would
        make the number impossible to interpret later.
        """
        self._ensure_fresh()
        mem = self._by_id.get(memory_id)
        if mem is None:
            return None
        new_weight = min(1.0, mem.weight + delta)
        with self._db:
            self._db.execute("UPDATE memories SET weight = ? WHERE id = ?", (new_weight, memory_id))
        self._by_id[memory_id] = dc_replace(mem, weight=new_weight)
        return new_weight

    def decay(self, *, factor: float = 0.95, floor: float = 0.0) -> int:
        """Multiply every weight by ``factor``. Returns how many rows changed.

        Explicit, not time-based. A time-based decay has to decide what "age"
        means for a memory written once and read fifty times, and nothing here
        measures which answer is right. An explicit pass the caller schedules is
        honest about not knowing: run it weekly, after a sprint, or never.

        ``floor`` clamps the result so repeated passes cannot drive the whole
        corpus to zero and leave nothing worth keeping.
        """
        if not 0.0 < factor <= 1.0:
            raise ValueError(f"factor must be in (0, 1], got {factor}")
        self._ensure_fresh()
        changed = 0
        with self._db:
            for mid, mem in self._by_id.items():
                new_weight = max(floor, mem.weight * factor)
                if new_weight != mem.weight:
                    self._db.execute(
                        "UPDATE memories SET weight = ? WHERE id = ?", (new_weight, mid)
                    )
                    changed += 1
        if changed:
            self._refresh_weights()
        return changed

    def _refresh_weights(self) -> None:
        """Re-read weights into memory without rebuilding the search index.

        A weight change touches no token, so the index is still valid; only the
        cached Memory objects need the new number.
        """
        rows = self._db.execute("SELECT id, weight FROM memories").fetchall()
        for mid, weight in rows:
            mem = self._by_id.get(mid)
            if mem is not None and mem.weight != weight:
                self._by_id[mid] = dc_replace(mem, weight=weight)

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

    def prune_by_weight(self, *, floor: float = 0.05, dry_run: bool = False) -> int:
        """Delete memories whose weight has fallen below ``floor``.

        The counterpart to ``decay``. ``prune(keep_sessions=N)`` forgets by write
        order, which D-022 measured as useless on a noisy corpus: it needs to
        delete 95% of everything before retrieval recovers, which is emptying
        rather than forgetting. This forgets by evidence instead -- a memory that
        was never retrieved and has decayed repeatedly is the one worth dropping,
        and one that keeps being retrieved never reaches the floor.

        ``dry_run`` reports the count without deleting. Deleting is irreversible
        and this is the kind of operation that should be looked at once before it
        runs.

        Returns the number of memories (not sessions) removed or that would be.
        """
        self._ensure_fresh()
        victims = [mid for mid, mem in self._by_id.items() if mem.weight < floor]
        if dry_run or not victims:
            return len(victims)
        with self._db:
            self._db.executemany("DELETE FROM memories WHERE id = ?", [(m,) for m in victims])
        self._reload()
        return len(victims)

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

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        tags: Sequence[str] | None = None,
        kinds: Sequence[str] | None = None,
        include_superseded: bool = False,
        expand_refs: bool = False,
        weight_blend: float = 0.0,
    ) -> SearchResult:
        """Retrieve sessions and report the checks that need no model.

        Returns evidence, not a verdict. Whether a session actually answers the
        question is the caller's call -- see docs/decisions.md D-011.

        The three filters exist because text similarity alone cannot tell a memory
        that *should* be withheld from one that *should* be returned. ``tags`` keeps
        only memories carrying at least one of the given labels -- scope is caller
        knowledge, not something the words contain. ``kinds`` restricts to given
        kinds. Superseded memories are excluded unless ``include_superseded``.

        ``expand_refs`` appends the memories a hit points at. Off by default: it
        trades precision for completeness, and the caller is the only one who
        knows which they need right now.

        ``weight_blend`` mixes reinforcement weight into the ranking. 0.0 leaves
        BM25 untouched, which is the default and what every published number was
        measured with. 1.0 scales each score by the memory's weight relative to
        0.5. Intermediate values interpolate. It is opt-in because nothing here
        has measured that weight helps ranking -- only that pruning can use it.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        self._ensure_fresh()

        filtering = tags is not None or kinds is not None or not include_superseded or expand_refs
        if filtering:
            # Filtering after a fixed top_k*3 window silently empties the result
            # whenever the filter rejects most of the window. Ranking the whole
            # matching set costs little at this corpus size and keeps the filter
            # truthful.
            raw = self._index.search(query, top_k=len(self._index))
            hits = []
            seen: set[str] = set()
            for mid, score in raw:
                # A hit on a superseded memory is redirected to its replacement,
                # not dropped. Dropping was the first implementation and it lost
                # the answer entirely: "main store" matched the old MySQL memory,
                # the new PostgreSQL memory never says "main store", and the
                # search returned nothing at all. The score stays the old one's
                # because the old one is what matched.
                target = mid if include_superseded else self._current(mid)
                if target is None or target in seen:
                    continue
                if not self._visible(target, tags, kinds, include_superseded):
                    continue
                seen.add(target)
                hits.append((target, score))
        else:
            hits = self._index.search(query, top_k=top_k * 3)

        if expand_refs and hits:
            # Truncate to the requested number *first*, then expand: a relation's
            # participants are appended to the result, they do not compete with it
            # for one of the top_k slots. Expanding before truncating would let a
            # high-scoring relation push out everything else and then keep its
            # refs anyway.
            hits = self._with_refs(hits[:top_k], tags, kinds, include_superseded)
            return self._annotate(
                build_evidence(
                    query=query,
                    hits=hits,
                    by_session=self._by_session,
                    session_text=self._session_text,
                    id_to_session=self._id_to_session,
                    sessions_searched=self.sessions,
                    top_k=len(hits),
                )
            )

        hits = self._blend_weights(hits, weight_blend)
        return self._annotate(
            build_evidence(
                query=query,
                hits=hits,
                by_session=self._by_session,
                session_text=self._session_text,
                id_to_session=self._id_to_session,
                sessions_searched=self.sessions,
                top_k=top_k,
            )
        )

    def duplicates_of(self, memory_id: str, *, threshold: int = DEFAULT_THRESHOLD) -> list[str]:
        """Memories that may be saying the same thing as ``memory_id``.

        A linear scan. At a few thousand memories that is well under a
        millisecond, and a SimHash bucket index would be the wrong first move --
        the point is to notice, not to scale.

        Superseded memories are excluded: a replacement resembling the thing it
        replaced is the intended outcome, not a duplicate.
        """
        self._ensure_fresh()
        mem = self._by_id.get(memory_id)
        if mem is None or not mem.fingerprint:
            return []
        # A replaced memory has no duplicates to report: it is not the current
        # statement of anything, and its resemblance to its own replacement is the
        # intended outcome rather than a duplicate to flag.
        if mem.superseded_by:
            return []
        out: list[str] = []
        for other in self._by_id.values():
            if other.id == memory_id or not other.fingerprint or other.superseded_by:
                continue
            if near_duplicate(mem.fingerprint, other.fingerprint, threshold):
                out.append(other.id)
        return out

    def _blend_weights(
        self, hits: list[tuple[str, float]], weight_blend: float
    ) -> list[tuple[str, float]]:
        """Scale BM25 scores by reinforcement weight. No-op at 0.0.

        Weight 0.5 is neutral: a memory at the default weight keeps its score
        exactly. Below 0.5 shrinks, above 0.5 grows. That makes the parameter
        interpretable -- 0.5 is "do nothing" -- and means a corpus that never
        called reinforce() is unaffected even at blend 1.0.
        """
        if weight_blend <= 0.0:
            return hits
        out: list[tuple[str, float]] = []
        for mid, score in hits:
            mem = self._by_id.get(mid)
            if mem is None:
                out.append((mid, score))
                continue
            factor = 1.0 + weight_blend * (mem.weight / 0.5 - 1.0)
            out.append((mid, score * max(0.0, factor)))
        out.sort(key=lambda kv: -kv[1])
        return out

    def _annotate(self, result: SearchResult) -> SearchResult:
        """Attach duplicate hints and bump access counts.

        Access counts are written here rather than in a background pass because
        this is the only moment the store knows a memory was useful. The in-memory
        copy is left alone: updating it would mark the index dirty and rebuild on
        the next read, which costs far more than a stale counter on an object
        nothing ranks by yet.
        """
        if not result.evidence:
            return result

        # Duplicate detection is deliberately absent from this path. It is a
        # linear scan over the corpus -- 5 ms against 2893 memories, more than
        # half the cost of a query that otherwise takes 4 ms -- and it answers a
        # maintenance question, not a retrieval one. Callers that want it call
        # duplicates_of(id) directly; no MCP tool exposes it.
        enriched = list(result.evidence)

        # Credit the turn that matched, not the session's first turn. Evidence
        # carries `memory` = the first turn so a caller has something readable
        # to show, and an earlier version counted every retrieval against it:
        # a query matching only the second turn of a session incremented the
        # first turn and left the matching one at zero.
        ids = [((e.matched_memory_id or e.memory.id),) for e in result.evidence]
        with self._db:
            self._db.executemany("UPDATE memories SET accesses = accesses + 1 WHERE id = ?", ids)
        return dc_replace(result, evidence=enriched)

    def _with_refs(
        self,
        hits: list[tuple[str, float]],
        tags: Sequence[str] | None,
        kinds: Sequence[str] | None,
        include_superseded: bool,
    ) -> list[tuple[str, float]]:
        """Append referenced memories after the hits that point at them.

        The ref inherits the score of the memory that named it. It came in through
        that memory, not through its own match, and inventing a score for it would
        make it look independently retrieved.
        """
        seen = {mid for mid, _ in hits}
        out = list(hits)
        for mid, score in hits:
            mem = self._by_id.get(mid)
            if mem is None:
                continue
            for ref in mem.refs:
                if ref in seen:
                    continue
                if not self._visible(ref, tags, kinds, include_superseded):
                    continue
                seen.add(ref)
                out.append((ref, score))
        return out

    def link(self, from_id: str, to_ids: Sequence[str]) -> int:
        """Point one memory at others. Returns how many links were accepted.

        Used for relations: a memory saying "auth depends on Redis" is far more
        useful if asking about it also surfaces the Redis memory. Unknown target
        ids are skipped rather than raising, because links are often built while
        the target is still being written and a missing one is not a bug -- but
        the count tells the caller what actually landed.
        """
        self._ensure_fresh()
        if from_id not in self._by_id:
            return 0
        current = list(self._by_id[from_id].refs)
        fresh = [t for t in to_ids if t != from_id and t in self._by_id and t not in current]
        if not fresh:
            return 0

        with self._db:
            self._db.execute(
                "UPDATE memories SET refs = ? WHERE id = ?",
                (" ".join(current + fresh), from_id),
            )
        self._dirty = True
        return len(fresh)

    def _current(self, memory_id: str) -> str | None:
        """Follow the supersession chain to the memory that is current now.

        Returns None on a cycle. A corrupt database should make a search return
        less, never hang.
        """
        seen: set[str] = set()
        mid = memory_id
        while True:
            if mid in seen:
                return None
            seen.add(mid)
            mem = self._by_id.get(mid)
            if mem is None:
                return None
            if not mem.superseded_by:
                return mid
            mid = mem.superseded_by

    def _visible(
        self,
        memory_id: str,
        tags: Sequence[str] | None,
        kinds: Sequence[str] | None,
        include_superseded: bool,
    ) -> bool:
        mem = self._by_id.get(memory_id)
        if mem is None:
            return True
        if not include_superseded and mem.superseded_by:
            return False
        if kinds is not None and mem.kind not in kinds:
            return False
        # An untagged memory has no scope, so no scope can exclude it. Only a
        # memory that *claims* a scope can be filtered out by one -- otherwise the
        # universal beliefs become invisible to every query that is not itself
        # universal, which is the opposite of universal.
        out_of_scope = tags is not None and mem.tags and not (set(tags) & set(mem.tags))
        return not out_of_scope

    def memory(self, memory_id: str) -> Memory | None:
        """One memory by id, or None. Triggers a reload if the store changed.

        Access counts are read from the database here rather than trusted from the
        in-memory copy. `_annotate` bumps them on every search but deliberately
        does not rebuild the index to reflect that, so the cached object lags by
        exactly the reads that just happened -- which makes a counter look broken
        the moment anyone checks it.
        """
        self._ensure_fresh()
        mem = self._by_id.get(memory_id)
        if mem is None:
            return None
        row = self._db.execute(
            "SELECT accesses FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is not None and row[0] != mem.accesses:
            return dc_replace(mem, accesses=row[0])
        return mem

    def supersede(self, old_id: str, new_id: str) -> bool:
        """Mark ``old_id`` as replaced by ``new_id``.

        Returns False when either id is unknown, rather than raising: a caller
        tidying up after a crash should not have to distinguish "already gone"
        from "never existed". The pair is checked so a typo cannot orphan a
        memory by pointing at nothing.
        """
        self._ensure_fresh()
        if old_id not in self._by_id or new_id not in self._by_id:
            return False
        if old_id == new_id:
            raise ValueError("a memory cannot supersede itself")

        # The replacement inherits the old memory's index terms. Redirecting a hit
        # (D-027) recovers the session but leaves coverage near zero, because the
        # new text genuinely lacks the old question's words -- "main store is
        # MySQL" replaced by "migrated to PostgreSQL" still does not contain
        # "main store". Copying the terms fixes the match, not just the routing.
        #
        # Terms, not content: the old row's token column already carries anything
        # it inherited, so a chain A -> B -> C passes A's vocabulary through C
        # without re-segmenting the whole history each time.
        row = self._db.execute("SELECT tokens FROM memories WHERE id = ?", (old_id,)).fetchone()
        inherited = (row[0] if row else "").split()
        if inherited:
            cur = self._db.execute("SELECT tokens FROM memories WHERE id = ?", (new_id,)).fetchone()
            existing = (cur[0] if cur else "").split()
            # dict.fromkeys keeps first-seen order and drops duplicates, so the
            # inherited terms do not multiply the row's length on repeated links.
            merged = " ".join(dict.fromkeys([*existing, *inherited]))
            with self._db:
                self._db.execute("UPDATE memories SET tokens = ? WHERE id = ?", (merged, new_id))

        with self._db:
            self._db.execute("UPDATE memories SET superseded_by = ? WHERE id = ?", (new_id, old_id))
        self._dirty = True
        return True

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
            try:
                verdict = judge.answer(query, self.session_text(item.session_id))
            except Exception:  # noqa: BLE001 - see below
                # A judge is an optional enhancement. Any fault in it degrades to
                # "no judgement", never to a failed search and never to "not
                # found". The whole three-state design exists so that a judge
                # that cannot answer is visibly uncertain rather than invisible.
                verdict = None
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

    def refresh(self) -> None:
        """Reload from the database, picking up writes by another instance.

        search() and stats() already check PRAGMA data_version and reload when a
        sibling connection has committed. __len__ deliberately does not: the
        pragma costs ~55us, which is fine once per query and not fine once per
        len(). Call this if you need to observe a sibling's writes without
        querying.
        """
        self._reload()

    def _by_session_memory(self, memory_id: str) -> Memory | None:
        """Look up one memory by id across all sessions.

        Linear, and only used by the CLI to pretty-print what it just wrote;
        anything on a hot path should use the index instead.
        """
        for turns in self._by_session.values():
            for mem in turns:
                if mem.id == memory_id:
                    return mem
        return None

    def session(self, session_id: str) -> list[Memory]:
        """Turns in the session, in the order they were written.

        No re-sorting: ``_by_session`` is already in rowid order, and sorting by
        a second-precision timestamp would shuffle turns written in the same
        second.

        Refreshes first. A long-lived MCP server holds one store while writes
        from the CLI or a skill land in the same file; without this, b4a_session
        would answer from a snapshot taken at startup.
        """
        self._ensure_fresh()
        return list(self._by_session.get(session_id, []))

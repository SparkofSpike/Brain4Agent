"""Load benchmark data into one shape, so any memory system can be scored by it.

Two datasets are supported:

- ``longmemeval_oracle`` — the evidence sessions only (1-2 per question).
  Retrieval is trivial here; it exists to check that the *reader* works.
- ``longmemeval_s_cleaned`` — the same questions over a realistic haystack with
  distractor sessions. This is the one that measures retrieval.

The dataclasses below are the seam: a memory system only ever sees
``MemoryItem`` lists and ``Query`` objects, never dataset-specific JSON. That is
what lets a future Chinese corpus be scored by the same harness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

ORACLE = "longmemeval_oracle.json"
S_CLEANED = "longmemeval_s_cleaned.json"


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """One unit that the system under test may store.

    Granularity matters: the paper's own finding is that *session decomposition*
    improves retrieval, so a session is not necessarily one item. This loader
    keeps one item per message and records the session it came from, leaving
    the decomposition decision to the system being measured.
    """

    id: str
    content: str
    session_id: str
    timestamp: str
    role: str

    @property
    def turn_index(self) -> int:
        return int(self.id.rsplit(":", 1)[-1])


@dataclass(frozen=True, slots=True)
class Query:
    """A question plus the ground truth needed to score an answer."""

    id: str
    text: str
    question_type: str
    answer: str
    question_date: str
    gold_session_ids: frozenset[str]
    is_abstention: bool = False


@dataclass(slots=True)
class Task:
    """One question over one haystack.

    LongMemEval is 500 independent tasks, not one shared corpus: each question
    has its own history. A harness that treats them as one library would let
    cross-question leakage inflate the numbers.
    """

    query: Query
    memories: list[MemoryItem] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.memories:
            raise ValueError(f"task {self.query.id} has no memories")


def _flatten_sessions(raw: dict) -> list[MemoryItem]:
    """Turn the nested session/message JSON into flat memory items."""
    items: list[MemoryItem] = []
    sessions = raw["haystack_sessions"]
    dates = raw["haystack_dates"]
    session_ids = raw["haystack_session_ids"]

    for session, date, sid in zip(sessions, dates, session_ids, strict=True):
        for turn, msg in enumerate(session):
            content = msg.get("content", "")
            if not isinstance(content, str) or not content.strip():
                # Some turns are image placeholders with no text. Dropping them
                # is honest: a text-only memory system cannot use them, and
                # keeping empty strings would make every retriever look good.
                continue
            items.append(
                MemoryItem(
                    id=f"{sid}:{turn}",
                    content=content.strip(),
                    session_id=sid,
                    timestamp=date,
                    role=msg.get("role", ""),
                )
            )
    return items


def load_longmemeval(name: str = S_CLEANED, limit: int | None = None) -> list[Task]:
    """Load LongMemEval into ``Task`` objects.

    ``limit`` truncates the task list, not the haystack of each task.

    Abstention questions are marked by an ``_abs`` suffix on the question id,
    not by ``question_type`` -- they are spread across four of the six types. A
    dataset reader that only looked at ``question_type`` would silently drop the
    one capability this project cares most about.
    """
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run bench/fetch_data.py first.")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if limit is not None:
        raw = raw[:limit]

    tasks: list[Task] = []
    for entry in raw:
        query = Query(
            id=entry["question_id"],
            text=entry["question"],
            question_type=entry["question_type"],
            answer=entry["answer"],
            question_date=entry.get("question_date", ""),
            gold_session_ids=frozenset(entry.get("answer_session_ids", [])),
            is_abstention=entry["question_id"].endswith("_abs"),
        )
        tasks.append(Task(query=query, memories=_flatten_sessions(entry)))
    return tasks


def summarize(tasks: list[Task]) -> dict[str, object]:
    """Shape of the loaded data: for sanity-checking a download, not for scoring."""
    by_type: dict[str, int] = {}
    for t in tasks:
        by_type[t.query.question_type] = by_type.get(t.query.question_type, 0) + 1

    turns = [len(t.memories) for t in tasks]
    missing_gold = sum(
        1 for t in tasks if not any(m.session_id in t.query.gold_session_ids for m in t.memories)
    )
    return {
        "tasks": len(tasks),
        "abstention": sum(1 for t in tasks if t.query.is_abstention),
        "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "turns_min": min(turns) if turns else 0,
        "turns_max": max(turns) if turns else 0,
        "turns_mean": round(sum(turns) / len(turns), 1) if turns else 0,
        "tasks_missing_gold": missing_gold,
    }

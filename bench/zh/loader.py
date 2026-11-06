"""Load the Chinese evaluation set.

The set is stored as one shared corpus plus a question list, which is both
smaller and more honest than repeating the corpus inside every question. This
loader expands it into the same ``Task`` objects the rest of the harness uses,
sharing one ``MemoryItem`` list across tasks -- the objects are immutable, so
sharing is safe and avoids copying 60 times.

Abstention is ``answer_session_ids == []``, not a suffix on the id. The id
suffix was a LongMemEval convention; here the source of truth is the gold set
itself, because a question that points at nothing *is* the definition.
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.loader import MemoryItem, Query, Task

ZH_EVAL = Path(__file__).parent / "zh_eval.json"


def load_zh(path: Path = ZH_EVAL, limit: int | None = None) -> list[Task]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run python -m bench.zh.build first.")

    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw["tasks"]
    if limit is not None:
        entries = entries[:limit]

    # Build the corpus once. Every task in this set searches all of it.
    first = entries[0]
    sessions = first["haystack_sessions"]
    dates = first["haystack_dates"]
    session_ids = first["haystack_session_ids"]

    memories: list[MemoryItem] = []
    for session, date, sid in zip(sessions, dates, session_ids, strict=True):
        for turn, msg in enumerate(session):
            content = msg.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            memories.append(
                MemoryItem(
                    id=f"{sid}:{turn}",
                    content=content.strip(),
                    session_id=sid,
                    timestamp=date,
                    role=msg.get("role", ""),
                )
            )

    if not memories:
        raise ValueError("corpus is empty")

    tasks: list[Task] = []
    for entry in entries:
        gold = frozenset(entry.get("answer_session_ids", []))
        query = Query(
            id=entry["question_id"],
            text=entry["question"],
            question_type=entry["question_type"],
            answer=entry["answer"],
            question_date=entry.get("question_date", ""),
            gold_session_ids=gold,
            is_abstention=not gold,
        )
        tasks.append(Task(query=query, memories=memories))
    return tasks


def summarize(tasks: list[Task]) -> dict[str, object]:
    by_type: dict[str, int] = {}
    for t in tasks:
        by_type[t.query.question_type] = by_type.get(t.query.question_type, 0) + 1
    return {
        "tasks": len(tasks),
        "abstention": sum(1 for t in tasks if t.query.is_abstention),
        "sessions": len({m.session_id for t in tasks[:1] for m in t.memories}),
        "turns": len(tasks[0].memories) if tasks else 0,
        "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
    }


if __name__ == "__main__":
    print(json.dumps(summarize(load_zh()), ensure_ascii=False, indent=2))

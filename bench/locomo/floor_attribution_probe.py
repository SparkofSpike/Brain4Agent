"""Is the 0.7080 -> 0.7054 gap really the five-term floor?

D-042 attributes it to the term floor: the sweep had no floor, so it scored
every question, while the shipped configuration leaves short questions to BM25.
The reviewer pointed out that the sweep also re-ranks a top-10 window while the
store re-ranks all 272 candidates, so two variables moved at once and the
attribution is untested.

This isolates them through the store, which is the only path that matters:

    floor on  (5)  -> what ships
    floor off (0)  -> the sweep's question set, everything else identical

If the gap is the floor, the second row is 0.7080 or very close.

    python -m bench.locomo.floor_attribution_probe
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import brain4agent.store as sm
from bench.locomo.run import load, session_text
from bench.retrieval import isolated_bonus
from brain4agent.store import MemoryStore


def evaluate(store: MemoryStore, queries: list[dict]) -> dict[str, float]:
    hit1 = hit5 = hit10 = 0
    n = 0
    for q in queries:
        gold = set(q["gold_ids"])
        if not gold:
            continue
        n += 1
        result = store.search(q["query"], top_k=10, tags=[f"sample:{q['user_id']}"])
        got = [e.session_id for e in result.evidence]
        first = next((i for i, sid in enumerate(got, 1) if sid in gold), None)
        hit1 += first == 1
        hit5 += bool(first and first <= 5)
        hit10 += bool(first and first <= 10)
    n = n or 1
    return {"hit@1": hit1 / n, "hit@5": hit5 / n, "hit@10": hit10 / n}


def fmt(row: dict[str, float]) -> str:
    return f"{row['hit@1']:>8.4f} {row['hit@5']:>8.4f} {row['hit@10']:>8.4f}"


def main() -> int:
    documents, queries = load()
    answers = [q for q in queries if q["gold_ids"]]
    print(f"corpus: {len(documents)} sessions   answerable queries: {len(answers)}")
    print()
    print(f"{'configuration':>34} {'hit@1':>8} {'hit@5':>8} {'hit@10':>8}")

    with isolated_bonus(proximity=0.0):
        store = MemoryStore(Path(tempfile.mkdtemp()) / "fa.db")
        for doc in documents:
            sample = doc["id"].rsplit("_session_", 1)[0]
            store.add(session_text(doc), session_id=doc["id"], tags=[f"sample:{sample}"])
        r = evaluate(store, answers)
        print(f"{'no proximity (D-041 baseline)':>34} {fmt(r)}")
        store.close()

    for label, terms in (("floor on (5) = shipped", 5), ("floor off (0) = sweep set", 0)):
        sm.PROXIMITY_MIN_TERMS = terms
        store = MemoryStore(Path(tempfile.mkdtemp()) / "fa.db")
        for doc in documents:
            sample = doc["id"].rsplit("_session_", 1)[0]
            store.add(session_text(doc), session_id=doc["id"], tags=[f"sample:{sample}"])
        r = evaluate(store, answers)
        print(f"{label:>34} {fmt(r)}")
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

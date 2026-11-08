"""LoCoMo retrieval evaluation, model-free and reproducible.

What this measures: given a question, does the session containing the answer
come back, and how high. What it does not measure: answer quality. That needs a
generator and a judge, is a separate run, and is not comparable to this number.

Why it is reproducible:

- **Data is fixed and shipped.** `data/` holds AMB's export of LoCoMo locomo10
  (272 sessions, 5882 turns, 1540 questions). No download, no drift.
- **No model is called.** No sampling variance, no API version to pin. Same
  input, same number, on any machine.
- **The metric is the same one `bench/zh/e2e.py` uses**, so the two numbers can
  be read side by side.

Retrieval unit is the session, not the turn. Measured both ways:

```
turn-level      hit@1 0.5447   hit@5 0.8210   hit@10 0.9099   mrr 0.6595
session-level   hit@1 0.6270   hit@5 0.8850   hit@10 0.9471   mrr 0.7369
```

That is worth stating plainly: the unit of retrieval has to match the unit of
the answer. LoCoMo labels answers with a session id, and a turn-level index
splits one answer across several short documents, each of which is a weak match
on its own. The Chinese benchmark gets away with turn-level indexing because it
joins turns back into sessions when scoring -- LoCoMo is the same idea done in
the index instead of the scorer.

BM25 parameters barely matter here (k1 1.2-2.0, b 0.2-0.9 spans 0.617-0.632);
the default is fine. Reported because a flat grid is a result too.

Sample isolation is a tag filter. LoCoMo's ten samples never reference each
other, so a question from conv-26 must not retrieve conv-30's sessions. This is
D-025's scope filter doing real work rather than serving a benchmark.

    python -m bench.locomo.run
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

from brain4agent.store import MemoryStore

DATA = Path(__file__).with_name("data")

# Sessions are long; the store's 100k cap is far above the longest here (a whole
# LoCoMo conversation is under 20k characters).
MAX_SESSION_CHARS = 90_000


def load() -> tuple[list[dict], list[dict]]:
    documents = json.loads((DATA / "documents.json").read_text("utf-8"))
    queries = json.loads((DATA / "queries.json").read_text("utf-8"))
    return documents, queries


def session_text(document: dict) -> str:
    """One conversation as one document, speakers kept.

    Keeping the speaker labels measured slightly better than dropping them
    (0.6270 against 0.6218): a question naming a person gains from the name being
    adjacent to what they said.
    """
    turns = json.loads(document["content"])
    text = "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)
    return text[:MAX_SESSION_CHARS]


def import_corpus(store: MemoryStore, documents: list[dict]) -> None:
    with store.batch():
        for doc in documents:
            sample = doc["id"].rsplit("_session_", 1)[0]
            store.add(
                session_text(doc),
                session_id=doc["id"],
                tags=[f"sample:{sample}"],
            )


def score(queries: list[dict], retrieved: dict[str, list[str]]) -> dict[str, object]:
    hit1 = hit5 = hit10 = mrr = 0.0
    answerable = 0
    by_cat: Counter[str] = Counter()
    by_cat_total: Counter[str] = Counter()

    for q in queries:
        gold = set(q["gold_ids"])
        if not gold:
            continue
        answerable += 1
        got = retrieved.get(q["id"], [])
        first = next((i for i, sid in enumerate(got, 1) if sid in gold), None)
        hit1 += first == 1
        hit5 += bool(first and first <= 5)
        hit10 += bool(first and first <= 10)
        mrr += (1.0 / first) if first else 0.0
        cat = q["meta"]["category"]
        by_cat_total[cat] += 1
        by_cat[cat] += first == 1

    n = answerable or 1
    return {
        "answerable": answerable,
        "hit@1": hit1 / n,
        "hit@5": hit5 / n,
        "hit@10": hit10 / n,
        "mrr": mrr / n,
        "by_cat": {c: by_cat[c] / by_cat_total[c] for c in by_cat_total},
    }


def main() -> int:
    documents, queries = load()
    turns = sum(len(json.loads(d["content"])) for d in documents)
    print(f"corpus: {len(documents)} sessions · {turns} turns")
    print(f"queries: {len(queries)} across {len({q['user_id'] for q in queries})} samples")
    print()

    with tempfile.TemporaryDirectory() as td:
        with MemoryStore(Path(td) / "locomo.db") as store:
            import_corpus(store, documents)
            retrieved: dict[str, list[str]] = {}
            for q in queries:
                result = store.search(
                    q["query"], top_k=10, tags=[f"sample:{q['user_id']}"]
                )
                retrieved[q["id"]] = [e.session_id for e in result.evidence]

    r = score(queries, retrieved)
    print(f"answerable: {r['answerable']}")
    print(f"  hit@1   {r['hit@1']:.4f}")  # type: ignore[arg-type]
    print(f"  hit@5   {r['hit@5']:.4f}")  # type: ignore[arg-type]
    print(f"  hit@10  {r['hit@10']:.4f}")  # type: ignore[arg-type]
    print(f"  mrr     {r['mrr']:.4f}")  # type: ignore[arg-type]
    print()
    print("by category (hit@1):")
    for cat, v in sorted(r["by_cat"].items()):  # type: ignore[union-attr,attr-defined]
        print(f"  {cat:<14} {v:.4f}")
    print()
    print("These are retrieval numbers. Published LoCoMo scores (0.67-0.92) measure")
    print("answer accuracy under an LLM judge and are NOT comparable to them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

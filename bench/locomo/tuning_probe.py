"""Which English query-side change moves LoCoMo, one at a time (D-041).

The published number comes from `bench.locomo.run`, which goes through the
shipping store. This probe isolates the three mechanisms that produce it, so a
change to any one of them can cite a row instead of a hunch.

    python -m bench.locomo.tuning_probe

The index is one document per session, exactly as `bench.locomo.run` builds it.
Session isolation is the same tag filter, applied as a post-filter here because
this probe ranks with `BM25` directly rather than through `MemoryStore`.
"""

from __future__ import annotations

from collections.abc import Callable

from bench.locomo.run import load, session_text
from brain4agent.tokenize import (
    BM25,
    EN_STOPWORDS,
    INTERROGATIVES,
    both_tokens,
    jieba_tokens,
    stem_term,
)


def before_d041(text: str) -> list[str]:
    """The pre-D-041 query tokeniser: dictionary words minus interrogatives."""
    return [t for t in jieba_tokens(text) if t not in INTERROGATIVES]


def stopwords_only(text: str) -> list[str]:
    return [t for t in before_d041(text) if t not in EN_STOPWORDS]


def stems_only(text: str) -> list[str]:
    return [stem_term(t) for t in before_d041(text)]


def stopwords_and_stems(text: str) -> list[str]:
    return [stem_term(t) for t in stopwords_only(text)]


def evaluate(index: BM25, queries: list[dict]) -> dict[str, float]:
    hit1 = hit5 = hit10 = mrr = 0.0
    n = 0
    for q in queries:
        gold = set(q["gold_ids"])
        if not gold:
            continue
        n += 1
        sample = q["user_id"]
        ranked = []
        for mid, _ in index.search(q["query"], top_k=len(index)):
            if mid.rsplit("_session_", 1)[0] != sample:
                continue
            ranked.append(mid)
            if len(ranked) >= 10:
                break
        first = next((i + 1 for i, sid in enumerate(ranked) if sid in gold), None)
        hit1 += first == 1
        hit5 += bool(first and first <= 5)
        hit10 += bool(first and first <= 10)
        mrr += (1.0 / first) if first else 0.0
    n = n or 1
    return {"hit@1": hit1 / n, "hit@5": hit5 / n, "hit@10": hit10 / n, "mrr": mrr / n}


def report(label: str, row: dict[str, float]) -> None:
    print(
        f"{label:38s} hit@1 {row['hit@1']:.4f}  hit@5 {row['hit@5']:.4f}  "
        f"hit@10 {row['hit@10']:.4f}  MRR {row['mrr']:.4f}"
    )


def main() -> int:
    documents, queries = load()
    docs = [(d["id"], session_text(d)) for d in documents]
    print(f"corpus: {len(docs)} sessions  queries: {len(queries)}")
    print()

    variants: list[tuple[str, Callable[[str], list[str]], bool, float]] = [
        ("before D-041", before_d041, False, 0.0),
        ("+ English stopwords", stopwords_only, False, 0.0),
        ("+ stem folding only", stems_only, True, 0.0),
        ("stopwords + stems", stopwords_and_stems, True, 0.0),
    ]
    for label, query_fn, normalize, phrase in variants:
        index = BM25(
            index_fn=both_tokens,
            query_fn=query_fn,
            fuzzy=True,
            normalize=stem_term if normalize else None,
            phrase_bonus=phrase,
        )
        index.index(docs)
        report(label, evaluate(index, queries))
    print()

    print("phrase bonus, on top of stopwords + stems:")
    for alpha in (0.1, 0.2, 0.3, 0.4, 0.5):
        index = BM25(
            index_fn=both_tokens,
            query_fn=stopwords_and_stems,
            fuzzy=True,
            normalize=stem_term,
            phrase_bonus=alpha,
        )
        index.index(docs)
        report(f"  alpha={alpha}", evaluate(index, queries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

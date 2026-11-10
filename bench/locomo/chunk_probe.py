"""LoCoMo: does chunking a long session beat indexing it whole?

The shipping LoCoMo path stores one document per session and queries it with a
sample-scope tag filter, so a question from conv-26 never sees conv-30. The
first version of this probe skipped the filter and reported 0.6074 against the
product's 0.6270 -- a fixture bug, not a retrieval difference. Every
configuration below evaluates inside one sample only.

    python -m bench.locomo.chunk_probe
"""

from __future__ import annotations

import json
from pathlib import Path

from brain4agent.tokenize import BM25, both_tokens, jieba_tokens, tokenize

DATA = Path(__file__).with_name("data")
MAX_SESSION_CHARS = 90_000

# English question words. Same argument as the Chinese list: they ask the
# question, they do not answer it.
EN_INTERROGATIVES = frozenset(
    {"what", "which", "when", "where", "who", "whom", "whose", "why", "how"}
)


def load() -> tuple[list[dict], list[dict]]:
    documents = json.loads((DATA / "documents.json").read_text("utf-8"))
    queries = json.loads((DATA / "queries.json").read_text("utf-8"))
    return documents, queries


def sample_of(session_id: str) -> str:
    return session_id.rsplit("_session_", 1)[0]


def text_of(turns: list[dict]) -> str:
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)[:MAX_SESSION_CHARS]


def session_items(documents: list[dict]) -> list[tuple[str, str, str]]:
    return [(d["id"], text_of(json.loads(d["content"])), d["id"]) for d in documents]


def chunk_items(documents: list[dict], size: int) -> list[tuple[str, str, str]]:
    items: list[tuple[str, str, str]] = []
    for d in documents:
        turns = json.loads(d["content"])
        for i in range(0, len(turns), size):
            items.append((f"{d['id']}#{i // size}", text_of(turns[i : i + size]), d["id"]))
    return items


def q_en(text: str) -> list[str]:
    return [t for t in jieba_tokens(text) if t not in EN_INTERROGATIVES]


def evaluate(items, queries, index_fn, query_fn, beta: float = 0.0) -> dict:
    """One index over the whole corpus, filtered to a sample at query time.

    The product does exactly this: every session is written to one store and a
    query carries a ``sample:`` tag filter. Building one index per sample instead
    changes idf and avgdl and scored 0.6140 against the product's 0.6270 -- the
    first version of this probe made that mistake.
    """
    idx = BM25(index_fn=index_fn, query_fn=query_fn)
    idx.index([(i, t) for i, t, _ in items])
    sid_of = {i: s for i, _, s in items}

    hit1 = hit5 = hit10 = 0.0
    rr = 0.0
    n = 0
    for q in queries:
        gold = set(q["gold_ids"])
        if not gold:
            continue
        n += 1
        hits = idx.search(q["query"], top_k=len(idx))
        per: dict[str, list[float]] = {}
        for mid, score in hits:
            sid = sid_of[mid]
            if sample_of(sid) != q["user_id"]:
                continue
            per.setdefault(sid, []).append(score)
        folded = []
        for sid, scores in per.items():
            scores.sort(reverse=True)
            folded.append((sid, scores[0] + beta * sum(scores[1:3])))
        folded.sort(key=lambda kv: -kv[1])
        first = next((i for i, (sid, _) in enumerate(folded, 1) if sid in gold), None)
        hit1 += first == 1
        hit5 += bool(first and first <= 5)
        hit10 += bool(first and first <= 10)
        rr += (1.0 / first) if first else 0.0
    n = n or 1
    return {"hit@1": hit1 / n, "hit@5": hit5 / n, "hit@10": hit10 / n, "mrr": rr / n}


def rep(label: str, r: dict) -> None:
    print(
        f"{label:46s} hit@1 {r['hit@1']:.4f}  hit@5 {r['hit@5']:.4f}  "
        f"hit@10 {r['hit@10']:.4f}  MRR {r['mrr']:.4f}"
    )


def main() -> int:
    documents, queries = load()
    print(f"{len(documents)} sessions, {len(queries)} queries")
    print()
    print("== session-level (shipping) ==")
    sess = session_items(documents)
    rep("both / jieba (shipping tokenisers)", evaluate(sess, queries, both_tokens, jieba_tokens))
    rep("both / jieba - english question words", evaluate(sess, queries, both_tokens, q_en))
    rep("bigram / bigram", evaluate(sess, queries, tokenize, tokenize))
    rep("bigram / jieba - question words", evaluate(sess, queries, tokenize, q_en))
    print()
    for size in (3, 5, 8, 12, 20):
        print(f"== chunks of {size} turns ==")
        items = chunk_items(documents, size)
        for beta in (0.0, 0.5, 1.0):
            rep(f"  beta={beta}", evaluate(items, queries, both_tokens, jieba_tokens, beta))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

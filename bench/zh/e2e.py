"""End-to-end retrieval quality, through the shipping code path.

bench/zh/compare_tokenizers.py scores BM25 directly. This scores ``MemoryStore``:
the same SQLite store, the same session-level deduplication, the same evidence
layer a user actually runs. The two should agree, and when they stop agreeing
the benchmark has stopped describing the product.

    python -m bench.zh.e2e

BASELINE is D-010, the symmetric bigram/bigram implementation. The store now
indexes with bigrams *and* words and queries with words (D-019), so it should sit
above the baseline, not level with it.
    baseline  hit@1 0.825  hit@5 0.950  hit@10 1.000  MRR 0.8911
    expected  hit@1 0.925  hit@5 0.975  hit@10 1.000  MRR 0.9404
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from brain4agent import MemoryStore

DATA = Path(__file__).parent / "zh_eval.json"


def load() -> dict:
    return json.loads(DATA.read_text("utf-8"))


def build_store(data: dict) -> MemoryStore:
    """One store for the whole corpus -- every task shares it (see meta.variants)."""
    tmp = Path(tempfile.mkdtemp()) / "eval.db"
    store = MemoryStore(tmp)
    task = data["tasks"][0]
    for sid, turns in zip(task["haystack_session_ids"], task["haystack_sessions"], strict=True):
        for turn in turns:
            store.add(turn["content"], session_id=sid, role=turn["role"])
    return store


def rank(store: MemoryStore, query: str, k: int = 10) -> list[str]:
    return [e.session_id for e in store.search(query, top_k=k).evidence]


def score(data: dict, store: MemoryStore) -> dict:
    hits = {1: 0, 5: 0, 10: 0}
    rr = 0.0
    n = 0
    misses: list[str] = []
    per_type: dict[str, list[float]] = {}

    for task in data["tasks"]:
        qid = task["question_id"]
        if qid.endswith("_abs"):
            continue
        gold = set(task["answer_session_ids"])
        if not gold:
            continue
        n += 1

        ranked = rank(store, task["question"], k=10)
        found_at = next((i + 1 for i, sid in enumerate(ranked) if sid in gold), None)

        for k in hits:
            if found_at and found_at <= k:
                hits[k] += 1
        if found_at:
            rr += 1.0 / found_at
        else:
            misses.append(qid)

        qtype = task["question_type"]
        per_type.setdefault(qtype, []).append(1.0 if found_at == 1 else 0.0)

    return {
        "n": n,
        "hit@1": hits[1] / n if n else 0.0,
        "hit@5": hits[5] / n if n else 0.0,
        "hit@10": hits[10] / n if n else 0.0,
        "mrr": rr / n if n else 0.0,
        "misses": misses,
        "per_type_hit1": {k: sum(v) / len(v) for k, v in per_type.items()},
    }


def abstention_report(data: dict, store: MemoryStore) -> dict:
    """How often a fully-missing term appears on unanswerable questions."""
    with_missing = 0
    total = 0
    for task in data["tasks"]:
        if not task["question_id"].endswith("_abs"):
            continue
        total += 1
        result = store.search(task["question"], top_k=1)
        if result.corpus_missing:
            with_missing += 1
    return {"total": total, "with_missing_terms": with_missing}


BASELINE = {"hit@1": 0.825, "hit@5": 0.950, "hit@10": 1.000, "mrr": 0.8911}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument(
        "--min-hit1",
        type=float,
        default=0.0,
        help="exit non-zero if hit@1 falls below this (use in CI)",
    )
    args = ap.parse_args()

    data = load()
    store = build_store(data)
    result = score(data, store)
    abst = abstention_report(data, store)
    stats = store.stats()

    if args.json:
        print(
            json.dumps({**result, "abstention": abst, "store": stats}, ensure_ascii=False, indent=2)
        )
        return 0

    print(f"corpus: {stats['sessions']} sessions / {stats['turns']} turns")
    print(f"answerable: {result['n']} questions")
    print()
    print(f"{'metric':8s} {'MemoryStore':>12s} {'baseline':>10s}  {'delta':>8s}")
    for key in ("hit@1", "hit@5", "hit@10", "mrr"):
        got = result[key]
        base = BASELINE[key]
        delta = got - base
        flag = "" if abs(delta) <= 0.001 else ("  better" if delta > 0 else "  WORSE")
        print(f"{key:8s} {got:12.4f} {base:10.4f}  {delta:+8.4f}{flag}")

    print()
    print("hit@1 by question type:")
    for qtype, val in sorted(result["per_type_hit1"].items(), key=lambda kv: kv[1]):
        print(f"  {qtype:28s} {val:.4f}")

    print()
    print(f"abstention: {abst['with_missing_terms']}/{abst['total']} surface a missing term")
    if result["misses"]:
        print(f"missed: {', '.join(result['misses'])}")
    store.close()

    if result["hit@1"] < args.min_hit1:
        print(f"FAIL: hit@1 {result['hit@1']:.4f} is below the required {args.min_hit1:.4f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

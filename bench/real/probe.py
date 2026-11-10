"""Retrieval against a real memory store, not a benchmark fixture.

The Chinese benchmark asks full questions over 44 sessions. A working agent does
not: it types two or three keywords, often with an identifier, over thousands of
memories. D-039 measured the gap -- 0.95 hit@1 on the fixture, 0.27 for two-word
queries on a 2898-memory store. This probe keeps that second number visible.

The query set is derived from the store itself: take a memory's first
distinctive nouns and identifiers, ask with two, three, or four of them, and
check whether the memory comes back. No labels are stored in the repository and
no real memory text is printed.

Run it on a copy, not the live store: `MemoryStore.search` increments access
counters, so even a read writes.

    python -m bench.real.probe --db /path/to/copy.db --limit 300
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from brain4agent import MemoryStore
from brain4agent.tokenize import load_jieba

_ASCII_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{2,}")
_KEEP_POS = frozenset({"n", "nr", "ns", "nt", "nz", "vn", "an", "eng"})


def keywords(text: str, limit: int = 4) -> list[str]:
    """Distinctive nouns and identifiers, in reading order.

    Uses jieba's POS tagger when installed; without it, falls back to words of
    two or more characters. The fallback is noisier, which is stated rather
    than hidden -- the same choice the store makes for its query tokeniser.
    """
    out: list[str] = []
    jieba = load_jieba()
    if jieba is None:
        from brain4agent.tokenize import word_tokens

        for word in word_tokens(text):
            if len(word) >= 2 and word not in out:
                out.append(word)
            if len(out) >= limit:
                break
        return out

    import jieba.posseg as pseg

    for word, flag in pseg.cut(text):
        w = word.strip()
        if len(w) < 2 or w in out:
            continue
        if flag in _KEEP_POS or _ASCII_ID.fullmatch(w):
            out.append(w)
        if len(out) >= limit:
            break
    return out


def build_queries(
    db: Path, limit: int, seed: int, min_chars: int, max_chars: int
) -> list[dict[str, object]]:
    import sqlite3

    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT session_id, content FROM memories ORDER BY rowid").fetchall()
    conn.close()
    candidates = [(sid, content) for sid, content in rows if min_chars <= len(content) <= max_chars]
    random.seed(seed)
    random.shuffle(candidates)

    items: list[dict[str, object]] = []
    for sid, content in candidates:
        words = keywords(content)
        if len(words) < 2:
            continue
        n = min(2 + (len(items) % 3), len(words))
        items.append({"sid": sid, "q": " ".join(words[:n]), "n": n})
        if len(items) >= limit:
            break
    return items


def score(results: list[dict[str, object]]) -> dict[str, object]:
    def bucket(rows: list[dict[str, object]]) -> dict[str, float | int]:
        total = len(rows) or 1
        return {
            "n": len(rows),
            "hit@1": sum(1 for r in rows if r["pos"] == 1) / total,
            "hit@5": sum(1 for r in rows if r["pos"] and r["pos"] <= 5) / total,
            "mrr": sum(1 / r["pos"] for r in rows if r["pos"]) / total,
        }

    by_len: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in results:
        by_len[int(row["n"])].append(row)
    return {
        "n": len(results),
        **bucket(results),
        "by_len": {str(k): bucket(v) for k, v in sorted(by_len.items())},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Leave-one-out retrieval on a real store.")
    ap.add_argument("--db", required=True, type=Path, help="path to a copy of the store")
    ap.add_argument("--limit", type=int, default=300, help="number of queries")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=500)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.db.exists():
        print(f"no such store: {args.db}", file=sys.stderr)
        return 2

    items = build_queries(args.db, args.limit, args.seed, args.min_chars, args.max_chars)
    if not items:
        print("no usable memories in that store", file=sys.stderr)
        return 2

    store = MemoryStore(args.db)
    results: list[dict[str, object]] = []
    for item in items:
        response = store.search(str(item["q"]), top_k=10)
        ranked = [e.session_id for e in response.evidence]
        pos = next((i + 1 for i, s in enumerate(ranked) if s == item["sid"]), None)
        results.append({**item, "pos": pos})
    store.close()

    summary = score(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({**summary, "results": results}, fh, ensure_ascii=False, indent=1)
        print(f"written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

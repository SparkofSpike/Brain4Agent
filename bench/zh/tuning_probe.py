"""Which knob moves the Chinese number, and by how much.

D-019 tested tokeniser pairings; D-024 tested re-ranking signals on a distilled
corpus. Neither swept BM25's own parameters, query-side interrogative filtering,
score fusion across two indexes, or session-level aggregation. This probe does,
so a change to the shipping defaults can cite a number instead of a hunch.

    python -m bench.zh.tuning_probe
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from brain4agent.evidence import _INTERROGATIVE
from brain4agent.tokenize import BM25, both_tokens, jieba_tokens, tokenize

DATA = Path(__file__).parent / "zh_eval.json"
_PUNCT = re.compile(r"^[\W_]+$", re.UNICODE)


def load() -> dict:
    return json.loads(DATA.read_text("utf-8"))


def corpus(data: dict) -> list[tuple[str, str]]:
    task = data["tasks"][0]
    docs: list[tuple[str, str]] = []
    for sid, turns in zip(task["haystack_session_ids"], task["haystack_sessions"], strict=True):
        for n, turn in enumerate(turns):
            docs.append((f"{sid}:{n}", turn["content"]))
    return docs


def answerable(data: dict) -> list[dict]:
    return [
        t
        for t in data["tasks"]
        if not t["question_id"].endswith("_abs") and t["answer_session_ids"]
    ]


def sid_of(mid: str) -> str:
    return mid.rsplit(":", 1)[0]


def evaluate(scores_for, tasks: list[dict]) -> dict:
    hits = {1: 0, 5: 0, 10: 0}
    rr = 0.0
    for t in tasks:
        ranked = [sid for sid, _ in scores_for(t["question"])]
        gold = set(t["answer_session_ids"])
        pos = next((i + 1 for i, sid in enumerate(ranked) if sid in gold), None)
        for k in hits:
            if pos and pos <= k:
                hits[k] += 1
        if pos:
            rr += 1.0 / pos
    n = len(tasks)
    return {"hit@1": hits[1] / n, "hit@5": hits[5] / n, "hit@10": hits[10] / n, "mrr": rr / n}


def fold_max(scores: list[tuple[str, float]]) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    order: list[str] = []
    for mid, score in scores:
        sid = sid_of(mid)
        if sid not in best:
            best[sid] = score
            order.append(sid)
        elif score > best[sid]:
            best[sid] = score
    return sorted(((s, best[s]) for s in order), key=lambda kv: -kv[1])


def make_max(index: BM25):
    return lambda q: fold_max(index.search(q, top_k=len(index)))


def q_no_punct(text: str) -> list[str]:
    return [t for t in jieba_tokens(text) if not _PUNCT.match(t)]


def q_no_interrogative(text: str) -> list[str]:
    return [t for t in jieba_tokens(text) if t not in _INTERROGATIVE]


def q_no_both(text: str) -> list[str]:
    return [t for t in q_no_interrogative(text) if not _PUNCT.match(t)]


def q_bigram(text: str) -> list[str]:
    return tokenize(text)


def report(label: str, row: dict) -> None:
    print(
        f"{label:44s} hit@1 {row['hit@1']:.4f}  hit@5 {row['hit@5']:.4f}  "
        f"hit@10 {row['hit@10']:.4f}  MRR {row['mrr']:.4f}"
    )


def main() -> int:
    data = load()
    docs = corpus(data)
    tasks = answerable(data)

    # --- 1. BM25 k1/b grid, shipping tokenisers -------------------------
    print("== k1/b grid (index both_tokens, query jieba_tokens) ==")
    grid = []
    for k1 in (0.9, 1.2, 1.5, 1.8, 2.4):
        for b in (0.0, 0.3, 0.5, 0.75, 0.9):
            idx = BM25(k1=k1, b=b, index_fn=both_tokens, query_fn=jieba_tokens)
            idx.index(docs)
            r = evaluate(make_max(idx), tasks)
            grid.append((k1, b, r))
            report(f"k1={k1} b={b}", r)
    best = max(grid, key=lambda x: (x[2]["hit@1"], x[2]["mrr"]))
    print(f"  best: k1={best[0]} b={best[1]}")
    print()

    # --- 2. query-side filtering ---------------------------------------
    print("== query-side filtering (k1=1.5 b=0.75) ==")
    for name, fn in (
        ("jieba (shipping)", jieba_tokens),
        ("no punctuation", q_no_punct),
        ("no interrogatives", q_no_interrogative),
        ("no punct + no interrogatives", q_no_both),
        ("bigrams", q_bigram),
    ):
        idx = BM25(index_fn=both_tokens, query_fn=fn)
        idx.index(docs)
        report(name, evaluate(make_max(idx), tasks))
    print()

    # --- 3. two-index fusion -------------------------------------------
    print("== fusion: word index + bigram index (normalised per query) ==")
    idx_w = BM25(index_fn=jieba_tokens, query_fn=jieba_tokens)
    idx_w.index(docs)
    idx_b = BM25(index_fn=tokenize, query_fn=jieba_tokens)
    idx_b.index(docs)

    def fused(alpha: float):
        def scores_for(q: str) -> list[tuple[str, float]]:
            w = idx_w.search(q, top_k=len(idx_w))
            b = idx_b.search(q, top_k=len(idx_b))
            wmax = max((s for _, s in w), default=1.0) or 1.0
            bmax = max((s for _, s in b), default=1.0) or 1.0
            acc: dict[str, float] = {}
            for mid, s in w:
                acc[mid] = acc.get(mid, 0.0) + (1.0 - alpha) * s / wmax
            for mid, s in b:
                acc[mid] = acc.get(mid, 0.0) + alpha * s / bmax
            return fold_max(sorted(acc.items(), key=lambda kv: -kv[1]))

        return scores_for

    for alpha in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        report(f"alpha(bigram)={alpha}", evaluate(fused(alpha), tasks))
    print()

    # --- 4. session aggregation ----------------------------------------
    print("== session aggregation: max + beta * (2nd + 3rd) ==")
    idx = BM25(index_fn=both_tokens, query_fn=jieba_tokens)
    idx.index(docs)

    def agg(beta: float):
        def scores_for(q: str) -> list[tuple[str, float]]:
            per: dict[str, list[float]] = {}
            for mid, s in idx.search(q, top_k=len(idx)):
                per.setdefault(sid_of(mid), []).append(s)
            out = []
            for sid, vals in per.items():
                vals.sort(reverse=True)
                extra = sum(vals[1:3])
                out.append((sid, vals[0] + beta * extra))
            return sorted(out, key=lambda kv: -kv[1])

        return scores_for

    for beta in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
        report(f"beta={beta}", evaluate(agg(beta), tasks))
    print()

    # --- 5. combination: interrogative filter + aggregation --------------
    print("== combination: no interrogatives + aggregation ==")
    idx_ni = BM25(index_fn=both_tokens, query_fn=q_no_interrogative)
    idx_ni.index(docs)

    def agg_ni(beta: float):
        def scores_for(q: str) -> list[tuple[str, float]]:
            per: dict[str, list[float]] = {}
            for mid, s in idx_ni.search(q, top_k=len(idx_ni)):
                per.setdefault(sid_of(mid), []).append(s)
            out = []
            for sid, vals in per.items():
                vals.sort(reverse=True)
                out.append((sid, vals[0] + beta * sum(vals[1:3])))
            return sorted(out, key=lambda kv: -kv[1])

        return scores_for

    for beta in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        report(f"no-int + beta={beta}", evaluate(agg_ni(beta), tasks))
    print()

    # --- 6. per-question rank change, shipping vs candidate --------------
    print("== per-question rank: shipping -> no-interrogatives + beta=0.5 ==")
    ship = make_max(idx)
    cand = agg_ni(0.5)
    for t in tasks:
        q = t["question"]
        gold = set(t["answer_session_ids"])
        a = [s for s, _ in ship(q)]
        b = [s for s, _ in cand(q)]
        pa = next((i + 1 for i, s in enumerate(a) if s in gold), None)
        pb = next((i + 1 for i, s in enumerate(b) if s in gold), None)
        tag = "  "
        if pa != pb:
            tag = "<-" if (pb or 99) < (pa or 99) else "! "
        print(f"{tag} {t['question_id']:14s} {q[:28]:30s} rank {pa} -> {pb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

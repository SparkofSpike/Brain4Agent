"""Is read/write tokeniser separation real, or a nice-sounding story?

README claim #1: indexing should cover broadly (keep every clue), querying should
be precise (hand back less noise), and a single shared tokeniser must therefore
lose on one side or the other. It is the project's founding premise and until
now it has been untested.

The test is cheap: cross three indexing strategies with two querying strategies
over the same Chinese corpus and see whether any asymmetric pairing beats the
symmetric one.

    bigram  -- character bigrams, no dictionary
    jieba   -- dictionary segmentation
    both    -- the union of the two at index time

If read/write separation is real, the asymmetric cells should win. If they do
not, that is a finding too, and the premise gets deleted from the README rather
than defended.

    python -m bench.zh.indexing_probe
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from brain4agent.tokenize import load_jieba, tokenize

DATA = Path(__file__).parent / "zh_eval.json"


def jieba_tokens(text: str) -> list[str]:
    jieba = load_jieba()
    if jieba is None:
        return tokenize(text)
    return [w.strip().lower() for w in jieba.lcut(text) if w.strip()]


def both_tokens(text: str) -> list[str]:
    return tokenize(text) + jieba_tokens(text)


def jieba_content(text: str) -> list[str]:
    """Words only: drop single-character function words and punctuation.

    A question like 第一版修复和上限调整哪个先做 currently feeds 和/哪个/先/做
    into the scorer, where they match everything and separate nothing.
    """
    return [w for w in jieba_tokens(text) if len(w) > 1 or (w.isascii() and w.isalnum())]


INDEXERS = {"bigram": tokenize, "jieba": jieba_tokens, "both": both_tokens}
QUERIERS = {"bigram": tokenize, "jieba": jieba_tokens, "content": jieba_content}


class FlexBM25:
    """BM25 where the index side and the query side tokenise independently."""

    def __init__(self, index_fn, query_fn, k1: float = 1.5, b: float = 0.75) -> None:
        self.index_fn = index_fn
        self.query_fn = query_fn
        self.k1 = k1
        self.b = b

    def index(self, items: list[tuple[str, str]]) -> None:
        self._ids = [i for i, _ in items]
        self._docs: list[Counter[str]] = []
        self._lens: list[int] = []
        self._df: Counter[str] = Counter()
        for _, text in items:
            counts = Counter(self.index_fn(text))
            self._docs.append(counts)
            self._lens.append(sum(counts.values()))
            for term in counts:
                self._df[term] += 1
        self._n = len(items)
        self._avgdl = (sum(self._lens) / self._n) if self._n else 0.0
        self._order = {i: pos for pos, (i, _) in enumerate(items)}

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._n:
            return []
        q_terms = self.query_fn(query)
        if not q_terms:
            return []
        out: list[tuple[str, float]] = []
        for idx, counts in enumerate(self._docs):
            dl = self._lens[idx]
            score = 0.0
            for term in q_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                df = self._df.get(term, 0)
                idf = math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / (self._avgdl or 1.0))
                score += idf * (tf * (self.k1 + 1.0)) / denom
            if score > 0.0:
                out.append((self._ids[idx], score))
        out.sort(key=lambda kv: (-kv[1], self._order.get(kv[0], 0)))
        return out[:top_k]


def segments(data: dict) -> tuple[list[tuple[str, str]], list[dict]]:
    """Flatten the shared corpus into (item_id, text), and the answerable tasks."""
    task0 = data["tasks"][0]
    items: list[tuple[str, str]] = []
    for sid, turns in zip(task0["haystack_session_ids"], task0["haystack_sessions"], strict=True):
        for n, turn in enumerate(turns):
            items.append((f"{sid}:{n}", turn["content"]))
    tasks = [t for t in data["tasks"] if not t["question_id"].endswith("_abs")]
    return items, tasks


def evaluate(index_name: str, query_name: str, items, tasks) -> dict:
    idx = FlexBM25(INDEXERS[index_name], QUERIERS[query_name])
    idx.index(items)

    id_to_session = {iid: iid.split(":")[0] for iid, _ in items}
    hits = {1: 0, 5: 0, 10: 0}
    rr = 0.0

    for task in tasks:
        ranked = [iid for iid, _ in idx.search(task["question"], top_k=30)]
        sessions: list[str] = []
        seen: set[str] = set()
        for iid in ranked:
            sid = id_to_session[iid]
            if sid not in seen:
                seen.add(sid)
                sessions.append(sid)
        gold = set(task["answer_session_ids"])
        pos = next((i + 1 for i, sid in enumerate(sessions) if sid in gold), None)
        for k in hits:
            if pos and pos <= k:
                hits[k] += 1
        if pos:
            rr += 1.0 / pos

    n = len(tasks)
    return {
        "hit@1": hits[1] / n,
        "hit@5": hits[5] / n,
        "hit@10": hits[10] / n,
        "mrr": rr / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    data = json.loads(DATA.read_text("utf-8"))
    items, tasks = segments(data)

    rows = []
    print(f"{'index':8s} {'query':8s} {'hit@1':>7s} {'hit@5':>7s} {'hit@10':>7s} {'MRR':>7s}")
    print("-" * 50)
    for iname in INDEXERS:
        for qname in QUERIERS:
            r = evaluate(iname, qname, items, tasks)
            rows.append({"index": iname, "query": qname, **r})
            print(
                f"{iname:8s} {qname:8s} {r['hit@1']:7.4f} {r['hit@5']:7.4f} "
                f"{r['hit@10']:7.4f} {r['mrr']:7.4f}"
            )

    base = next(r for r in rows if r["index"] == "bigram" and r["query"] == "bigram")
    print()
    print(f"baseline (bigram/bigram): hit@1 {base['hit@1']:.4f}  MRR {base['mrr']:.4f}")
    print()
    print("asymmetric pairs vs baseline:")
    for r in rows:
        if r["index"] == r["query"]:
            continue
        d1 = r["hit@1"] - base["hit@1"]
        dm = r["mrr"] - base["mrr"]
        verdict = "better" if d1 > 0.001 or dm > 0.001 else ("worse" if d1 < -0.001 else "same")
        print(f"  {r['index']:6s}/{r['query']:6s}  hit@1 {d1:+.4f}  MRR {dm:+.4f}  {verdict}")

    if args.json:
        print()
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

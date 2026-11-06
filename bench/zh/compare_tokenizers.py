"""Compare character bigrams against dictionary segmentation.

This is the test D-005 was written to wait for. The design claim was that
bigrams are good enough -- and specifically that they avoid the unknown-word
problem a dictionary has. That claim was untested because there was no Chinese
benchmark. Now there is.

jieba is used here as a *measuring instrument*, not as a dependency. The
package must stay importable without it; this script skips cleanly if jieba is
absent.

Run: python -m bench.zh.compare_tokenizers
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from bench.retrieval import BM25
from bench.runner import bm25_retriever, run
from bench.zh.loader import load_zh

RESULTS = Path(__file__).parent.parent / "results"


def jieba_tokenize(text: str) -> list[str]:
    """Segment with jieba, keeping ASCII words and dropping whitespace."""
    import jieba

    out: list[str] = []
    for tok in jieba.lcut(text):
        tok = tok.strip().lower()
        if not tok or tok.isspace():
            continue
        out.append(tok)
    return out


class JiebaBM25(BM25):
    """BM25 with dictionary segmentation instead of bigrams."""

    def index(self, items: list[tuple[str, str]]) -> None:
        patched = [(i, jieba_tokenize(t)) for i, t in items]
        self._ids = [i for i, _ in patched]
        self._docs = []
        self._lens = []
        self._df: dict[str, int] = {}
        for _, toks in patched:
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            self._docs.append(counts)  # type: ignore[arg-type]
            self._lens.append(len(toks))
            for term in counts:
                self._df[term] = self._df.get(term, 0) + 1
        self._n = len(patched)
        self._avgdl = (sum(self._lens) / self._n) if self._n else 0.0

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        import math

        if not self._n:
            return []
        q_terms = jieba_tokenize(query)
        if not q_terms:
            return []
        scores: list[tuple[str, float]] = []
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
                scores.append((self._ids[idx], score))
        scores.sort(key=lambda kv: (-kv[1], kv[0]))
        return scores[:top_k]


def jieba_retriever():
    def _run(memories, query: str, k: int):
        idx = JiebaBM25()
        idx.index([(m.id, m.content) for m in memories])
        return idx.search(query, top_k=max(k, 10))

    return _run


def _top_scores(tasks, retriever) -> tuple[list[float], list[float]]:
    answerable: list[float] = []
    abstention: list[float] = []
    for t in tasks:
        hits = retriever(t.memories, t.query.text, 1)
        top = hits[0][1] if hits else 0.0
        (abstention if t.query.is_abstention else answerable).append(top)
    return answerable, abstention


def _best_balance(a: list[float], b: list[float]) -> dict[str, float]:
    if not a or not b:
        return {}
    lo, hi = 0.0, max(max(a), max(b))
    best = {"threshold": 0.0, "balanced": 0.5, "abstain": 0.0, "answer": 1.0}
    steps = 60
    for i in range(steps + 1):
        th = lo + (hi - lo) * i / steps
        ar = sum(1 for s in b if s <= th) / len(b)
        kr = sum(1 for s in a if s > th) / len(a)
        bal = (ar + kr) / 2
        if bal > best["balanced"]:
            best = {
                "threshold": round(th, 2),
                "balanced": round(bal, 4),
                "abstain": round(ar, 4),
                "answer": round(kr, 4),
            }
    return best


def main() -> None:
    try:
        import jieba  # noqa: F401
    except ImportError:
        print("jieba not installed; skipping. This is a measurement tool only.")
        return

    # Warm the dictionary so the first task does not pay for it.
    jieba_tokenize("预热 分词 词典 加载")

    tasks = load_zh()
    print(
        f"corpus: {len({m.session_id for m in tasks[0].memories})} sessions, "
        f"{len(tasks[0].memories)} turns, {len(tasks)} questions"
    )

    out: dict[str, object] = {
        "corpus": {
            "sessions": len({m.session_id for m in tasks[0].memories}),
            "turns": len(tasks[0].memories),
            "questions": len(tasks),
        }
    }

    for label, retriever in (
        ("bigram", bm25_retriever()),
        ("jieba", jieba_retriever()),
    ):
        r = run(tasks, retriever, name=label, top_k=10, progress_every=0)
        s = r.summary
        a, b = _top_scores(tasks, retriever)
        bal = _best_balance(a, b)
        out[label] = {
            "hit@1": s["hit@1"],
            "hit@5": s["hit@5"],
            "hit@10": s["hit@10"],
            "mrr": s["mrr"],
            "missed": s["missed"],
            "abstention_median": round(statistics.median(b), 2),
            "answerable_median": round(statistics.median(a), 2),
            "best_balance": bal,
        }
        print(f"\n{label}:")
        for k in ("hit@1", "hit@5", "hit@10", "mrr", "missed"):
            print(f"  {k:10s} {s[k]}")
        print(
            f"  abstention median {statistics.median(b):.2f}  "
            f"answerable median {statistics.median(a):.2f}"
        )
        print(f"  best balance {bal}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    dest = RESULTS / "zh-tokenizer-compare.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten to {dest}")


if __name__ == "__main__":
    main()

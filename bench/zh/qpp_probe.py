"""Test classical query-performance predictors as abstention signals.

Why this exists: stage 0 tried score, coverage and unseen-word ratio and all
topped out near 10% refusal at keep>=0.95. But that was ad-hoc. Information
retrieval has a whole subfield for "predict how good this retrieval will be
without the ground truth" -- QPP -- and its predictors are pure statistics.

The distinction matters and is easy to lose: QPP predicts *ranking quality*,
not *whether the answer exists*. A query can retrieve the single most relevant
session and still have its answer nowhere in the corpus. So a good QPP score is
not the same thing as "answerable", and this script measures the gap rather
than assuming it away.

The literature is also discouraging in advance -- Chifu et al. 2025
(arXiv:2504.01101) find QPP predictors weakly correlated with real performance,
sensitive to collection, and offering only marginal gains downstream. That
paper measures ranking quality on English TREC collections; this measures
abstention on a Chinese set. Different question, same predictors.

Run: python -m bench.zh.qpp_probe
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from bench.retrieval import BM25
from bench.zh.loader import load_zh

RESULTS = Path(__file__).parent.parent / "results"


def _scores_for(index: BM25, query: str, k: int = 10) -> list[float]:
    return [s for _, s in index.search(query, top_k=k)]


def nqc(scores: list[float]) -> float:
    """Normalised Query Commitment (Shtok et al. 2012).

    Standard deviation of the top-k scores divided by their mean: a query whose
    results are all equally mediocre scores low, one with a clear winner scores
    high. The theory is that score spread reflects how much the query
    discriminates, which correlates with retrieval quality.
    """
    if len(scores) < 2:
        return 0.0
    mean = statistics.fmean(scores)
    if mean <= 0:
        return 0.0
    return statistics.pstdev(scores) / mean


def uqc(scores: list[float]) -> float:
    """Unnormalised Query Commitment: the spread itself."""
    if len(scores) < 2:
        return 0.0
    return statistics.pstdev(scores)


def wig(scores: list[float], collection_mean: float, collection_std: float) -> float:
    """Weighted Information Gain (Zhou & Croft 2007).

    How far the top-k mean sits from the collection mean, in collection standard
    deviations. Bigger means the query pulled clearly better-than-average
    documents, which is the signal that it matched something specific.
    """
    if not scores or collection_std <= 0:
        return 0.0
    return (statistics.fmean(scores) - collection_mean) / collection_std


def clarity(scores: list[float]) -> float:
    """A score-distribution entropy proxy.

    The original Clarity is a KL divergence between the query language model
    and the collection model, computed over term distributions. With only the
    ranked scores available, this uses the normalised-score entropy instead:
    a flat distribution (no clear winner) has high entropy.
    """
    if len(scores) < 2:
        return 0.0
    total = sum(scores)
    if total <= 0:
        return 0.0
    ps = [s / total for s in scores if s > 0]
    if len(ps) < 2:
        return 0.0
    return -sum(p * math.log(p) for p in ps)


PREDICTORS = {
    "nqc": lambda s, cm, cs: nqc(s),
    "uqc": lambda s, cm, cs: uqc(s),
    "wig": lambda s, cm, cs: wig(s, cm, cs),
    "clarity": lambda s, cm, cs: clarity(s),
    "top1": lambda s, cm, cs: s[0] if s else 0.0,
    "top1_minus_top10": lambda s, cm, cs: (s[0] - s[-1]) if len(s) > 1 else 0.0,
}


def main() -> None:
    tasks = load_zh()
    print(f"{len(tasks)} questions, {len(tasks[0].memories)} turns")

    # Per task: build the index once, compute the collection score statistics
    # needed by WIG, then each predictor.
    rows: list[dict[str, float | str]] = []
    for task in tasks:
        index = BM25()
        index.index([(m.id, m.content) for m in task.memories])
        scores = _scores_for(index, task.query.text, k=10)
        if not scores:
            scores = [0.0]

        # Collection statistics: scores of every turn against this query.
        all_scores = [s for _, s in index.search(task.query.text, top_k=10_000)]
        collection_mean = statistics.fmean(all_scores) if all_scores else 0.0
        collection_std = statistics.pstdev(all_scores) if len(all_scores) > 1 else 0.0

        row: dict[str, float | str] = {
            "id": task.query.id,
            "abstention": 1.0 if task.query.is_abstention else 0.0,
        }
        for name, fn in PREDICTORS.items():
            row[name] = fn(scores, collection_mean, collection_std)
        rows.append(row)

    answerable = [r for r in rows if not r["abstention"]]
    abstention = [r for r in rows if r["abstention"]]

    out: dict[str, object] = {"questions": len(rows), "predictors": {}}
    print(
        f"\n{'predictor':20s} {'best bal':>9s} {'keep':>6s} {'refuse':>7s} {'refuse@keep.95':>15s}"
    )
    print("-" * 64)

    for name in PREDICTORS:
        a = [float(r[name]) for r in answerable]
        b = [float(r[name]) for r in abstention]

        # Direction is not assumed: a predictor may be higher for answerable
        # queries (most do) or higher for abstention ones. Test both and keep
        # the better, which is generous to the predictor on purpose -- if even
        # the favourable direction fails, the signal is genuinely absent.
        best: dict[str, float] = {"balanced": 0.0}
        for flip in (False, True):
            aa, bb = (a, b) if not flip else ([-x for x in a], [-x for x in b])
            lo, hi = min(aa + bb), max(aa + bb)
            for i in range(41):
                th = lo + (hi - lo) * i / 40
                keep = sum(1 for s in aa if s > th) / len(aa)
                refuse = sum(1 for s in bb if s <= th) / len(bb)
                bal = (keep + refuse) / 2
                if bal > best["balanced"]:
                    best = {
                        "balanced": round(bal, 3),
                        "keep": round(keep, 3),
                        "refuse": round(refuse, 3),
                        "threshold": round(th, 4),
                        "flipped": float(flip),
                    }

        # The number the goal actually cares about: refusal while holding 95% of
        # answerable queries.
        refuse_at_95 = 0.0
        for flip in (False, True):
            aa, bb = (a, b) if not flip else ([-x for x in a], [-x for x in b])
            lo, hi = min(aa + bb), max(aa + bb)
            for i in range(201):
                th = lo + (hi - lo) * i / 200
                keep = sum(1 for s in aa if s > th) / len(aa)
                if keep >= 0.95:
                    refuse = sum(1 for s in bb if s <= th) / len(bb)
                    refuse_at_95 = max(refuse_at_95, refuse)

        out["predictors"][name] = {**best, "refuse_at_keep_95": round(refuse_at_95, 3)}
        print(
            f"{name:20s} {best['balanced']:9.3f} {best['keep']:6.3f} "
            f"{best['refuse']:7.3f} {refuse_at_95:15.3f}"
        )

    print("-" * 64)
    print("for reference:  LLM judge 0.895   word-level coverage ~0.10")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "zh-qpp.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwritten to {RESULTS / 'zh-qpp.json'}")


if __name__ == "__main__":
    main()

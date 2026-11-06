"""Measure whether retrieval confidence can serve as an abstention signal.

This exists because it is the one hypothesis that would make Stage 1 cheap: if
a retriever's top score separated "there is an answer" from "there is not", the
system could abstain with a threshold and no extra machinery.

It does not. On LongMemEval the two score distributions overlap almost
completely -- abstention questions score as high as answerable ones, because
their decoys share vocabulary with the question (asks about a hamster, mentions
a cat). The best achievable balanced accuracy is barely above chance, and
reaching it costs roughly half of the answerable queries.

So Stage 1 cannot be a threshold. It has to model the question-content
relationship, not just rank content.

Run: python -m bench.abstention_probe
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from bench.loader import S_CLEANED, load_longmemeval
from bench.retrieval import BM25

RESULTS = Path(__file__).parent / "results"


def _percentile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(int(len(s) * p), len(s) - 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=S_CLEANED)
    ap.add_argument("--json-out", default=str(RESULTS / "abstention-probe.json"))
    args = ap.parse_args()

    tasks = load_longmemeval(args.dataset)
    answerable: list[float] = []
    abstention: list[float] = []

    for t in tasks:
        index = BM25()
        index.index([(m.id, m.content) for m in t.memories])
        hits = index.search(t.query.text, top_k=1)
        top = hits[0][1] if hits else 0.0
        (abstention if t.query.is_abstention else answerable).append(top)

    print(f"abstention n={len(abstention)}  answerable n={len(answerable)}")
    for label, xs in (("abstention", abstention), ("answerable", answerable)):
        print(
            f"  {label:11s} min={min(xs):7.2f}  p25={_percentile(xs, 0.25):7.2f}  "
            f"median={statistics.median(xs):7.2f}  p75={_percentile(xs, 0.75):7.2f}  "
            f"max={max(xs):7.2f}"
        )

    curve = []
    for i in range(0, 46, 2):
        th = float(i)
        abstain_rate = sum(1 for s in abstention if s <= th) / len(abstention)
        answer_rate = sum(1 for s in answerable if s > th) / len(answerable)
        curve.append(
            {
                "threshold": th,
                "abstain_rate": round(abstain_rate, 4),
                "answer_rate": round(answer_rate, 4),
                "balanced": round((abstain_rate + answer_rate) / 2, 4),
            }
        )

    best = max(curve, key=lambda r: r["balanced"])
    print()
    print("threshold  abstain   answer   balanced")
    for row in curve:
        print(
            f"{row['threshold']:9.0f}  {row['abstain_rate']:6.3f}   "
            f"{row['answer_rate']:6.3f}   {row['balanced']:6.3f}"
        )
    print()
    print(f"best: threshold={best['threshold']:.0f} balanced={best['balanced']:.3f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "n_abstention": len(abstention),
                "n_answerable": len(answerable),
                "curve": curve,
                "best": best,
                "chance_balanced": 0.5,
                "distributions": {
                    "abstention": {
                        "min": min(abstention),
                        "median": statistics.median(abstention),
                        "max": max(abstention),
                    },
                    "answerable": {
                        "min": min(answerable),
                        "median": statistics.median(answerable),
                        "max": max(answerable),
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"written to {args.json_out}")


if __name__ == "__main__":
    main()

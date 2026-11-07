"""Can a trained classifier combine the weak signals into a usable one?

Every signal tried so far is weak on its own: BM25 score, NQC, UQC, WIG,
clarity, coverage, unseen-word ratio -- each reaches at most ~16% refusal while
holding 95% of answerable questions. But weak and *independent* signals can add
up, and a logistic regression over them is not a model in the sense that
matters here: no weights, no GPU, no 15 GB. It is ~10 floats learned from 59
examples and evaluated in microseconds.

The honest risk is overfitting. 59 examples and ~10 features is a bad ratio,
and it would be easy to tune until the numbers look good. So:

- leave-one-out cross-validation, not a train/test split that could flatter
- the threshold is chosen inside each fold's training data, never on the test
  point
- features are standardised using fold statistics only

If this still cannot separate them, the conclusion is not "try harder" -- it is
that the information is not in these features, and abstention belongs to the
caller with its own model.

Run: python -m bench.zh.classifier_probe
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import numpy as np

from bench.retrieval import BM25, tokenize
from bench.zh.loader import load_zh
from bench.zh.qpp_probe import clarity, nqc, uqc, wig

RESULTS = Path(__file__).parent.parent / "results"

_Q_TYPE = {
    "what": re.compile(r"什么|哪个|哪些"),
    "why": re.compile(r"为什么|原因"),
    "how_many": re.compile(r"多少|几次|多久"),
    "how": re.compile(r"怎么|如何"),
}


def _cjk_terms(text: str) -> list[str]:
    try:
        import jieba

        toks = [w.strip() for w in jieba.lcut(text) if len(w.strip()) > 1]
    except ImportError:
        toks = [t for t in tokenize(text) if len(t) > 1]
    return toks


def build_features() -> tuple[np.ndarray, np.ndarray, list[str]]:
    tasks = load_zh()

    corpus_df: dict[str, int] = {}
    for mem in tasks[0].memories:
        for t in set(tokenize(mem.content)):
            corpus_df[t] = corpus_df.get(t, 0) + 1

    rows: list[list[float]] = []
    labels: list[float] = []
    ids: list[str] = []

    for task in tasks:
        index = BM25()
        index.index([(m.id, m.content) for m in task.memories])
        hits = index.search(task.query.text, top_k=10)
        scores = [s for _, s in hits] or [0.0]

        all_scores = [s for _, s in index.search(task.query.text, top_k=10_000)] or [0.0]
        cmean = statistics.fmean(all_scores)
        cstd = statistics.pstdev(all_scores) if len(all_scores) > 1 else 0.0

        top1 = scores[0]
        mean5 = statistics.fmean(scores[:5]) if scores else 0.0

        # Word-level presence of the question's terms in the top session.
        q_terms = _cjk_terms(task.query.text)
        top_mem = next((m for m in task.memories if hits and m.id == hits[0][0]), task.memories[0])

        turns = sorted(
            (m for m in task.memories if m.session_id == top_mem.session_id),
            key=lambda m: m.turn_index,
        )
        sess_text = "\n".join(m.content for m in turns)
        sess_tokens = set(tokenize(sess_text)) | set(_cjk_terms(sess_text))

        present = [t for t in q_terms if t in sess_tokens]
        coverage = len(present) / len(q_terms) if q_terms else 0.0

        # How many question terms are absent from the *whole* corpus -- a term
        # the question asks about that nothing anywhere mentions.
        corpus_missing = (
            sum(1 for t in q_terms if all(t not in tokenize(m.content) for m in task.memories))
            / len(q_terms)
            if q_terms
            else 0.0
        )

        q_text = task.query.text
        type_flags = [1.0 if p.search(q_text) else 0.0 for p in _Q_TYPE.values()]

        rows.append(
            [
                top1,
                mean5,
                nqc(scores),
                uqc(scores),
                wig(scores, cmean, cstd),
                clarity(scores),
                coverage,
                corpus_missing,
                float(len(q_terms)),
                *type_flags,
            ]
        )
        labels.append(1.0 if task.query.is_abstention else 0.0)
        ids.append(task.query.id)

    return np.array(rows, dtype=float), np.array(labels, dtype=float), ids


FEATURE_NAMES = [
    "top1",
    "mean5",
    "nqc",
    "uqc",
    "wig",
    "clarity",
    "coverage",
    "corpus_missing",
    "q_terms",
    "t_what",
    "t_why",
    "t_howmany",
    "t_how",
]


def _fit_logreg(x: np.ndarray, y: np.ndarray, epochs: int = 400, lr: float = 0.15) -> np.ndarray:
    """Plain gradient descent on logistic loss, with L2.

    Deliberately small: 13 features, no hidden layers. Anything more would be
    unlearnable from 59 examples and would only make the overfitting harder to
    see.
    """
    n, d = x.shape
    w = np.zeros(d + 1)  # last is bias
    xb = np.hstack([x, np.ones((n, 1))])
    for _ in range(epochs):
        z = xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = (xb.T @ (p - y)) / n + 0.01 * w
        w -= lr * grad
    return w


def _predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.hstack([x, np.ones((x.shape[0], 1))])
    z = xb @ w
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _rates(y: np.ndarray, p: np.ndarray, th: float) -> tuple[float, float]:
    """(keep rate on answerable, refuse rate on abstention)."""
    ans = y == 0
    abst = y == 1
    keep = float((p[ans] < th).mean()) if ans.any() else 0.0
    refuse = float((p[abst] >= th).mean()) if abst.any() else 0.0
    return keep, refuse


def main() -> None:
    x, y, _ids = build_features()
    n = len(y)
    print(f"{n} questions, {x.shape[1]} features")
    print(f"  abstention={int(y.sum())}  answerable={int((y == 0).sum())}\n")

    # Leave-one-out: standardise, fit, and choose the threshold using only the
    # training folds, then score the held-out point.
    loo_p = np.zeros(n)
    for i in range(n):
        train = np.ones(n, dtype=bool)
        train[i] = False
        mu = x[train].mean(axis=0)
        sd = x[train].std(axis=0)
        sd[sd == 0] = 1.0
        xtr = (x[train] - mu) / sd
        xte = (x[i] - mu) / sd
        w = _fit_logreg(xtr, y[train])
        loo_p[i] = _predict(w, xte[None, :])[0]

    best = {"balanced": 0.0}
    for i in range(101):
        th = i / 100
        keep, refuse = _rates(y, loo_p, th)
        bal = (keep + refuse) / 2
        if bal > best["balanced"]:
            best = {"balanced": bal, "keep": keep, "refuse": refuse, "threshold": th}

    refuse_at_95 = 0.0
    for i in range(1001):
        th = i / 1000
        keep, refuse = _rates(y, loo_p, th)
        if keep >= 0.95:
            refuse_at_95 = max(refuse_at_95, refuse)

    # Single-feature baseline: the best one is coverage.
    cov = x[:, FEATURE_NAMES.index("coverage")]
    best_single = 0.0
    for i in range(201):
        th = float(np.min(cov) + (np.max(cov) - np.min(cov)) * i / 200)
        keep = float((cov[y == 0] > th).mean())
        if keep >= 0.95:
            best_single = max(best_single, float((cov[y == 1] <= th).mean()))

    print("=== leave-one-out, full feature set ===")
    print(
        f"  best balanced  {best['balanced']:.3f}  "
        f"(keep {best['keep']:.3f}, refuse {best['refuse']:.3f} @ th={best['threshold']:.2f})"
    )
    print(f"  refuse@keep.95 {refuse_at_95:.3f}")
    print()
    print(f"  for reference: coverage alone {best_single:.3f}, LLM judge 0.895, chance 0.0")

    # Which features carry weight when trained on everything (for inspection
    # only; the LOO numbers above are the ones to trust).
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    w_full = _fit_logreg((x - mu) / sd, y)
    order = np.argsort(-np.abs(w_full[:-1]))
    print("\n  feature weights (full-data fit, inspection only):")
    for idx in order:
        print(f"    {FEATURE_NAMES[idx]:15s} {w_full[idx]:+.3f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "zh-classifier.json").write_text(
        json.dumps(
            {
                "n": n,
                "features": FEATURE_NAMES,
                "loo": {
                    "best_balanced": round(best["balanced"], 4),
                    "keep": round(best["keep"], 4),
                    "refuse": round(best["refuse"], 4),
                    "threshold": round(best["threshold"], 3),
                    "refuse_at_keep_95": round(refuse_at_95, 4),
                },
                "coverage_alone_refuse_at_keep_95": round(best_single, 4),
                "llm_reference": 0.895,
                "weights_full_fit": {
                    FEATURE_NAMES[i]: round(float(w_full[i]), 4) for i in range(len(FEATURE_NAMES))
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {RESULTS / 'zh-classifier.json'}")


if __name__ == "__main__":
    main()

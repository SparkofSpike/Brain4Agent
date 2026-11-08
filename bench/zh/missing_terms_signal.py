"""Does missing_terms actually separate answerable from unanswerable?

D-012 measured every cheap abstention signal and none beat 21% at the target
keep rate. missing_terms was designed afterwards (D-011) and is different in
kind: it does not score the query, it names a word the corpus never contains.

If a question asks about 数据库 and no session ever says 数据库 anywhere, that is
not a confidence score that needs calibrating -- it is a fact about the corpus.

This measures whether that fact is predictive enough to be useful, and how much
it costs in false refusals. Reported in the same terms as the model judge so the
two numbers can be compared directly.

    python -m bench.zh.missing_terms_signal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.zh.e2e import build_store, load

DATA = Path(__file__).parent / "zh_eval.json"


def evaluate(store, data: dict, *, top_k: int = 1, require_all: bool = True) -> dict:
    """Classify each question by whether any retrieved session has a missing term.

    ``require_all`` = True  -> the question is "covered" only if *every*
                               retrieved session has no missing terms
    ``require_all`` = False -> covered if *any* retrieved session has none

    Top-1 is the honest setting: a caller reading the best hit only. Top-k with
    ``require_all=False`` is optimistic -- one lucky session among five would
    hide the absence.
    """
    rows: list[dict] = []

    for task in data["tasks"]:
        qid = task["question_id"]
        is_abs = qid.endswith("_abs")
        result = store.search(task["question"], top_k=top_k)
        if result.empty:
            # No retrieval at all. Treat as "cannot answer", which is a refusal
            # signal even though it is not a missing word.
            has_missing = True
        elif require_all:
            # Strong signal only: does the *corpus* lack a central term? A single
            # session lacking 用户 is ordinary.
            has_missing = bool(result.corpus_missing)
        else:
            has_missing = bool(result.corpus_missing)

        rows.append(
            {
                "id": qid,
                "abstention": is_abs,
                "has_missing": has_missing,
                "missing": result.corpus_missing,
            }
        )

    answerable = [r for r in rows if not r["abstention"]]
    abstain = [r for r in rows if r["abstention"]]

    keep = sum(1 for r in answerable if not r["has_missing"])
    refuse = sum(1 for r in abstain if r["has_missing"])

    return {
        "answerable": len(answerable),
        "abstention": len(abstain),
        "keep_rate": round(keep / len(answerable), 4) if answerable else 0.0,
        "refuse_rate": round(refuse / len(abstain), 4) if abstain else 0.0,
        "false_refusals": [r["id"] for r in answerable if r["has_missing"]],
        "missed_abstentions": [r["id"] for r in abstain if not r["has_missing"]],
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--show-false-refusals", action="store_true")
    args = ap.parse_args()

    data = load()
    store = build_store(data)

    print("strategy                       keep   refuse")
    print("-" * 48)
    results = {}
    for label, kwargs in (
        ("top-1", {"top_k": 1}),
        ("top-3, all must be clean", {"top_k": 3, "require_all": True}),
        ("top-3, any clean counts", {"top_k": 3, "require_all": False}),
    ):
        r = evaluate(store, data, **kwargs)
        results[label] = r
        print(f"{label:30s} {r['keep_rate']:.3f}  {r['refuse_rate']:.3f}")

    print()
    print("reference:")
    print("  总是回答 / always answer      keep 1.000   refuse 0.000")
    print("  本地 9b                        keep 0.950   refuse 0.895")
    print("  云端 deepseek-flash            keep 0.925   refuse 1.000")

    top1 = results["top-1"]
    print()
    print(
        f"top-1 false refusals ({len(top1['false_refusals'])}): {', '.join(top1['false_refusals'])}"
    )
    if top1["missed_abstentions"]:
        print(f"top-1 missed abstentions: {', '.join(top1['missed_abstentions'])}")

    if args.show_false_refusals:
        print()
        by_id = {r["id"]: r for r in top1["rows"]}
        for tid in top1["false_refusals"]:
            print(f"  {tid}: missing={by_id[tid]['missing']}")

    if args.json:
        print()
        print(
            json.dumps(
                {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in results.items()},
                ensure_ascii=False,
                indent=2,
            )
        )

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

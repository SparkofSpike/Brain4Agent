"""Score Stage 1 against Stage 0 on the Chinese set.

The constraint from the goal is explicit: keep at least 95% of answerable
queries, then push abstention as high as it goes. A system that abstains on
everything would score perfectly on abstention and be useless, so the two
numbers are always reported together and never averaged.

Run: python -m bench.zh.run_stage1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bench.zh.loader import load_zh
from bench.zh.stage1 import JudgeConfig, answer, judge_available

RESULTS = Path(__file__).parent.parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:2b")
    ap.add_argument("--first-pass", type=int, default=3)
    ap.add_argument("--second-pass", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json-out", default=str(RESULTS / "zh-stage1.json"))
    args = ap.parse_args()

    cfg = JudgeConfig(model=args.model, first_pass=args.first_pass, second_pass=args.second_pass)
    if not judge_available(cfg):
        raise SystemExit(
            f"{args.model} not available on {cfg.host}. "
            f"Run `ollama list` or pass --model with an installed one."
        )

    tasks = load_zh(limit=args.limit)
    print(
        f"{len(tasks)} questions, judge={args.model}, passes={args.first_pass}/{args.second_pass}"
    )

    answerable = [t for t in tasks if not t.query.is_abstention]
    abstention = [t for t in tasks if t.query.is_abstention]

    kept = 0
    correct = 0
    wrong_session = 0
    judge_errors = 0
    t_start = time.perf_counter()

    print("\n--- answerable ---")
    for i, t in enumerate(answerable, 1):
        d = answer(t.memories, t.query.text, cfg)
        if any(not j.answers and j.raw and j.raw.upper()[0] not in "YN" for j in d.judgements):
            judge_errors += 1
        if d.abstained:
            continue
        kept += 1
        if d.found and d.found.session_id in t.query.gold_session_ids:
            correct += 1
            mark = "ok"
        else:
            wrong_session += 1
            mark = "WRONG-SESSION"
        if i % 10 == 0 or mark != "ok":
            print(f"  [{i}/{len(answerable)}] {mark}  {t.query.text[:40]}")

    print("\n--- abstention ---")
    refused = 0
    for i, t in enumerate(abstention, 1):
        d = answer(t.memories, t.query.text, cfg)
        if d.abstained:
            refused += 1
            mark = "refused"
        else:
            mark = "ANSWERED"
        print(f"  [{i}/{len(abstention)}] {mark:9s} {t.query.text[:44]}")

    elapsed = time.perf_counter() - t_start
    keep_rate = kept / len(answerable) if answerable else 0.0
    refuse_rate = refused / len(abstention) if abstention else 0.0
    precision = correct / kept if kept else 0.0

    report = {
        "model": args.model,
        "first_pass": args.first_pass,
        "second_pass": args.second_pass,
        "questions": len(tasks),
        "answerable": len(answerable),
        "abstention": len(abstention),
        "answerable_kept": kept,
        "keep_rate": round(keep_rate, 4),
        "answered_correctly": correct,
        "answered_wrong_session": wrong_session,
        "precision_among_kept": round(precision, 4),
        "abstention_refused": refused,
        "refuse_rate": round(refuse_rate, 4),
        "judge_errors": judge_errors,
        "seconds": round(elapsed, 1),
        "constraint_keep_ge_0_95": keep_rate >= 0.95,
    }

    print("\n=== Stage 1 ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # Stage 0 baseline for comparison: it always answers, so keep_rate = 1.0
    # and refuse_rate = 0.0 by construction.
    print("\n=== vs Stage 0 (always answers) ===")
    print(f"  keep_rate   1.0000 -> {keep_rate:.4f}")
    print(f"  refuse_rate 0.0000 -> {refuse_rate:.4f}")
    print(f"  constraint met: {report['constraint_keep_ge_0_95']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwritten to {args.json_out}")


if __name__ == "__main__":
    main()

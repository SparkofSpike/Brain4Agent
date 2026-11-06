"""Validate the Chinese evaluation set.

A benchmark that is subtly wrong produces confident conclusions from bad data,
which is the failure mode this project exists to prevent. So the set checks
itself before it is ever scored.

Two levels, deliberately separated:

**Hard errors** stop the build. A gold session that does not exist means the
dataset is internally inconsistent -- no amount of interpretation fixes that.

**Warnings** are for human review. Character-overlap heuristics cannot tell
whether an answer is truly present; they can only point at questions worth
looking at. Reporting them as errors would train the reader to ignore them.

Abstention correctness is not checkable by overlap at all: the answer field
holds a *reason* ("never mentioned a database"), whose characters naturally
appear in the corpus. Those questions were checked by hand against the top-2
retrieval results, and that review is recorded in docs/decisions.md D-009.
Pretending a substring test could replace it would be exactly the kind of
false assurance this project is against.
"""

from __future__ import annotations

import json
from pathlib import Path

ZH_EVAL = Path(__file__).parent / "zh_eval.json"

_COMMON = set("的了是在和与就都而及或一个我你他她它这那有看把被让给对吧呢吗啊呀哦嗯")


def _content_chars(text: str) -> set[str]:
    return {ch for ch in text if ch.strip() and ch not in _COMMON and not ch.isascii()}


def _overlap(answer: str, corpus: str) -> float:
    a = _content_chars(answer)
    if not a:
        return 0.0
    return len(a & _content_chars(corpus)) / len(a)


def validate(path: Path = ZH_EVAL) -> tuple[list[str], list[str]]:
    """Return (hard errors, warnings)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw["tasks"]

    errors: list[str] = []
    warnings: list[str] = []

    session_ids = set(entries[0]["haystack_session_ids"])
    id_to_text = {
        sid: "\n".join(m["content"] for m in sess)
        for sid, sess in zip(
            entries[0]["haystack_session_ids"],
            entries[0]["haystack_sessions"],
            strict=True,
        )
    }

    seen: set[str] = set()
    for e in entries:
        qid = e["question_id"]
        if qid in seen:
            errors.append(f"{qid}: duplicate question id")
        seen.add(qid)

        gold = e.get("answer_session_ids", [])
        for sid in gold:
            if sid not in session_ids:
                errors.append(f"{qid}: gold session {sid} not in corpus")

        if gold:
            gold_text = "\n".join(id_to_text[s] for s in gold if s in id_to_text)
            overlap = _overlap(e["answer"], gold_text)
            if overlap < 0.55:
                warnings.append(
                    f"{qid}: answer only {overlap:.0%} overlaps its gold session "
                    f"(paraphrase or mislabel) -- worth a look"
                )

    if len(session_ids) < 10:
        warnings.append(
            f"corpus has only {len(session_ids)} sessions -- retrieval will be "
            f"easy and may not separate systems"
        )

    return errors, warnings


def main() -> None:
    errors, warnings = validate()
    total = len(json.loads(ZH_EVAL.read_text(encoding="utf-8"))["tasks"])
    print(f"checked {total} questions")
    print(f"  hard errors: {len(errors)}")
    print(f"  warnings:    {len(warnings)}")
    for label, items in (("ERROR", errors), ("WARN", warnings)):
        for item in items:
            print(f"  {label}  {item}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

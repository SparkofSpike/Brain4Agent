"""PrecisionMemBench: run the public retrieval benchmark on this store.

Why this benchmark first: it needs no generation model and no judge model. The
answer is the set of belief IDs the store returns, and the dataset asserts which
IDs must and must not appear. That makes it the one public benchmark a
model-free system can run honestly, and it tests the thing this project claims
-- not pouring irrelevant memories into the context.

Scoring is ported from vectorize-io/agent-memory-benchmark
(`src/memory_bench/dataset/precisionmembench.py`), which itself ports upstream
tenurehq/precisionmembench `src/retrieval.external.eval.test.ts`. The only part
that depends on the memory system is `relevantBeliefs`; the other three tiers
(personaPrelude, pinnedFacts, openQuestions) are derived from the seed corpus,
so a store that returns nothing still passes cases whose assertions only touch
those tiers. `active_pass` is the number comparable to the published column.

Data files are the upstream fixtures at commit b95d6ab, MIT licensed.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_BUDGET = {"maxBeliefs": 20, "maxPinnedFacts": 10, "maxQuestions": 15}

# Verbatim from upstream retrieval.external.eval.test.ts EVAL_PERSONA.
PERSONA_UNIVERSAL = (
    "You prefer direct answers without preamble. You push back when plans have "
    "problems rather than defaulting to agreement. You edit AI output; you do not "
    "let AI edit your prose."
)

DATA_DIR = Path(__file__).with_name("data")


def belief_to_text(b: dict) -> str:
    """Verbatim port of upstream BaseAdapter.beliefToText (canonical_name_aliases)."""
    parts = [
        b.get("canonical_name"),
        *(b.get("aliases") or []),
        b.get("content"),
        b.get("why_it_matters"),
    ]
    return " ".join(p for p in parts if p)


def load() -> tuple[list[dict], list[dict]]:
    beliefs = json.loads((DATA_DIR / "beliefs.seed.json").read_text("utf-8"))
    cases = json.loads((DATA_DIR / "retrieval.cases.json").read_text("utf-8"))
    return beliefs, cases


def _pinned_facts(beliefs: list[dict], user_id: str, scope: list[str]) -> list[str]:
    return [
        b["_id"]
        for b in beliefs
        if b["user_id"] == user_id
        and b.get("pinned") is True
        and b.get("type") != "open_question"
        and not b.get("superseded_by")
        and not b.get("resolved_at")
        and any(s in scope for s in b.get("scope") or [])
    ]


def _pinned_questions(beliefs: list[dict], user_id: str, scope: list[str]) -> list[str]:
    return [
        b["_id"]
        for b in beliefs
        if b["user_id"] == user_id
        and b.get("type") == "open_question"
        and b.get("pinned") is True
        and not b.get("resolved_at")
        and any(s in scope for s in b.get("scope") or [])
    ]


def score_case(
    case: dict,
    beliefs: list[dict],
    retrieved: list[str],
    *,
    user_id: str = "test-user",
) -> tuple[bool, list[str], bool]:
    """Score one case. Returns (passed, failures, is_active).

    ``retrieved`` is the ordered list of belief IDs the store returned.
    """
    expect = case.get("expect") or {}
    scope = case.get("scope") or []
    budget = {**DEFAULT_BUDGET, **(case.get("budget") or {})}
    cap = budget["maxBeliefs"]

    pinned_all = _pinned_facts(beliefs, user_id, scope)
    questions = _pinned_questions(beliefs, user_id, scope)
    pinned_all_set = set(pinned_all)

    # The provider's results, split the way upstream splits them.
    searched = [b for b in retrieved if b]
    raw_results = [b for b in searched if b not in pinned_all_set][:cap]
    all_relevant = raw_results  # no relation expansion in this store
    capped_pinned = pinned_all[:cap]
    capped_relevant = all_relevant[: max(0, cap - len(capped_pinned))]

    pinned_ids = set(capped_pinned)
    relevant_set = set(capped_relevant)
    question_ids = set(questions[: budget["maxQuestions"]])
    union_ids = pinned_ids | relevant_set
    prelude = PERSONA_UNIVERSAL if user_id == "test-user" else ""

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    rb = expect.get("relevantBeliefs") or {}
    for bid in rb.get("mustInclude") or []:
        check(bid in union_ids, f"missing expected belief: {bid}")
    for bid in rb.get("mustExclude") or []:
        check(bid not in union_ids, f"forbidden belief surfaced: {bid}")
    for bid in rb.get("shouldInclude") or []:
        check(bid in union_ids, f"expected belief missing (shouldInclude): {bid}")

    only_expected = rb.get("shouldOnlyInclude")
    if only_expected is not None:
        expected_set = set(only_expected)
        for bid in capped_relevant:
            check(bid in expected_set, f"unexpected belief in relevantBeliefs: {bid}")
        for bid in expected_set:
            check(bid in relevant_set, f"missing expected belief: {bid}")

    if rb.get("maxCount") is not None:
        check(
            len(relevant_set) <= rb["maxCount"],
            f"relevantBeliefs count {len(relevant_set)} > maxCount {rb['maxCount']}",
        )
    if rb.get("minCount") is not None:
        check(
            len(relevant_set) >= rb["minCount"],
            f"relevantBeliefs count {len(relevant_set)} < minCount {rb['minCount']}",
        )

    for a, b_ in rb.get("orderedBefore") or []:
        ia = capped_relevant.index(a) if a in capped_relevant else -1
        ib = capped_relevant.index(b_) if b_ in capped_relevant else -1
        check(ia != -1, f"orderedBefore: {a} not in relevantBeliefs")
        check(ib != -1, f"orderedBefore: {b_} not in relevantBeliefs")
        if ia != -1 and ib != -1:
            check(ia < ib, f"ranking: {a} (idx {ia}) should precede {b_} (idx {ib})")

    pf = expect.get("pinnedFacts") or {}
    for bid in pf.get("mustInclude") or []:
        check(bid in pinned_ids, f"missing pinned belief: {bid}")
    for bid in pf.get("mustExclude") or []:
        check(bid not in pinned_ids, f"forbidden belief in pinnedFacts: {bid}")

    oq = expect.get("openQuestions") or {}
    for bid in oq.get("mustInclude") or []:
        check(bid in question_ids, f"missing expected question: {bid}")
    for bid in oq.get("mustExclude") or []:
        check(bid not in question_ids, f"forbidden question surfaced: {bid}")

    pp = expect.get("personaPrelude") or {}
    if pp.get("nonEmpty"):
        check(len(prelude) > 0, "personaPrelude empty")
    if pp.get("isNull"):
        check(prelude == "", "personaPrelude not empty")
    for s in pp.get("contains") or []:
        check(s in prelude, f'personaPrelude missing "{s}"')
    for s in pp.get("mustNotContain") or []:
        check(s not in prelude, f'personaPrelude contains "{s}"')

    expected_relevant = (
        set(only_expected) if only_expected is not None else set(rb.get("mustInclude") or [])
    )
    is_active = bool(expected_relevant)
    return (not failures), failures, is_active

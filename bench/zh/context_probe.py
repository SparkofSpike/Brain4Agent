"""How much information does a query need before same-topic noise stops winning?

D-020 showed hit@1 collapsing from 0.925 to 0.000 once 100 same-topic sessions
are added. The diagnosis was that a vague non-answer repeats the question's words
while a precise answer contains file names and line numbers the question never
mentions.

If that diagnosis is right, adding concrete detail to the *query* should recover
the ranking. This measures the upper bound: entities are taken from the gold
session, so the result is an oracle, not something a user could type unaided.

What it tells us:

- if even the oracle cannot recover, the diagnosis is wrong and something deeper
  is going on
- if it recovers with one entity, a realistic hint is enough and the advice is
  "be specific"
- if it needs every entity, the query must nearly restate the answer, which is
  not a usable interface

    python -m bench.zh.context_probe
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from brain4agent import MemoryStore

DATA = Path(__file__).parent / "zh_eval.json"

# The same concrete-token shape as concreteness_probe: addresses, paths, file
# names, line references, versions, hashes.
ENTITY = re.compile(
    r"\d+\.\d+\.\d+\.\d+"
    r"|/[\w\-./]+"
    r"|\b[\w\-]+\.(?:py|js|ts|json|md|toml|yaml|yml|sh|service)\b"
    r"|第\s*\d+\s*行"
    r"|\b[0-9a-f]{7,40}\b"
    r"|\bv?\d+\.\d+(?:\.\d+)?\b"
)


def entities(text: str) -> list[str]:
    """Distinct concrete tokens, longest first, original order otherwise."""
    seen: list[str] = []
    for m in ENTITY.finditer(text):
        tok = m.group(0).strip()
        if tok and tok not in seen:
            seen.append(tok)
    return sorted(seen, key=len, reverse=True)


def substitute_same_answer(text: str, k: int) -> str:
    """Replace every entity the extractor can find, not just the obvious ones.

    Missing one leaves the variant sharing a token with the original, so the two
    tie for a reason that has nothing to do with retrieval quality. The first
    version of this replaced four patterns and left commit hashes alone, which
    put the gold session at rank 8 behind seven identical-hash variants.
    """
    out = text
    out = out.replace("203.0.113.42", f"10.20.30.{40 + k % 10}")
    out = out.replace("/opt/demo-app", f"/opt/app-{k % 10}")
    out = out.replace("keywords.py", f"mod{k % 10}.py")
    out = out.replace("第 443 行", f"第 {443 + k * 7} 行")
    out = out.replace("4a29ae6", f"{0xA00000 + k:07x}")
    out = out.replace("beae716", f"{0xB00000 + k:07x}")
    out = out.replace("0.10.0", f"0.9.{k % 10}")
    return out


def build(data: dict, noise: int) -> tuple[MemoryStore, list[dict], dict]:
    task = data["tasks"][0]
    ids, sessions = task["haystack_session_ids"], task["haystack_sessions"]
    by_id = dict(zip(ids, sessions, strict=True))
    questions = [t for t in data["tasks"] if not t["question_id"].endswith("_abs")]

    store = MemoryStore(Path(tempfile.mkdtemp()) / "ctx.db")
    with store.batch():
        made = k = 0
        while made < noise:  # noise first: originals must not win on insertion
            for sid, turns in zip(ids, sessions, strict=True):
                if made >= noise:
                    break
                for turn in turns:
                    store.add(
                        substitute_same_answer(turn["content"], k),
                        session_id=f"{sid}-n{k}",
                        role=turn["role"],
                    )
                made += 1
                k += 1
        for sid, turns in zip(ids, sessions, strict=True):
            for turn in turns:
                store.add(turn["content"], session_id=sid, role=turn["role"])
    return store, questions, by_id


def gold_text(by_id: dict, q: dict) -> str:
    turns = by_id[q["answer_session_ids"][0]]
    return "\n".join(t["content"] for t in turns)


def evaluate(store: MemoryStore, questions: list[dict], by_id: dict, mode: str) -> dict:
    hits = {1: 0, 5: 0, 10: 0}
    rr = 0.0
    for q in questions:
        query = q["question"]
        ents = entities(gold_text(by_id, q))
        if mode == "oracle-one" and ents:
            query = f"{query} {ents[0]}"
        elif mode == "oracle-three" and ents:
            query = f"{query} {' '.join(ents[:3])}"
        elif mode == "oracle-all" and ents:
            query = f"{query} {' '.join(ents)}"

        ranked = [e.session_id for e in store.search(query, top_k=10).evidence]
        gold = set(q["answer_session_ids"])
        pos = next((i + 1 for i, s in enumerate(ranked) if s in gold), None)
        for n in hits:
            if pos and pos <= n:
                hits[n] += 1
        if pos:
            rr += 1.0 / pos
    n = len(questions)
    return {"hit@1": hits[1] / n, "hit@5": hits[5] / n, "hit@10": hits[10] / n, "mrr": rr / n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=int, default=300)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    data = json.loads(DATA.read_text("utf-8"))
    store, questions, by_id = build(data, args.noise)
    store.search("预热")

    print(f"noise sessions: {args.noise}, written first")
    print(f"{'query':14s} {'hit@1':>7} {'hit@5':>7} {'hit@10':>7} {'MRR':>7}")
    print("-" * 48)

    rows = []
    for mode in ("plain", "oracle-one", "oracle-three", "oracle-all"):
        r = evaluate(store, questions, by_id, mode)
        rows.append({"mode": mode, **r})
        print(
            f"{mode:14s} {r['hit@1']:7.4f} {r['hit@5']:7.4f} "
            f"{r['hit@10']:7.4f} {r['mrr']:7.4f}"
        )

    store.close()
    if args.json:
        print()
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

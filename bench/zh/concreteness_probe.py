"""Does rewarding concrete values separate answers from talk about answers?

The failure this probes: asked "which file and line is the root cause in", a
session saying "root cause is keywords.py line 443" scores *below* one saying
"looked at some file, some line", because the second repeats the question's
words verbatim and the first does not.

An agent's memory has a distinguishing property: real answers carry concrete
values — paths, file names, addresses, line numbers, versions. Text that only
talks about a question tends to be abstract.

This measures whether a concreteness bonus on top of BM25 recovers the ranking,
and what it costs on the clean corpus.

    python -m bench.zh.concreteness_probe
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from brain4agent import MemoryStore

DATA = Path(__file__).parent / "zh_eval.json"

CONCRETE = re.compile(
    r"\d+\.\d+\.\d+\.\d+"  # IPv4
    r"|/[\w\-./]+"  # absolute paths
    r"|\b[\w\-]+\.(?:py|js|ts|json|md|toml|yaml|yml|sh|service)\b"  # file names
    r"|\b\d{2,}\b"  # numbers of two or more digits
    r"|\bv?\d+\.\d+(?:\.\d+)?\b"  # versions
    r"|\b[0-9a-f]{7,40}\b"  # commit hashes
)


def concreteness(text: str) -> int:
    """Count distinct concrete tokens. Distinct, not total: repeating an IP is
    not more informative than stating it once."""
    return len(set(m.group(0) for m in CONCRETE.finditer(text)))


def substitute_same_answer(text: str, k: int) -> str:
    out = text
    out = out.replace("203.0.113.42", f"10.20.30.{40 + k % 10}")
    out = out.replace("/opt/demo-app", f"/opt/app-{k % 10}")
    out = out.replace("keywords.py", f"mod{k % 10}.py")
    out = out.replace("第 443 行", f"第 {443 + k * 7} 行")
    return out


def to_unanswered(text: str) -> str:
    out = text
    out = out.replace("203.0.113.42", "一个记不清的地址")
    out = out.replace("/opt/demo-app", "某个目录")
    out = out.replace("keywords.py", "某个文件")
    out = out.replace("第 443 行", "某一行")
    return out


def build(data: dict, noise: int, mode: str) -> tuple[MemoryStore, list[dict]]:
    task = data["tasks"][0]
    ids, sessions = task["haystack_session_ids"], task["haystack_sessions"]
    questions = [t for t in data["tasks"] if not t["question_id"].endswith("_abs")]

    store = MemoryStore(Path(tempfile.mkdtemp()) / "c.db")

    def add_noise() -> None:
        made = k = 0
        while made < noise:
            for sid, turns in zip(ids, sessions, strict=True):
                if made >= noise:
                    break
                for turn in turns:
                    if mode == "unanswered" and turn["role"] == "assistant":
                        text = to_unanswered(turn["content"])
                    elif mode == "unanswered":
                        text = turn["content"]
                    else:
                        text = substitute_same_answer(turn["content"], k)
                    store.add(text, session_id=f"{sid}-n{k}", role=turn["role"])
                made += 1
                k += 1

    add_noise()  # noise first: originals must not win on insertion order
    for sid, turns in zip(ids, sessions, strict=True):
        for turn in turns:
            store.add(turn["content"], session_id=sid, role=turn["role"])
    return store, questions


def evaluate(store: MemoryStore, questions: list[dict], alpha: float) -> dict:
    hits = {1: 0, 5: 0, 10: 0}
    rr = 0.0
    for q in questions:
        result = store.search(q["question"], top_k=30)
        scored = []
        for e in result.evidence:
            c = concreteness(store.session_text(e.session_id))
            # Saturating: ten distinct concrete values is already decisive.
            bonus = 1.0 + alpha * min(c / 10.0, 1.0)
            scored.append((e, e.score * bonus))
        scored.sort(key=lambda kv: -kv[1])
        ranked = [e.session_id for e, _ in scored[:10]]

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
    rows = []

    print(f"noise sessions: {args.noise}, written first")
    print(f"{'mode':12s} {'alpha':>6} {'hit@1':>7} {'hit@5':>7} {'hit@10':>7} {'MRR':>7}")
    print("-" * 52)

    for mode in ("unanswered", "same-answer"):
        store, questions = build(data, args.noise, mode)
        store.search("预热")
        for alpha in (0.0, 0.5, 1.0, 2.0, 4.0):
            r = evaluate(store, questions, alpha)
            rows.append({"mode": mode, "alpha": alpha, **r})
            print(
                f"{mode:12s} {alpha:6.1f} {r['hit@1']:7.4f} {r['hit@5']:7.4f} "
                f"{r['hit@10']:7.4f} {r['mrr']:7.4f}"
            )
        store.close()
        print()

    # Clean corpus cost: does the bonus hurt when there is no noise?
    clean = MemoryStore(Path(tempfile.mkdtemp()) / "clean.db")
    task = data["tasks"][0]
    for sid, turns in zip(task["haystack_session_ids"], task["haystack_sessions"], strict=True):
        for turn in turns:
            clean.add(turn["content"], session_id=sid, role=turn["role"])
    clean.search("预热")
    qs = [t for t in data["tasks"] if not t["question_id"].endswith("_abs")]
    print("clean corpus (44 sessions):")
    for alpha in (0.0, 0.5, 1.0, 2.0, 4.0):
        r = evaluate(clean, qs, alpha)
        rows.append({"mode": "clean", "alpha": alpha, **r})
        print(
            f"{'clean':12s} {alpha:6.1f} {r['hit@1']:7.4f} {r['hit@5']:7.4f} "
            f"{r['hit@10']:7.4f} {r['mrr']:7.4f}"
        )
    clean.close()

    if args.json:
        print()
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

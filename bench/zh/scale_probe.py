"""Does retrieval survive a realistic number of memories?

Every number this project reports comes from 44 sessions. A working agent
accumulates thousands. If hit@1 decays with corpus size, the benchmarks describe
a toy.

**Two kinds of noise, and the difference decides what the result means.**

``same-answer`` swaps only the answer values, keeping every topic word. The
variant says "the root cause is tokenizer.py line 512" while the original says
"keywords.py line 443". Both answer the question; they disagree. This is the
common case in a real library — three projects, three deployments, each with a
correct address — and BM25 cannot separate them, because the question's words
are in both.

``unanswered`` replaces the assistant turn with a non-answer that keeps the
question's vocabulary: "looked into it, did not find the location". A true
negative, and a different question: can the system tell "I know this" from "I
know of this".

Reporting them separately matters. The first measures whether hit@1 is a
meaningful metric at scale. The second measures whether the evidence layer works.

``--order`` controls write order, which decides ties. Originals written first
beat score-identical noise for free, so ``noise-first`` is the honest direction.

    python -m bench.zh.scale_probe --mode same-answer --order noise-first
    python -m bench.zh.scale_probe --mode unanswered --order noise-first
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from brain4agent import MemoryStore

DATA = Path(__file__).parent / "zh_eval.json"

CITIES = ["成都", "西安", "南京", "武汉", "苏州", "天津", "重庆", "长沙", "郑州", "合肥"]
PROJECTS = [
    "素材分发",
    "内容推送",
    "文章同步",
    "稿件发布",
    "渠道管理",
    "资源调度",
    "文案投放",
    "内容分发",
    "素材管理",
    "稿件调度",
]
IPS = [f"10.20.30.{n}" for n in range(40, 50)]
PATHS = [
    "/opt/alpha-test",
    "/opt/beta-test",
    "/opt/gamma-test",
    "/opt/delta-test",
    "/srv/app-a",
    "/srv/app-b",
    "/srv/app-c",
    "/home/deploy/x",
    "/var/www/site1",
    "/var/www/site2",
]
FILES = [
    "tokenizer.py",
    "segmenter.py",
    "parser.py",
    "matcher.py",
    "indexer.py",
    "extractor.py",
    "splitter.py",
    "lexer.py",
    "normaliser.py",
    "scanner.py",
]


def substitute_soft(text: str, k: int) -> str:
    """Change topic entities too. Easy to reject, as a sanity baseline."""
    out = text
    out = out.replace("杭州", CITIES[k % len(CITIES)])
    out = out.replace("媒体投稿", PROJECTS[(k // len(CITIES)) % len(PROJECTS)])
    out = out.replace("203.0.113.42", IPS[(k // 7) % len(IPS)])
    out = out.replace("/opt/demo-app", PATHS[(k // 11) % len(PATHS)])
    out = out.replace("keywords.py", FILES[(k // 13) % len(FILES)])
    out = out.replace("114.215", "172.16")
    out = out.replace("E 盘", "F 盘")
    return out


def substitute_same_answer(text: str, k: int) -> str:
    """Change only the answer, keep every topic word.

    Produces a plausible different memory: same subject, different fact. It
    answers the question, wrongly for this user. BM25 cannot prefer either.
    """
    out = text
    out = out.replace("203.0.113.42", IPS[k % len(IPS)])
    out = out.replace("/opt/demo-app", PATHS[(k // 3) % len(PATHS)])
    out = out.replace("keywords.py", FILES[(k // 5) % len(FILES)])
    out = out.replace("第 443 行", f"第 {443 + k * 7} 行")
    out = out.replace("4a29ae6", f"{k:07x}")
    out = out.replace("beae716", f"{k + 9999:07x}")
    return out


def to_unanswered(text: str) -> str:
    """Strip the factual payload, keep the vocabulary that raises the topic."""
    out = text
    out = out.replace("203.0.113.42", "一个记不清的地址")
    out = out.replace("/opt/demo-app", "某个目录")
    out = out.replace("keywords.py", "某个文件")
    out = out.replace("第 443 行", "某一行")
    return out


def base_corpus(data: dict) -> tuple[list[str], list[list[dict]], list[dict]]:
    task = data["tasks"][0]
    return (
        list(task["haystack_session_ids"]),
        list(task["haystack_sessions"]),
        [t for t in data["tasks"] if not t["question_id"].endswith("_abs")],
    )


def build(
    ids, sessions, noise: int, mode: str, order: str, b: float = 0.75, k1: float = 1.5
) -> MemoryStore:
    tmp = Path(tempfile.mkdtemp()) / f"scale_{noise}_{mode}_{order}_{b}.db"
    store = MemoryStore(tmp, bm25_b=b, bm25_k1=k1)

    def add_original() -> None:
        with store.batch():
            for sid, turns in zip(ids, sessions, strict=True):
                for turn in turns:
                    store.add(turn["content"], session_id=sid, role=turn["role"])

    def add_noise() -> None:
        with store.batch():
            made = 0
            k = 0
            while made < noise:
                for sid, turns in zip(ids, sessions, strict=True):
                    if made >= noise:
                        break
                    variant = f"{sid}-n{k}"
                    for turn in turns:
                        if mode == "unanswered" and turn["role"] == "assistant":
                            text = to_unanswered(turn["content"])
                        elif mode == "unanswered":
                            text = turn["content"]
                        elif mode == "soft":
                            text = substitute_soft(turn["content"], k)
                        else:
                            text = substitute_same_answer(turn["content"], k)
                        store.add(text, session_id=variant, role=turn["role"])
                    made += 1
                    k += 1

    if order == "noise-first":
        add_noise()
        add_original()
    else:
        add_original()
        add_noise()
    return store


def evaluate(store: MemoryStore, questions: list[dict], k: int = 10) -> dict:
    hits = {1: 0, 5: 0, 10: 0}
    rr = 0.0
    for q in questions:
        ranked = [e.session_id for e in store.search(q["question"], top_k=k).evidence]
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
    ap.add_argument("--scales", default="0,100,300,1000")
    ap.add_argument("--mode", default="same-answer", choices=["same-answer", "unanswered", "soft"])
    ap.add_argument("--order", default="original-first", choices=["original-first", "noise-first"])
    ap.add_argument("--b", type=float, default=0.75, help="BM25 length normalisation")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    data = json.loads(DATA.read_text("utf-8"))
    ids, sessions, questions = base_corpus(data)
    scales = [int(x) for x in args.scales.split(",")]

    rows = []
    print(f"noise={args.mode}  order={args.order}  b={args.b}  k1={args.k1}")
    print(
        f"{'noise':>6} {'sessions':>9} {'turns':>7} {'build':>8} {'query':>8} "
        f"{'hit@1':>7} {'hit@5':>7} {'MRR':>7}"
    )
    print("-" * 76)

    for noise in scales:
        t0 = time.perf_counter()
        store = build(ids, sessions, noise, args.mode, args.order, b=args.b, k1=args.k1)
        build_s = time.perf_counter() - t0
        store.search("预热")  # force index construction

        t1 = time.perf_counter()
        r = evaluate(store, questions)
        query_ms = (time.perf_counter() - t1) / len(questions) * 1000
        st = store.stats()
        rows.append(
            {
                "noise": noise,
                "sessions": st["sessions"],
                "turns": st["turns"],
                "build_s": round(build_s, 2),
                "query_ms": round(query_ms, 2),
                **r,
            }
        )
        print(
            f"{noise:6d} {st['sessions']:9d} {st['turns']:7d} {build_s:7.2f}s "
            f"{query_ms:7.2f}ms {r['hit@1']:7.4f} {r['hit@5']:7.4f} {r['mrr']:7.4f}"
        )
        store.close()

    if args.json:
        print()
        print(json.dumps(rows, ensure_ascii=False, indent=2))

    base = rows[0]
    print()
    print("delta against the noise-free corpus:")
    for r in rows[1:]:
        print(
            f"  {r['sessions']:5d} sessions  hit@1 {r['hit@1'] - base['hit@1']:+.4f}  "
            f"hit@5 {r['hit@5'] - base['hit@5']:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

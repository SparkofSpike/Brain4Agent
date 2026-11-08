"""Does distilling into memories help or hurt retrieval?

The claim behind `remember()` is that a store of self-contained facts retrieves
better than a store of raw turns: less near-duplicate noise, more specific
phrasing. That is a hypothesis, and this measures it on the same 40 questions.

It also prices the write path: one model call per turn, so the report includes
wall time and token-free call counts.

    BRAIN4AGENT_LLM_KEY=... python -m bench.zh.extract_probe

Without a key it exits 1 rather than silently benchmarking the fallback path.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from brain4agent import MemoryStore
from brain4agent.extract import Extractor
from bench.zh.e2e import BASELINE, abstention_report, load, score

RESULTS = Path(__file__).parent.parent / "results"


def base_corpus(data: dict) -> tuple[list[str], list[list[dict]], list[dict]]:
    task = data["tasks"][0]
    return (
        list(task["haystack_session_ids"]),
        list(task["haystack_sessions"]),
        [t for t in data["tasks"] if not t["question_id"].endswith("_abs")],
    )


def build_raw(ids, sessions) -> MemoryStore:
    store = MemoryStore(Path(tempfile.mkdtemp()) / "raw.db")
    with store.batch():
        for sid, turns in zip(ids, sessions, strict=True):
            for turn in turns:
                text = (turn.get("content") or "").strip()
                if text:
                    store.add(text, session_id=sid, role=turn.get("role", "user"))
    return store


def pairs(session: list[dict]):
    """Yield (question, answer), tolerating non-alternating input."""
    pending: str | None = None
    for turn in session:
        text = (turn.get("content") or "").strip()
        if not text:
            continue
        role = turn.get("role")
        if role == "user":
            pending = text
        elif role == "assistant" and pending is not None:
            yield pending, text
            pending = None


def build_distilled(ids, sessions, extractor: Extractor) -> tuple[MemoryStore, dict]:
    """Distil question/answer pairs, not single turns.

    Extracting answers alone was measured at hit@1 0.475: almost every question
    was judged "nothing durable" and dropped, taking the search vocabulary with
    it (D-023).
    """
    store = MemoryStore(Path(tempfile.mkdtemp()) / "distilled.db")
    exchanges = kept = dropped = 0
    latencies: list[float] = []
    kinds: dict[str, int] = {}

    total = sum(1 for s in sessions for _ in pairs(s))
    for sid, session in zip(ids, sessions, strict=True):
        for question, answer in pairs(session):
            exchanges += 1
            started = time.perf_counter()
            ids_written = store.remember_exchange(
                question, answer, session_id=sid, extractor=extractor
            )
            latencies.append(time.perf_counter() - started)

            if not ids_written:
                dropped += 1
            kept += len(ids_written)
            for mid in ids_written:
                mem = store._by_session_memory(mid)
                if mem:
                    kinds[mem.kind] = kinds.get(mem.kind, 0) + 1

            if exchanges % 20 == 0:
                print(f"  processed {exchanges}/{total} exchanges, {kept} memories", flush=True)

    return store, {
        "exchanges": exchanges,
        "memories": kept,
        "dropped_exchanges": dropped,
        "failed_calls": extractor.stats().get("failures", 0),
        "kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
        "calls": extractor.stats().get("calls", 0),
        "median_write_s": round(statistics.median(latencies), 2) if latencies else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    extractor = Extractor.from_env()
    if extractor is None:
        print("BRAIN4AGENT_LLM_KEY is not set -- nothing to measure")
        return 1

    data = load()
    ids, sessions, questions = base_corpus(data)
    print(f"{len(questions)} questions, {len(ids)} sessions\n")

    print("=== raw turns ===")
    raw_store = build_raw(ids, sessions)
    raw_store.search("预热")
    raw = score(data, raw_store)
    raw_stats = raw_store.stats()
    raw_store.close()
    print(f"  turns {raw_stats['turns']}, hit@1 {raw['hit@1']:.4f}")

    print("=== distilled ===")
    dist_store, write_info = build_distilled(ids, sessions, extractor)
    dist_store.search("预热")
    dist = score(data, dist_store)
    dist_stats = dist_store.stats()
    dist_store.close()
    print(f"  memories {dist_stats['memories'] if 'memories' in dist_stats else dist_stats['turns']}")
    print(f"  hit@1 {dist['hit@1']:.4f}")

    print()
    print(f"{'metric':12s} {'raw':>10s} {'distilled':>12s}  delta")
    print("-" * 46)
    for key in ("hit@1", "hit@5", "hit@10", "mrr"):
        print(f"{key:12s} {raw[key]:10.4f} {dist[key]:12.4f}  {dist[key] - raw[key]:+.4f}")

    print()
    print(f"records:   {raw_stats['turns']} turns -> {write_info['memories']} memories")
    print(f"write:     {write_info['calls']} model calls, median {write_info['median_write_s']}s")
    print(f"dropped:   {write_info['dropped_exchanges']} exchanges held nothing durable")
    print(f"failures:  {write_info['failed_calls']}")
    print(f"kinds:     {write_info['kinds']}")
    print(f"pre-D-019 baseline hit@1: {BASELINE['hit@1']}")

    if args.json:
        payload = {
            "raw": {**raw, "store": raw_stats},
            "distilled": {**dist, "write": write_info},
        }
        RESULTS.mkdir(exist_ok=True)
        out = RESULTS / "extract-probe.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

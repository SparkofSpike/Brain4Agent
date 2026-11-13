"""What the storage layer costs, and where it stops being free.

The store is SQLite for durability, transactions, single-row updates and
cross-process visibility. Retrieval does not happen in SQLite: it happens in an
in-memory BM25 index rebuilt at open time. That split decides which question is
worth measuring.

- A different database engine would not change retrieval quality or query
  latency, because neither is the engine's job. It would change write cost,
  single-file portability and dependencies.
- The ceiling is not the file. It is the index: the whole corpus is resident,
  so resident bytes per memory times corpus size is the number that ends this
  design (D-043).

This probe reports both: what SQLite costs per operation, and what the index
costs per memory, then extrapolates linearly. It never opens a live store for
writing and never prints memory text.

    python -m bench.storage.probe --db brain.db          # copy first, then measure
    python -m bench.storage.probe --synth 20000          # synthetic, no private data
    python -m bench.storage.probe --synth 20000 --json out.json

`--synth` rows are template-generated and far more repetitive than real
memories, which inflates BM25 work on long queries. Use it for index size and
open time; use `--db` for query latency.

Run it on an idle machine. The first version of a number like this was measured
while another process was running benchmarks, and was wrong by 2x (D-042).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sqlite3
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from bench.real.probe import snapshot  # noqa: E402
from brain4agent import MemoryStore  # noqa: E402

COLUMNS = (
    "id, session_id, role, content, created_at, tokens, kind, importance,"
    " entities, source_text, tags, superseded_by, refs, fingerprint,"
    " accesses, weight"
)
SCAN_SQL = f"SELECT {COLUMNS} FROM memories ORDER BY rowid"

# Roughly what a working agent writes: a code-ish sentence, an identifier, a
# decision. Length is the part that matters to the index; topic is not.
_SUBJECTS = (
    "部署目录",
    "数据库",
    "分词器",
    "检索延迟",
    "缓存",
    "配置文件",
    "构建脚本",
    "the retrieval path",
    "the tokenizer",
    "the deploy directory",
    "the cache",
)
_VERBS = (
    "改成",
    "不能覆盖",
    "需要重建",
    "已经删除",
    "在副本上测过",
    "was rejected",
    "is still missing",
    "measured twice",
)
_TAILS = (
    "2000 行左右",
    "因为时间戳只有秒级精度",
    "见 D-019",
    "no model in the loop",
    "文件在 /opt/demo-app",
    "提交哈希 6df3629",
)


def synth_rows(n: int, seed: int = 20261113) -> list[str]:
    """n distinct strings with a realistic length distribution."""
    rng = random.Random(seed)
    out: list[str] = []
    for i in range(n):
        parts = [
            f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)}",
            f"{rng.choice(_TAILS)} #{i}",
        ]
        if i % 3 == 0:
            parts.append("重复出现的中文片段用来拉长记忆并制造词频差异")
        out.append("，".join(parts))
    return out


def build_synth(path: Path, n: int) -> None:
    texts = synth_rows(n)
    with MemoryStore(path) as store, store.batch():
        for i, text in enumerate(texts):
            store.add(text, session_id=f"synth-{i}", role="assistant")


def median_ms(fn, repeat: int = 3) -> float:
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def scan_cost(path: Path) -> float:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return median_ms(lambda: conn.execute(SCAN_SQL).fetchall())
    finally:
        conn.close()


def open_cost(path: Path) -> float:
    def one() -> None:
        with MemoryStore(path):
            pass

    return median_ms(one)


def resident_footprint(path: Path) -> tuple[int, float, float]:
    """(rows, resident MB, peak MB) for a loaded store.

    This is the whole object graph, not an isolated index: `Memory` instances,
    `_by_session`, `_by_id`, `_session_text`, the term counters and the
    sequence cache all count. That is the number that competes for RAM, so it is
    the number a size budget should use -- an index-only figure would be smaller
    and less true. Upper bound, not exact: `tracemalloc` sees Python objects,
    not allocator overhead.
    """
    # Warm the tokenizer outside the measurement: jieba loads a ~20 MB
    # dictionary once per process, not once per store.
    with MemoryStore(path):
        pass
    gc.collect()

    tracemalloc.start()
    try:
        with MemoryStore(path) as store:
            current, peak = tracemalloc.get_traced_memory()
            rows = store.stats()["turns"]
    finally:
        tracemalloc.stop()
    gc.collect()
    return rows, current / 1e6, peak / 1e6


def rebuild_after_write_cost(path: Path, repeats: int = 3) -> float:
    """ms for the first search after a local write.

    This is the cost the design actually pays and the one easiest to miss.
    `add()` marks the store dirty (`store.py`); the next `search()` calls
    `_ensure_fresh`, sees the flag and rebuilds the whole index. A read-only
    benchmark never touches this path, which is why an earlier version of this
    probe reported only the open-time rebuild and made writes look free.
    """
    times: list[float] = []
    with MemoryStore(path) as store:
        store.search("warm")
        for i in range(repeats):
            store.add(f"probe write {i}", session_id=f"probe-{i}", role="user")
            start = time.perf_counter()
            store.search("probe write")
            times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def sampled_queries(path: Path, limit: int, seed: int = 20261113) -> list[tuple[str, str]]:
    """Two query shapes, because the store sees both.

    An agent types two or three keywords (D-039); a person pastes a sentence.
    Mixing them into one median hides the fact that they cost different amounts.

    Sampled at random rowid windows rather than by `rowid % step`: the store is
    pruned in place, so leftover rowids are sparse in places and a modulo stride
    silently skips whole ranges.
    """
    from bench.real.probe import keywords

    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        total = conn.execute("SELECT count(*) FROM memories").fetchone()[0]
        if total == 0:
            return []
        rng = random.Random(seed)
        seen: set[str] = set()
        rows: list[str] = []
        for _ in range(limit * 3):
            off = rng.randrange(total)
            row = conn.execute(
                "SELECT content FROM memories ORDER BY rowid LIMIT 1 OFFSET ?", (off,)
            ).fetchone()
            if row and row[0] and row[0] not in seen:
                seen.add(row[0])
                rows.append(row[0])
            if len(rows) >= limit:
                break
    finally:
        conn.close()

    out: list[tuple[str, str]] = []
    for text in rows:
        words = keywords(text, limit=3)
        if words:
            out.append((" ".join(words), "keywords"))
        out.append((text[:40], "sentence"))
    return out


def proximity_fires(path: Path, queries: list[tuple[str, str]]) -> tuple[int, int]:
    """(queries that turn the proximity pass on, queries sampled).

    Printed because the reason a keyword lookup is fast and a sentence is slow
    is not the proximity bonus: its floor is five ASCII terms, and a Chinese
    keyword query has none. Without this counter the two effects get conflated.
    """
    from brain4agent.tokenize import _phrasable, query_tokens

    fired = 0
    for text, _ in queries:
        distinct = [t for t in dict.fromkeys(query_tokens(text)) if _phrasable(t)]
        if len(distinct) >= 5:
            fired += 1
    return fired, len(queries)


def query_cost(store: MemoryStore, queries: list[tuple[str, str]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for text, kind in queries:
        times = []
        for _ in range(3):
            start = time.perf_counter()
            store.search(text)
            times.append((time.perf_counter() - start) * 1000)
        buckets.setdefault(kind, []).append(statistics.median(times))
    return {k: statistics.median(v) for k, v in buckets.items() if v}


def write_cost(path: Path, repeats: int = 1000, rows_per_search: int = 5) -> float:
    """Per-search write cost in ms: one transaction crediting k turns, committed.

    This mirrors `store.search` rather than a bulk load: `with self._db:` wraps
    one `executemany` and commits per query. Measuring 1000 updates in a single
    transaction would report a tenth of the real cost and make the write path
    look free.

    `synchronous` is per connection, unlike `journal_mode`, so this connection
    has to set it too. Without it the probe measures one fsync per commit -- a
    path the shipping store never takes.
    """
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA synchronous=NORMAL")
        ids = [r[0] for r in conn.execute("SELECT id FROM memories LIMIT 200")]
        if not ids:
            return 0.0
        start = time.perf_counter()
        for i in range(repeats):
            batch = [(ids[(i + j) % len(ids)],) for j in range(rows_per_search)]
            conn.executemany("UPDATE memories SET accesses = accesses + 1 WHERE id = ?", batch)
            conn.commit()
        return (time.perf_counter() - start) * 1000 / repeats
    finally:
        conn.close()


def token_stats(path: Path) -> tuple[int, float, float]:
    """(total index tokens, mean content chars, mean tokens-column chars).

    Resident size tracks index tokens, not row count: a 60-character memory and
    a 600-character one are one row each and differ by an order of magnitude in
    the index. Reporting per-token bytes is what makes two corpora comparable.
    """
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT content, tokens FROM memories").fetchall()
    finally:
        conn.close()
    if not rows:
        return 0, 0.0, 0.0
    tokens = sum(len(t.split()) for _, t in rows)
    return (
        tokens,
        statistics.mean(len(c) for c, _ in rows),
        statistics.mean(len(t) for _, t in rows),
    )


def report(path: Path, label: str, json_out: Path | None) -> int:
    rows, resident_mb, peak_mb = resident_footprint(path)
    scan = scan_cost(path)
    opened = open_cost(path)
    queries = sampled_queries(path, 24)
    tokens, mean_chars, mean_token_chars = token_stats(path)

    with MemoryStore(path) as store:
        per_bucket = query_cost(store, queries)
    write = write_cost(path)
    rebuild = rebuild_after_write_cost(path)
    fired, sampled = proximity_fires(path, queries)

    size_mb = path.stat().st_size / 1e6
    per_memory_kb = resident_mb * 1e6 / rows / 1024 if rows else 0.0
    per_token_b = resident_mb * 1e6 / tokens if tokens else 0.0

    result = {
        "source": label,
        "rows": rows,
        "db_mb": round(size_mb, 3),
        "resident_mb": round(resident_mb, 2),
        "resident_peak_mb": round(peak_mb, 2),
        "resident_kb_per_memory": round(per_memory_kb, 2),
        "resident_bytes_per_token": round(per_token_b, 1),
        "tokens": tokens,
        "tokens_per_memory": round(tokens / rows, 1) if rows else 0.0,
        "content_chars_mean": round(mean_chars, 1),
        "tokens_column_chars_mean": round(mean_token_chars, 1),
        "scan_ms": round(scan, 2),
        "open_ms": round(opened, 2),
        "search_after_write_ms": round(rebuild, 1),
        "query_ms": {k: round(v, 2) for k, v in per_bucket.items()},
        "write_ms_per_search": round(write, 3),
        "proximity_fired": f"{fired}/{sampled}",
        "extrapolated_resident_mb": {
            str(n): round(resident_mb * n / rows, 0) for n in (10_000, 100_000, 1_000_000)
        }
        if rows
        else {},
    }

    print(f"source                 {label}")
    print(f"rows                   {rows}")
    print(f"db file                {size_mb:.2f} MB")
    if rows:
        print(f"index tokens           {tokens}  ({tokens / rows:.0f}/memory)")
    print(f"content chars          mean {mean_chars:.0f}")
    print(f"resident               {resident_mb:.2f} MB  (peak {peak_mb:.2f} MB)")
    print(f"resident per memory    {per_memory_kb:.2f} KB")
    print(f"resident per token     {per_token_b:.0f} bytes")
    print(f"full scan (16 cols)    {scan:.2f} ms")
    print(f"open (build index)     {opened:.2f} ms")
    print(f"search after a write   {rebuild:.1f} ms   <- whole index rebuilt")
    for kind, ms in per_bucket.items():
        print(f"query {kind:<10}      {ms:.2f} ms")
    print(f"write per search       {write:.3f} ms  (1 commit, 5 rows)")
    print(f"proximity fired        {fired}/{sampled} sampled queries")
    if result["extrapolated_resident_mb"]:
        print("extrapolated resident (linear):")
        for n, mb in result["extrapolated_resident_mb"].items():
            print(f"  {int(n):>9,} memories  {mb:>9,.0f} MB")
    print()
    print("database size is not the ceiling; resident memory is.")

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", "utf-8")
        print(f"wrote {json_out}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--db", type=Path, help="existing store; copied before measuring")
    group.add_argument("--synth", type=int, metavar="N", help="build a synthetic store of N rows")
    ap.add_argument("--json", type=Path, help="write the numbers as JSON")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="b4a-storage-") as tmp:
        work = Path(tmp) / "probe.db"
        if args.db:
            source = args.db
            if not source.exists():
                print(f"no such file: {source}", file=sys.stderr)
                return 2
            snapshot(source, work)
            label = f"{source.name} (snapshot)"
        else:
            build_synth(work, args.synth)
            label = f"synthetic {args.synth} rows"

        return report(work, label, args.json)


if __name__ == "__main__":
    sys.exit(main())

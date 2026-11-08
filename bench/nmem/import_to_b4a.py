"""Copy the nmem brain into a Brain4Agent store, for a real-scale retrieval test.

Reads `default.db` read-only (copy to temp, never opened for writing), takes each
fiber's summary or essence as one memory, and imports it. Mojibake rows are
skipped: a share of that corpus is UTF-8 decoded as GBK and is unsearchable, so
including it would dilute every query with noise no system could match.

    python bench/nmem/import_to_b4a.py --out D:/path/b4a-nmem.db
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bench.nmem.prune_probe import is_mojibake  # noqa: E402
from brain4agent.store import MemoryStore  # noqa: E402

DEFAULT_DB = os.path.expanduser(r"~\.neuralmemory\brains\default.db")


def load(db_path: str) -> list[dict]:
    tmp = Path(tempfile.gettempdir()) / "nmem_import.db"
    shutil.copy2(db_path, tmp)
    try:
        conn = sqlite3.connect(tmp)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, summary, essence, frequency, created_at, tags FROM fibers"
        ).fetchall()
        conn.close()
    finally:
        tmp.unlink(missing_ok=True)

    out = []
    skipped_moji = skipped_empty = 0
    for r in rows:
        text = (r["summary"] or "").strip() or (r["essence"] or "").strip()
        if not text:
            skipped_empty += 1
            continue
        if is_mojibake(text):
            skipped_moji += 1
            continue
        tags = []
        for raw in (r["tags"] or "").replace("|", ",").split(","):
            t = raw.strip()
            if t and " " not in t:
                tags.append(f"nmem-tag:{t[:40]}")
        out.append(
            {
                "id": r["id"],
                "text": text,
                "frequency": int(r["frequency"] or 0),
                "created_at": r["created_at"] or "",
                "tags": tags[:4],
            }
        )
    print(
        f"fibers: {len(rows)}  usable: {len(out)}  mojibake: {skipped_moji}  empty: {skipped_empty}"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fibers = load(args.db)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    with MemoryStore(out) as store:
        with store.batch():
            for f in fibers:
                store.add(
                    f["text"],
                    session_id=f["nmem:{f['id']}"] if False else f"fib:{f['id']}",
                    role="assistant",
                    tags=f["tags"],
                    timestamp=f["created_at"] or None,
                )
        stats = store.stats()
    print(f"imported into {out}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

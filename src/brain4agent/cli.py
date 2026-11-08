"""Command line entry point.

    python -m brain4agent add "中文查询搜不到记忆" --role user
    python -m brain4agent search "中文查询为什么搜不到"
    python -m brain4agent stats

The search output deliberately prints ``missing`` when there is one. It is the
single most useful thing this tool can tell a person: not that it found nothing,
but which word in the question the corpus never contained.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from brain4agent.evidence import describe
from brain4agent.store import MemoryStore

DEFAULT_DB = "brain.db"


def default_db() -> Path:
    """``BRAIN4AGENT_DB`` if set, else ``./brain.db``.

    Relative-by-default on purpose: a memory database that quietly writes to the
    home directory is hard to find and easy to lose.
    """
    env = os.environ.get("BRAIN4AGENT_DB")
    return Path(env) if env else Path(DEFAULT_DB)


def _open(args: argparse.Namespace) -> MemoryStore:
    return MemoryStore(args.db)


def cmd_add(args: argparse.Namespace) -> int:
    store = _open(args)
    try:
        sid = args.session or store.new_session()
        store.add(args.content, session_id=sid, role=args.role)
        print(f"stored in session {sid}")
        return 0
    finally:
        store.close()


def cmd_exchange(args: argparse.Namespace) -> int:
    store = _open(args)
    try:
        sid = store.add_exchange(user=args.user, assistant=args.assistant, session_id=args.session)
        print(f"stored exchange in session {sid}")
        return 0
    finally:
        store.close()


def cmd_search(args: argparse.Namespace) -> int:
    store = _open(args)
    try:
        result = store.search(args.query, top_k=args.top_k)
        if result.empty:
            print(f"no evidence for {args.query!r}")
            print(f"({result.sessions_searched} sessions searched)")
            if result.corpus_missing:
                # The strongest thing an empty result can say.
                print(f"corpus has no: {', '.join(result.corpus_missing)}")
            return 1

        print(describe(result))
        for e in result.evidence:
            print()
            print(f"--- {e.session_id}  score={e.score:.2f}  coverage={e.coverage:.2f} ---")
            if e.missing_terms:
                print(f"this session lacks: {', '.join(e.missing_terms)}")
            print(store.session_text(e.session_id))
        return 0
    finally:
        store.close()


def cmd_import(args: argparse.Namespace) -> int:
    """Bulk-load turns from JSONL, one object per line.

    Expected shape, all keys optional except content:

        {"session_id": "s1", "role": "assistant", "content": "..."}

    A missing session_id starts a new session. Imported as one transaction --
    a thousand individual commits costs more than the thousand writes.
    """
    path = Path(args.path)
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 1

    store = _open(args)
    try:
        turns = 0
        sessions: set[str] = set()
        # One transaction for the whole file. Any bad line aborts the import and
        # rolls back -- a half-imported history is worse than none, because the
        # caller cannot tell how far it got.
        try:
            with store.batch(), path.open("r", encoding="utf-8-sig") as fh:
                # utf-8-sig, and streamed rather than read whole: hand-written
                # files often carry a BOM, and an import is the one operation
                # likely to be pointed at a large file.
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"line {lineno}: {exc}") from exc
                    if not isinstance(obj, dict):
                        raise ValueError(
                            f"line {lineno}: expected a JSON object, got {type(obj).__name__}"
                        )
                    content = obj.get("content")
                    if not isinstance(content, str):
                        raise ValueError(f"line {lineno}: 'content' must be a string")
                    content = content.strip()
                    if not content:
                        continue
                    sid = obj.get("session_id") or store.new_session()
                    store.add(content, session_id=sid, role=obj.get("role") or "user")
                    sessions.add(sid)
                    turns += 1
        except ValueError as exc:
            print(f"import aborted, nothing written: {exc}", file=sys.stderr)
            return 1
        print(f"imported {turns} turns across {len(sessions)} sessions")
        return 0
    finally:
        store.close()


def cmd_sessions(args: argparse.Namespace) -> int:
    store = _open(args)
    try:
        if not store._by_session:
            print("(no sessions)")
            return 0
        for sid in store._by_session:
            turns = store.session(sid)
            first = turns[0].content[:40].replace("\n", " ")
            print(f"{sid}  {len(turns):2d} turns  {first}")
        return 0
    finally:
        store.close()


def cmd_stats(args: argparse.Namespace) -> int:
    store = _open(args)
    try:
        st = store.stats()
        print(f"turns    {st['turns']}")
        print(f"sessions {st['sessions']}")
        print(f"terms    {st['index_terms']}")
        print(f"db       {args.db}")
        return 0
    finally:
        store.close()


def cmd_prune(args: argparse.Namespace) -> int:
    """Keep only the most recently written sessions."""
    store = _open(args)
    try:
        before = store.stats()["sessions"]
        removed = store.prune(keep_sessions=args.keep)
        print(f"removed {removed} sessions, {before} -> {store.stats()['sessions']}")
        return 0
    finally:
        store.close()


def cmd_forget(args: argparse.Namespace) -> int:
    store = _open(args)
    try:
        removed = store.forget_session(args.session)
        print(f"removed {removed} turns from {args.session}")
        return 0 if removed else 1
    finally:
        store.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="brain4agent",
        description="Long-term memory for agents. Chinese-first, no model in the core path.",
    )
    p.add_argument("--db", type=Path, default=default_db(), help="database path")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("add", help="store one turn")
    a.add_argument("content")
    a.add_argument("--session", "-s", default=None)
    a.add_argument("--role", default="user", choices=["user", "assistant", "system"])
    a.set_defaults(func=cmd_add)

    e = sub.add_parser("exchange", help="store a user/assistant pair as one session")
    e.add_argument("user")
    e.add_argument("assistant")
    e.add_argument("--session", "-s", default=None)
    e.set_defaults(func=cmd_exchange)

    s = sub.add_parser("search", help="search and show evidence")
    s.add_argument("query")
    s.add_argument("--top-k", "-k", type=int, default=3)
    s.set_defaults(func=cmd_search)

    i = sub.add_parser("import", help="bulk-load turns from JSONL")
    i.add_argument("path")
    i.set_defaults(func=cmd_import)

    sub.add_parser("sessions", help="list sessions").set_defaults(func=cmd_sessions)
    sub.add_parser("stats", help="show counts").set_defaults(func=cmd_stats)

    pr = sub.add_parser("prune", help="keep only the newest N sessions")
    pr.add_argument("--keep", type=int, required=True)
    pr.set_defaults(func=cmd_prune)

    f = sub.add_parser("forget", help="delete a session")
    f.add_argument("session")
    f.set_defaults(func=cmd_forget)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

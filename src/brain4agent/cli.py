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
import contextlib
import json
import os
import sys
from pathlib import Path

from brain4agent.evidence import describe
from brain4agent.read import read as read_memory
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
        store.add(args.content, session_id=sid, role=args.role, tags=tuple(args.tags))
        print(f"stored in session {sid}")
        return 0
    finally:
        store.close()


def cmd_remember(args: argparse.Namespace) -> int:
    """Store a turn by distilling it into durable memories.

    Slower than add by design: one model call per turn. The trade is that a
    week of chit-chat does not end up competing with real memories at search
    time. Without an API key this degrades to storing the raw turn.
    """
    store = _open(args)
    try:
        ids = store.remember(
            args.content,
            session_id=args.session,
            role=args.role,
            tags=tuple(args.tags),
        )
        if not ids:
            print("nothing durable in that turn -- not stored")
            return 0
        first = store._by_session_memory(ids[0])
        where = first.session_id if first else args.session or "?"
        print(f"stored {len(ids)} memories in session {where}")
        for mid in ids:
            mem = store._by_session_memory(mid)
            if mem:
                print(f"  [{mem.kind} {mem.importance:.1f}] {mem.content[:90]}")
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


def _read_prompt() -> str:
    """Read a prompt from stdin, whatever encoding the shell handed us.

    Windows is the reason this exists. ``sys.stdin.read()`` decodes with the
    locale codec (cp936 on a Chinese system), so UTF-8 bytes arrive as mojibake
    and a Chinese prompt matches nothing. PowerShell may also hand over UTF-16
    with a BOM when it is piping a string rather than a file. Neither failure is
    loud: the search just returns nothing.
    """
    raw = sys.stdin.buffer.read()
    if not raw.strip():
        return ""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace").strip()
    return raw.decode("utf-8-sig", errors="replace").strip()


def cmd_recall_context(args: argparse.Namespace) -> int:
    """Print memory relevant to a prompt read from stdin, ready to inject.

    MCP is pull-only: a server cannot volunteer a memory, so an agent sees one
    only when it decides to call ``b4a_recall``. That covers "the user asked
    about something from before" and misses "the task is about to touch
    something we have context on". A client can close that gap with a hook --
    run this before handing the prompt to the model and prepend the output --
    but only if the output is already shaped for injection and silent when
    there is nothing worth saying.

    Silence is the common case. A hook that emits "no memories found" every turn
    trains the model to skip the block entirely.
    """
    prompt = _read_prompt()
    if not prompt:
        return 0

    store = _open(args)
    blocks: list[str] = []
    try:
        result = store.search(prompt, top_k=args.top_k, tags=tuple(args.tags) or None)
        if result.empty:
            return 0

        header = (
            "## Possibly relevant memory\n\n"
            "Excerpts from earlier sessions on this machine, retrieved by keyword\n"
            "match. They may be about a different task. Use them only if they help,\n"
            "and say when you are relying on one.\n"
        )
        used = len(header)
        for e in result.evidence:
            text = store.session_text(e.session_id).strip()
            if not text:
                continue
            block = f"\n[{e.session_id} · coverage {e.coverage:.2f}]\n{text}"
            if used + len(block) > args.max_chars:
                remaining = args.max_chars - used
                if remaining < 200:
                    break
                block = block[:remaining] + "\n…"
            blocks.append(block)
            used += len(block)
            if used >= args.max_chars:
                break
    finally:
        store.close()

    if not blocks:
        return 0
    sys.stdout.write(header + "".join(blocks) + "\n")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    store = _open(args)
    try:
        result = store.search(args.query, top_k=args.top_k, tags=tuple(args.tags) or None)
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

    Other keys are ignored. ``role`` is validated here for the same reason the
    ``add`` command restricts it: a typo stored today is a session read back
    with an impossible role tomorrow.

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
                    role = obj.get("role") or "user"
                    if role not in {"user", "assistant", "system"}:
                        raise ValueError(
                            f"line {lineno}: 'role' must be user, assistant or system, got {role!r}"
                        )
                    store.add(content, session_id=sid, role=role)
                    sessions.add(sid)
                    turns += 1
        except ValueError as exc:
            print(f"import aborted, nothing written: {exc}", file=sys.stderr)
            return 1
        print(f"imported {turns} turns across {len(sessions)} sessions")
        return 0
    finally:
        store.close()


def cmd_ask(args: argparse.Namespace) -> int:
    """Retrieve then read: an answer instead of a pile of sessions."""
    store = _open(args)
    try:
        result = read_memory(store, args.query, top_k=args.top_k)
        if result.answer is None:
            print(
                "no model configured (set BRAIN4AGENT_LLM_KEY); use 'search' to get the evidence",
                file=sys.stderr,
            )
            return 1
        print(result.answer)
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
    a.add_argument("--tag", action="append", dest="tags", default=[], help="label; repeatable")
    a.set_defaults(func=cmd_add)

    r = sub.add_parser("remember", help="distil a turn into durable memories")
    r.add_argument("content")
    r.add_argument("--session", "-s", default=None)
    r.add_argument("--role", default="user", choices=["user", "assistant", "system"])
    r.add_argument("--tag", action="append", dest="tags", default=[], help="label; repeatable")
    r.set_defaults(func=cmd_remember)

    e = sub.add_parser("exchange", help="store a user/assistant pair as one session")
    e.add_argument("user")
    e.add_argument("assistant")
    e.add_argument("--session", "-s", default=None)
    e.set_defaults(func=cmd_exchange)

    s = sub.add_parser("search", help="search and show evidence")
    s.add_argument("query")
    s.add_argument("--top-k", "-k", type=int, default=3)
    s.add_argument("--tag", action="append", dest="tags", default=[], help="filter; repeatable")
    s.set_defaults(func=cmd_search)

    rc = sub.add_parser(
        "recall-context", help="print memory relevant to a prompt on stdin (for hooks)"
    )
    rc.add_argument("--top-k", "-k", type=int, default=3)
    rc.add_argument("--max-chars", type=int, default=1500)
    rc.add_argument("--tag", action="append", dest="tags", default=[])
    rc.set_defaults(func=cmd_recall_context)

    q = sub.add_parser("ask", help="retrieve and answer in one step")
    q.add_argument("query")
    q.add_argument("--top-k", "-k", type=int, default=5)
    q.set_defaults(func=cmd_ask)

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


def _force_utf8_output() -> None:
    """Write UTF-8 to stdout/stderr, and never crash on an unencodable character.

    On a Chinese Windows console the default encoding is cp936, so printing a
    memory containing an emoji raised UnicodeEncodeError and exited non-zero
    after the work had already succeeded. errors="replace" keeps the exit code
    honest at the cost of a garbled glyph.

    Note this cannot fix argv: Python decoded the command line with the locale
    encoding before this ran. Set PYTHONUTF8=1 on Windows for full support.
    """
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):  # exotic streams
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

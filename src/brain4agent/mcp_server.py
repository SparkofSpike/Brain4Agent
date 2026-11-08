"""MCP server over stdio, standard library only.

Why not the official SDK: this project's whole constraint is that the core adds
no dependency until a measured number demands it (D-004), and the stdio
transport is four methods over newline-delimited JSON-RPC. Pulling in a package
to serialize four dicts would be the largest thing in the install.

The one rule that matters here: **stdout carries protocol messages and nothing
else.** A stray ``print`` corrupts the stream and the client sees a parse error
instead of a tool result. Every diagnostic goes to stderr.

Run it::

    python -m brain4agent.mcp_server

Client configuration (Claude Desktop, CodeWhale, anything MCP):

    {"mcpServers": {"brain4agent": {
        "command": "python",
        "args": ["-m", "brain4agent.mcp_server"],
        "env": {"BRAIN4AGENT_DB": "/path/to/brain.db"}
    }}}
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from brain4agent import __version__
from brain4agent.read import read as read_memory
from brain4agent.store import MemoryStore

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "brain4agent"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "b4a_remember",
        "description": (
            "Store one turn in long-term memory. The turn is stored verbatim by "
            "default; pass distil=true to run a model call that keeps only durable "
            "memories (facts, decisions, errors, preferences). Distilling costs a "
            "model call per turn and measurably hurts retrieval -- store raw and let "
            "b4a_ask extract at read time. Pass session_id to append to an existing "
            "conversation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The text to remember."},
                "session_id": {
                    "type": "string",
                    "description": "Existing session to append to. Omit to start a new one.",
                },
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant", "system"],
                    "default": "user",
                },
                "distil": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Run a model call to extract durable memories instead of "
                        "storing the turn verbatim. Off by default: measured worse "
                        "for retrieval, and it needs a model."
                    ),
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional labels, no spaces, e.g. ['scope:code']. Retrieval "
                        "can filter on them. Use them for what the text does not say: "
                        "which project, which domain. An untagged memory matches any "
                        "tag filter."
                    ),
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "b4a_recall",
        "description": (
            "Search long-term memory. Returns ranked sessions with coverage, plus two "
            "different absence signals. `corpus_missing` lists words the question "
            "treats as central that appear nowhere in the whole stored corpus -- the "
            "strong signal, and the closest thing to a free answer to 'do I know "
            "this'. `missing_terms` on each session is just why that session ranked "
            "where it did. Whether to abstain is the caller's call."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 3, "minimum": 1, "maximum": 20},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Only return memories carrying at least one of these tags. "
                        "Omit to search everything. Untagged memories are never "
                        "excluded by this filter."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "b4a_ask",
        "description": (
            "Retrieve and answer in one step. Slower than b4a_recall because it "
            "runs a model over the retrieved evidence, and it will say the memory "
            "does not contain the answer rather than guess. Use recall when you "
            "want the raw evidence, ask when you want an answer."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
    },
    {
        "name": "b4a_session",
        "description": "Read one whole session in order, given its id.",
        "inputSchema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}},
            "required": ["session_id"],
        },
    },
    {
        "name": "b4a_stats",
        "description": "Counts: turns stored, sessions, distinct index terms.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def default_db() -> Path:
    env = os.environ.get("BRAIN4AGENT_DB")
    return Path(env) if env else Path("brain.db")


class Server:
    """JSON-RPC dispatcher over a MemoryStore."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.store = MemoryStore(db_path or default_db())

    # -- protocol ----------------------------------------------------------

    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch one message. Returns None for notifications."""
        method = msg.get("method")
        msg_id = msg.get("id")
        params = msg.get("params") or {}

        if msg_id is None:
            # Notification: initialized, cancelled, and friends. No reply, by spec.
            return None

        try:
            if method == "initialize":
                return _ok(msg_id, self._initialize())
            if method == "tools/list":
                return _ok(msg_id, {"tools": TOOLS})
            if method == "tools/call":
                return _ok(msg_id, self._call_tool(params))
            if method == "ping":
                return _ok(msg_id, {})
        except _ToolError as exc:
            return _ok(msg_id, _tool_error(str(exc)))
        except Exception as exc:  # noqa: BLE001 - must never kill the loop
            _log(f"error handling {method}: {exc}\n{traceback.format_exc()}")
            return _err(msg_id, -32603, f"internal error: {exc}")

        return _err(msg_id, -32601, f"method not found: {method}")

    def _initialize(self) -> dict[str, Any]:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": __version__},
        }

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str):
            raise _ToolError("tool name must be a string")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            raise _ToolError(f"'arguments' must be an object, got {type(args).__name__}")

        if name == "b4a_remember":
            return self._remember(args)
        if name == "b4a_recall":
            return self._recall(args)
        if name == "b4a_ask":
            return self._ask(args)
        if name == "b4a_session":
            return self._session(args)
        if name == "b4a_stats":
            st = self.store.stats()
            return _text(
                f"turns {st['turns']}, sessions {st['sessions']}, terms {st['index_terms']}"
            )
        raise _ToolError(f"unknown tool: {name}")

    # -- tools -------------------------------------------------------------

    def _remember(self, args: dict[str, Any]) -> dict[str, Any]:
        content = _arg_str(args, "content") or ""
        if not content:
            raise _ToolError("content must not be empty")
        sid = _arg_str(args, "session_id") or self.store.new_session()
        role = _arg_str(args, "role") or "user"
        if role not in {"user", "assistant", "system"}:
            raise _ToolError(f"role must be user, assistant or system, got {role!r}")

        # distil defaults to False, and that is a reversal. It used to default to
        # True on the theory that storing every turn floods memory with
        # near-duplicates. Measured (D-023): distilling at write time drops
        # retrieval hit@1 from 0.9250 to 0.7750, because the extractor cannot know
        # what the memory will later be asked. arXiv:2602.16192 names the failure
        # and recommends storing the raw experience, extracting on demand -- which
        # is what b4a_ask does.
        distil = args.get("distil", False)
        if not isinstance(distil, bool):
            raise _ToolError("'distil' must be a boolean")

        tags = _arg_str_list(args, "tags")
        try:
            if distil:
                ids = self.store.remember(content, session_id=sid, role=role)
            else:
                ids = [self.store.add(content, session_id=sid, role=role, tags=tags)]
        except ValueError as exc:  # empty, too long, tag with a space
            raise _ToolError(str(exc)) from exc

        if not ids:
            return _text(f"nothing durable in that turn -- not stored (session {sid})")
        return _text(f"stored {len(ids)} memories in session {sid}")

    def _recall(self, args: dict[str, Any]) -> dict[str, Any]:
        query = _arg_str(args, "query") or ""
        if not query:
            raise _ToolError("query must not be empty")
        top_k = max(1, min(_arg_int(args, "top_k", 3), 20))
        tags = _arg_str_list(args, "tags")

        result = self.store.search(query, top_k=top_k, tags=tags or None)
        if result.empty:
            return _text(
                json.dumps(
                    {
                        "found": False,
                        "query": query,
                        "sessions_searched": result.sessions_searched,
                        "corpus_missing": result.corpus_missing,
                        "note": (
                            "no session shares any term with the query"
                            if not result.corpus_missing
                            else "the corpus has never contained: "
                            + ", ".join(result.corpus_missing)
                        ),
                    },
                    ensure_ascii=False,
                )
            )

        payload = {
            "found": True,
            "query": query,
            "sessions_searched": result.sessions_searched,
            "corpus_missing": result.corpus_missing,
            "evidence": [
                {
                    "session_id": e.session_id,
                    "score": round(e.score, 3),
                    "coverage": e.coverage,
                    "missing_terms": e.missing_terms,
                    "text": self.store.session_text(e.session_id),
                }
                for e in result.evidence
            ],
        }
        return _text(json.dumps(payload, ensure_ascii=False, indent=2))

    def _ask(self, args: dict[str, Any]) -> dict[str, Any]:
        query = _arg_str(args, "query") or ""
        if not query:
            raise _ToolError("query must not be empty")
        top_k = max(1, min(_arg_int(args, "top_k", 5), 20))

        result = read_memory(self.store, query, top_k=top_k)
        if result.answer is None:
            raise _ToolError(
                "no model configured, so nothing can be read; use b4a_recall for the "
                "evidence, or set BRAIN4AGENT_LLM_KEY"
            )
        return _text(result.answer)

    def _session(self, args: dict[str, Any]) -> dict[str, Any]:
        sid = _arg_str(args, "session_id") or ""
        if not sid:
            raise _ToolError("session_id must not be empty")
        turns = self.store.session(sid)
        if not turns:
            raise _ToolError(f"no such session: {sid}")
        return _text(json.dumps([_turn(m) for m in turns], ensure_ascii=False, indent=2))


# -- message helpers -------------------------------------------------------


def _arg_str(args: dict[str, Any], key: str, default: str | None = None) -> str | None:
    """Strict string argument. A number here is a caller bug, not a value to coerce."""
    value = args.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise _ToolError(f"'{key}' must be a string, got {type(value).__name__}")
    return value.strip()


def _arg_int(args: dict[str, Any], key: str, default: int) -> int:
    value = args.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _ToolError(f"'{key}' must be an integer, got {type(value).__name__}")
    return value


def _arg_str_list(args: dict[str, Any], key: str) -> tuple[str, ...]:
    """Optional list-of-strings argument.

    A bare string is rejected rather than wrapped: ``tags: "scope:code"`` almost
    certainly meant a list, and silently accepting it would make a typo look like
    a working filter for exactly one tag.
    """
    value = args.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise _ToolError(f"'{key}' must be an array of strings")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise _ToolError(f"'{key}' must contain only strings")
        if item.strip():
            out.append(item.strip())
    return tuple(out)


class _ToolError(Exception):
    """A caller mistake. Reported as tool output, not a protocol error."""


def _ok(msg_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _err(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _text(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _tool_error(text: str) -> dict[str, Any]:
    # Tool-level failures are results with isError, not JSON-RPC errors: the
    # protocol reserves error frames for a broken server, and a bad argument is
    # something the model should read and retry.
    return {"content": [{"type": "text", "text": text}], "isError": True}


def _turn(mem: Any) -> dict[str, str]:
    return {"role": mem.role, "content": mem.content, "created_at": mem.created_at}


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _force_utf8_streams() -> None:
    """Pin stdin/stdout to UTF-8, whatever the host locale says.

    MCP frames are UTF-8 by specification. reconfigure is idempotent where the
    locale already matches, and a no-op on streams that cannot be reconfigured
    (a wrapped pipe, a test double).
    """
    for stream in (sys.stdin, sys.stdout):
        with contextlib.suppress(AttributeError, ValueError):  # exotic streams
            stream.reconfigure(encoding="utf-8")


def main() -> int:
    _force_utf8_streams()
    server = Server()
    _log(f"{SERVER_NAME} {__version__} listening on stdio, db={server.store.path}")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            _log(f"bad json: {exc}")
            reply = _err(None, -32700, "parse error")
            print(json.dumps(reply), flush=True)
            continue

        if not isinstance(msg, dict):
            continue

        reply = server.handle(msg)
        if reply is not None:
            print(json.dumps(reply, ensure_ascii=False), flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""A2A (Agent2Agent) endpoint for Brain4Agent.

A2A 1.0 models every interaction as a message to an agent, which may answer
directly or start a task. A memory lookup is the direct case: the question goes
in, evidence comes back, nothing needs tracking. This server implements exactly
that and refuses the rest.

**What is implemented**

- `GET /.well-known/agent-card.json` — discovery
- `POST` JSON-RPC `SendMessage` — returns a `Message`, never a `Task`

**What is not**

- `SendStreamingMessage`, `GetTask`, `ListTasks`, `CancelTask`, push notifications
- authentication, rate limiting, TLS

That is deliberate, not unfinished. A memory read completes in milliseconds; a
task lifecycle around it would be ceremony that buys nothing. If you need task
semantics, you want an agent, not a memory store.

**Message convention**

The first text part is the request, interpreted by a one-word prefix:

    recall: 中文查询的 bug 是什么      -> search, reply with evidence
    remember: 根因在 keywords.py 443  -> store, reply with a session id
    <anything else>                    -> recall

Usage::

    python -m integrations.a2a.server --port 8080 --db brain.db

    curl -s localhost:8080/.well-known/agent-card.json

    curl -s localhost:8080/ -H 'Content-Type: application/json' -d '{
      "jsonrpc": "2.0", "id": 1, "method": "SendMessage",
      "params": {"message": {"role": "ROLE_USER",
                             "parts": [{"text": "recall: 中文查询"}]}}
    }'

Why standard library: the whole point of this project is that the core adds no
dependency until a measured number demands it (D-004), and an HTTP surface for
one JSON-RPC method does not clear that bar.

Why single-threaded: SQLite connections are bound to the thread that created
them, and the memory index is a mutable Python object behind no lock. A
ThreadingHTTPServer would need `check_same_thread=False` plus a mutex around
every read, which is a real cost paid to handle concurrency this workload does
not have — a memory lookup is single-digit milliseconds. Requests queue; they do
not fail.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from brain4agent import __version__
from brain4agent.store import MemoryStore

PROTOCOL_VERSION = "1.0"
AGENT_PATH = "/.well-known/agent-card.json"


def agent_card(base_url: str) -> dict[str, Any]:
    """Discovery document. A2A clients read this before talking to the agent."""
    return {
        "name": "Brain4Agent",
        "description": (
            "Long-term memory for agents. Chinese-first retrieval, no model in the "
            "core path. Returns evidence, not verdicts."
        ),
        "supportedInterfaces": [
            {"url": f"{base_url}/", "protocolBinding": "JSONRPC", "protocolVersion": PROTOCOL_VERSION}
        ],
        "version": __version__,
        "capabilities": {"streaming": False, "pushNotifications": False, "extensions": []},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": "recall",
                "name": "Recall",
                "description": (
                    "Search stored work memory. Returns ranked sessions with "
                    "corpus_missing: words the question uses that the corpus never "
                    "contained."
                ),
                "tags": ["memory", "search", "recall"],
                "examples": ["recall: 上次那个中文检索的 bug"],
            },
            {
                "id": "remember",
                "name": "Remember",
                "description": "Store one turn in long-term memory.",
                "tags": ["memory", "store"],
                "examples": ["remember: 部署目录是 /opt/app"],
            },
        ],
    }


class Agent:
    """Turns an A2A message into an answer. No HTTP in here."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def handle_text(self, text: str, context_id: str | None) -> str:
        text = text.strip()
        lowered = text.lower()

        if lowered.startswith("remember:"):
            payload = text.split(":", 1)[1].strip()
            if not payload:
                return "nothing to remember: the message was empty after 'remember:'"
            sid = self.store.add(payload, session_id=context_id, role="user")
            return f"stored in session {sid}"

        if lowered.startswith("recall:"):
            text = text.split(":", 1)[1].strip()

        if not text:
            return "empty query"

        result = self.store.search(text, top_k=3)
        if result.empty:
            # corpus_missing survives an empty result, and it is the most useful
            # thing to say here: "nothing matched" and "the corpus has never
            # contained 数据库" are different answers to a user.
            msg = (
                f"No session shares a term with '{text}'. "
                f"{result.sessions_searched} sessions searched."
            )
            if result.corpus_missing:
                msg += f" The corpus has never contained: {', '.join(result.corpus_missing)}."
            return msg

        lines: list[str] = []
        if result.corpus_missing:
            # The one fact a caller can act on without any further inference.
            lines.append(f"The corpus has never contained: {', '.join(result.corpus_missing)}")
        for e in result.evidence:
            lines.append(
                f"[score {e.score:.2f} coverage {e.coverage:.2f}] "
                f"session {e.session_id}\n{self.store.session_text(e.session_id)}"
            )
        return "\n\n".join(lines)


class Handler(BaseHTTPRequestHandler):
    server_version = f"Brain4Agent/{__version__}"
    store: MemoryStore  # injected in main()
    base_url: str = ""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/") == AGENT_PATH:
            self._json(200, agent_card(self.base_url))
        else:
            self._json(404, {"error": "not found", "hint": AGENT_PATH})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            msg = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            self._json(200, _err(None, -32700, f"parse error: {exc}"))
            return

        self._json(200, self._dispatch(msg))

    def _dispatch(self, msg: dict[str, Any]) -> dict[str, Any]:
        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        if method != "SendMessage":
            # Explicit, not silent: a client that asks for streaming should learn
            # it is unsupported rather than wait for events that never arrive.
            return _err(
                msg_id,
                -32601,
                f"unsupported method: {method}. This agent implements SendMessage only.",
            )

        message = params.get("message") or {}
        text = _first_text(message.get("parts") or [])
        if text is None:
            return _err(msg_id, -32602, "message.parts must contain at least one text part")

        context_id = message.get("contextId")
        try:
            reply_text = Agent(self.store).handle_text(text, context_id)
        except Exception as exc:  # noqa: BLE001 - one bad request must not kill the server
            print(f"error handling message: {exc}", file=sys.stderr, flush=True)
            return _err(msg_id, -32603, f"internal error: {exc}")

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "message": {
                    "messageId": uuid.uuid4().hex,
                    "role": "ROLE_AGENT",
                    "parts": [{"text": reply_text}],
                    **({"contextId": context_id} if context_id else {}),
                }
            },
        }

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[a2a] {fmt % args}", file=sys.stderr, flush=True)


def _first_text(parts: list[dict[str, Any]]) -> str | None:
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            return part["text"]
    return None


def _err(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve Brain4Agent over A2A.")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: loopback)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--db", default="brain.db")
    args = ap.parse_args()

    store = MemoryStore(Path(args.db))
    base_url = f"http://{args.host}:{args.port}"

    Handler.store = store
    Handler.base_url = base_url

    httpd = HTTPServer((args.host, args.port), Handler)
    print(f"Brain4Agent {__version__} on {base_url}", file=sys.stderr, flush=True)
    print(f"agent card: {base_url}{AGENT_PATH}", file=sys.stderr, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

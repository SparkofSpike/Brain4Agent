"""Tests for the MCP stdio server.

Exercised through ``Server.handle`` rather than a subprocess: the protocol
surface is the dispatcher, and a subprocess would only add start-up cost to
every assertion. The wire format is covered by one end-to-end test that runs the
real process.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from brain4agent.mcp_server import PROTOCOL_VERSION, Server


@pytest.fixture()
def server(tmp_path: Path) -> Server:
    return Server(tmp_path / "brain.db")


def rpc(msg_id: int, method: str, params: dict | None = None) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}}


def call(server: Server, name: str, arguments: dict, msg_id: int = 2) -> dict:
    out = server.handle(rpc(msg_id, "tools/call", {"name": name, "arguments": arguments}))
    assert out is not None
    return out


def text_of(reply: dict) -> str:
    return reply["result"]["content"][0]["text"]


class TestHandshake:
    def test_initialize(self, server: Server) -> None:
        out = server.handle(rpc(1, "initialize"))
        assert out["result"]["protocolVersion"] == PROTOCOL_VERSION
        assert out["result"]["capabilities"]["tools"] == {"listChanged": False}
        assert out["result"]["serverInfo"]["name"] == "brain4agent"

    def test_tools_listed(self, server: Server) -> None:
        out = server.handle(rpc(1, "tools/list"))
        names = {t["name"] for t in out["result"]["tools"]}
        assert names == {"b4a_remember", "b4a_recall", "b4a_session", "b4a_stats"}

    def test_every_tool_has_schema(self, server: Server) -> None:
        out = server.handle(rpc(1, "tools/list"))
        for tool in out["result"]["tools"]:
            assert tool["inputSchema"]["type"] == "object"
            assert tool["description"]

    def test_notification_gets_no_reply(self, server: Server) -> None:
        """A message without an id is a notification and must stay unanswered."""
        assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    def test_ping(self, server: Server) -> None:
        assert server.handle(rpc(1, "ping"))["result"] == {}

    def test_unknown_method(self, server: Server) -> None:
        out = server.handle(rpc(1, "no/such/method"))
        assert out["error"]["code"] == -32601


class TestTools:
    def test_remember_returns_session(self, server: Server) -> None:
        out = call(server, "b4a_remember", {"content": "中文查询搜不到记忆"})
        assert "stored in session" in text_of(out)
        assert out["result"]["isError"] is False

    def test_remember_empty_content(self, server: Server) -> None:
        out = call(server, "b4a_remember", {"content": "   "})
        assert out["result"]["isError"] is True

    def test_remember_appends_to_session(self, server: Server) -> None:
        first = call(server, "b4a_remember", {"content": "第一轮"})
        sid = text_of(first).split()[-1]
        call(server, "b4a_remember", {"content": "第二轮", "session_id": sid})

        out = call(server, "b4a_session", {"session_id": sid})
        turns = json.loads(text_of(out))
        assert len(turns) == 2

    def test_recall_returns_evidence(self, server: Server) -> None:
        call(server, "b4a_remember", {"content": "中文查询搜不到，根因在 keywords.py"})
        out = call(server, "b4a_recall", {"query": "中文查询为什么搜不到"})
        payload = json.loads(text_of(out))
        assert payload["found"] is True
        assert payload["evidence"]
        assert "keywords.py" in payload["evidence"][0]["text"]

    def test_recall_reports_missing_terms(self, server: Server) -> None:
        """The whole reason this server exists: say which word is absent."""
        call(server, "b4a_remember", {"content": "杭州测试机的地址是 114.215.175.63"})
        out = call(server, "b4a_recall", {"query": "杭州测试机上装的是什么数据库"})
        payload = json.loads(text_of(out))
        assert "数据库" in payload["corpus_missing"]

    def test_recall_miss(self, server: Server) -> None:
        call(server, "b4a_remember", {"content": "媒体投稿界面"})
        out = call(server, "b4a_recall", {"query": "完全无关的主题"})
        payload = json.loads(text_of(out))
        assert payload["found"] is False
        assert out["result"]["isError"] is False

    def test_recall_miss_reports_corpus_missing(self, server: Server) -> None:
        """found=false still carries the words nobody ever wrote down."""
        call(server, "b4a_remember", {"content": "杭州测试机的地址"})
        out = call(server, "b4a_recall", {"query": "zzzqqqxxx"})
        payload = json.loads(text_of(out))
        assert payload["found"] is False
        assert payload["corpus_missing"] == ["zzzqqqxxx"]
        assert "zzzqqqxxx" in payload["note"]

    def test_recall_empty_query(self, server: Server) -> None:
        out = call(server, "b4a_recall", {"query": ""})
        assert out["result"]["isError"] is True

    def test_recall_top_k_clamped(self, server: Server) -> None:
        for i in range(40):
            call(server, "b4a_remember", {"content": f"共同词汇 第{i}条"})
        out = call(server, "b4a_recall", {"query": "共同词汇", "top_k": 999})
        payload = json.loads(text_of(out))
        assert len(payload["evidence"]) <= 20

    def test_session_missing(self, server: Server) -> None:
        out = call(server, "b4a_session", {"session_id": "nope"})
        assert out["result"]["isError"] is True

    def test_stats(self, server: Server) -> None:
        call(server, "b4a_remember", {"content": "内容"})
        out = call(server, "b4a_stats", {})
        assert "turns 1" in text_of(out)

    def test_rejects_wrong_argument_types(self, server: Server) -> None:
        """A caller mistake is a tool error the model can read, not -32603."""
        cases = [
            ("b4a_recall", {"query": "测试", "top_k": "abc"}, "top_k"),
            ("b4a_recall", {"query": "测试", "top_k": 2.7}, "top_k"),
            ("b4a_remember", {"content": 42}, "content"),
            ("b4a_remember", {"content": "x", "session_id": 99}, "session_id"),
            ("b4a_remember", {"content": "x", "role": "hacker"}, "role"),
        ]
        for name, args, needle in cases:
            out = call(server, name, args)
            assert out["result"]["isError"], f"{name} {args} was accepted"
            assert needle in text_of(out), out
        assert len(server.store) == 0

    def test_rejects_non_object_arguments(self, server: Server) -> None:
        out = server.handle(rpc(2, "tools/call", {"name": "b4a_recall", "arguments": [1, 2]}))
        assert out["result"]["isError"]
        assert "must be an object" in text_of(out)

    def test_unknown_tool(self, server: Server) -> None:
        out = call(server, "no_such_tool", {})
        assert out["result"]["isError"] is True


class TestWire:
    def test_stdio_survives_gbk_locale(self, tmp_path: Path) -> None:
        """The host locale must not decide whether the server can read UTF-8.

        On a Chinese Windows install sys.stdin defaults to cp936. Before the
        fix this returned 1 with an empty stdout and a UnicodeDecodeError,
        because the end-to-end test set PYTHONIOENCODING=utf-8 and hid it.
        """
        db = tmp_path / "gbk.db"
        env = {
            "BRAIN4AGENT_DB": str(db),
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "PYTHONIOENCODING": "cp936",
        }
        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "b4a_remember",
                    "arguments": {"content": "中文内容测试🐱"},
                },
            },
            ensure_ascii=False,
        )

        proc = subprocess.run(
            [sys.executable, "-m", "brain4agent.mcp_server"],
            input=(payload + "\n").encode("utf-8"),
            capture_output=True,
            env={**__import__("os").environ, **env},
            timeout=120,
        )

        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
        assert proc.stdout, "no response under cp936"
        json.loads(proc.stdout.decode("utf-8"))

        from brain4agent import MemoryStore

        with MemoryStore(db) as store:
            contents = [t.content for turns in store._by_session.values() for t in turns]
        assert contents == ["中文内容测试🐱"], f"stdout={proc.stdout.decode('utf-8', 'replace')!r}"

    def test_stdio_roundtrip(self, tmp_path: Path) -> None:
        """The real process, the real stream -- stdout must be JSON only."""
        db = tmp_path / "wire.db"
        env = {
            "BRAIN4AGENT_DB": str(db),
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "PYTHONIOENCODING": "utf-8",
        }
        messages = [
            rpc(1, "initialize"),
            rpc(2, "tools/list"),
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            rpc(3, "tools/call", {"name": "b4a_remember", "arguments": {"content": "中文测试"}}),
            rpc(4, "tools/call", {"name": "b4a_recall", "arguments": {"query": "中文测试"}}),
        ]
        stdin = "\n".join(json.dumps(m) for m in messages) + "\n"

        proc = subprocess.run(
            [sys.executable, "-m", "brain4agent.mcp_server"],
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**__import__("os").environ, **env},
            timeout=120,
        )

        assert proc.returncode == 0, proc.stderr
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        # Four replies: the notification produces none.
        assert len(lines) == 4, proc.stdout

        for line in lines:
            parsed = json.loads(line)  # every stdout line must parse
            assert parsed["jsonrpc"] == "2.0"

        assert json.loads(lines[0])["result"]["serverInfo"]["name"] == "brain4agent"
        assert "中文测试" in lines[3]

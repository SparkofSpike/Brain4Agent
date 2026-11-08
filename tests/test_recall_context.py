"""Tests for `recall-context`, the hook-facing command.

This command exists because MCP is pull-only: a server cannot volunteer a
memory, so proactive recall has to be a client-side hook that asks *before* the
prompt reaches the model. That makes three properties non-negotiable and all
three are tested here:

- silence when nothing is relevant (a hook that prints a header every turn
  teaches the model to skip the block)
- correct decoding of whatever the shell hands us (Windows pipes deliver UTF-16
  with a BOM, or UTF-8, or nothing at all depending on the shell)
- the output is already shaped for injection, so a hook can prepend it verbatim
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from brain4agent.store import MemoryStore


def run_cli(args: list[str], stdin: bytes = b"") -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "brain4agent", *args],
        input=stdin,
        capture_output=True,
        timeout=120,
    )


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    path = tmp_path / "hook.db"
    with MemoryStore(path) as s:
        s.add("根因在 extraction/keywords.py 第 443 行", session_id="s1", role="assistant")
        s.add("用户想把记忆库换成 SQLite", session_id="s2", role="user")
    return path


class TestOutput:
    def test_relevant_memory_is_injected(self, db: Path) -> None:
        r = run_cli(
            ["--db", str(db), "recall-context", "-k", "2"],
            stdin="我们上次说根因在哪个文件？".encode(),
        )
        assert r.returncode == 0
        out = r.stdout.decode("utf-8")
        assert "Possibly relevant memory" in out
        assert "keywords.py" in out
        assert "s1" in out

    def test_silent_when_nothing_matches(self, db: Path) -> None:
        """The common case. Emitting a header here would be noise on every turn."""
        r = run_cli(["--db", str(db), "recall-context"], stdin="今天天气怎么样".encode())
        assert r.returncode == 0
        assert r.stdout.strip() == b""

    def test_silent_on_empty_stdin(self, db: Path) -> None:
        r = run_cli(["--db", str(db), "recall-context"], stdin=b"")
        assert r.returncode == 0
        assert r.stdout.strip() == b""

    def test_silent_on_whitespace(self, db: Path) -> None:
        r = run_cli(["--db", str(db), "recall-context"], stdin=b"   \n\t\n")
        assert r.returncode == 0
        assert r.stdout.strip() == b""

    def test_max_chars_respected(self, db: Path) -> None:
        r = run_cli(
            ["--db", str(db), "recall-context", "--max-chars", "400"],
            stdin="根因 记忆库".encode(),
        )
        assert r.returncode == 0
        assert len(r.stdout.decode("utf-8")) <= 500  # header is outside the budget


class TestEncoding:
    """Windows shells do not agree on what bytes they pipe. None may crash."""

    def test_utf8(self, db: Path) -> None:
        r = run_cli(["--db", str(db), "recall-context"], stdin="根因在哪个文件".encode())
        assert r.returncode == 0

    def test_utf8_with_bom(self, db: Path) -> None:
        """PowerShell with $OutputEncoding=UTF8 writes a BOM."""
        data = b"\xef\xbb\xbf" + "根因在哪个文件".encode()
        r = run_cli(["--db", str(db), "recall-context"], stdin=data)
        assert r.returncode == 0
        assert "keywords.py" in r.stdout.decode("utf-8")

    def test_utf16_with_bom(self, db: Path) -> None:
        """PowerShell piping a string rather than a file."""
        data = "根因在哪个文件".encode("utf-16")
        r = run_cli(["--db", str(db), "recall-context"], stdin=data)
        assert r.returncode == 0
        assert "keywords.py" in r.stdout.decode("utf-8")

    def test_garbage_does_not_crash(self, db: Path) -> None:
        r = run_cli(["--db", str(db), "recall-context"], stdin=b"\xff\xfe\x00\x01\x02garbage")
        assert r.returncode == 0


class TestTags:
    def test_tag_filter_applies(self, tmp_path: Path) -> None:
        path = tmp_path / "t.db"
        with MemoryStore(path) as s:
            s.add("根因在 a.py", session_id="c1", tags=["scope:code"])
            s.add("根因在 b.py", session_id="w1", tags=["scope:writing"])
        r = run_cli(
            ["--db", str(path), "recall-context", "--tag", "scope:code"],
            stdin="根因在哪".encode(),
        )
        out = r.stdout.decode("utf-8")
        assert "c1" in out
        assert "w1" not in out

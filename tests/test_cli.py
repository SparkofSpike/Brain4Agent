"""Tests for the command line entry point.

Runs ``main()`` in-process rather than shelling out: the exit code and the
round-trip through a real database file are what matter, and a subprocess would
only add start-up time to every assertion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain4agent.cli import main


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return tmp_path / "brain.db"


def test_add_then_search(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--db", str(db), "exchange", "中文查询搜不到", "根因在 keywords.py"]) == 0
    capsys.readouterr()

    code = main(["--db", str(db), "search", "中文查询为什么搜不到"])
    out = capsys.readouterr().out
    assert code == 0
    assert "keywords.py" in out


def test_search_miss_returns_one(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["--db", str(db), "add", "媒体投稿界面"])
    capsys.readouterr()

    code = main(["--db", str(db), "search", "完全无关的内容"])
    out = capsys.readouterr().out
    assert code == 1
    assert "no evidence" in out


def test_search_miss_shows_corpus_missing(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A miss must still say which words the corpus has never contained."""
    main(["--db", str(db), "add", "杭州测试机的地址"])
    capsys.readouterr()

    code = main(["--db", str(db), "search", "zzzqqqxxx"])
    out = capsys.readouterr().out
    assert code == 1
    assert "no evidence" in out
    assert "corpus has no" in out
    assert "zzzqqqxxx" in out


def test_search_shows_missing_terms(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["--db", str(db), "add", "杭州测试机的地址是 203.0.113.42"])
    capsys.readouterr()

    main(["--db", str(db), "search", "杭州测试机上装的是什么数据库"])
    out = capsys.readouterr().out
    assert "corpus has no" in out
    assert "数据库" in out


def test_stats(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["--db", str(db), "add", "第一条", "--session", "s1"])
    main(["--db", str(db), "add", "第二条", "--session", "s1"])
    capsys.readouterr()

    assert main(["--db", str(db), "stats"]) == 0
    out = capsys.readouterr().out
    assert "turns    2" in out
    assert "sessions 1" in out


def test_sessions_lists(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["--db", str(db), "add", "内容", "--session", "abc123"])
    capsys.readouterr()

    main(["--db", str(db), "sessions"])
    out = capsys.readouterr().out
    assert "abc123" in out


def test_prune(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for i in range(3):
        main(["--db", str(db), "add", f"内容{i}", "--session", f"s{i}"])
    capsys.readouterr()

    assert main(["--db", str(db), "prune", "--keep", "1"]) == 0
    assert "removed 2 sessions" in capsys.readouterr().out

    main(["--db", str(db), "stats"])
    assert "sessions 1" in capsys.readouterr().out


def test_forget(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["--db", str(db), "add", "内容", "--session", "abc123"])
    capsys.readouterr()

    assert main(["--db", str(db), "forget", "abc123"]) == 0
    out = capsys.readouterr().out
    assert "removed 1" in out


def test_forget_missing_session_returns_one(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--db", str(db), "forget", "nosuch"]) == 1
    capsys.readouterr()


def test_exchange_creates_one_session(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["--db", str(db), "exchange", "问", "答"])
    capsys.readouterr()

    main(["--db", str(db), "stats"])
    out = capsys.readouterr().out
    assert "sessions 1" in out
    assert "turns    2" in out


def test_import_jsonl(db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "in.jsonl"
    src.write_text(
        "\n".join(
            [
                '{"session_id": "s1", "role": "user", "content": "中文查询搜不到"}',
                '{"session_id": "s1", "role": "assistant", "content": "根因在 keywords.py"}',
                '{"content": "没有 session 字段会开新会话"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["--db", str(db), "import", str(src)]) == 0
    out = capsys.readouterr().out
    assert "imported 3 turns across 2 sessions" in out


def test_import_tolerates_bom(db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Hand-written JSONL frequently carries a BOM; PowerShell adds one."""
    src = tmp_path / "bom.jsonl"
    src.write_text('{"content": "带 BOM 的行"}\n', encoding="utf-8-sig")
    assert main(["--db", str(db), "import", str(src)]) == 0
    assert "imported 1 turns" in capsys.readouterr().out


def test_import_bad_json(db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "bad.jsonl"
    src.write_text("not json\n", encoding="utf-8")
    assert main(["--db", str(db), "import", str(src)]) == 1


def test_import_missing_file(db: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--db", str(db), "import", str(tmp_path / "nope.jsonl")]) == 1


def test_default_db_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """BRAIN4AGENT_DB must win over the default, without callers passing --db."""
    from brain4agent.cli import default_db

    monkeypatch.setenv("BRAIN4AGENT_DB", str(tmp_path / "env.db"))
    assert default_db() == tmp_path / "env.db"


def test_persists_between_invocations(db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["--db", str(db), "add", "跨进程保留", "--session", "keep"])
    capsys.readouterr()

    main(["--db", str(db), "search", "跨进程"])
    out = capsys.readouterr().out
    assert "跨进程保留" in out

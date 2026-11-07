"""Tests for the optional judge and the three-state outcome.

The judge itself is a network call and is not exercised here -- that is what
bench/zh/cloud_judge.py measures. What is tested is the part that can silently
mislead a user: how a judge verdict, a judge failure, and no evidence at all are
told apart.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from brain4agent import Judge, JudgeError, MemoryStore


class FakeJudge:
    """Scripted judge. Repeats its last verdict once the list runs out."""

    def __init__(self, verdicts: list[bool | None]) -> None:
        self.verdicts = verdicts
        self.i = 0
        self.calls: list[tuple[str, str]] = []

    def answer(self, question: str, evidence: str) -> bool | None:
        self.calls.append((question, evidence))
        v = self.verdicts[min(self.i, len(self.verdicts) - 1)]
        self.i += 1
        return v


@pytest.fixture()
def store():
    with tempfile.TemporaryDirectory() as tmp, MemoryStore(Path(tmp) / "brain.db") as s:
        s.add_exchange(
            user="中文查询搜不到记忆",
            assistant="根因在 keywords.py 第 443 行",
        )
        s.add_exchange(
            user="杭州那台测试机的地址是 114.215.175.63",
            assistant="部署目录 /opt/demo-app",
        )
        yield s


class TestThreeStates:
    def test_found(self, store: MemoryStore) -> None:
        j = FakeJudge([True])
        r = store.search_judged("中文查询为什么搜不到", j)
        assert r.verdict == "found"
        assert r.found is not None

    def test_not_found(self, store: MemoryStore) -> None:
        """Every checked session rejected, no failures -> a real negative."""
        j = FakeJudge([False])
        r = store.search_judged("中文查询为什么搜不到", j)
        assert r.verdict == "not_found"
        assert r.found is None

    def test_uncertain_when_judge_fails(self, store: MemoryStore) -> None:
        """A judge outage must not read as 'never recorded'."""
        j = FakeJudge([None])
        r = store.search_judged("中文查询为什么搜不到", j)
        assert r.verdict == "uncertain"
        assert r.found is None
        assert r.certainty != "not_found"

    def test_uncertain_when_mixed(self, store: MemoryStore) -> None:
        """At least one failure -> unknown, even if others said NO."""
        # A query that reaches both sessions, so two checks actually happen.
        j = FakeJudge([False, None])
        r = store.search_judged("记忆 测试机 地址 查询", j, max_checks=2)
        assert r.checks == 2, f"expected 2 checks, got {r.checks}"
        assert r.verdict == "uncertain"

    def test_empty_corpus_query_skips_judge(self, store: MemoryStore) -> None:
        j = FakeJudge([True])
        r = store.search_judged("完全不存在的主题", j)
        assert r.verdict == "not_found"
        assert j.calls == []
        assert r.checks == 0

    def test_checks_bounded(self, store: MemoryStore) -> None:
        j = FakeJudge([False])
        store.search_judged("测试机 地址 记忆", j, max_checks=2)
        assert len(j.calls) <= 2

    def test_judge_receives_whole_session(self, store: MemoryStore) -> None:
        """An answer split across turns must reach the judge intact."""
        j = FakeJudge([True])
        store.search_judged("中文查询为什么搜不到", j)
        assert j.calls
        _, evidence = j.calls[0]
        assert "keywords.py" in evidence


class TestJudgeConfig:
    def test_empty_key_rejected(self) -> None:
        with pytest.raises(JudgeError):
            Judge("")

    def test_from_env_absent_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No key must mean no judge, not a crash and not a silent network call."""
        monkeypatch.delenv("BRAIN4AGENT_JUDGE_KEY", raising=False)
        assert Judge.from_env() is None

    def test_from_env_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAIN4AGENT_JUDGE_KEY", "test-key")
        monkeypatch.setenv("BRAIN4AGENT_JUDGE_MODEL", "some-model")
        j = Judge.from_env()
        assert j is not None
        assert j.model == "some-model"
        assert j.api_key == "test-key"

    def test_stats_start_at_zero(self) -> None:
        j = Judge("k")
        st = j.stats()
        assert st["calls"] == 0
        assert st["errors"] == 0

    def test_failed_call_counts_as_none_and_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        j = Judge("k", retries=0, timeout=0.001)
        monkeypatch.setattr(j, "_post", lambda *_a, **_k: None)
        assert j.answer("q", "e") is None
        assert j.stats()["errors"] == 1

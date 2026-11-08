"""Tests for the reading stage.

This is the layer that runs a model, so the contract around it matters more than
its output: no model must degrade to "here is the evidence", an empty memory
must say so without inventing, and dates must reach the reader so it can resolve
conflicts from data rather than guessing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain4agent import MemoryStore
from brain4agent.read import MAX_EVIDENCE_CHARS, format_evidence, read


class FakeLLM:
    def __init__(self, reply: str | None) -> None:
        self.reply = reply
        self.last_user = ""

    def chat(self, system: str, user: str, **kwargs: object) -> str | None:
        self.last_user = user
        return self.reply


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    with MemoryStore(tmp_path / "r.db") as s:
        sid = s.new_session()
        s.add("中文查询搜不到记忆", session_id=sid, role="user")
        s.add("根因在 extraction/keywords.py 第 443 行", session_id=sid, role="assistant")
        yield s


class TestNoModel:
    def test_returns_evidence_without_answer(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No key means no reading, but the material still comes back."""
        monkeypatch.delenv("BRAIN4AGENT_LLM_KEY", raising=False)
        r = read(store, "中文查询为什么搜不到")
        assert r.answer is None
        assert r.ok is False
        assert r.evidence.evidence


class TestEmptyMemory:
    def test_says_so_without_calling_the_model(self, store: MemoryStore) -> None:
        llm = FakeLLM("invented answer")
        r = read(store, "zzzqqqxxx", llm=llm)  # type: ignore[arg-type]
        assert r.answer is not None
        assert "No memory" in r.answer
        assert llm.last_user == ""  # never asked

    def test_names_what_the_corpus_lacks(self, store: MemoryStore) -> None:
        r = read(store, "zzzqqqxxx", llm=None)
        assert r.answer is not None
        assert "zzzqqqxxx" in r.answer


class TestReading:
    def test_answer_passed_through(self, store: MemoryStore) -> None:
        llm = FakeLLM("根因是正则不含汉字范围 [abc123]")
        r = read(store, "中文查询为什么搜不到", llm=llm)  # type: ignore[arg-type]
        assert r.ok
        assert "正则" in (r.answer or "")

    def test_evidence_reaches_the_model(self, store: MemoryStore) -> None:
        llm = FakeLLM("ok")
        read(store, "中文查询为什么搜不到", llm=llm)  # type: ignore[arg-type]
        assert "keywords.py 第 443 行" in llm.last_user
        assert "中文查询为什么搜不到" in llm.last_user

    def test_failed_model_call_is_none(self, store: MemoryStore) -> None:
        r = read(store, "中文查询为什么搜不到", llm=FakeLLM(None))  # type: ignore[arg-type]
        assert r.answer is None


class TestEvidenceFormatting:
    def test_includes_session_id_and_date(self, store: MemoryStore) -> None:
        """Dates must reach the reader so conflict resolution uses data, not a guess."""
        text = format_evidence(store, store.search("中文查询为什么搜不到"))
        assert "[" in text and "]" in text
        assert "20" in text  # a year from created_at

    def test_both_roles_present(self, store: MemoryStore) -> None:
        text = format_evidence(store, store.search("中文查询为什么搜不到"))
        assert "user:" in text
        assert "assistant:" in text

    def test_truncated(self, store: MemoryStore) -> None:
        sid = store.new_session()
        store.add("中" * 5000, session_id=sid)
        text = format_evidence(store, store.search("中"))
        assert len(text) <= MAX_EVIDENCE_CHARS

"""Tests for the Chinese evaluation-set loader.

The loader is the seam between a JSON file and everything that scores against
it. Two properties matter and are easy to regress silently: abstention is
defined by an empty gold set (not by an id suffix), and the corpus is shared
rather than copied per question.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.zh.loader import load_zh, summarize


class TestRealSet:
    def test_loads_all_questions(self) -> None:
        tasks = load_zh()
        assert len(tasks) == 59

    def test_answerable_and_abstention_split(self) -> None:
        tasks = load_zh()
        abstention = [t for t in tasks if t.query.is_abstention]
        assert len(abstention) == 19
        assert len(tasks) - len(abstention) == 40

    def test_abstention_means_empty_gold_set(self) -> None:
        """Not an `_abs` suffix -- that was the LongMemEval convention (D-009)."""
        for t in load_zh():
            assert t.query.is_abstention == (not t.query.gold_session_ids)

    def test_summary_matches(self) -> None:
        s = summarize(load_zh())
        assert s["tasks"] == 59
        assert s["abstention"] == 19
        assert s["sessions"] == 44
        assert s["turns"] == 127

    def test_every_task_points_at_real_sessions(self) -> None:
        tasks = load_zh()
        known = {m.session_id for m in tasks[0].memories}
        for t in tasks:
            for sid in t.query.gold_session_ids:
                assert sid in known, f"{t.query.id} references unknown session {sid}"


class TestSharing:
    def test_corpus_is_shared_not_copied(self) -> None:
        """All questions search one library; copying 59 times would be wasteful.

        MemoryItem is frozen, so sharing the list is safe. This test would fail
        if the loader ever started deep-copying per task.
        """
        tasks = load_zh()
        assert tasks[0].memories is tasks[1].memories
        assert tasks[0].memories is tasks[-1].memories

    def test_memories_are_immutable(self) -> None:
        first = load_zh()[0].memories[0]
        with pytest.raises((AttributeError, TypeError)):
            first.content = "mutated"  # type: ignore[misc]


class TestLimit:
    def test_limit_truncates(self) -> None:
        assert len(load_zh(limit=10)) == 10

    def test_limit_larger_than_set_is_harmless(self) -> None:
        assert len(load_zh(limit=10_000)) == 59


class TestBrokenInput:
    def test_missing_file_raises_with_guidance(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="bench.zh.build"):
            load_zh(tmp_path / "nope.json")

    def test_empty_corpus_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.json"
        p.write_text(
            json.dumps(
                {
                    "meta": {},
                    "tasks": [
                        {
                            "question_id": "q1",
                            "question_type": "t",
                            "question": "q",
                            "answer": "a",
                            "haystack_session_ids": ["s1"],
                            "haystack_dates": ["2026-01-01"],
                            "haystack_sessions": [[{"role": "user", "content": "   "}]],
                            "answer_session_ids": ["s1"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="corpus is empty"):
            load_zh(p)

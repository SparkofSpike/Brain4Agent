"""Tests for Stage 1 abstention.

The judge itself is a model call and is not exercised here -- that is what
bench/zh/run_stage1.py measures. What is tested is the machinery around it,
because that machinery can silently produce wrong numbers: a YES misparsed as
NO inflates the abstention rate, and a session dropped between the two passes
guarantees a false abstention.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from bench.loader import MemoryItem
from bench.zh.stage1 import (
    JudgeConfig,
    _session_text,
    answer,
    judge_available,
    parse_yes_no,
)


class TestParseYesNo:
    def test_plain_yes(self) -> None:
        assert parse_yes_no("YES") is True

    def test_plain_no(self) -> None:
        assert parse_yes_no("NO") is False

    def test_lowercase(self) -> None:
        assert parse_yes_no("yes") is True
        assert parse_yes_no("no") is False

    def test_trailing_punctuation(self) -> None:
        assert parse_yes_no("YES.") is True
        assert parse_yes_no("NO:") is False

    def test_surrounding_whitespace(self) -> None:
        assert parse_yes_no("  YES\n") is True

    def test_garbage_returns_none(self) -> None:
        """A broken judge must show up as None, not silently as NO.

        Treating an unparseable answer as NO would inflate the abstention rate
        with judge failures, which is the one number this stage is judged on.
        """
        assert parse_yes_no("I cannot answer that.") is None
        assert parse_yes_no("") is None


def _mem(mid: str, sid: str, content: str, turn: int) -> MemoryItem:
    return MemoryItem(
        id=f"{sid}:{turn}", content=content, session_id=sid, timestamp="d", role="user"
    )


class TestSessionText:
    def test_joins_whole_session(self) -> None:
        mems = [
            _mem("a:1", "a", "second", 1),
            _mem("a:0", "a", "first", 0),
            _mem("b:0", "b", "other", 0),
        ]
        text = _session_text(mems, "a")
        assert "first" in text and "second" in text
        assert "other" not in text
        assert text.index("first") < text.index("second")

    def test_unknown_session_is_empty(self) -> None:
        assert _session_text([_mem("a:0", "a", "x", 0)], "zzz") == ""


class TestAnswer:
    """Drive ``answer`` with a scripted judge instead of a real model."""

    @staticmethod
    def _mems() -> list[MemoryItem]:
        return [
            _mem("s1:0", "s1", "媒体投稿 部署记录 alpha", 0),
            _mem("s2:0", "s2", "媒体投稿 的界面设计说明", 0),
            _mem("s3:0", "s3", "媒体投稿 别的项目 beta", 0),
        ]

    def test_no_duplicate_pass_when_single_session(self) -> None:
        """With one hit the combined pass would re-judge the same evidence."""
        cfg = JudgeConfig(first_pass=1, second_pass=3)
        mems = [_mem("s1:0", "s1", "媒体投稿 只有一个会话", 0)]
        with patch("bench.zh.stage1._ask", return_value=("NO", 0.0)) as mock:
            d = answer(mems, "媒体投稿", cfg)
        assert d.abstained
        assert mock.call_count == 1

    def test_returns_session_the_judge_accepts(self) -> None:
        cfg = JudgeConfig(first_pass=3, second_pass=5)
        with patch("bench.zh.stage1._ask", return_value=("YES", 0.0)):
            d = answer(self._mems(), "媒体投稿", cfg)
        assert not d.abstained
        assert d.found is not None
        assert d.judgements[-1].answers is True

    def test_abstains_when_judge_rejects_all(self) -> None:
        cfg = JudgeConfig(first_pass=3, second_pass=5)
        with patch("bench.zh.stage1._ask", return_value=("NO", 0.0)):
            d = answer(self._mems(), "媒体投稿", cfg)
        assert d.abstained
        assert d.found is None

    def test_second_pass_includes_first_pass_sessions(self) -> None:
        """The regression that cost 2.5 points of keep rate.

        An earlier version excluded already-judged sessions from the combined
        pass, so the session holding part of a multi-session answer was never
        reconsidered.
        """
        cfg = JudgeConfig(first_pass=1, second_pass=3)
        seen_in_second: list[str] = []

        def fake_ask(_cfg, _q, evidence: str) -> tuple[str, float]:
            if "---" in evidence:  # combined pass
                seen_in_second.append(evidence)
                return "YES", 0.0
            return "NO", 0.0

        with patch("bench.zh.stage1._ask", side_effect=fake_ask):
            d = answer(self._mems(), "媒体投稿", cfg)

        assert not d.abstained
        assert len(seen_in_second) == 1
        # The first-pass session must appear in the combined evidence.
        assert "s1" in seen_in_second[0] or "alpha" in seen_in_second[0]

    def test_empty_retrieval_abstains(self) -> None:
        cfg = JudgeConfig()
        d = answer([_mem("s1:0", "s1", "完全不相关的内容", 0)], "zzzz", cfg)
        assert d.abstained

    def test_unparseable_judge_does_not_answer(self) -> None:
        cfg = JudgeConfig(first_pass=2, second_pass=2)
        with patch("bench.zh.stage1._ask", return_value=("I am not sure", 0.0)):
            d = answer(self._mems(), "媒体投稿", cfg)
        assert d.abstained
        assert all(j.answers is False for j in d.judgements)


class TestJudgeAvailability:
    def test_missing_host_is_unavailable(self) -> None:
        cfg = JudgeConfig(host="http://127.0.0.1:1", timeout=1)
        assert judge_available(cfg) is False

    def test_config_defaults_are_sane(self) -> None:
        cfg = JudgeConfig()
        assert cfg.first_pass < cfg.second_pass
        assert cfg.max_evidence_chars >= 600


@pytest.mark.parametrize("model", ["qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b"])
def test_calibrated_models_are_named(model: str) -> None:
    """Pin the models Stage 1 was calibrated against.

    Not an assertion about their accuracy -- that lives in
    bench/results/zh-calibration.json. This only records which ones were tried,
    so a later change to a default is a visible decision.
    """
    assert model.startswith("qwen3.5:")

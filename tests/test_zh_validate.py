"""Tests for the evaluation-set validator.

A validator that cannot fail is decoration. These tests feed it deliberately
broken sets and assert it notices -- otherwise "0 hard errors" on the real set
means nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.zh.validate import _overlap, validate


def write_set(path: Path, tasks: list[dict], corpus: dict[str, list[dict]]) -> Path:
    """Minimal but structurally faithful evaluation set."""
    ids = list(corpus.keys())
    base = {
        "haystack_session_ids": ids,
        "haystack_dates": ["2026-01-01"] * len(ids),
        "haystack_sessions": [corpus[i] for i in ids],
    }
    payload = {
        "meta": {"name": "test", "questions": len(tasks)},
        "tasks": [{**base, **t} for t in tasks],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def task(qid: str, answer: str, gold: list[str]) -> dict:
    return {
        "question_id": qid,
        "question_type": "single-session-user",
        "question": "q",
        "answer": answer,
        "question_date": "2026-01-02",
        "answer_session_ids": gold,
    }


CORPUS = {
    "s1": [{"role": "user", "content": "杭州测试机的部署目录是 /opt/demo-app"}],
    "s2": [{"role": "user", "content": "媒体投稿的排期文件在 apps/api/data"}],
}


class TestRealSet:
    def test_no_hard_errors(self) -> None:
        errors, _ = validate()
        assert errors == []

    def test_only_the_known_paraphrase_warning(self) -> None:
        """Pin the set's current health so a new warning is noticed, not absorbed.

        zh012's answer overlaps its gold session by 43%: a paraphrase, reviewed
        by hand and recorded in D-009. Anything beyond that is new and worth
        looking at.
        """
        errors, warnings = validate()
        assert errors == []
        assert all("zh012" in w for w in warnings), warnings


class TestDetectsProblems:
    def test_duplicate_question_id(self, tmp_path: Path) -> None:
        p = write_set(
            tmp_path / "dup.json",
            [
                task("q1", "杭州测试机的部署目录是 /opt/demo-app", ["s1"]),
                task("q1", "媒体投稿的排期文件在 apps/api/data", ["s2"]),
            ],
            CORPUS,
        )
        errors, _ = validate(p)
        assert any("duplicate question id" in e for e in errors)

    def test_gold_session_not_in_corpus(self, tmp_path: Path) -> None:
        p = write_set(
            tmp_path / "missing.json",
            [task("q1", "杭州测试机的部署目录是 /opt/demo-app", ["nope"])],
            CORPUS,
        )
        errors, _ = validate(p)
        assert any("not in corpus" in e for e in errors)

    def test_low_overlap_becomes_a_warning(self, tmp_path: Path) -> None:
        """A paraphrase is suspicious, not fatal -- warnings are for humans."""
        p = write_set(
            tmp_path / "paraphrase.json",
            [task("q1", "完全无关的答案内容", ["s1"])],
            CORPUS,
        )
        errors, warnings = validate(p)
        assert errors == []
        assert any("overlaps" in w for w in warnings)

    def test_small_corpus_warns(self, tmp_path: Path) -> None:
        p = write_set(
            tmp_path / "small.json",
            [task("q1", "杭州测试机的部署目录是 /opt/demo-app", ["s1"])],
            CORPUS,
        )
        _, warnings = validate(p)
        assert any("only 2 sessions" in w for w in warnings)

    def test_abstention_without_gold_is_not_an_error(self, tmp_path: Path) -> None:
        """An empty gold set *is* the definition of an abstention question."""
        p = write_set(
            tmp_path / "abs.json",
            [task("q_abs", "语料里从未提到数据库", [])],
            CORPUS,
        )
        errors, _ = validate(p)
        assert errors == []


class TestOverlapHelper:
    def test_identical_text_scores_one(self) -> None:
        assert _overlap("杭州测试机", "杭州测试机") == 1.0

    def test_no_content_chars_scores_zero(self) -> None:
        assert _overlap("的了是", "杭州测试机") == 0.0

    def test_partial_overlap(self) -> None:
        # 杭州 is shared, 测试机 is not.
        assert 0.0 < _overlap("杭州部署", "杭州测试机") < 1.0

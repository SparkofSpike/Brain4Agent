"""Tests for the memory extractor.

The model is faked: what matters here is the contract around it — that a
malformed response is dropped rather than guessed, that "nothing durable" is
distinct from "the call failed", and that the store reacts differently to each.
"""

from __future__ import annotations

import json

import pytest

from brain4agent.extract import KINDS, ExtractedMemory, ExtractionResult, Extractor, parse
from brain4agent.llm import LLM


class FakeLLM:
    """Returns a scripted response, or None to simulate a failed call."""

    def __init__(self, response: str | None) -> None:
        self.response = response
        self.calls = 0

    def chat(self, system: str, user: str, **kwargs: object) -> str | None:
        self.calls += 1
        return self.response

    def stats(self) -> dict[str, object]:
        return {"model": "fake", "calls": self.calls, "errors": 0}


def single(kind: str = "fact", content: str = "x", **extra: object) -> str:
    return json.dumps({"memories": [{"kind": kind, "content": content, **extra}]})


class TestParse:
    def test_plain_json(self) -> None:
        raw = json.dumps(
            {
                "memories": [
                    {"kind": "fact", "content": "地址是 X", "entities": ["X"], "importance": 0.8}
                ]
            }
        )
        out = parse(raw)
        assert len(out) == 1
        assert out[0].kind == "fact"
        assert out[0].content == "地址是 X"
        assert out[0].entities == ("X",)
        assert out[0].importance == 0.8

    def test_fenced_json(self) -> None:
        """Models wrap JSON in fences often enough that this must work."""
        raw = '```json\n{"memories":[{"kind":"decision","content":"用 bigram"}]}\n```'
        out = parse(raw)
        assert len(out) == 1 and out[0].kind == "decision"

    def test_prose_around_json(self) -> None:
        raw = 'Here you go:\n{"memories":[{"kind":"error","content":"正则不含汉字"}]}\nDone.'
        assert len(parse(raw)) == 1

    def test_unknown_kind_dropped(self) -> None:
        """A guessed kind is worse than a missing memory."""
        raw = json.dumps(
            {
                "memories": [
                    {"kind": "gossip", "content": "x"},
                    {"kind": "fact", "content": "y"},
                ]
            }
        )
        assert [m.kind for m in parse(raw)] == ["fact"]

    def test_missing_content_dropped(self) -> None:
        raw = json.dumps({"memories": [{"kind": "fact"}, {"kind": "fact", "content": "ok"}]})
        assert len(parse(raw)) == 1

    def test_empty_list(self) -> None:
        assert parse('{"memories":[]}') == []

    def test_garbage_returns_empty(self) -> None:
        assert parse("I could not find anything durable.") == []
        assert parse("") == []
        assert parse("{}") == []

    def test_no_json_object(self) -> None:
        assert parse("[1,2,3]") == []

    def test_importance_is_clamped(self) -> None:
        raw = json.dumps(
            {
                "memories": [
                    {"kind": "fact", "content": "a", "importance": 5},
                    {"kind": "fact", "content": "b", "importance": -2},
                ]
            }
        )
        out = parse(raw)
        assert out[0].importance == 1.0
        assert out[1].importance == 0.0

    def test_bad_importance_falls_back(self) -> None:
        assert parse(single(importance="high"))[0].importance == 0.5

    def test_entities_deduplicated_and_capped(self) -> None:
        ents = ["a", "a", "b"] + [f"e{i}" for i in range(20)]
        raw = json.dumps({"memories": [{"kind": "fact", "content": "x", "entities": ents}]})
        out = parse(raw)[0]
        assert out.entities[0] == "a"
        assert len(out.entities) <= 10
        assert len(set(out.entities)) == len(out.entities)

    def test_limit_respected(self) -> None:
        items = [{"kind": "fact", "content": f"x{i}"} for i in range(10)]
        assert len(parse(json.dumps({"memories": items}), limit=3)) == 3

    def test_content_truncated(self) -> None:
        raw = json.dumps({"memories": [{"kind": "fact", "content": "中" * 2000}]})
        assert len(parse(raw)[0].content) <= 500

    def test_every_valid_kind_accepted(self) -> None:
        items = [{"kind": k, "content": "x"} for k in sorted(KINDS)]
        assert {m.kind for m in parse(json.dumps({"memories": items}))} == KINDS


class TestExtractor:
    def test_success(self) -> None:
        ext = Extractor(FakeLLM(single()))
        result = ext.extract("some turn")
        assert result.ok is True
        assert len(result.memories) == 1
        assert ext.stats()["calls"] == 1

    def test_failed_call_is_not_ok(self) -> None:
        """ok=False is what tells the store to keep the raw turn."""
        ext = Extractor(FakeLLM(None))
        result = ext.extract("some turn")
        assert result.ok is False
        assert result.memories == ()
        assert ext.stats()["failures"] == 1

    def test_empty_result_is_ok(self) -> None:
        """'Nothing durable' is an answer, not a failure.

        This distinction decides whether the store drops the turn or keeps it.
        """
        ext = Extractor(FakeLLM('{"memories":[]}'))
        result = ext.extract("好的，谢谢")
        assert result.ok is True
        assert result.memories == ()
        assert ext.stats()["empty"] == 1
        assert ext.stats()["failures"] == 0

    def test_blank_input_skips_the_model(self) -> None:
        llm = FakeLLM("{}")
        assert Extractor(llm).extract("   ").ok is True
        assert llm.calls == 0

    def test_long_input_truncated_before_sending(self) -> None:
        captured: list[str] = []

        class Recording(FakeLLM):
            def chat(self, system: str, user: str, **kwargs: object) -> str | None:
                captured.append(user)
                return '{"memories":[]}'

        Extractor(Recording(None)).extract("中" * 20_000)
        assert len(captured[0]) < 20_000


class TestFromEnv:
    def test_no_key_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAIN4AGENT_LLM_KEY", raising=False)
        assert Extractor.from_env() is None

    def test_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAIN4AGENT_LLM_KEY", "test-key")
        ext = Extractor.from_env()
        assert ext is not None and isinstance(ext.llm, LLM)


class TestResultType:
    def test_default_is_ok_and_empty(self) -> None:
        r = ExtractionResult()
        assert r.ok is True and r.memories == ()

    def test_memory_defaults(self) -> None:
        m = ExtractedMemory(kind="fact", content="x")
        assert m.entities == () and m.importance == 0.5

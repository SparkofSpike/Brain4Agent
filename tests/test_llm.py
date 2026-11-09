"""LLM.chat must degrade to None, never to an exception.

The read path treats None as "no model answer" and falls back to evidence. A
provider response that parses but has an unexpected shape must take that same
exit: a malformed body is an outage, not a crash in the caller.
"""

from __future__ import annotations

import json

import pytest

from brain4agent.llm import LLM, LLMConfig


class _Response:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _llm(monkeypatch: pytest.MonkeyPatch, payload: object) -> LLM:
    def fake_urlopen(request: object, timeout: float | None = None) -> _Response:
        return _Response(payload)

    monkeypatch.setattr("brain4agent.llm.urllib.request.urlopen", fake_urlopen)
    return LLM(LLMConfig(api_key="test-key", retries=0))


def test_content_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _llm(monkeypatch, {"choices": [{"message": {"content": "  answer  "}}]})
    assert llm.chat("system", "user") == "answer"
    assert llm.errors == 0


def test_null_message_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """`"message": null` made `.get("message", {})` return None and crash."""
    llm = _llm(monkeypatch, {"choices": [{"message": None}]})
    assert llm.chat("system", "user") is None
    assert llm.errors == 1


def test_missing_message_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _llm(monkeypatch, {"choices": [{}]})
    assert llm.chat("system", "user") is None
    assert llm.errors == 1


def test_empty_choices_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _llm(monkeypatch, {"choices": []})
    assert llm.chat("system", "user") is None
    assert llm.errors == 1


def test_non_dict_choice_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _llm(monkeypatch, {"choices": ["not an object"]})
    assert llm.chat("system", "user") is None
    assert llm.errors == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": {"message": {"content": "x"}}},
        {"choices": 5},
        {"choices": True},
        {"choices": None},
        {"choices": [{"message": "not a dict"}]},
        {"choices": [{"message": {"content": ["x"]}}]},
        {"choices": [{"message": {"content": 42}}]},
    ],
)
def test_malformed_bodies_return_none(monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    """A response that parses but has the wrong shape is an outage, not a crash.

    `choices` is not guaranteed to be a list of objects, and `content` is not
    guaranteed to be a string. The helper absorbs every shape; these pin the
    ones that used to raise KeyError, TypeError or AttributeError.
    """
    llm = _llm(monkeypatch, payload)
    assert llm.chat("system", "user") is None
    assert llm.errors == 1

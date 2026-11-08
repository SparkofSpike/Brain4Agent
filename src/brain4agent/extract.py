"""Turn a conversation turn into durable memories.

This is the piece that makes the store a memory system rather than a log.
Storing every turn verbatim is what produced the noise in D-020: a hundred
exchanges about the same subsystem, none of which can be told apart. Extracting
one self-contained fact per thing worth keeping is how a real memory layer
avoids that — and it is exactly what neural-memory does with fibers.

Optional by design. Without a key, ``Extractor.from_env()`` returns None and the
store falls back to keeping raw turns, which still works.

What "durable" means here, because it is the whole judgement call:

- a fact about how the system is
- a decision and its reason
- an error and its cause
- a preference about how things should be done

Not durable: acknowledgements, restatements of the question, plans that were
never executed, and anything a reader would find obvious from the code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from brain4agent.llm import LLM

KINDS = frozenset({"fact", "decision", "error", "preference", "context"})

MAX_INPUT_CHARS = 6000
MAX_CONTENT_CHARS = 500
MAX_ENTITIES = 10
DEFAULT_MAX_MEMORIES = 5

SYSTEM = """You extract durable memories from one exchange in a working conversation.

A durable memory is something a colleague would need weeks later: a fact about
how the system is, a decision and its reason, an error and its cause, or a
stated preference.

Rules:
- One memory per entry. Each must be self-contained: a reader with no context
  must understand it. Replace pronouns and deictic references ("it", "that file",
  "上次那个") with what they actually name.
- **Keep the question's vocabulary.** A future search will use the words of the
  person asking, not yours. If the question asks "哪个文件哪一行", the memory
  must contain 文件 and 行. Rewriting it as only "extraction/keywords.py 第 443 行"
  makes it unfindable by the very question it answers.
- Keep specifics verbatim: file names, line numbers, addresses, versions, commit
  hashes, error messages.
- Drop acknowledgements, pure pleasantries, and plans never executed.
- If the exchange has nothing durable, return an empty list. Returning nothing
  is a correct and common answer.

Output strict JSON and nothing else:
{"memories":[{"kind":"fact","content":"...","entities":["..."],"importance":0.7}]}

kind is one of: fact, decision, error, preference, context.
importance is 0.0-1.0: how likely a future question depends on this entry."""

USER = """Exchange:
---
{exchange}
---"""


@dataclass(frozen=True, slots=True)
class ExtractedMemory:
    """One durable item distilled from a turn."""

    kind: str
    content: str
    entities: tuple[str, ...] = ()
    importance: float = 0.5


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Outcome of one extraction.

    The distinction between an empty result and a failed one decides what the
    store does: empty means "nothing durable here, drop the turn", failed means
    "the model was unavailable, keep the raw text". Collapsing them would
    either lose data on an outage or keep chit-chat when the model says it is
    chit-chat.
    """

    memories: tuple[ExtractedMemory, ...] = ()
    ok: bool = True


class Extractor:
    """Distils a turn into zero or more :class:`ExtractedMemory`.

    Every public method returns a list, never raises for model problems. A
    failed extraction must degrade to "store the raw turn", which the caller
    decides; it must not take the write down with it.
    """

    def __init__(self, llm: LLM, *, max_memories: int = DEFAULT_MAX_MEMORIES) -> None:
        self.llm = llm
        self.max_memories = max_memories
        self.calls = 0
        self.failures = 0
        self.empty = 0

    @classmethod
    def from_env(cls) -> Extractor | None:
        llm = LLM.from_env()
        return cls(llm) if llm is not None else None

    def extract(
        self,
        text: str,
        *,
        question: str | None = None,
        max_memories: int | None = None,
    ) -> ExtractionResult:
        """Distil one turn, or one question/answer pair.

        Passing ``question`` is strongly preferred when the text is an answer:
        the question carries the words a future search will use, and dropping
        them was measured to halve hit@1 (D-023).
        """
        body = text.strip()
        if not body:
            return ExtractionResult((), ok=True)

        if question and question.strip():
            exchange = f"question: {question.strip()}\nanswer: {body}"
        else:
            exchange = body

        limit = max_memories or self.max_memories
        self.calls += 1
        raw = self.llm.chat(
            SYSTEM, USER.format(exchange=exchange[:MAX_INPUT_CHARS]), max_tokens=1024
        )
        if raw is None:
            self.failures += 1
            return ExtractionResult((), ok=False)

        memories = parse(raw, limit=limit)
        if not memories:
            # "Nothing durable here" is a legitimate answer, not a failure, and
            # must not count against health.
            self.empty += 1
        return ExtractionResult(tuple(memories), ok=True)

    def stats(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "empty": self.empty,
            **self.llm.stats(),
        }


def parse(raw: str, *, limit: int = DEFAULT_MAX_MEMORIES) -> list[ExtractedMemory]:
    """Parse a model response into memories, dropping anything malformed.

    Models wrap JSON in prose or fences often enough that a strict
    ``json.loads`` on the whole string is not enough. Anything that does not
    survive validation is dropped rather than repaired: a guessed memory is
    worse than a missing one.
    """
    blob = _find_json(raw)
    if blob is None:
        return []

    items = blob.get("memories")
    if not isinstance(items, list):
        return []

    out: list[ExtractedMemory] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue

        kind = str(item.get("kind", "")).strip().lower()
        if kind not in KINDS:
            continue

        content = str(item.get("content", "")).strip()
        if not content:
            continue

        raw_entities = item.get("entities")
        entities: list[str] = []
        if isinstance(raw_entities, list):
            for value in raw_entities:
                text = str(value).strip()
                if text and text not in entities:
                    entities.append(text)
        entities = entities[:MAX_ENTITIES]

        try:
            importance = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        importance = min(1.0, max(0.0, importance))

        out.append(
            ExtractedMemory(
                kind=kind,
                content=content[:MAX_CONTENT_CHARS],
                entities=tuple(entities),
                importance=importance,
            )
        )
    return out


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def _find_json(raw: str) -> dict | None:
    text = _FENCE.sub("", raw.strip())
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None

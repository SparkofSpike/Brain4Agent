"""The reading stage: answer from retrieved evidence, on demand.

LongMemEval splits long-term memory into indexing → retrieval → reading. This
project had the first two and stopped at the boundary, handing raw sessions to
the caller. That was defensible while the core was model-free, but it left the
last stage to whoever called us, and D-023 showed what happens when extraction
is moved to the wrong end of the pipeline.

STONE (arXiv:2602.16192) is the reason this module exists. Its argument: deciding
what is useful at *write* time necessarily discards what a different question
would have needed. Extract at read time instead, when the question is actually
known.

So the sequence is:

    store raw turn  →  BM25 finds it  →  this module extracts the answer

Not the reverse. Writing distilled facts and searching them scored 0.775 on the
same questions that raw turns score 0.925.

The read stage also does conflict resolution without asking a model to judge
recency — arXiv:2606.01435 makes the point that "which fact is newer" is data,
not inference, and that entangling it with generation is why memory systems
retrieve correctly and answer wrongly. Dates come from the store; the prompt
tells the reader to prefer the later one and say so.
"""

from __future__ import annotations

from dataclasses import dataclass

from brain4agent.evidence import SearchResult
from brain4agent.llm import LLM

MAX_EVIDENCE_CHARS = 8000

SYSTEM = """You answer strictly from the evidence provided.

Rules:
- If the evidence does not contain the answer, say so plainly and name what is
  missing. Do not fill the gap from your own knowledge.
- Cite the session id the answer came from, in square brackets.
- If two pieces of evidence disagree, prefer the one with the later date and say
  that you did. The dates are given; do not infer them.
- Be terse. One short paragraph or a single sentence is usually right."""

PROMPT = """Question: {query}

Evidence (each block starts with its session id and date):
{evidence}"""


@dataclass(frozen=True, slots=True)
class ReadResult:
    """An answer, or an explicit reason there is none.

    ``answer`` is None only when the model could not be reached. "The evidence
    does not contain this" is a successful read with a negative answer, and the
    two must stay distinguishable — the same three-way split as everywhere else
    in this project.
    """

    query: str
    answer: str | None
    evidence: SearchResult

    @property
    def ok(self) -> bool:
        return self.answer is not None

    @property
    def definitive(self) -> bool:
        """True when the reader ran and produced an answer, positive or negative."""
        return self.answer is not None


def format_evidence(store: object, result: SearchResult) -> str:
    """Render sessions for the reader, newest date first within each block.

    Dates are included because the reader is told to resolve conflicts by them;
    withholding the data and then asking for a recency judgement would be the
    mistake arXiv:2606.01435 describes.
    """
    blocks: list[str] = []
    for item in result.evidence:
        session_id = item.session_id
        turns = store.session(session_id)  # type: ignore[attr-defined]
        if not turns:
            continue
        date = turns[0].created_at[:10]
        body = "\n".join(f"{t.role}: {t.content}" for t in turns)
        blocks.append(f"[{session_id}] {date}\n{body}")
    text = "\n\n".join(blocks)
    return text[:MAX_EVIDENCE_CHARS]


def read(
    store: object,
    query: str,
    *,
    top_k: int = 5,
    llm: LLM | None = None,
) -> ReadResult:
    """Retrieve, then extract an answer from what was retrieved.

    With no model configured this returns the evidence and ``answer=None``, so
    a caller without a key still gets the material — it just has to read it
    itself.
    """
    result = store.search(query, top_k=top_k)  # type: ignore[attr-defined]

    if result.empty:
        return ReadResult(
            query=query,
            answer=(
                "No memory shares any term with this question."
                + (
                    f" The corpus has never contained: {', '.join(result.corpus_missing)}."
                    if result.corpus_missing
                    else ""
                )
            ),
            evidence=result,
        )

    model = llm if llm is not None else LLM.from_env()
    if model is None:
        return ReadResult(query=query, answer=None, evidence=result)

    text = format_evidence(store, result)
    if not text:
        return ReadResult(query=query, answer=None, evidence=result)

    answer = model.chat(SYSTEM, PROMPT.format(query=query, evidence=text), max_tokens=600)
    return ReadResult(query=query, answer=answer, evidence=result)

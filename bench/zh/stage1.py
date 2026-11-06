"""Stage 1: abstention by asking whether the evidence answers the question.

Stage 0 measured two cheap signals and both failed:

    BM25 top-1 score      best balanced 0.688 (chance 0.5), costs 46% of answers
    IDF token coverage    best balanced 0.697, same problem
    unseen-word ratio     best balanced 0.730 with jieba, still not enough

They fail for the same reason: whether a fact is *present* is not a property of
lexical overlap. A question about a database scores highly against a session
about the same server because the server name, the city, the deployment path all
match -- the one thing asked for is what is missing.

So Stage 1 asks a model the narrow question instead: given this evidence, is
the answer here? That is a different question from "answer this", and phrasing
it as judgement rather than generation is what makes small local models usable.

The judge is optional. Without ollama the system falls back to the Stage 0
behaviour (always answer). A memory system that refuses to start because a
local model is missing would be worse than one that cannot abstain.

**Resource cost is the real constraint.** Measured on the 59-question Chinese
set: ``qwen3.5:9b`` is what meets the keep-rate target, and it holds 15.2 GB of
RAM while running -- more than half of a 23.6 GB machine, to the point of
freezing it. ``qwen3.5:4b`` needs about 9 GB and only reaches keep 0.92.
So this path is **off by default** and has to be enabled deliberately; the
memory system itself runs in tens of megabytes. See docs/decisions.md D-011.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from bench.loader import MemoryItem
from bench.retrieval import BM25

HOST = "http://127.0.0.1:11434"

SYSTEM = "You answer with exactly one word: YES or NO. Nothing else."

TEMPLATE = (
    "Evidence: {evidence}\n\n"
    "Question: {question}\n\n"
    "Is the answer present in the evidence? YES or NO:"
)

# Long evidence wastes context and slows small models without improving the
# judgement; the answer, when present, is local.
MAX_EVIDENCE_CHARS = 600


@dataclass(slots=True)
class JudgeConfig:
    model: str = "qwen3.5:9b"
    host: str = HOST
    # First pass: a small window, which is cheap and is where the answer
    # usually is. Second pass: a wider window, only if the first found nothing.
    #
    # A flat top_k was tried and was worse in both directions: at 3 the answer
    # was sometimes at rank 6 and the judge never saw it; at 5 or more every
    # abstention question paid for four extra judgements it did not need, and
    # the wider net raises the chance of a false YES.
    first_pass: int = 3
    second_pass: int = 10
    # Generous: a 9b model loading cold against a 2.4k-character evidence block
    # can exceed two minutes, and a timeout here silently becomes an abstention.
    timeout: int = 300
    # The combined second pass carries several whole sessions, so it needs a
    # larger budget than a single one. Truncating it defeats the purpose.
    max_evidence_chars: int = 2400


@dataclass(slots=True)
class Judgement:
    session_id: str
    memory_id: str
    answers: bool
    raw: str


@dataclass(slots=True)
class Decision:
    """What the system decided, and why.

    ``found`` carries the memory that was judged to answer the question.
    ``None`` with ``abstained=True`` means "there is no memory that answers
    this" -- which is the distinction Stage 0 could not express.
    """

    found: MemoryItem | None
    abstained: bool
    judgements: list[Judgement] = field(default_factory=list)
    seconds: float = 0.0


def _ask(cfg: JudgeConfig, question: str, evidence: str) -> tuple[str, float]:
    """Ask the judge. Returns (raw_text, seconds).

    Failures return an empty string rather than raising: one slow model call
    must not abort a 59-question run, and an empty string parses to None, which
    the caller counts as "no judgement" instead of silently as "no answer".
    """
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": TEMPLATE.format(
                    question=question, evidence=evidence[: cfg.max_evidence_chars]
                ),
            },
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 8},
    }
    req = urllib.request.Request(
        f"{cfg.host}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
            body = json.loads(resp.read())
        return body.get("message", {}).get("content", "").strip(), time.perf_counter() - t0
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        elapsed = time.perf_counter() - t0
        print(f"    [judge error] {type(exc).__name__} after {elapsed:.0f}s")
        return "", elapsed


def parse_yes_no(text: str) -> bool | None:
    """Parse a YES/NO answer, tolerating punctuation and casing.

    Returns None when the model said something else -- treated as "no
    judgement" rather than silently as NO, so a broken judge shows up in the
    error count instead of inflating the abstention rate.
    """
    t = text.strip().upper()
    if t.startswith("YES"):
        return True
    if t.startswith("NO"):
        return False
    return None


def judge_available(cfg: JudgeConfig) -> bool:
    try:
        req = urllib.request.Request(f"{cfg.host}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            models = {m["name"] for m in json.loads(resp.read()).get("models", [])}
        return cfg.model in models
    except (urllib.error.URLError, OSError, TimeoutError, KeyError, ValueError):
        return False


def _session_text(memories: list[MemoryItem], session_id: str) -> str:
    """Join a whole session as evidence.

    A single turn is often not enough: "why did the Chinese query fail" is
    answered across an exchange, and judging one turn at a time made the judge
    reject 40% of answerable questions in a first run. The unit of evidence is
    the session, not the turn.
    """
    turns = [m for m in memories if m.session_id == session_id]
    turns.sort(key=lambda m: m.turn_index)
    return "\n".join(f"{m.role}: {m.content}" for m in turns)


def answer(
    memories: list[MemoryItem],
    question: str,
    cfg: JudgeConfig,
) -> Decision:
    """Retrieve, then decide whether anything actually answers the question.

    Two passes. The first looks at a narrow window because that is where the
    answer usually is, and most questions cost one judgement. If everything in
    that window is rejected, a second pass widens the net before concluding
    that nothing answers the question -- a false abstention is the failure this
    whole stage exists to avoid.
    """
    t0 = time.perf_counter()

    index = BM25()
    index.index([(m.id, m.content) for m in memories])
    hits = index.search(question, top_k=cfg.second_pass * 3)

    if not hits:
        return Decision(found=None, abstained=True, seconds=time.perf_counter() - t0)

    by_id = {m.id: m for m in memories}
    judgements: list[Judgement] = []
    seen_sessions: set[str] = set()

    def judge_until(limit: int) -> MemoryItem | None:
        for memory_id, _score in hits:
            mem = by_id.get(memory_id)
            if mem is None or mem.session_id in seen_sessions:
                continue
            seen_sessions.add(mem.session_id)
            if len(seen_sessions) > limit:
                return None

            evidence = _session_text(memories, mem.session_id)
            raw, _ = _ask(cfg, question, evidence)
            verdict = parse_yes_no(raw)
            judgements.append(
                Judgement(
                    session_id=mem.session_id,
                    memory_id=mem.id,
                    answers=bool(verdict),
                    raw=raw,
                )
            )
            if verdict:
                return mem
        return None

    def judge_combined(limit: int) -> MemoryItem | None:
        """Second pass: judge the top sessions together, as one evidence block.

        Multi-session questions -- "what steps led from the bug to the PR" --
        are answered across several sessions and no single one contains the
        answer. Judging them individually guarantees a false abstention, which
        is the failure this stage exists to prevent.

        The block **includes** sessions the first pass already rejected. An
        earlier version excluded them, on the theory that they had been ruled
        out; that made this pass useless for its actual purpose, because the
        session holding half a multi-session answer was always one of them.
        """
        session_ids: list[str] = []
        first_mem: MemoryItem | None = None
        for memory_id, _score in hits:
            mem = by_id.get(memory_id)
            if mem is None or mem.session_id in session_ids:
                continue
            session_ids.append(mem.session_id)
            if first_mem is None:
                first_mem = mem
            if len(session_ids) >= limit:
                break

        if not session_ids or first_mem is None:
            return None

        evidence = "\n\n---\n\n".join(_session_text(memories, s) for s in session_ids)
        raw, _ = _ask(cfg, question, evidence)
        verdict = parse_yes_no(raw)
        judgements.append(
            Judgement(
                session_id="+".join(session_ids),
                memory_id=first_mem.id,
                answers=bool(verdict),
                raw=raw,
            )
        )
        return first_mem if verdict else None

    found = judge_until(cfg.first_pass)
    if found is None:
        # Only worth a combined pass if there are sessions the first pass did
        # not cover. With one or two hits the block would be the same evidence
        # judged twice -- a wasted call that also counted a duplicate
        # judgement, which the tests caught.
        hit_ids = {h[0] for h in hits}
        available = {m.session_id for m in memories if m.id in hit_ids}
        if len(available) > cfg.first_pass:
            found = judge_combined(cfg.second_pass)

    if found is not None:
        return Decision(
            found=found,
            abstained=False,
            judgements=judgements,
            seconds=time.perf_counter() - t0,
        )

    return Decision(
        found=None, abstained=True, judgements=judgements, seconds=time.perf_counter() - t0
    )

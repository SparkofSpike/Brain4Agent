"""Brain4Agent -- long-term memory for agents.

Chinese-first, single SQLite file, no model in the core path.

    from brain4agent import MemoryStore

    with MemoryStore("brain.db") as store:
        sid = store.new_session()
        store.add("中文查询搜不到记忆", session_id=sid, role="user")
        store.add("根因在 keywords.py 第 443 行", session_id=sid, role="assistant")

        result = store.search("中文查询为什么搜不到")
        print(result.best().missing_terms)

The search returns evidence, not a verdict: whether a session answers the
question is the caller's call. See docs/decisions.md D-011 for why.
"""

from brain4agent.evidence import Evidence, JudgedResult, SearchResult, describe
from brain4agent.judge import Judge, JudgeError
from brain4agent.store import Memory, MemoryStore
from brain4agent.tokenize import BM25, tokenize, word_tokens

__version__ = "0.1.0"

__all__ = [
    "BM25",
    "Evidence",
    "Judge",
    "JudgeError",
    "JudgedResult",
    "Memory",
    "MemoryStore",
    "SearchResult",
    "describe",
    "tokenize",
    "word_tokens",
]

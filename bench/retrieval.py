"""Benchmark-side re-export of the production tokeniser and ranker.

This file used to hold its own copy. That is a trap: the benchmark would keep
scoring a version of BM25 that no longer matched the one shipping, and the two
would drift silently. Everything here now comes from ``brain4agent`` so that a
number in bench/results/ describes the code a user actually runs.

Kept as a module rather than deleted so the existing benchmark code and tests
keep importing ``bench.retrieval`` unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import brain4agent.store as _store
from brain4agent.tokenize import BM25, tokenize, word_tokens

__all__ = ["BM25", "tokenize", "word_tokens", "isolated_bonus"]


@contextmanager
def isolated_bonus(
    *,
    proximity: float | None = 0.0,
    phrase: float | None = None,
    min_terms: int | None = None,
    min_doc_tokens: int | None = None,
) -> Iterator[None]:
    """Turn a store-level ranking knob off (or elsewhere) for the duration of a probe.

    A probe that varies a ranking signal has to start from the same baseline the
    published number was measured against. Once a bonus is on in ``MemoryStore``,
    ``store.search()`` returns *already-boosted* scores, and a probe that applies
    the same signal again measures its own output twice -- the alpha=0 row stops
    reproducing the published number and every row becomes a delta on itself.

    ``bench/locomo/rerank_probe.py`` and ``proximity_sweep.py`` both depend on
    this: their docstrings promise that a disagreement with ``bench.locomo.run``
    is a bug in the probe. Without the reset the promise was false (D-042).

    The two guard knobs exist for the same reason in the other direction: the
    cost of a guard is a measurement, and a measurement needs a way to switch it
    off. ``min_terms`` is the five-term query floor; ``min_doc_tokens`` overrides
    the three-window length floor, where 0 keeps the built-in rule.

    ``None`` leaves a knob alone, so a probe isolating proximity keeps the phrase
    bonus and both guards its published baseline was measured with. Restores
    whatever the module had on entry, including when the block raises.
    """
    saved = (
        _store.PROXIMITY_BONUS,
        _store.PHRASE_BONUS,
        _store.PROXIMITY_MIN_TERMS,
        _store.PROXIMITY_MIN_DOC_TOKENS,
    )
    if proximity is not None:
        _store.PROXIMITY_BONUS = proximity
    if phrase is not None:
        _store.PHRASE_BONUS = phrase
    if min_terms is not None:
        _store.PROXIMITY_MIN_TERMS = min_terms
    if min_doc_tokens is not None:
        _store.PROXIMITY_MIN_DOC_TOKENS = min_doc_tokens
    try:
        yield
    finally:
        (
            _store.PROXIMITY_BONUS,
            _store.PHRASE_BONUS,
            _store.PROXIMITY_MIN_TERMS,
            _store.PROXIMITY_MIN_DOC_TOKENS,
        ) = saved

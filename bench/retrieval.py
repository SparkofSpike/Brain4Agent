"""Benchmark-side re-export of the production tokeniser and ranker.

This file used to hold its own copy. That is a trap: the benchmark would keep
scoring a version of BM25 that no longer matched the one shipping, and the two
would drift silently. Everything here now comes from ``brain4agent`` so that a
number in bench/results/ describes the code a user actually runs.

Kept as a module rather than deleted so the existing benchmark code and tests
keep importing ``bench.retrieval`` unchanged.
"""

from __future__ import annotations

from brain4agent.tokenize import BM25, tokenize, word_tokens

__all__ = ["BM25", "tokenize", "word_tokens"]

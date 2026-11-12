# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-12

First minor release. Six decision-log entries ship here (D-037 to D-042), all
retrieval, all measured, and three of them move a published benchmark.

### Added

- `SearchResult.term_frequencies` reports how many sessions each query term
  appears in. It is the evidence an agent needs when a two-word query returns a
  pile of same-topic memories: the word is too common, not missing and not
  misspelt (D-039).
- `SearchResult.related_terms` lists words the top sessions use that the query
  did not, and `b4a_recall` rows carry `tags`, `created_at` and `kind` from the
  turn that matched (D-040).
- `bench/real/probe.py` measures keyword-query retrieval against a real store
  without committing its contents.

### Changed

- `b4a_remember` writes `date:`, `kind:` and `project:` tags automatically.
  `project` comes from the argument or `BRAIN4AGENT_PROJECT`. The library's
  `add()` is unchanged; only the MCP tool adds the labels (D-040).
- Chinese retrieval: hit@1 0.9250 → 0.9500, hit@5 0.9750 → 1.0000, MRR
  0.9404 → 0.9613. Question words (哪个/为什么/哪里/什么) are dropped from the
  query side, and a session is scored from its best three turns instead of its
  best one (D-037).
- PrecisionMemBench: 34/43 active and 64/77 total → 42/43 and 72/77. The result
  set is sized from the names a query contains at any length, restricted to
  beliefs the case's scope can see; name matching is whole-word rather than
  substring; and an unknown English word is matched one edit away from a corpus
  word (D-038).
- LoCoMo: hit@1 0.6270 → 0.6310, MRR 0.7369 → 0.7402, from the same fuzzy path.
  LongMemEval is unchanged; its harness keeps bigrams, which measured better and
  8× faster than the store's tokenisers on the same tasks.
- English retrieval: LoCoMo hit@1 0.6310 → 0.6826, hit@5 0.8870 → 0.9020,
  hit@10 0.9484 → 0.9549, MRR 0.7402 → 0.7784. English stopwords are dropped
  from the query side, inflections are folded on both sides at index time
  (`camped` matches `camping`), and two adjacent query words earn a small bonus
  when a memory contains them adjacent (`pottery class`). PrecisionMemBench total
  72/77 → 74/77. Chinese and LongMemEval are unchanged (D-041).
- LoCoMo: hit@1 0.6826 → 0.7054, hit@5 0.9020 → 0.9086, MRR 0.7784 → 0.7921,
  hit@10 unchanged. A session is rewarded for keeping the question's words in
  one passage — the fraction of distinct query terms that share a 40-token
  window, multiplied into the score. Only ASCII words take part, documents
  shorter than three windows and queries shorter than five terms are skipped;
  each guard exists because its absence regressed a different benchmark.
  Chinese, PrecisionMemBench, LongMemEval and keyword queries against the real
  store were all measured unchanged (D-042).

### Fixed

- The proximity re-rank was first written with two pruning bounds. Both were
  wrong in ways that returned fewer results than asked for — `>` where `>=` was
  needed never assigned the frontier and skipped every document after the first;
  `<=` where `<` was needed dropped documents that tie the k-th place. Fixing them
  was not enough: instrumenting `window_proximity` showed 271.6 calls per query
  out of 272 candidates, in both call shapes, so neither bound ever fired at the
  shipped alpha. The bounds are removed rather than kept as dead code — sixty
  lines that never fire and had already produced two wrong answers.

- `supersede(old, new)` redirects hits to the current memory instead of
  returning nothing when the replacement uses different wording (D-027).
- The PrecisionMemBench harness sizes `top_k` from the names a query contains at
  every length, not only below eight words, and counts only beliefs the case's
  scope can see (D-038).
- `import` rejects an explicit null role, non-string roles and unknown roles
  without a traceback.
- The hosted judge treats a null `message` as no answer and absorbs malformed
  provider bodies.
- Access counts are credited to the turn that actually matched.
- MCP writes keep `tags` on the distilling path.
- `Evidence` keeps positional compatibility for callers that construct it
  directly.

## [0.0.3] - 2026-11-09

- The release workflow scans the built artifact for machine-local content
  before publishing.
- CI fails when PrecisionMemBench or LoCoMo regresses.

## [0.0.2] - 2026-11-09

- Removed author-specific project names from the published package.

## [0.0.1] - 2026-11-09

- First published release: SQLite-backed session retrieval, CLI, MCP server,
  and BM25 with separate read/write tokenizers.

# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `SearchResult.term_frequencies` reports how many sessions each query term
  appears in. It is the evidence an agent needs when a two-word query returns a
  pile of same-topic memories: the word is too common, not missing and not
  misspelt (D-039).

### Changed

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

### Fixed

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

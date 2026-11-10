# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `supersede(old, new)` redirects hits to the current memory instead of
  returning nothing when the replacement uses different wording (D-027).
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

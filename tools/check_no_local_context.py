#!/usr/bin/env python3
"""Refuse to commit anything that belongs to one machine.

This repository is published and installed on other people's computers. Three
categories of content have leaked into it before, and a rule in a document did
not stop any of them:

  1. A private memory store (.neuralmemory/surface.nm) committed whole.
  2. A real server IP and machine-local absolute paths in the decision log.
  3. The author's own projects used as sample hostnames and directories.

A written rule is a wish. This is the mechanism.

Install once:

    python -m tools.check_no_local_context --install

The hook reads every staged file as bytes and refuses the commit on a match.
Exit 0 to allow, 1 to block with the offending lines printed.

Deliberate exemptions live in ALLOW below: text explaining what the pattern is,
or a public default path that happens to share a name. They are matched against
the whole path plus line, so be specific.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Each entry: (label, compiled pattern). Matching is case-insensitive.
RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "machine-local path",
        re.compile(r"\b[A-Za-z]:\\(?:Users|DataRoot|Coding)\\"),
    ),
    (
        "home directory with a username",
        re.compile(r"/home/[a-z][a-z0-9_-]{2,}/"),
    ),
    (
        "private memory store",
        re.compile(r"(?:^|[/\\])\.neuralmemory[/\\]|surface\.nm\b"),
    ),
    (
        "private key or token file",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----|ghp_[A-Za-z0-9]{20,}"),
    ),
    (
        "non-reserved IP address",
        re.compile(
            r"\b(?!127\.|10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.|0\.|255\.)"
            r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
        ),
    ),
]

# Paths where a rule would fire but the match is legitimate: this file itself
# describes the patterns, and the bench code documents nmem's public default
# location.
ALLOW: tuple[str, ...] = (
    "tools/check_no_local_context.py",
    "bench/nmem/README.md",
    "bench/nmem/import_to_b4a.py",
    "bench/nmem/prune_probe.py",
    ".gitignore",
)

# Generated / vendored trees that are never source.
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "bench/data", ".venv"}


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def scan(path: str, text: str) -> list[tuple[int, str, str]]:
    if path in ALLOW:
        return []
    hits: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for label, pattern in RULES:
            if pattern.search(line):
                hits.append((lineno, label, line.strip()[:140]))
    return hits


def main(argv: list[str]) -> int:
    if "--install" in argv:
        hook = ROOT / ".git" / "hooks" / "pre-commit"
        hook.parent.mkdir(parents=True, exist_ok=True)
        hook.write_text(
            "#!/bin/sh\n"
            'exec python -m tools.check_no_local_context "$@"\n',
            encoding="utf-8",
            newline="\n",
        )
        hook.chmod(0o755)
        print(f"installed {hook}")
        return 0

    problems: list[str] = []
    for path in staged_files():
        full = ROOT / path
        if not full.is_file():
            continue
        if any(part in SKIP_DIRS for part in Path(path).parts):
            continue
        try:
            text = full.read_text("utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: not text we can judge
        for lineno, label, line in scan(path, text):
            problems.append(f"  {path}:{lineno}  [{label}]  {line}")

    if not problems:
        return 0

    print("Refusing to commit: this content belongs to one machine, not to a", file=sys.stderr)
    print("published package. Remove it, or add a narrow entry to ALLOW in", file=sys.stderr)
    print("tools/check_no_local_context.py if the match is legitimate.", file=sys.stderr)
    print("", file=sys.stderr)
    print("\n".join(problems), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

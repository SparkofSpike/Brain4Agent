"""No control characters in text files, and no broken code fences.

A backtick is an escape character in PowerShell here-strings: a backtick followed
by `b` becomes backspace, and a run of three backticks can collapse. Writing
markdown through one produced a U+0008 inside `docs/decisions.md` and turned two
fences into single backticks -- invisible in an editor, and both shipped in a
public document. The discipline ("write with the write tool, not an inline
string") is in AGENTS.md; this is the mechanism that catches the next slip.

Three checks, all exact:

- C0 control characters other than tab/newline/CR
- an odd number of code fences in a markdown file
- a line holding nothing but one backtick

A fourth idea -- verifying every backticked ``bench.\\..`` path exists -- was
dropped: the decision log deliberately names files it removed (D-035 records
deleting the old bench-side evidence module), so that check reports history as
breakage.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Control characters that should never appear in a text file. Tab, newline and
# carriage return are legitimate; everything else in C0 is a mistake or binary.
_ALLOWED = {0x09, 0x0A, 0x0D}
_CTRL = {chr(c): f"U+{c:04X}" for c in range(0x20) if c not in _ALLOWED}
_CTRL[chr(0x7F)] = "U+007F"

_TEXT_SUFFIXES = {".md", ".py", ".toml", ".txt", ".json", ".yml", ".yaml", ".cfg", ".ini"}
_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "data", ".venv", "build"}


def _text_files(suffixes: set[str] | None = None) -> list[Path]:
    wanted = suffixes or _TEXT_SUFFIXES
    return [
        path
        for path in ROOT.rglob("*")
        if not any(part in _SKIP_DIRS for part in path.parts)
        if path.is_file()
        if path.suffix.lower() in wanted
    ]


def test_no_control_characters_in_text_files() -> None:
    hits: list[str] = []
    for path in _text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary masquerading under a text suffix: not this check's job
        for lineno, line in enumerate(text.splitlines(), 1):
            for ch, name in _CTRL.items():
                if ch in line:
                    rel = path.relative_to(ROOT)
                    hits.append(f"{rel}:{lineno} {name} in {line.strip()[:60]!r}")
    assert not hits, "control characters in text files:\n" + "\n".join(hits)


def test_markdown_code_fences_are_balanced() -> None:
    """An odd fence count means one block swallows everything after it."""
    offenders: list[str] = []
    for path in _text_files({".md"}):
        fences = sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith("```")
        )
        if fences % 2:
            offenders.append(f"{path.relative_to(ROOT)}: {fences} fences")
    assert not offenders, "unbalanced code fences:\n" + "\n".join(offenders)


def test_no_bare_backtick_lines_in_markdown() -> None:
    """A line holding one backtick is a fence that lost two characters."""
    offenders: list[str] = []
    for path in _text_files({".md"}):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip() == "`":
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}")
    assert not offenders, "bare backtick lines (broken fences):\n" + "\n".join(offenders)

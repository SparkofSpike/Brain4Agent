"""Tests for bench/zh/e2e.py.

This script is the CI regression gate. If its own logic were wrong -- scoring
the wrong thing, or returning 0 below the threshold -- the gate would be a
rubber stamp, which is worse than no gate.

Each test builds the 44-session corpus and scores 40 questions, so the file
takes a couple of seconds. That is the price of testing the real path rather
than a mock of it.
"""

from __future__ import annotations

import json

from bench.zh.e2e import BASELINE, main


def test_default_run_passes(capsys) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "hit@1" in out
    assert "corpus:" in out


def test_reports_the_recorded_baseline(capsys) -> None:
    """D-019's adopted numbers, so a regression shows up as a diff."""
    main([])
    out = capsys.readouterr().out
    assert "0.9500" in out  # current hit@1
    assert f"{BASELINE['hit@1']:.4f}" in out  # the pre-D-019 baseline


def test_threshold_above_actual_fails(capsys) -> None:
    assert main(["--min-hit1", "0.99"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_threshold_at_actual_passes(capsys) -> None:
    assert main(["--min-hit1", "0.92"]) == 0
    assert "FAIL" not in capsys.readouterr().out


def test_json_mode_emits_only_json(capsys) -> None:
    """A caller piping this should not have to strip a human table first."""
    assert main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    for key in ("hit@1", "hit@5", "hit@10", "mrr", "abstention", "store"):
        assert key in payload, key
    assert payload["store"]["sessions"] == 44
    assert payload["store"]["turns"] == 127


def test_json_mode_still_reports_exit_code(capsys) -> None:
    """The exit code must not be lost just because output is machine-readable."""
    assert main(["--json", "--min-hit1", "0.99"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["hit@1"] < 0.99

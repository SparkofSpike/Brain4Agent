"""Measure the DeepSeek judge: latency, accuracy, and token cost.

Local models were unusable for two independent reasons -- 9b freezes this
machine, and 2b (the only one that fits comfortably) scores 0.75 keep. A hosted
API removes the memory problem but not necessarily the latency problem, so both
are measured here before any conclusion.

`deepseek-chat` returned 503 ("service is too busy") during this run;
`deepseek-flash` works and is a reasoning model -- it emits
`reasoning_content` before the answer, so `max_tokens` must leave room or the
answer is truncated to nothing. That cost a confusing empty response before
being diagnosed.

Run: python -m bench.zh.cloud_judge --limit 12
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from bench.zh.loader import load_zh

RESULTS = Path(__file__).parent.parent / "results"
# Config location is resolved at runtime, never hard-coded: this file
# ships in a public repo and must not name anyone's home directory.
CONFIG = Path(
    os.environ.get(
        "BRAIN4AGENT_JUDGE_CONFIG",
        Path.home() / ".codewhale" / "config.toml",
    )
)

SYSTEM = "You answer with exactly one word: YES or NO. Nothing else."
TEMPLATE = (
    "Evidence:\n{evidence}\n\nQuestion: {question}\n\n"
    "Is the answer present in the evidence? YES or NO:"
)


def _api_key() -> str:
    """Read the DeepSeek key from the local CodeWhale config.

    Read at call time and never written anywhere -- not into the repo, not into
    a result file, not into an error message.
    """
    cfg = CONFIG.read_text(encoding="utf-8")
    seg = cfg.split("[providers.deepseek]")[1].split("[")[0]
    m = re.search(r'api_key\s*=\s*"([^"]+)"', seg)
    if not m:
        raise RuntimeError("no api_key under [providers.deepseek]")
    return m.group(1)


def _session_text(memories, session_id: str) -> str:
    turns = sorted((m for m in memories if m.session_id == session_id), key=lambda m: m.turn_index)
    return "\n".join(f"{m.role}: {m.content}" for m in turns)


def judge(
    key: str,
    question: str,
    evidence: str,
    *,
    model: str = "deepseek-flash",
    timeout: int = 30,
    retries: int = 1,
) -> tuple[bool | None, float, int, int]:
    """Returns (verdict, seconds, prompt_tokens, completion_tokens).

    Retries once on a transport or server error. The hosted endpoint returns
    500/503/null under load at roughly 5%, and a lost call becomes a false
    abstention in the caller -- which is the expensive direction. One retry cuts
    that to ~0.25% at the cost of one extra second only when it happens.

    ``timeout`` is short on purpose: a judge that takes longer than a few
    seconds is worse than no judge for an interactive path, so the call is
    abandoned rather than waited on.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": TEMPLATE.format(question=question, evidence=evidence[:2400]),
            },
        ],
        "max_tokens": 256,  # reasoning tokens come out of this
        "temperature": 0,
    }
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    t0 = time.perf_counter()
    last_error = ""
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:160].decode("utf-8", "replace")
            last_error = f"HTTP {exc.code}: {detail}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:100]}"
        if attempt < retries:
            time.sleep(0.5)
    else:
        print(f"    [api error] {last_error}")
        return None, time.perf_counter() - t0, 0, 0

    dt = time.perf_counter() - t0
    try:
        body = json.loads(raw)
    except ValueError:
        print(f"    [api error] non-JSON body: {raw[:100]!r}")
        return None, dt, 0, 0
    if not isinstance(body, dict):
        print(f"    [api error] body was {type(body).__name__}: {raw[:100]!r}")
        return None, dt, 0, 0

    choices = body.get("choices") or [{}]
    text = (choices[0].get("message", {}).get("content") or "").strip().upper()
    usage = body.get("usage") or {}
    verdict = True if text.startswith("YES") else False if text.startswith("NO") else None
    return verdict, dt, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max answerable questions")
    ap.add_argument("--abs-limit", type=int, default=None, help="max abstention questions")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--json-out", default=str(RESULTS / "zh-cloud-judge.json"))
    args = ap.parse_args()

    key = _api_key()
    tasks = load_zh()
    answerable = [t for t in tasks if not t.query.is_abstention]
    abstention = [t for t in tasks if t.query.is_abstention]
    # Clustered sampling: the abstention questions all sit near the end of the
    # file, so a plain ``limit`` would silently score answerable-only and report
    # a 100% keep rate on half the evidence.
    if args.limit:
        answerable = answerable[: args.limit]
    if args.abs_limit:
        abstention = abstention[: args.abs_limit]
    tasks = answerable + abstention
    print(f"{len(tasks)} questions ({len(answerable)} answerable / {len(abstention)} abstention)")
    print(f"model={args.model}\n")

    kept, refused = 0, 0
    errors = 0
    lat: list[float] = []
    ptok = ctok = 0

    print("--- answerable (a NO here is a false abstention) ---")
    for i, t in enumerate(answerable, 1):
        # Judge the top-3 sessions in order, stop at the first YES.
        from bench.retrieval import BM25

        idx = BM25()
        idx.index([(m.id, m.content) for m in t.memories])
        hits = idx.search(t.query.text, top_k=9)

        found = False
        seen: set[str] = set()
        for mid, _ in hits:
            sid = next((m.session_id for m in t.memories if m.id == mid), None)
            if sid is None or sid in seen:
                continue
            seen.add(sid)
            if len(seen) > 3:
                break
            v, dt, pt, ct = judge(
                key, t.query.text, _session_text(t.memories, sid), model=args.model
            )
            lat.append(dt)
            ptok += pt
            ctok += ct
            if v is None:
                errors += 1
            if v:
                found = True
                break
        if found:
            kept += 1
        elif i <= 5 or i % 10 == 0:
            print(f"  [{i}] rejected: {t.query.text[:40]}")

    print("\n--- abstention (all NO is correct) ---")
    for i, t in enumerate(abstention, 1):
        from bench.retrieval import BM25

        idx = BM25()
        idx.index([(m.id, m.content) for m in t.memories])
        hits = idx.search(t.query.text, top_k=9)

        said_yes = False
        seen: set[str] = set()
        for mid, _ in hits:
            sid = next((m.session_id for m in t.memories if m.id == mid), None)
            if sid is None or sid in seen:
                continue
            seen.add(sid)
            if len(seen) > 3:
                break
            v, dt, pt, ct = judge(
                key, t.query.text, _session_text(t.memories, sid), model=args.model
            )
            lat.append(dt)
            ptok += pt
            ctok += ct
            if v is None:
                errors += 1
            if v:
                said_yes = True
                break
        if not said_yes:
            refused += 1
        else:
            print(f"  [{i}] answered: {t.query.text[:44]}")

    keep_rate = kept / len(answerable) if answerable else 0.0
    refuse_rate = refused / len(abstention) if abstention else 0.0
    calls = len(lat) or 1

    report = {
        "model": args.model,
        "questions": len(tasks),
        "answerable": len(answerable),
        "abstention": len(abstention),
        "keep_rate": round(keep_rate, 4),
        "refuse_rate": round(refuse_rate, 4),
        "constraint_keep_ge_0_95": keep_rate >= 0.95,
        "judge_calls": len(lat),
        "errors": errors,
        "latency_mean_s": round(sum(lat) / calls, 3),
        "latency_min_s": round(min(lat), 3) if lat else 0,
        "latency_max_s": round(max(lat), 3) if lat else 0,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "tokens_per_call": round((ptok + ctok) / calls, 1),
    }

    print("\n=== result ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    RESULTS.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwritten to {args.json_out}")


if __name__ == "__main__":
    main()

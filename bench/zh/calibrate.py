"""Measure judge accuracy on balanced answerable/abstention cases.

PowerShell's Invoke-RestMethod mangles UTF-8 in both directions, which produced
unreadable output and, worse, silently different inputs. Everything here goes
through urllib with explicit UTF-8 so what is measured is the model and not the
shell.

Run: python -m bench.zh.calibrate
"""

from __future__ import annotations

import json
import time
import urllib.request

from bench.zh.loader import load_zh
from bench.zh.stage1 import TEMPLATE, _session_text

HOST = "http://127.0.0.1:11434"


def chat(model: str, system: str, user: str, num_predict: int = 8) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    req = urllib.request.Request(
        f"{HOST}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read()).get("message", {}).get("content", "").strip()


def parse(text: str) -> bool | None:
    t = text.strip().upper()
    if t.startswith("YES"):
        return True
    if t.startswith("NO"):
        return False
    return None


SYSTEMS = {
    "strict": "You answer with exactly one word: YES or NO. Nothing else.",
    "chinese": "你只输出一个词：YES 或 NO。不要输出其他任何内容。",
}


def main() -> None:
    tasks = load_zh()
    answerable = [t for t in tasks if not t.query.is_abstention][:12]
    abstention = [t for t in tasks if t.query.is_abstention][:12]

    models = ["qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b"]
    print(
        f"testing {len(models)} models x {len(SYSTEMS)} prompts "
        f"on {len(answerable)}+{len(abstention)} cases\n"
    )

    for system_name, system in SYSTEMS.items():
        for model in models:
            t0 = time.perf_counter()
            kept, refused, errors = 0, 0, 0
            for task in answerable:
                sess = max(
                    task.query.gold_session_ids, key=lambda s: len(_session_text(task.memories, s))
                )
                ev = _session_text(task.memories, sess)[:800]
                raw = chat(model, system, TEMPLATE.format(question=task.query.text, evidence=ev))
                v = parse(raw)
                if v is None:
                    errors += 1
                elif v:
                    kept += 1
            for task in abstention:
                # Decoy session: the one the retriever would surface.
                ev = _session_text(task.memories, task.memories[0].session_id)[:800]
                raw = chat(model, system, TEMPLATE.format(question=task.query.text, evidence=ev))
                v = parse(raw)
                if v is None:
                    errors += 1
                elif not v:
                    refused += 1
            dt = time.perf_counter() - t0
            keep_rate = kept / len(answerable)
            refuse_rate = refused / len(abstention)
            print(
                f"{system_name:8s} {model:12s} keep={keep_rate:.2f} "
                f"refuse={refuse_rate:.2f} errors={errors} "
                f"{dt / (len(answerable) + len(abstention)):.1f}s/call"
            )


if __name__ == "__main__":
    main()

"""Optional hosted judge. Off unless configured.

The core path never calls a model: retrieval is BM25 over SQLite and the caller
decides whether the evidence answers its question. This module exists for
callers that want a second opinion and cannot spare the reasoning, and it is
measured honestly in docs/decisions.md D-014:

    keep 0.925 / refuse 1.000 / mean 0.81 s / 214 tokens per call

Three things make it safe to ship as an option rather than a dependency:

- **No key, no judge.** ``from_env`` returns ``None`` when nothing is
  configured, and callers fall back to plain evidence. Importing this module
  must never require a key or reach the network.
- **Short timeout, one retry.** The endpoint returned 500/503/null on ~5% of
  calls when measured; one retry took 109 calls to zero failures. Without a
  cap, a single call once took 363 seconds -- unacceptable on an interactive
  path.
- **Three outcomes, not two.** A judge that fails is ``None`` (uncertain), which
  is different from ``False`` (the evidence does not answer it). Collapsing
  those would turn an outage into a confident "not found", which is the exact
  failure this project exists to prevent.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from brain4agent.llm import first_choice_text

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"

SYSTEM = "You answer with exactly one word: YES or NO. Nothing else."
TEMPLATE = (
    "Evidence:\n{evidence}\n\nQuestion: {question}\n\n"
    "Is the answer present in the evidence? YES or NO:"
)

# Reasoning models emit `reasoning_content` before the answer, and that comes out
# of the same budget. 8 tokens returned an empty answer during measurement.
MAX_TOKENS = 256
MAX_EVIDENCE_CHARS = 2400


class JudgeError(RuntimeError):
    """Raised only for configuration mistakes, never for a failed call."""


class Judge:
    """A hosted yes/no judge over retrieved evidence."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 30.0,
        retries: int = 1,
    ) -> None:
        if not api_key:
            raise JudgeError("api_key must not be empty")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.last_latency: float = 0.0
        self.calls: int = 0
        self.errors: int = 0

    @classmethod
    def from_env(cls) -> Judge | None:
        """Build from ``BRAIN4AGENT_JUDGE_KEY``; return None when unset.

        Deliberately does not read any other project's config file. A memory
        library that silently borrows credentials from elsewhere is a surprise
        waiting to happen.
        """
        key = os.environ.get("BRAIN4AGENT_JUDGE_KEY")
        if not key:
            return None
        return cls(
            key,
            base_url=os.environ.get("BRAIN4AGENT_JUDGE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("BRAIN4AGENT_JUDGE_MODEL", DEFAULT_MODEL),
        )

    def _post(self, question: str, evidence: str) -> dict[str, object] | None:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": TEMPLATE.format(
                        question=question, evidence=evidence[:MAX_EVIDENCE_CHARS]
                    ),
                },
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read())
                if isinstance(body, dict):
                    return body
            # HTTPError and URLError both derive from OSError, as does
            # TimeoutError from 3.10 on, so this covers the transport failures
            # this call can actually produce.
            except (OSError, ValueError):
                pass
            if attempt < self.retries:
                time.sleep(0.5)
        return None

    def answer(self, question: str, evidence: str) -> bool | None:
        """True / False / None.

        ``None`` means the judge could not decide -- a failed call, a timeout, or
        unparseable output. Callers must not treat it as False.
        """
        self.calls += 1
        started = time.perf_counter()
        body = self._post(question, evidence)
        self.last_latency = time.perf_counter() - started

        if body is None:
            self.errors += 1
            return None

        text = first_choice_text(body).upper()
        if text.startswith("YES"):
            return True
        if text.startswith("NO"):
            return False
        return None

    def stats(self) -> dict[str, object]:
        return {
            "model": self.model,
            "calls": self.calls,
            "errors": self.errors,
            "error_rate": round(self.errors / self.calls, 4) if self.calls else 0.0,
            "last_latency_s": round(self.last_latency, 3),
        }

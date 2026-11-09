"""A minimal OpenAI-compatible chat client.

Used by two optional features — the abstention judge and the memory extractor —
so the transport, retry and timeout rules live in one place instead of being
copied and diverging.

Nothing here runs unless a key is configured. The core path stays model-free;
this module is an enhancement layer that degrades to "not configured" rather
than to an exception.

Environment (all optional):

    BRAIN4AGENT_LLM_KEY      API key. Unset -> LLM.from_env() returns None.
    BRAIN4AGENT_LLM_URL      Base URL, default https://api.deepseek.com
    BRAIN4AGENT_LLM_MODEL    Model name, default deepseek-chat
    BRAIN4AGENT_LLM_TIMEOUT  Seconds, default 30
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 1


@dataclass(frozen=True, slots=True)
class LLMConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    retries: int = DEFAULT_RETRIES


def first_choice_text(body: object) -> str:
    """The first choice's message text, or "" for any other shape.

    A provider response is not a contract: ``choices`` can be absent, null, a
    dict or a scalar, and ``content`` can be null or a non-string. Every caller
    that reads a completion goes through here, so a malformed body becomes an
    empty answer instead of a crash.
    """
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content.strip() if isinstance(content, str) else ""


class LLM:
    """Synchronous chat completion over HTTP.

    Deliberately not async: every caller here is a single short request on a
    path that is already optional, and an event loop would be a second thing to
    keep correct.
    """

    def __init__(self, config: LLMConfig) -> None:
        if not config.api_key:
            raise ValueError("api_key must not be empty")
        self.config = config
        self.calls = 0
        self.errors = 0
        self.last_latency = 0.0

    @classmethod
    def from_env(cls) -> LLM | None:
        """Build from the environment, or return None.

        Returning None rather than raising is the whole point: "no key" is a
        supported configuration, not an error. Callers check for None and take
        the model-free path.
        """
        key = os.environ.get("BRAIN4AGENT_LLM_KEY")
        if not key:
            return None
        return cls(
            LLMConfig(
                api_key=key,
                base_url=os.environ.get("BRAIN4AGENT_LLM_URL", DEFAULT_BASE_URL),
                model=os.environ.get("BRAIN4AGENT_LLM_MODEL", DEFAULT_MODEL),
                timeout=float(os.environ.get("BRAIN4AGENT_LLM_TIMEOUT", DEFAULT_TIMEOUT)),
            )
        )

    def chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str | None:
        """Return the assistant message, or None on any failure.

        None means "no answer", never "empty answer". Transport errors, HTTP
        errors, unparseable bodies and an empty content field all collapse to
        None so callers have one case to handle.
        """
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )

        self.calls += 1
        started = time.perf_counter()
        for attempt in range(self.config.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as resp:
                    body = json.loads(resp.read())
                # HTTPError and URLError derive from OSError; so does TimeoutError
                # from 3.10 on. This covers the transport failures this can hit.
            except (OSError, ValueError):
                body = None
            finally:
                self.last_latency = time.perf_counter() - started

            if isinstance(body, dict):
                text = first_choice_text(body)
                if text:
                    return text
                # Empty content with a non-empty reasoning budget is a truncation,
                # not a refusal -- but either way there is nothing to return.
                break
            if attempt < self.config.retries:
                time.sleep(0.5)

        self.errors += 1
        return None

    def stats(self) -> dict[str, object]:
        return {
            "model": self.config.model,
            "calls": self.calls,
            "errors": self.errors,
            "error_rate": round(self.errors / self.calls, 4) if self.calls else 0.0,
            "last_latency_s": round(self.last_latency, 3),
        }

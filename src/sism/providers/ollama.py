"""Local self-hosted models via Ollama.

The paper's model panel includes "a locally hosted model run on commodity
hardware" as a self-hosted reference point, to test whether provider-side
alignment tuning changes the size of the effect. This provider is that arm.

It speaks Ollama's OpenAI-compatible endpoint, so a model here is configured
exactly like an OpenRouter one -- only the ``provider`` field and the model
slug change. No API key, no cost, and the same on-disk cache.
"""

from __future__ import annotations

import os
from typing import Sequence

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from ..cache import ResponseCache
from .base import ChatMessage, Completion, ProviderError

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class OllamaProvider:
    provenance = "live-local"

    def __init__(self, *, host: str = DEFAULT_HOST, cache_dir: str = ".cache/ollama",
                 timeout: float = 600.0, concurrency: int = 2, **_: object) -> None:
        # Local inference is compute-bound, not rate-limited: high concurrency
        # slows the whole run down rather than speeding it up.
        self.host = host.rstrip("/")
        self._cache = ResponseCache(cache_dir)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))
        self._concurrency = max(1, min(concurrency, 4))
        self.total_cost_usd = 0.0
        self.n_calls = 0
        self.n_cache_hits = 0

    async def health(self) -> list[str]:
        """Return locally available model tags, or raise a legible error."""
        try:
            r = await self._client.get(f"{self.host}/api/tags", timeout=10.0)
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(
                f"cannot reach Ollama at {self.host}. Start it with `ollama serve`, "
                f"then pull a model, e.g. `ollama pull llama3.1:8b`."
            ) from exc
        return [m["name"] for m in r.json().get("models", [])]

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, ProviderError)),
        wait=wait_exponential_jitter(initial=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def _post(self, payload: dict) -> dict:
        r = await self._client.post(f"{self.host}/v1/chat/completions", json=payload)
        if r.status_code >= 500:
            raise ProviderError(f"ollama {r.status_code}: {r.text[:300]}")
        if r.status_code >= 400:
            raise RuntimeError(f"ollama {r.status_code}: {r.text[:400]}")
        return r.json()

    async def complete(self, messages: Sequence[ChatMessage], *, model: str,
                       temperature: float, max_tokens: int,
                       seed: int | None = None) -> Completion:
        payload = {
            "model": model,
            "messages": [m.as_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed

        key = self._cache.key({**payload, "_host": "ollama"})
        if (hit := self._cache.get(key)) is not None:
            self.n_cache_hits += 1
            return Completion(text=hit["text"], model=model, cached=True)

        data = await self._post(payload)
        try:
            text = (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"malformed ollama response: {str(data)[:300]}") from exc

        usage = data.get("usage") or {}
        self.n_calls += 1
        self._cache.put(key, {"text": text})
        return Completion(
            text=text, model=model,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            cost_usd=0.0,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

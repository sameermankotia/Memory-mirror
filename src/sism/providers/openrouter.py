"""OpenRouter chat-completions client with retry, cost accounting and an on-disk cache."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Sequence

import httpx
from tenacity import (
    retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter,
)

from ..cache import ResponseCache
from ..secrets import get_openrouter_key
from .base import ChatMessage, Completion, ProviderError

API_URL = "https://openrouter.ai/api/v1/chat/completions"

RETRYABLE = (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError, ProviderError)


class RateLimited(ProviderError):
    pass


class OpenRouterProvider:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        cache_dir: str = ".cache/openrouter",
        timeout: float = 180.0,
        concurrency: int = 8,
    ) -> None:
        self._key = api_key or get_openrouter_key()
        self._cache = ResponseCache(cache_dir)
        self._sem = asyncio.Semaphore(concurrency)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=20.0),
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.environ.get(
                    "OPENROUTER_APP_URL", "https://github.com/sameermankotia/sism-eval"
                ),
                "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "SISM-Eval"),
            },
        )
        self.total_cost_usd = 0.0
        self.n_calls = 0
        self.n_cache_hits = 0

    @retry(
        retry=retry_if_exception_type(RETRYABLE),
        wait=wait_exponential_jitter(initial=2, max=45),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _post(self, payload: dict) -> dict:
        resp = await self._client.post(API_URL, json=payload)
        if resp.status_code == 429:
            raise RateLimited("rate limited by OpenRouter")
        if resp.status_code >= 500:
            raise ProviderError(f"upstream {resp.status_code}: {resp.text[:300]}")
        if resp.status_code >= 400:
            # 4xx other than 429 will not get better on retry.
            raise RuntimeError(f"OpenRouter {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> Completion:
        payload = {
            "model": model,
            "messages": [m.as_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        if seed is not None:
            payload["seed"] = seed

        key = self._cache.key(payload)
        if (hit := self._cache.get(key)) is not None:
            self.n_cache_hits += 1
            return Completion(
                text=hit["text"], model=model, cached=True,
                prompt_tokens=hit.get("prompt_tokens", 0),
                completion_tokens=hit.get("completion_tokens", 0),
                cost_usd=0.0,
            )

        async with self._sem:
            data = await self._post(payload)

        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"malformed response: {json.dumps(data)[:400]}") from exc

        usage = data.get("usage") or {}
        cost = float(usage.get("cost", 0.0) or 0.0)
        comp = Completion(
            text=text.strip(),
            model=model,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            cost_usd=cost,
        )
        self.total_cost_usd += cost
        self.n_calls += 1
        self._cache.put(key, {
            "text": comp.text,
            "prompt_tokens": comp.prompt_tokens,
            "completion_tokens": comp.completion_tokens,
        })
        return comp

    async def aclose(self) -> None:
        await self._client.aclose()

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class Completion:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False


class Provider(Protocol):
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> Completion: ...

    async def aclose(self) -> None: ...

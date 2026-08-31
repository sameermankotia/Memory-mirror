from .base import ChatMessage, Completion, Provider, ProviderError
from .ollama import OllamaProvider
from .openrouter import OpenRouterProvider
from .synthetic import SyntheticProvider

_REGISTRY = {
    "openrouter": OpenRouterProvider,
    "ollama": OllamaProvider,
    "synthetic": SyntheticProvider,
}


def make_provider(kind: str, **kw) -> Provider:
    try:
        return _REGISTRY[kind](**kw)
    except KeyError:
        raise ValueError(
            f"unknown provider {kind!r}; expected one of {sorted(_REGISTRY)}"
        ) from None


__all__ = [
    "ChatMessage", "Completion", "Provider", "ProviderError",
    "OpenRouterProvider", "OllamaProvider", "SyntheticProvider", "make_provider",
]

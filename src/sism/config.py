"""Experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelSpec:
    """One system under test.

    ``provider`` is per-model so a single run can mix a proprietary model, an
    open-weight model served through OpenRouter, and a locally hosted one --
    the three-arm panel the paper describes.
    """

    id: str                          # provider-side slug, e.g. "openai/gpt-4o-mini"
    label: str                       # short name used in tables and figures
    arm: str = "open-weight"         # "proprietary" | "open-weight" | "local"
    provider: str = "openrouter"     # "openrouter" | "ollama" | "synthetic"
    temperature: float = 0.7
    max_tokens: int = 700

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JudgeSpec:
    id: str
    label: str
    provider: str = "openrouter"
    temperature: float = 0.0
    max_tokens: int = 400

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunConfig:
    name: str
    models: list[ModelSpec]
    judges: list[JudgeSpec]
    conditions: list[str] = field(
        default_factory=lambda: ["none", "neutral", "flattering", "critical"])
    domains: list[str] = field(
        default_factory=lambda: ["competence", "moral", "decision"])
    n_items: int | None = None       # per domain; None = all
    n_samples: int = 3               # k replies per (item, condition, model)
    seed: int = 20260831
    escalation_turns: int = 3
    run_escalation: bool = True
    elicit_e_star: bool = True
    e_star_source: str = "authored"  # "authored" | "elicited"
    concurrency: int = 8
    out_dir: Path = Path("results")
    notes: str = ""

    @staticmethod
    def load(path: str | Path) -> "RunConfig":
        raw = yaml.safe_load(Path(path).read_text())
        kwargs: dict[str, Any] = {
            k: v for k, v in raw.items() if k not in ("models", "judges", "out_dir")
        }
        return RunConfig(
            models=[ModelSpec(**m) for m in raw["models"]],
            judges=[JudgeSpec(**j) for j in raw["judges"]],
            out_dir=Path(raw.get("out_dir", "results")),
            **kwargs,
        )

    @property
    def run_dir(self) -> Path:
        return self.out_dir / self.name

    @property
    def provider_kinds(self) -> list[str]:
        return sorted({m.provider for m in self.models} | {j.provider for j in self.judges})

    @property
    def provenance(self) -> str:
        """'synthetic' if any model output came from the offline stand-in."""
        return "synthetic" if any(m.provider == "synthetic" for m in self.models) else "live"

    def validate(self) -> None:
        if not self.models:
            raise ValueError("config lists no models")
        if not self.judges:
            raise ValueError("config lists no judges")
        if "none" not in self.conditions:
            raise ValueError("the 'none' (clean context) condition is required: "
                             "Bias0 and SRS are both defined against it")
        if "flattering" not in self.conditions:
            raise ValueError("the 'flattering' condition is required to compute SRS")
        if self.e_star_source == "elicited" and not self.elicit_e_star:
            raise ValueError("e_star_source='elicited' requires elicit_e_star: true")
        labels = [m.label for m in self.models]
        if len(set(labels)) != len(labels):
            raise ValueError(f"model labels must be unique, got {labels}")

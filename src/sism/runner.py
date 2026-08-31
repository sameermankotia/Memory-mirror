"""Orchestration: generate across the condition grid, then judge blind.

Implements Algorithm 1. One deviation from the pseudocode, made deliberately:
the algorithm computes per-item metrics inside the loop, but here generation,
judging and metric computation are separate passes over separate files. That
makes each stage independently resumable and lets the judge run be repeated
with a different judge without regenerating a single reply.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn,
)

from .config import ModelSpec, RunConfig
from .judge import judge_one
from .probes import build_messages, load_items, strip_meta
from .providers import make_provider
from .providers.base import ChatMessage, Provider
from .warrant import calibration, elicit

console = Console()


class ProviderPool:
    """Lazily constructs one provider per kind and closes them all at the end."""

    def __init__(self, concurrency: int) -> None:
        self._concurrency = concurrency
        self._by_kind: dict[str, Provider] = {}

    def get(self, kind: str) -> Provider:
        if kind not in self._by_kind:
            self._by_kind[kind] = make_provider(kind, concurrency=self._concurrency)
        return self._by_kind[kind]

    @property
    def totals(self) -> dict:
        return {
            "total_cost_usd": round(sum(getattr(p, "total_cost_usd", 0.0)
                                        for p in self._by_kind.values()), 4),
            "api_calls": sum(getattr(p, "n_calls", 0) for p in self._by_kind.values()),
            "cache_hits": sum(getattr(p, "n_cache_hits", 0) for p in self._by_kind.values()),
        }

    async def aclose(self) -> None:
        for p in self._by_kind.values():
            try:
                await p.aclose()
            except Exception:  # noqa: BLE001,S110
                pass


@dataclass
class Cell:
    spec: ModelSpec
    item: dict
    condition: str
    sample: int


def _row(spec: ModelSpec, item: dict, condition: str, sample: int, turn: int, comp) -> dict:
    return {
        "model": spec.label, "model_id": spec.id, "arm": spec.arm,
        "item_id": item["id"], "domain": item["domain"],
        "condition": condition, "sample": sample, "turn": turn,
        "warranted": item["warranted"], "counter_strength": item["counter_strength"],
        "checkability": item["checkability"],
        "response": comp.text, "cached": comp.cached,
        "prompt_tokens": comp.prompt_tokens, "completion_tokens": comp.completion_tokens,
        "cost_usd": comp.cost_usd,
    }


async def run_generation(cfg: RunConfig, items: list[dict], pool: ProviderPool) -> pd.DataFrame:
    cells = [
        Cell(m, it, cond, s)
        for m in cfg.models for it in items
        for cond in cfg.conditions for s in range(cfg.n_samples)
    ]
    sem = asyncio.Semaphore(cfg.concurrency)
    results: list[dict] = []

    async def one(c: Cell):
        async with sem:
            live = c.spec.provider != "synthetic"
            msgs = build_messages(c.item, c.condition, sample=c.sample, turn=1,
                                  for_live_api=live)
            try:
                comp = await pool.get(c.spec.provider).complete(
                    msgs, model=c.spec.id, temperature=c.spec.temperature,
                    max_tokens=c.spec.max_tokens, seed=cfg.seed + c.sample,
                )
            except Exception as exc:  # noqa: BLE001 - one bad cell must not kill a run
                console.print(f"[red]gen fail[/] {c.spec.label}/{c.item['id']}/{c.condition}: "
                              f"{type(exc).__name__}: {exc}")
                return None
            return _row(c.spec, c.item, c.condition, c.sample, 1, comp)

    results = await _gather(cells, one, "generating", cfg.concurrency)
    return pd.DataFrame(results)


async def run_escalation(cfg: RunConfig, items: list[dict], pool: ProviderPool) -> pd.DataFrame:
    """Multi-turn: the user re-asserts the self-claim after each reply.

    Beyond the paper's single-turn protocol; supports the escalation slope ES.
    """
    conds = [c for c in ("flattering", "critical") if c in cfg.conditions]
    jobs = [(m, it, c) for m in cfg.models for it in items for c in conds]
    sem = asyncio.Semaphore(cfg.concurrency)

    async def one(job):
        spec, item, cond = job
        async with sem:
            live = spec.provider != "synthetic"
            rows, history = [], []
            for turn in range(1, cfg.escalation_turns + 1):
                msgs = build_messages(item, cond, sample=0, turn=turn,
                                      history=history or None, for_live_api=live)
                try:
                    comp = await pool.get(spec.provider).complete(
                        msgs, model=spec.id, temperature=spec.temperature,
                        max_tokens=spec.max_tokens, seed=cfg.seed,
                    )
                except Exception as exc:  # noqa: BLE001
                    console.print(f"[red]esc fail[/] {spec.label}/{item['id']}: {exc}")
                    break
                rows.append(_row(spec, item, cond, 0, turn, comp))
                history = [*history, ChatMessage("user", strip_meta(msgs[-1].content)),
                           ChatMessage("assistant", comp.text)]
            return rows

    nested = await _gather(jobs, one, "multi-turn escalation", cfg.concurrency)
    flat = [r for rows in nested if rows for r in rows]
    # Turn 1 duplicates the single-turn cell; the slope only needs turns >= 2.
    return pd.DataFrame([r for r in flat if r["turn"] > 1])


async def run_judging(cfg: RunConfig, df: pd.DataFrame, items: list[dict],
                      pool: ProviderPool) -> pd.DataFrame:
    by_id = {it["id"]: it for it in items}
    jobs = [(i, row, j) for i, row in df.iterrows() for j in cfg.judges]
    sem = asyncio.Semaphore(cfg.concurrency)

    async def one(job):
        i, row, jspec = job
        async with sem:
            try:
                v = await judge_one(
                    pool.get(jspec.provider), by_id[row["item_id"]], row["response"],
                    model=jspec.id, label=jspec.label,
                    temperature=jspec.temperature, max_tokens=jspec.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]judge fail[/] row {i}: {exc}")
                return None
            return {
                "model": row["model"], "arm": row["arm"], "item_id": row["item_id"],
                "domain": row["domain"], "condition": row["condition"],
                "sample": int(row["sample"]), "turn": int(row["turn"]),
                "warranted": row["warranted"],
                "counter_strength": row["counter_strength"],
                "checkability": row["checkability"], **v.to_dict(),
            }

    rows = await _gather(jobs, one, "judging", cfg.concurrency)
    return pd.DataFrame([r for r in rows if r is not None])


async def _gather(jobs, fn, label: str, concurrency: int):
    out = []
    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
                  BarColumn(), TaskProgressColumn(), TimeElapsedColumn(),
                  console=console) as prog:
        task = prog.add_task(label, total=len(jobs))
        chunk = max(concurrency * 4, 32)
        for i in range(0, len(jobs), chunk):
            batch = jobs[i:i + chunk]
            out.extend(await asyncio.gather(*(fn(j) for j in batch)))
            prog.advance(task, len(batch))
    return [r for r in out if r is not None]


async def execute(cfg: RunConfig) -> dict:
    cfg.validate()
    items = load_items(cfg.domains, cfg.n_items)
    pool = ProviderPool(cfg.concurrency)
    run_dir = cfg.run_dir
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    console.rule(f"[bold]SISM run: {cfg.name}[/]")
    console.print(
        f"{len(items)} items x {len(cfg.conditions)} conditions x k={cfg.n_samples} "
        f"x {len(cfg.models)} models  ->  "
        f"{len(items) * len(cfg.conditions) * cfg.n_samples * len(cfg.models)} generations, "
        f"x{len(cfg.judges)} judges"
    )
    if cfg.provenance == "synthetic":
        console.print("[yellow]provider=synthetic: pipeline test, NOT findings[/]")

    calib: dict = {}
    try:
        gen = await run_generation(cfg, items, pool)
        if gen.empty:
            raise RuntimeError("generation produced no rows; check credentials and model slugs")

        if cfg.run_escalation:
            esc = await run_escalation(cfg, items, pool)
            esc.to_json(run_dir / "raw" / "escalation.jsonl", orient="records", lines=True)
        else:
            esc = pd.DataFrame()

        all_gen = pd.concat([gen, esc], ignore_index=True) if not esc.empty else gen
        all_gen.to_json(run_dir / "raw" / "generations.jsonl", orient="records", lines=True)

        jud = await run_judging(cfg, all_gen, items, pool)
        jud.to_json(run_dir / "raw" / "verdicts.jsonl", orient="records", lines=True)

        if cfg.elicit_e_star:
            console.print("eliciting evidence-warranted targets e*(x)")
            warr = await elicit(pool.get(cfg.judges[0].provider), items, cfg.judges,
                                concurrency=cfg.concurrency)
            warr.to_json(run_dir / "raw" / "warrant.jsonl", orient="records", lines=True)
            calib = calibration(warr)
    finally:
        totals = pool.totals
        await pool.aclose()

    meta = {
        "run": cfg.name,
        "provenance": cfg.provenance,
        "providers": cfg.provider_kinds,
        "models": [m.as_dict() for m in cfg.models],
        "judges": [j.as_dict() for j in cfg.judges],
        "conditions": cfg.conditions,
        "domains": cfg.domains,
        "n_items": len(items),
        "k_samples": cfg.n_samples,
        "seed": cfg.seed,
        "e_star_source": cfg.e_star_source,
        "n_generations": int(len(all_gen)),
        "n_verdicts": int(len(jud)),
        "parse_failures": int((~jud["parse_ok"]).sum()) if "parse_ok" in jud else 0,
        "e_star_calibration": calib,
        "wall_seconds": round(time.time() - t0, 1),
        "notes": cfg.notes,
        **totals,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    console.print(f"[green]run complete[/] {meta['wall_seconds']}s  "
                  f"${meta['total_cost_usd']}  {meta['n_verdicts']} verdicts  -> {run_dir}")
    return meta

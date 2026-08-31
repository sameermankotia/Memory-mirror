"""Command line interface.

    sism doctor                       check credentials and provider reachability
    sism probes                       probe-set stats and the manipulation check
    sism run      --config C          generate + judge, write raw JSONL
    sism analyse  --run R             metrics, tests, CSV + LaTeX tables
    sism figures  --run R             render the paper figures
    sism all      --config C          run, analyse, figures
    sism export-human --run R         stratified subset for human judge validation
    sism human    --run R --ratings F agreement between humans and the LLM judge
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
def cmd_doctor(args) -> int:
    from .secrets import get_openrouter_key, redact
    ok = True

    key = None
    try:
        key = get_openrouter_key(required=False)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]credential lookup raised[/]: {exc}")
    if key:
        console.print(f"[green]OK[/]  OpenRouter key resolved: {redact(key)}")
    else:
        ok = False
        console.print("[yellow]--[/]  no OpenRouter key. Set OPENROUTER_API_KEY, "
                      "write .env, or unlock 1Password and set SISM_OP_SECRET_REF.")

    async def _ollama():
        from .providers.ollama import OllamaProvider
        p = OllamaProvider()
        try:
            tags = await p.health()
            console.print(f"[green]OK[/]  Ollama reachable, {len(tags)} model(s): "
                          f"{', '.join(tags[:6]) or '(none pulled)'}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]--[/]  Ollama unavailable: {exc}")
        finally:
            await p.aclose()

    asyncio.run(_ollama())

    from .probes import load_items
    try:
        items = load_items()
        console.print(f"[green]OK[/]  probe set: {len(items)} items")
    except Exception as exc:  # noqa: BLE001
        ok = False
        console.print(f"[red]FAIL[/] probe set: {exc}")

    console.print("\n[dim]The synthetic provider always works offline: "
                  "`sism all --config configs/smoke.yaml`[/]")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
def cmd_probes(args) -> int:
    from .memory import build_memory, memory_parity_report
    from .probes import load_items

    items = load_items()
    t = Table(title="probe set", header_style="bold")
    for c in ("domain", "n", "mean e*", "min", "max", "strong counter-evidence"):
        t.add_column(c, justify="right" if c != "domain" else "left")
    for dom in ("competence", "moral", "decision"):
        sub = [i for i in items if i["domain"] == dom]
        if not sub:
            continue
        w = [i["warranted"] for i in sub]
        strong = sum(1 for i in sub if i["counter_strength"] == "strong")
        t.add_row(dom, str(len(sub)), f"{sum(w)/len(w):.1f}", str(min(w)), str(max(w)),
                  f"{strong}/{len(sub)}")
    console.print(t)

    console.print("\n[bold]manipulation check[/] "
                  "(memory blocks must be matched across conditions)")
    p = Table(header_style="bold")
    for c in ("condition", "mean words", "sd", "min", "max"):
        p.add_column(c, justify="right" if c != "condition" else "left")
    for cond, s in memory_parity_report(items).items():
        p.add_row(cond, str(s["mean_words"]), str(s["sd_words"]),
                  str(s["min"]), str(s["max"]))
    console.print(p)

    if args.show:
        it = next(i for i in items if i["id"] == args.show)
        console.print(f"\n[bold]{it['id']}[/]  ({it['domain']})")
        console.print(f"[dim]self-claim:[/] {it['self_claim']}")
        console.print(f"[dim]e* (authored):[/] {it['warranted']}  "
                      f"[dim]-- {it['warrant_rationale']}[/]")
        for cond in ("none", "neutral", "flattering", "critical"):
            m = build_memory(it, cond)
            console.print(f"\n[bold cyan]{cond}[/]\n{m.text or '(no memory block)'}")
    return 0


# ---------------------------------------------------------------------------
def cmd_run(args) -> int:
    from .config import RunConfig
    from .runner import execute

    cfg = RunConfig.load(args.config)
    if args.name:
        cfg.name = args.name
    if args.limit:
        cfg.n_items = args.limit
    if args.no_escalation:
        cfg.run_escalation = False
    meta = asyncio.run(execute(cfg))
    console.print_json(json.dumps({k: v for k, v in meta.items()
                                   if k not in ("models", "judges")}, default=str))
    return 0


# ---------------------------------------------------------------------------
def _prepared(run_dir: str, judge: str | None = None, e_star: str | None = None):
    from .analysis import load_run, prepare
    run = load_run(run_dir)
    a = prepare(run, judge=judge, e_star_source=e_star)
    return run, a


def cmd_analyse(args) -> int:
    from .analysis import write_all
    run, a = _prepared(args.run, args.judge, args.e_star)

    arms = {m["label"]: m.get("arm", "") for m in a["meta"].get("models", [])}
    written = write_all(a, args.run, arms)

    t = Table(title=f"SISM results  ({a['provenance']}, judge={a['primary_judge']}, "
                    f"e*={a['e_star_source']})", header_style="bold")
    for c in ("model", "arm", "Bias0", "SRS", "beta", "MemPresence", "EF"):
        t.add_column(c, justify="right" if c != "model" else "left")
    for _, r in a["summary"].iterrows():
        m = r["model"]
        f = lambda ci: (f"{ci[m].point:+.3f} [{ci[m].lo:+.3f},{ci[m].hi:+.3f}]"  # noqa: E731
                        if m in ci else "--")
        t.add_row(m, arms.get(m, "--"), f(a["bias0_ci"]), f(a["srs_ci"]),
                  f(a["beta_ci"]), f(a["presence_ci"]), f"{r['EF']:.3f}")
    console.print(t)

    if a["judge_agreement"].get("n"):
        ja = a["judge_agreement"]
        console.print(f"\n[bold]judge reliability[/] r={ja.get('pearson_r', float('nan')):.3f}  "
                      f"ICC(2,1)={ja.get('icc2_1', float('nan')):.3f}  "
                      f"MAD={ja.get('mean_abs_diff', float('nan')):.1f}pts  n={ja['n']}")
    if a["e_star_calibration"].get("n"):
        c = a["e_star_calibration"]
        console.print(f"[bold]e* calibration[/] authored vs elicited: "
                      f"r={c.get('pearson_r', float('nan')):.3f}  "
                      f"MAD={c.get('mean_abs_diff', float('nan')):.1f}pts  n={c['n']}")

    console.print("\n[bold]hypotheses[/]")
    for k, txt in a["hypotheses"]["verdict"].items():
        console.print(f"  {k}: {txt}")
    console.print(f"  [dim]H3 caveat: {a['hypotheses']['confound_note']}[/]")

    if a["n_refusals"]:
        console.print(f"\n[bold]refusals[/] {a['n_refusals']} "
                      f"({a['refusal_rate']*100:.2f}% of judged replies)")

    sig = a["tests"][a["tests"]["p_holm"] < 0.05] if not a["tests"].empty else None
    if sig is not None:
        console.print(f"\n[bold]{len(sig)}/{len(a['tests'])}[/] contrasts significant "
                      f"after Holm correction")
    console.print(f"\nwrote {len(written)} files under {args.run}")
    if a["provenance"] == "synthetic":
        console.print("[yellow]provenance=synthetic: these are pipeline numbers, "
                      "not findings. Do not put them in the paper.[/]")
    return 0


def cmd_appendix(args) -> int:
    """Emit the data-derived appendices so the paper cannot drift from the run."""
    from .paper import write_appendices
    written = write_appendices(args.run, args.out)
    for p in written:
        console.print(f"  {p}")
    return 0


def cmd_figures(args) -> int:
    from .figures import render_all
    run, a = _prepared(args.run, args.judge, args.e_star)
    paths = render_all(a, args.out, warrant=run.get("warrant"))
    console.print(f"rendered {len(paths)//2} figures (PDF + PNG) into {args.out}")
    for p in paths:
        if p.suffix == ".pdf":
            console.print(f"  {p}")
    return 0


def cmd_all(args) -> int:
    rc = cmd_run(args)
    if rc:
        return rc
    from .config import RunConfig
    cfg = RunConfig.load(args.config)
    if args.name:
        cfg.name = args.name
    args.run = str(cfg.run_dir)
    args.judge = None
    args.e_star = None
    rc = cmd_analyse(args)
    if rc:
        return rc
    args.out = args.out or "paper/figures"
    rc = cmd_figures(args)
    if rc:
        return rc
    args.out = "paper/generated"
    return cmd_appendix(args)


# ---------------------------------------------------------------------------
def cmd_export_human(args) -> int:
    """Stratified sample for human validation of the judge (paper Sec. 4)."""
    import pandas as pd
    run, a = _prepared(args.run)
    v = a["verdicts"]
    v = v[(v["judge"] == a["primary_judge"]) & (v["turn"] == 1)]
    gen = run["generations"]
    merged = v.merge(gen[["model", "item_id", "condition", "sample", "response"]],
                     on=["model", "item_id", "condition", "sample"], how="left")

    rng = __import__("numpy").random.default_rng(args.seed)
    picks = []
    for (_, _), g in merged.groupby(["condition", "domain"]):
        take = min(args.n_per_cell, len(g))
        picks.append(g.iloc[rng.choice(len(g), take, replace=False)])
    out = pd.concat(picks, ignore_index=True)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    from .probes import load_items
    by_id = {i["id"]: i for i in load_items()}
    out["self_claim"] = out["item_id"].map(lambda i: by_id[i]["self_claim"])
    out["scenario"] = out["item_id"].map(lambda i: by_id[i]["context"])
    out["human_endorsement"] = ""  # annotator fills 0-100
    out["human_notes"] = ""

    # Condition and model are withheld so the human rates blind, exactly as the
    # LLM judge does. The key file restores them for scoring.
    sheet = out[["item_id", "self_claim", "scenario", "response",
                 "human_endorsement", "human_notes"]]
    dest = Path(args.run) / "human"
    dest.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(dest / "rating_sheet.csv", index=False)
    out[["item_id", "model", "condition", "sample", "endorsement"]].to_csv(
        dest / "rating_key.csv", index=False)
    console.print(f"wrote {len(sheet)} blind rows -> {dest/'rating_sheet.csv'}")
    console.print(f"[dim]key (do not open before rating): {dest/'rating_key.csv'}[/]")
    return 0


def cmd_human(args) -> int:
    import numpy as np
    import pandas as pd
    from scipy import stats as sps
    from .stats import _icc2_1

    sheet = pd.read_csv(args.ratings)
    key = pd.read_csv(Path(args.run) / "human" / "rating_key.csv")
    if "human_endorsement" not in sheet:
        console.print("[red]rating file has no human_endorsement column[/]")
        return 1
    df = sheet.join(key[["model", "condition", "endorsement"]])
    df = df[pd.to_numeric(df["human_endorsement"], errors="coerce").notna()]
    if len(df) < 5:
        console.print("[red]fewer than 5 completed ratings[/]")
        return 1
    h = df["human_endorsement"].astype(float).to_numpy()
    m = df["endorsement"].astype(float).to_numpy()
    res = {
        "n": int(len(df)),
        "pearson_r": float(sps.pearsonr(h, m).statistic),
        "spearman_rho": float(sps.spearmanr(h, m).statistic),
        "mean_abs_diff": float(np.abs(h - m).mean()),
        "icc2_1": _icc2_1(np.c_[h, m]),
        "judge_minus_human": float((m - h).mean()),
    }
    out = Path(args.run) / "human" / "human_agreement.json"
    out.write_text(json.dumps(res, indent=2))
    console.print_json(json.dumps(res))
    console.print(f"\nwrote {out}")
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sism", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check credentials and providers").set_defaults(fn=cmd_doctor)

    pp = sub.add_parser("probes", help="probe-set stats and manipulation check")
    pp.add_argument("--show", metavar="ITEM_ID", help="print one item in all conditions")
    pp.set_defaults(fn=cmd_probes)

    def add_run_args(x):
        x.add_argument("--config", required=True)
        x.add_argument("--name", help="override the run name")
        x.add_argument("--limit", type=int, help="items per domain (smoke tests)")
        x.add_argument("--no-escalation", action="store_true")

    pr = sub.add_parser("run", help="generate and judge")
    add_run_args(pr); pr.set_defaults(fn=cmd_run)

    def add_read_args(x):
        x.add_argument("--run", required=True, help="run directory, e.g. results/pilot")
        x.add_argument("--judge", help="judge label for the headline numbers")
        x.add_argument("--e-star", choices=("authored", "elicited"), dest="e_star")

    pa = sub.add_parser("analyse", help="metrics, tests, tables")
    add_read_args(pa); pa.set_defaults(fn=cmd_analyse)

    pf = sub.add_parser("figures", help="render paper figures")
    add_read_args(pf)
    pf.add_argument("--out", default="paper/figures")
    pf.set_defaults(fn=cmd_figures)

    px = sub.add_parser("appendix", help="emit data-derived LaTeX appendices")
    px.add_argument("--run", required=True)
    px.add_argument("--out", default="paper/generated")
    px.set_defaults(fn=cmd_appendix)

    pl = sub.add_parser("all", help="run + analyse + figures + appendices")
    add_run_args(pl)
    pl.add_argument("--out", default="paper/figures")
    pl.set_defaults(fn=cmd_all)

    pe = sub.add_parser("export-human", help="blind rating sheet for judge validation")
    pe.add_argument("--run", required=True)
    pe.add_argument("--n-per-cell", type=int, default=4)
    pe.add_argument("--seed", type=int, default=20260831)
    pe.set_defaults(fn=cmd_export_human)

    ph = sub.add_parser("human", help="human vs judge agreement")
    ph.add_argument("--run", required=True)
    ph.add_argument("--ratings", required=True, help="completed rating_sheet.csv")
    ph.set_defaults(fn=cmd_human)

    return p


def app() -> None:
    args = build_parser().parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    app()

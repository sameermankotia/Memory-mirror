"""Analysis pass: raw verdicts -> the tables the paper reports.

Reads the JSONL a run wrote, computes the paper's three quantities with
item-clustered uncertainty, and emits both machine-readable CSV and
submission-ready LaTeX (booktabs, matching ``tab:main``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import hypotheses as H
from . import metrics as M
from . import stats as S


def load_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    raw = run_dir / "raw"
    if not (raw / "verdicts.jsonl").exists():
        raise FileNotFoundError(f"no verdicts.jsonl under {raw}; run `sism run` first")

    out: dict = {
        "dir": run_dir,
        "verdicts": pd.read_json(raw / "verdicts.jsonl", lines=True),
        "meta": json.loads((run_dir / "run_meta.json").read_text())
        if (run_dir / "run_meta.json").exists() else {},
    }
    for name in ("generations", "warrant", "escalation"):
        p = raw / f"{name}.jsonl"
        out[name] = pd.read_json(p, lines=True) if p.exists() else pd.DataFrame()
    return out


def prepare(run: dict, *, judge: str | None = None,
            e_star_source: str | None = None) -> dict:
    """Build every frame the figures and tables need.

    ``judge`` selects one judge for the headline numbers; the others are
    retained for the reliability analysis. Defaults to the first judge, which
    keeps the primary result from depending on a judge chosen after seeing it.
    """
    v = run["verdicts"].copy()
    v = v[v.get("parse_ok", True).astype(bool)] if "parse_ok" in v else v

    judges = sorted(v["judge"].unique())
    primary = judge or (run["meta"].get("judges") or [{}])[0].get("label") or judges[0]
    if primary not in judges:
        raise ValueError(f"judge {primary!r} not in run; have {judges}")

    src = e_star_source or run["meta"].get("e_star_source", "authored")
    if src == "elicited" and not run["warrant"].empty:
        from .warrant import merged_e_star
        lookup = merged_e_star(run["warrant"])
        v["warranted"] = v["item_id"].map(lookup).fillna(v["warranted"])

    single = v[(v["judge"] == primary) & (v["turn"] == 1)]
    turns = v[v["judge"] == primary]

    item = M.item_level(single)
    wide = M.wide_by_condition(item)
    per_item = M.per_item_metrics(wide)

    refusal_rate = float(v["refused"].mean()) if "refused" in v else 0.0

    return {
        "primary_judge": primary,
        "judges": judges,
        "e_star_source": src,
        "verdicts": v,
        "item": item,
        "wide": wide,
        "per_item": per_item,
        "turns": turns,
        "summary": M.summary_table(item, turns),
        "by_domain": M.domain_table(item),
        "tests": S.condition_tests(wide),
        "srs_ci": S.metric_with_ci(per_item, "SRS"),
        "beta_ci": S.metric_with_ci(per_item, "beta"),
        "bias0_ci": S.metric_with_ci(per_item, "Bias0"),
        "presence_ci": S.metric_with_ci(per_item, "MemPresence"),
        "judge_agreement": S.judge_agreement(v[v["turn"] == 1]) if len(judges) > 1 else {},
        "e_star_calibration": run["meta"].get("e_star_calibration", {}),
        "refusal_rate": refusal_rate,
        "n_refusals": int(v["refused"].sum()) if "refused" in v else 0,
        "hypotheses": H.run_all(per_item),
        "provenance": run["meta"].get("provenance", "live"),
        "meta": run["meta"],
    }


# --------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------
_ARM_LABEL = {"proprietary": "Proprietary", "open-weight": "Open-weight",
              "local": "Local (self-hosted)", "synthetic": "Synthetic"}


def _esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def main_table_tex(a: dict, arms: dict[str, str] | None = None) -> str:
    """``tab:main``: Bias0, SRS and beta per model, with bootstrap CIs."""
    summ = a["summary"]
    arms = arms or {}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Self-image sycophancy by model. $\mathrm{Bias}_0$ is endorsement of the "
        r"self-claim in clean context relative to what the evidence warrants; $\mathrm{SRS}$ is "
        r"the marginal effect of a flattering self-image in memory; $\beta$ is the "
        r"memory-tracking coefficient. Brackets give item-clustered 95\% bootstrap CIs "
        r"($N=" + str(int(summ["n_items"].max())) + r"$ items). Endorsement is on $[0,1]$.}",
        r"\label{tab:main}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Model & Arm & $\mathrm{Bias}_0$ & $\mathrm{SRS}$ & $\beta$ \\",
        r"\midrule",
    ]
    for _, r in summ.iterrows():
        m = r["model"]
        arm = _ARM_LABEL.get(arms.get(m, ""), arms.get(m, "--"))
        cells = []
        for key, ci in (("Bias0", a["bias0_ci"]), ("SRS", a["srs_ci"]), ("beta", a["beta_ci"])):
            est = ci.get(m)
            cells.append(f"{est.point:+.3f} [{est.lo:+.3f}, {est.hi:+.3f}]"
                         if est else f"{r[key]:+.3f}")
        lines.append(f"{_esc(m)} & {arm} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def domain_table_tex(a: dict) -> str:
    """SRS per (model, domain) -- the numeric companion to srs_by_domain.pdf."""
    d = a["by_domain"].pivot_table(index="model", columns="domain", values="SRS")
    cols = [c for c in ("competence", "moral", "decision") if c in d.columns]
    d = d[cols]
    head = " & ".join(M_DOMAIN.get(c, c) for c in cols)
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Self-Reinforcement Score by self domain. Higher values mean a "
        r"flattering stored self-image moves the model further from its clean-context "
        r"assessment.}",
        r"\label{tab:domain}",
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule", f"Model & {head} " + r"\\", r"\midrule",
    ]
    for model, row in d.iterrows():
        vals = " & ".join(f"{row[c]:+.3f}" for c in cols)
        lines.append(f"{_esc(model)} & {vals} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


M_DOMAIN = {"competence": "Competence", "moral": "Moral character",
            "decision": "Personal decisions"}


def reliability_tex(a: dict) -> str:
    """The judge-validation paragraph's numbers, as a small table."""
    ja, cal = a["judge_agreement"], a["e_star_calibration"]
    rows = []
    if ja.get("n"):
        rows.append(("Cross-judge endorsement", f"$r={ja.get('pearson_r', float('nan')):.3f}$, "
                     f"ICC(2,1)$={ja.get('icc2_1', float('nan')):.3f}$, "
                     f"MAD$={ja.get('mean_abs_diff', float('nan')):.1f}$ pts", ja["n"]))
    if cal.get("n"):
        rows.append((r"Authored vs.\ elicited $e^{\star}$",
                     f"$r={cal.get('pearson_r', float('nan')):.3f}$, "
                     f"MAD$={cal.get('mean_abs_diff', float('nan')):.1f}$ pts", cal["n"]))
    if not rows:
        return "% reliability table: needs >1 judge and elicit_e_star: true\n"
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Measurement reliability. MAD is mean absolute difference on the "
        r"judge's 0--100 scale.}", r"\label{tab:reliability}",
        r"\begin{tabular}{llc}", r"\toprule",
        r"Check & Agreement & $n$ \\", r"\midrule",
    ]
    for name, val, n in rows:
        lines.append(f"{name} & {val} & {n} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def write_all(a: dict, outdir: str | Path, arms: dict[str, str] | None = None) -> list[Path]:
    outdir = Path(outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    frames = {"summary": a["summary"], "by_domain": a["by_domain"],
              "tests": a["tests"], "per_item": a["per_item"]}
    for key in ("H1", "H2", "H3_domain", "H3_checkability_pooled",
                "H3_checkability_within_domain"):
        frames[f"hypothesis_{key}"] = a["hypotheses"][key]
    for name, frame in frames.items():
        p = outdir / f"{name}.csv"
        frame.to_csv(p, index=False)
        written.append(p)

    for name, tex in (("main_table", main_table_tex(a, arms)),
                      ("domain_table", domain_table_tex(a)),
                      ("reliability_table", reliability_tex(a))):
        p = outdir / "tables" / f"{name}.tex"
        p.write_text(tex + "\n")
        written.append(p)

    report = {
        "primary_judge": a["primary_judge"], "judges": a["judges"],
        "e_star_source": a["e_star_source"], "provenance": a["provenance"],
        "srs": {k: [v.point, v.lo, v.hi] for k, v in a["srs_ci"].items()},
        "beta": {k: [v.point, v.lo, v.hi] for k, v in a["beta_ci"].items()},
        "bias0": {k: [v.point, v.lo, v.hi] for k, v in a["bias0_ci"].items()},
        "mem_presence": {k: [v.point, v.lo, v.hi] for k, v in a["presence_ci"].items()},
        "judge_agreement": a["judge_agreement"],
        "e_star_calibration": a["e_star_calibration"],
        "refusal_rate": a["refusal_rate"],
        "n_refusals": a["n_refusals"],
        "hypothesis_verdicts": a["hypotheses"]["verdict"],
        "h3_confound_note": a["hypotheses"]["confound_note"],
    }
    p = outdir / "report.json"
    p.write_text(json.dumps(report, indent=2, default=float))
    written.append(p)
    return written

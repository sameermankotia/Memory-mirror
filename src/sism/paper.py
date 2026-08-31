"""Generate the paper's appendices from the artifacts themselves.

Appendices B, C and E restate things that already exist in code and data: the
judge rubric, worked probe examples, and the reproducibility record. Writing
them by hand guarantees they drift from what was actually run. These emitters
read the live rubric string, the real probe items, and the run metadata, so the
appendix cannot describe a protocol the code does not implement.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from .judge import JUDGE_SYSTEM, RUBRIC
from .memory import MEMORY_HEADER, build_memory
from .probes import BASE_SYSTEM, build_user_turn, load_items

DOMAIN_TITLE = {"competence": "Competence", "moral": "Moral character",
                "decision": "Personal decisions"}


def _tex(s: str) -> str:
    """Escape a plain string for LaTeX."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s.replace("'", "'").replace("'", "'").replace('"', "``").replace('"', "''")


def rubric_tex() -> str:
    """Appendix: the exact rubric text the judge receives."""
    return "\n".join([
        r"\section{Endorsement Rubric}", r"\label{app:rubric}",
        "The judge is condition-blind: it sees the self-claim, the evidence the",
        "person supplied, and the reply, but never the memory block or which",
        "condition produced the reply. Scores are elicited on 0--100 and divided",
        "by 100 for reporting; the anchors below are the ones in the prompt.",
        "", r"\paragraph{System message.}",
        r"\begin{quote}\small\ttfamily", _tex(JUDGE_SYSTEM), r"\end{quote}",
        "", r"\paragraph{Scoring instruction.}",
        r"\begin{quote}\small\ttfamily",
        "\\\\\n".join(_tex(l) if l.strip() else r"\\" for l in RUBRIC.splitlines()),
        r"\end{quote}",
    ])


def worked_examples_tex(items: list[dict] | None = None) -> str:
    """Appendix: one real item per domain, in every memory condition."""
    items = items or load_items()
    out = [r"\section{Probe Construction and Worked Examples}", r"\label{app:items}",
           "Each item pairs a scenario $x$ whose facts under-determine the self-claim",
           "$s$ with memory notes that differ only in what they say about how the user",
           "sees themselves. Every note carries the same factual line and the same two",
           "shared preference lines, so the conditions are matched on length and",
           "content and differ only in valence. Items below are verbatim from the",
           "released probe set.", ""]

    for dom in ("competence", "moral", "decision"):
        pick = next((i for i in items if i["domain"] == dom), None)
        if not pick:
            continue
        out += [
            rf"\paragraph{{{DOMAIN_TITLE[dom]} ({_tex(pick['id'])}).}}",
            rf"\textbf{{Scenario $x$:}} ``{_tex(pick['context'])}''",
            "",
            rf"\textbf{{Self-claim $s$:}} ``{_tex(pick['self_claim'])}''",
            "",
            rf"\textbf{{Question put to the model:}} ``{_tex(pick['question'])}''",
            "",
            rf"\textbf{{Memory $+$:}} {_tex(build_memory(pick, 'flattering').text.splitlines()[1][2:])}",
            "",
            rf"\textbf{{Memory $-$:}} {_tex(build_memory(pick, 'critical').text.splitlines()[1][2:])}",
            "",
            rf"\textbf{{Shared across conditions:}} {_tex(pick['memory_detail'])}",
            "",
            rf"\textbf{{Warranted level $e^{{\star}}$:}} {pick['warranted']/100:.2f} "
            rf"--- {_tex(pick['warrant_rationale'])}",
            "",
            rf"\textbf{{Checkability:}} {pick['checkability']}",
            "",
        ]
    return "\n".join(out)


def prompts_tex(items: list[dict] | None = None) -> str:
    """Appendix: the exact system context a target model receives."""
    items = items or load_items()
    it = items[0]
    mem = build_memory(it, "flattering")
    return "\n".join([
        r"\section{Target-Model Prompt Format}", r"\label{app:prompts}",
        "The memory block is placed in the system context ahead of the turn, in the",
        "note format used by mainstream assistants. Under $m = \\varnothing$ the block",
        "is omitted entirely and the system message is the first line alone.",
        "", r"\paragraph{System message ($m = +$).}",
        r"\begin{quote}\small\ttfamily",
        "\\\\\n".join(_tex(l) for l in (BASE_SYSTEM + "\n\n" + mem.text).splitlines()),
        r"\end{quote}",
        "", r"\paragraph{User turn.}",
        r"\begin{quote}\small\ttfamily",
        "\\\\\n".join(_tex(l) for l in textwrap.wrap(build_user_turn(it), 88)),
        r"\end{quote}",
    ])


def repro_tex(run_dir: str | Path) -> str:
    """Appendix: the reproducibility record, read from run metadata."""
    meta = json.loads((Path(run_dir) / "run_meta.json").read_text())
    rep = {}
    p = Path(run_dir) / "report.json"
    if p.exists():
        rep = json.loads(p.read_text())

    lines = [r"\section{Reproducibility}", r"\label{app:repro}"]
    if meta.get("provenance") == "synthetic":
        lines += [r"\textcolor{red}{\textbf{This record describes a synthetic "
                  r"pipeline test, not a model evaluation.}}", ""]

    lines += [r"\paragraph{Models under test.}", r"\begin{itemize}\itemsep0pt"]
    for m in meta.get("models", []):
        lines.append(rf"\item \texttt{{{_tex(m['id'])}}} ({_tex(m.get('arm',''))}, "
                     rf"via {_tex(m.get('provider',''))}), temperature "
                     rf"{m.get('temperature')}, max tokens {m.get('max_tokens')}")
    lines += [r"\end{itemize}", "", r"\paragraph{Judges.}", r"\begin{itemize}\itemsep0pt"]
    for j in meta.get("judges", []):
        lines.append(rf"\item \texttt{{{_tex(j['id'])}}}, temperature {j.get('temperature')}")
    lines += [r"\end{itemize}", ""]

    ja = rep.get("judge_agreement") or {}
    cal = rep.get("e_star_calibration") or {}
    facts = [
        ("Probe items", meta.get("n_items")),
        ("Memory conditions", ", ".join(meta.get("conditions", []))),
        ("Samples per cell ($k$)", meta.get("k_samples")),
        ("Random seed", meta.get("seed")),
        ("Source of $e^{\\star}$", meta.get("e_star_source")),
        ("Total model generations", meta.get("n_generations")),
        ("Total judge verdicts", meta.get("n_verdicts")),
        ("Unparseable judge outputs", meta.get("parse_failures")),
        ("Model refusals", f"{rep.get('n_refusals', 0)} "
                           f"({100*rep.get('refusal_rate', 0):.2f}\\%)"),
        ("Cross-judge ICC(2,1)", f"{ja['icc2_1']:.3f}" if ja.get("icc2_1") else "--"),
        ("Authored vs.\\ elicited $e^{\\star}$ ($r$)",
         f"{cal['pearson_r']:.3f}" if cal.get("pearson_r") else "--"),
        ("Total API cost (USD)", f"\\${meta.get('total_cost_usd', 0)}"),
        ("Wall-clock seconds", meta.get("wall_seconds")),
    ]
    lines += [r"\begin{table}[h]\centering\small", r"\begin{tabular}{ll}", r"\toprule"]
    for k, v in facts:
        lines.append(f"{k} & {v} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    lines += [
        "Refusals are counted by the judge, which sets a dedicated flag when a reply",
        "declines to engage at all; a reply that disagrees with the user is not a",
        "refusal. Refused replies are retained in the raw logs and excluded from no",
        "aggregate, since a refusal is a legitimate low-endorsement response.",
        "", r"\paragraph{Artifacts.}",
        "The probe set, rubric, judge prompts, run configuration and analysis code",
        "are released together; every number above is regenerated by",
        r"\texttt{sism run \&\& sism analyse}.",
    ]
    return "\n".join(lines)


def write_appendices(run_dir: str | Path, out: str | Path) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    items = load_items()
    files = {
        "app_rubric.tex": rubric_tex(),
        "app_items.tex": worked_examples_tex(items),
        "app_prompts.tex": prompts_tex(items),
        "app_repro.tex": repro_tex(run_dir),
    }
    written = []
    for name, body in files.items():
        p = out / name
        p.write_text(body + "\n")
        written.append(p)
    return written

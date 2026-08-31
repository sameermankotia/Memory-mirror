"""Publication figures.

Every figure is written for a 4-page two-column-ish layout: legible at
\\linewidth, vector PDF for the paper, 400 dpi PNG for slides. Any figure built
from the offline synthetic provider is watermarked by ``theme.save``.

Colour decisions follow the reference palette:
* conditions carry a *diverging* pair (self-doubting <- neutral -> flattering),
  because they have a real polarity;
* single-measure charts use one hue, since a second hue would encode nothing;
* scatter forms use one hue plus direct labels, which sidesteps the all-pairs
  CVD cap on categorical sets entirely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import theme as T
from .theme import (
    AQUA, BLUE, CONDITION_COLOR, CONDITION_LABEL, DIVERGING, DOMAIN_LABEL,
    DOMAIN_ORDER, GRAY_DARK, GRAY_MID, GRID, INK, INK_2, INK_MUTED, ORANGE,
    RED, SEQUENTIAL, VIOLET, despine, save,
)

BAR_GAP = 0.02  # ~2px surface gap between adjacent bars at typical figure sizes


# ---------------------------------------------------------------------------
# Fig 1 -- protocol schematic
# ---------------------------------------------------------------------------
def fig_design(outdir: Path, provenance: str = "live") -> list[Path]:
    """The measurement, drawn: one item, four memory conditions, three metrics."""
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.set_xlim(0, 11.4); ax.set_ylim(-1.55, 6.5); ax.axis("off")

    def box(x, y, w, h, label, sub, ec, tc=INK, fc="#ffffff"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.11",
                                    fc=fc, ec=ec, lw=1.1, zorder=3))
        ax.text(x + w / 2, y + h * 0.63, label, ha="center", va="center",
                fontsize=8.6, weight="bold", color=tc, zorder=4)
        if sub:
            ax.text(x + w / 2, y + h * 0.25, sub, ha="center", va="center",
                    fontsize=6.9, color=INK_2, zorder=4, linespacing=1.3)

    # column geometry
    CX, CW, CH, GAP = 2.55, 1.95, 1.05, 0.20
    conds = [("flattering", "$m = +$", "flattering self-image"),
             ("neutral",    "control", "memory, no valence"),
             ("none",       "$m = \\varnothing$", "clean context"),
             ("critical",   "$m = -$", "self-doubting image")]
    top = 5.15
    ys = [top - i * (CH + GAP) - CH for i in range(len(conds))]
    mid = (ys[0] + CH + ys[-1]) / 2

    box(0.15, mid - 0.68, 1.95, 1.36, "Item $(x, s)$", "scenario +\nself-claim", GRAY_MID)

    for y, (key, lab, sub) in zip(ys, conds):
        c = CONDITION_COLOR[key]
        tc = c if key != "neutral" else GRAY_DARK
        box(CX, y, CW, CH, lab, sub, c, tc=tc)
        ax.add_patch(FancyArrowPatch((2.14, mid), (CX - 0.06, y + CH / 2),
                                     arrowstyle="-|>", mutation_scale=7,
                                     color=GRAY_MID, lw=0.85, zorder=1,
                                     shrinkA=0, shrinkB=2))

    TX = CX + CW + 0.55
    box(TX, mid - 1.05, 1.55, 2.1, "Target\nmodel", "$k$ replies\nper condition", BLUE, tc=BLUE, fc="#f7f9fc")
    for y, (key, *_ ) in zip(ys, conds):
        ax.add_patch(FancyArrowPatch((CX + CW + 0.06, y + CH / 2), (TX - 0.06, mid),
                                     arrowstyle="-|>", mutation_scale=7,
                                     color=CONDITION_COLOR[key], lw=0.85, alpha=0.7,
                                     zorder=1, shrinkA=2, shrinkB=2))

    JX = TX + 1.55 + 0.5
    box(JX, mid - 1.05, 1.5, 2.1, "Blind\njudge", "rubric score\n$e \\in [0,1]$",
        AQUA, tc="#0f7a55", fc="#f6fbf9")
    ax.add_patch(FancyArrowPatch((TX + 1.55 + 0.05, mid), (JX - 0.05, mid),
                                 arrowstyle="-|>", mutation_scale=8, color=GRAY_DARK, lw=1.0))
    ax.text(JX + 0.75, mid - 1.28, "never sees the condition", ha="center",
            fontsize=6.5, color=INK_MUTED, style="italic")

    EX = JX + 1.5 + 0.42
    ax.add_patch(FancyArrowPatch((JX + 1.5 + 0.05, mid), (EX - 0.12, mid),
                                 arrowstyle="-|>", mutation_scale=8, color=GRAY_DARK, lw=1.0))
    for dy, txt, c in ((1.02, r"$\mathrm{Bias}_0 = e(\varnothing) - e^{\star}$", GRAY_DARK),
                       (0.00, r"$\mathrm{SRS} = e(+) - e(\varnothing)$", RED),
                       (-1.02, r"$\beta = \frac{1}{2}\,[\,e(+) - e(-)\,]$", VIOLET)):
        ax.text(EX, mid + dy, txt, ha="left", va="center", fontsize=8.8, color=c)

    # e*(x): rated from the facts alone, feeding Bias_0 only.
    ex_y = ys[-1] - 1.55
    box(0.15, ex_y, 1.95, 1.30, "$e^{\\star}(x)$", "warranted level,\nno claimant",
        ORANGE, tc="#a8461a", fc="#fdfaf5")
    ax.add_patch(FancyArrowPatch((1.12, mid - 0.70), (1.12, ex_y + 1.34),
                                 arrowstyle="-", color=GRAY_MID, lw=0.9,
                                 linestyle=(0, (2.5, 2.5)), zorder=1))
    # Route e* along the bottom channel and up the right margin, so the link to
    # Bias_0 never crosses the condition stack.
    ax.add_patch(FancyArrowPatch(
        (2.16, ex_y + 0.62), (EX - 0.16, mid + 1.02),
        arrowstyle="-|>", mutation_scale=7, color=ORANGE, lw=0.9, alpha=0.65,
        linestyle=(0, (2.5, 2.5)), zorder=0, shrinkA=2, shrinkB=3,
        connectionstyle="angle,angleA=0,angleB=-90,rad=10"))

    ax.text(0.15, 6.20, "Memory-conditioned self-image sycophancy",
            fontsize=10.5, weight="bold", color=INK, va="top")
    ax.text(0.15, 5.80, "The self-claim is held fixed; only what memory says about the user varies.",
            fontsize=8, color=INK_2, va="top")
    return save(fig, outdir, "fig1_design", provenance)


# ---------------------------------------------------------------------------
# Fig 2 -- SRS by domain and model  (referenced as srs_by_domain.pdf)
# ---------------------------------------------------------------------------
def fig_srs_by_domain(a: dict, outdir: Path) -> list[Path]:
    d = a["by_domain"]
    models = list(a["summary"]["model"])
    doms = [x for x in DOMAIN_ORDER if x in set(d["domain"])]
    piv = d.pivot_table(index="model", columns="domain", values="SRS").reindex(models)

    fig, ax = plt.subplots(figsize=(7.0, 0.62 * len(models) + 1.75))
    n = len(doms)
    h = 0.78 / n
    # One hue per domain from the sequential ramp: domains are facets of one
    # measure, not independent identities, so a single ramp reads correctly.
    shades = [SEQUENTIAL(v) for v in np.linspace(0.32, 0.92, n)]

    for j, dom in enumerate(doms):
        ys = np.arange(len(models)) + (j - (n - 1) / 2) * (h + BAR_GAP)
        vals = piv[dom].to_numpy(float)
        ax.barh(ys, vals, height=h, color=shades[j], edgecolor="white", linewidth=0.8,
                label=DOMAIN_LABEL.get(dom, dom), zorder=3)
        for y, v in zip(ys, vals):
            if np.isfinite(v):
                ax.text(v + (0.004 if v >= 0 else -0.004), y, f"{v:+.2f}",
                        va="center", ha="left" if v >= 0 else "right",
                        fontsize=6.8, color=INK_2, zorder=4)

    ax.axvline(0, color=GRAY_DARK, lw=1.0, zorder=2)
    ax.set_yticks(np.arange(len(models))); ax.set_yticklabels(models, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Self-Reinforcement Score   SRS $= e(+) - e(\\varnothing)$")
    # Descriptive, not a conclusion: the harness cannot know the direction of the
    # effect before the run, and a headline that asserts one would be wrong for
    # any model that resists.
    ax.set_title("Effect of a flattering stored self-image, by self domain",
                 loc="left", pad=10)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.13)
    despine(ax, keep=("bottom",))
    ax.legend(loc="lower right", ncols=n, fontsize=7.6,
              bbox_to_anchor=(1.0, -0.16 - 0.02 * len(models)))
    fig.text(0.005, 0.005, "Positive = memory moves the model away from its clean-context view.",
             fontsize=6.8, color=INK_MUTED)
    return save(fig, outdir, "srs_by_domain", a.get("provenance", "live"))


# ---------------------------------------------------------------------------
# Fig 3 -- condition profiles against the warranted level
# ---------------------------------------------------------------------------
def fig_condition_profiles(a: dict, outdir: Path) -> list[Path]:
    item = a["item"]
    models = list(a["summary"]["model"])
    conds = [c for c in ("critical", "none", "neutral", "flattering") if c in set(item["condition"])]

    ncol = min(3, len(models))
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.55 * ncol, 2.5 * nrow),
                             sharey=True, squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // ncol][idx % ncol]
        g = item[item["model"] == model]
        e_star = g.drop_duplicates("item_id")["e_star"].mean()

        xs = np.arange(len(conds))
        means, los, his = [], [], []
        for c in conds:
            vals = g[g["condition"] == c]["e"].to_numpy(float)
            from ..stats import bootstrap_mean
            est = bootstrap_mean(vals)
            means.append(est.point); los.append(est.lo); his.append(est.hi)

        ax.axhspan(0, e_star, color=GRID, alpha=0.55, zorder=0)
        ax.axhline(e_star, color=GRAY_DARK, lw=1.0, ls=(0, (3, 2)), zorder=2)
        for x, c, m, lo, hi in zip(xs, conds, means, los, his):
            ax.vlines(x, lo, hi, color=CONDITION_COLOR[c], lw=1.6, zorder=3)
            ax.plot(x, m, "o", ms=7, color=CONDITION_COLOR[c],
                    mec="white", mew=1.4, zorder=4)
        ax.plot(xs, means, color=GRAY_MID, lw=1.0, zorder=1)

        ax.set_xticks(xs)
        ax.set_xticklabels([{"critical": "$-$", "none": "$\\varnothing$",
                             "neutral": "ctrl", "flattering": "$+$"}[c] for c in conds],
                           fontsize=9)
        ax.set_title(model, fontsize=8.5)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
        despine(ax)
        if idx % ncol == 0:
            ax.set_ylabel("endorsement $e$")

    for k in range(len(models), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    handles = [Line2D([], [], color=GRAY_DARK, ls=(0, (3, 2)), lw=1.0,
                      label="$e^{\\star}$: warranted by the evidence")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), fontsize=7.6)
    fig.suptitle("Endorsement by memory condition, against what the evidence warrants",
                 fontsize=10, weight="bold", x=0.012, ha="left", y=1.005)
    fig.tight_layout()
    return save(fig, outdir, "fig3_condition_profiles", a.get("provenance", "live"))


# ---------------------------------------------------------------------------
# Fig 4 -- forest plot of the three headline quantities
# ---------------------------------------------------------------------------
def fig_forest(a: dict, outdir: Path) -> list[Path]:
    models = list(a["summary"]["model"])
    panels = [("Bias0", a["bias0_ci"], r"$\mathrm{Bias}_0 = e(\varnothing) - e^{\star}$", GRAY_DARK),
              ("SRS", a["srs_ci"], r"$\mathrm{SRS} = e(+) - e(\varnothing)$", RED),
              ("beta", a["beta_ci"], r"$\beta = \frac{1}{2}[e(+) - e(-)]$", VIOLET)]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 0.42 * len(models) + 1.9), sharey=True)
    for ax, (key, ci, title, color) in zip(axes, panels):
        ys = np.arange(len(models))
        for y, m in zip(ys, models):
            est = ci.get(m)
            if est is None:
                continue
            crosses = est.lo <= 0 <= est.hi
            ax.hlines(y, est.lo, est.hi, color=color, lw=2.0,
                      alpha=0.35 if crosses else 1.0, zorder=3)
            ax.plot(est.point, y, "o", ms=6.5, color=color, mec="white", mew=1.3,
                    alpha=0.45 if crosses else 1.0, zorder=4)
        ax.axvline(0, color=GRAY_DARK, lw=1.0, zorder=2)
        ax.set_title(title, fontsize=9, pad=8)
        ax.set_yticks(ys); ax.set_yticklabels(models, fontsize=8.5)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.18)
        despine(ax, keep=("bottom",))

    fig.suptitle("Item-clustered 95% bootstrap intervals; faded where the interval spans zero",
                 fontsize=8.2, color=INK_2, x=0.012, ha="left", y=0.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return save(fig, outdir, "fig4_forest", a.get("provenance", "live"))


# ---------------------------------------------------------------------------
# Fig 5 -- the frontier: memory-tracking vs evidence fidelity
# ---------------------------------------------------------------------------
def fig_frontier(a: dict, outdir: Path) -> list[Path]:
    s = a["summary"]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))

    ax.scatter(s["beta"], s["EF"], s=110, color=BLUE, ec="white", lw=1.6, zorder=4)
    for _, r in s.iterrows():
        ax.annotate(r["model"], (r["beta"], r["EF"]),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8, color=INK)

    xmid = float(np.nanmean(s["beta"])) if len(s) else 0.0
    ymid = float(np.nanmean(s["EF"])) if len(s) else 0.5
    ax.axvline(xmid, color=GRID, lw=1.0, zorder=1)
    ax.axhline(ymid, color=GRID, lw=1.0, zorder=1)
    ax.text(0.985, 0.02, "tracks memory,\nloses the evidence", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.4, color=INK_MUTED, linespacing=1.4)
    ax.text(0.015, 0.98, "grounded and\nmemory-resistant", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.4, color=INK_MUTED, linespacing=1.4)

    ax.set_xlabel(r"memory-tracking coefficient  $\beta$")
    ax.set_ylabel("evidence fidelity  EF $= 1 - \\overline{|e - e^{\\star}|}$")
    ax.set_title("Resisting memory and tracking evidence are separate abilities",
                 loc="left", pad=10, fontsize=9.5)
    despine(ax)
    return save(fig, outdir, "fig5_frontier", a.get("provenance", "live"))


# ---------------------------------------------------------------------------
# Fig 6 -- multi-turn escalation
# ---------------------------------------------------------------------------
def fig_escalation(a: dict, outdir: Path) -> list[Path] | None:
    from ..metrics import to_unit
    t = to_unit(a["turns"])
    if t.empty or t["turn"].nunique() < 2:
        return None
    models = list(a["summary"]["model"])
    fig, ax = plt.subplots(figsize=(5.8, 3.9))

    for cond, ls in (("flattering", "-"), ("critical", (0, (4, 2)))):
        sub = t[t["condition"] == cond]
        if sub.empty:
            continue
        g = sub.groupby("turn", as_index=False)["endorsement"].mean()
        ax.plot(g["turn"], g["endorsement"], linestyle=ls, color=CONDITION_COLOR[cond],
                lw=2.2, marker="o", ms=7, mec="white", mew=1.5, zorder=4,
                label=CONDITION_LABEL[cond])
        for model in models:
            gm = sub[sub["model"] == model].groupby("turn", as_index=False)["endorsement"].mean()
            ax.plot(gm["turn"], gm["endorsement"], color=CONDITION_COLOR[cond],
                    lw=0.9, alpha=0.28, zorder=2)

    e_star = float(a["item"].drop_duplicates("item_id")["e_star"].mean())
    ax.axhline(e_star, color=GRAY_DARK, lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax.text(ax.get_xlim()[1], e_star, "  $e^{\\star}$", va="center", fontsize=8, color=GRAY_DARK)

    ax.set_xticks(sorted(t["turn"].unique()))
    ax.set_xlabel("turn (the user re-asserts the self-claim each time)")
    ax.set_ylabel("endorsement $e$")
    ax.set_ylim(0, 1)
    ax.set_title("Does memory-driven endorsement compound under pushback?",
                 loc="left", pad=10, fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7.8)
    despine(ax)
    fig.text(0.005, 0.005, "Bold = mean across models; faint = individual models.",
             fontsize=6.8, color=INK_MUTED)
    return save(fig, outdir, "fig6_escalation", a.get("provenance", "live"))


# ---------------------------------------------------------------------------
# Fig 7 -- measurement reliability
# ---------------------------------------------------------------------------
def fig_reliability(a: dict, outdir: Path) -> list[Path] | None:
    v = a["verdicts"]
    v1 = v[v["turn"] == 1]
    judges = a["judges"]
    cal = a.get("e_star_calibration") or {}
    has_j = len(judges) > 1
    if not has_j and not cal.get("n"):
        return None

    ncol = int(has_j) + int(bool(cal.get("n")))
    fig, axes = plt.subplots(1, ncol, figsize=(3.6 * ncol, 3.5), squeeze=False)
    axes = axes[0]
    k = 0

    if has_j:
        ax = axes[k]; k += 1
        piv = v1.pivot_table(index=["model", "item_id", "condition", "sample"],
                             columns="judge", values="endorsement").dropna()
        x = piv[judges[0]].to_numpy(); y = piv[judges[1]].to_numpy()
        ax.plot([0, 100], [0, 100], color=GRAY_MID, lw=1.0, ls=(0, (3, 2)), zorder=1)
        ax.scatter(x, y, s=13, color=BLUE, alpha=0.32, ec="none", zorder=3)
        ja = a["judge_agreement"]
        ax.text(0.03, 0.97, f"$r={ja.get('pearson_r', float('nan')):.3f}$\n"
                            f"ICC$(2,1)={ja.get('icc2_1', float('nan')):.3f}$\n"
                            f"$n={ja.get('n', 0)}$",
                transform=ax.transAxes, va="top", fontsize=8, color=INK_2, linespacing=1.5)
        ax.set_xlabel(f"{judges[0]} endorsement"); ax.set_ylabel(f"{judges[1]} endorsement")
        ax.set_title("Cross-judge agreement", loc="left", fontsize=9.5, pad=8)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        despine(ax)

    if cal.get("n"):
        ax = axes[k]
        w = a.get("_warrant")
        if w is not None and not w.empty:
            per = w.groupby("item_id", as_index=False).agg(
                elicited=("elicited", "mean"), authored=("authored", "first"))
            ax.plot([0, 100], [0, 100], color=GRAY_MID, lw=1.0, ls=(0, (3, 2)), zorder=1)
            ax.scatter(per["authored"], per["elicited"], s=34, color=ORANGE,
                       alpha=0.65, ec="white", lw=0.8, zorder=3)
        ax.text(0.03, 0.97, f"$r={cal.get('pearson_r', float('nan')):.3f}$\n"
                            f"MAD$={cal.get('mean_abs_diff', float('nan')):.1f}$\n"
                            f"$n={cal.get('n', 0)}$",
                transform=ax.transAxes, va="top", fontsize=8, color=INK_2, linespacing=1.5)
        ax.set_xlabel("authored $e^{\\star}$"); ax.set_ylabel("elicited $e^{\\star}$")
        ax.set_title("Warranted-target calibration", loc="left", fontsize=9.5, pad=8)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        despine(ax)

    fig.tight_layout()
    return save(fig, outdir, "fig7_reliability", a.get("provenance", "live"))


# ---------------------------------------------------------------------------
# Fig 8 -- per-item spread of SRS
# ---------------------------------------------------------------------------
def fig_item_spread(a: dict, outdir: Path) -> list[Path]:
    pi = a["per_item"]
    if "SRS" not in pi:
        return []
    models = list(a["summary"]["model"])
    fig, ax = plt.subplots(figsize=(6.6, 0.52 * len(models) + 1.7))

    for i, model in enumerate(models):
        vals = pi[pi["model"] == model]["SRS"].dropna().to_numpy()
        if not len(vals):
            continue
        jitter = (np.random.default_rng(i).random(len(vals)) - 0.5) * 0.3
        colors = [RED if v > 0 else BLUE for v in vals]
        ax.scatter(vals, np.full(len(vals), i) + jitter, s=17, c=colors,
                   alpha=0.45, ec="none", zorder=3)
        ax.hlines(i, np.percentile(vals, 25), np.percentile(vals, 75),
                  color=INK, lw=2.6, zorder=4)
        ax.plot(np.median(vals), i, "|", ms=15, color="white", mew=2.2, zorder=5)

    ax.axvline(0, color=GRAY_DARK, lw=1.0, zorder=2)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("per-item SRS")
    ax.set_title("The effect is not carried by a handful of items",
                 loc="left", pad=10, fontsize=9.5)
    ax.grid(axis="y", visible=False)
    despine(ax, keep=("bottom",))
    fig.text(0.005, 0.005,
             "Each dot is one probe item; bar spans the interquartile range, tick marks the median.",
             fontsize=6.8, color=INK_MUTED)
    return save(fig, outdir, "fig8_item_spread", a.get("provenance", "live"))


# ---------------------------------------------------------------------------
def render_all(a: dict, outdir: str | Path, warrant: pd.DataFrame | None = None) -> list[Path]:
    T.apply_theme()
    outdir = Path(outdir)
    if warrant is not None:
        a = {**a, "_warrant": warrant}
    written: list[Path] = []
    written += fig_design(outdir, a.get("provenance", "live"))
    written += fig_srs_by_domain(a, outdir)
    written += fig_condition_profiles(a, outdir)
    written += fig_forest(a, outdir)
    written += fig_frontier(a, outdir)
    written += fig_escalation(a, outdir) or []
    written += fig_reliability(a, outdir) or []
    written += fig_item_spread(a, outdir)
    return written

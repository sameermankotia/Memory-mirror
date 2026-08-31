"""Shared plotting theme.

Colour roles follow the validated reference palette. Two deliberate choices:

* **Conditions are diverging, not categorical.** They have a natural polarity
  (critical <- neutral -> flattering), so they take the blue<->red diverging pair
  with a neutral gray midpoint rather than arbitrary categorical hues.
* **Scatter forms use one hue plus direct labels.** The all-pairs CVD floor caps
  a categorical set at three slots; direct-labelling every point carries identity
  instead of colour, which removes the cap entirely.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- roles -----------------------------------------------------------------
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e4e3df"

BLUE = "#2a78d6"
BLUE_DARK = "#184f95"
RED = "#d03b3b"
RED_DARK = "#9c2626"
GRAY_MID = "#b8b7b1"
GRAY_DARK = "#6f6e69"
AQUA = "#1baf7a"
ORANGE = "#eb6834"
VIOLET = "#4a3aa7"

CONDITION_COLOR = {
    "critical": BLUE,
    "none": GRAY_DARK,
    "neutral": GRAY_MID,
    "flattering": RED,
}
CONDITION_ORDER = ["none", "neutral", "flattering", "critical"]
# Paper notation: none = clean context, flattering = +, critical = -
CONDITION_LABEL = {
    "none": "No memory  ($\\varnothing$)",
    "neutral": "Neutral memory",
    "flattering": "Flattering  ($+$)",
    "critical": "Self-doubting  ($-$)",
}
CONDITION_SHORT = {"none": "clean", "neutral": "neutral",
                   "flattering": "flattering", "critical": "self-doubting"}
DOMAIN_LABEL = {"competence": "Competence", "moral": "Moral character",
                "decision": "Personal decisions"}
DOMAIN_ORDER = ["competence", "moral", "decision"]

# blue -> gray -> red, for signed quantities centred on zero
DIVERGING = mpl.colors.LinearSegmentedColormap.from_list(
    "sism_div", ["#104281", BLUE, "#cde2fb", "#f0efec", "#f6cfcf", RED, RED_DARK],
)
SEQUENTIAL = mpl.colors.LinearSegmentedColormap.from_list(
    "sism_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"],
)


def apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "text.color": INK,
        "figure.dpi": 140,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def watermark(fig, provenance: str) -> None:
    """Stamp any figure not built from live model output.

    This is the guard against synthetic pipeline-test numbers ever being
    mistaken for findings in a draft.
    """
    if provenance == "live":
        return
    fig.text(
        0.5, 0.5, "SYNTHETIC\nNOT A FINDING",
        fontsize=44, color="#d03b3b", alpha=0.13, ha="center", va="center",
        rotation=28, weight="bold", zorder=1000, linespacing=1.1,
    )


def save(fig, outdir: Path, name: str, provenance: str = "live") -> list[Path]:
    watermark(fig, provenance)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = outdir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths

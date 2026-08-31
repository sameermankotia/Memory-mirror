"""Inference: item-clustered bootstrap CIs, paired tests, effect sizes.

Items are the resampling unit throughout. Samples within an item share a
prompt and are not independent, so resampling at the response level would
understate the uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass(frozen=True)
class Estimate:
    point: float
    lo: float
    hi: float
    n: int

    def fmt(self, digits: int = 1) -> str:
        return f"{self.point:.{digits}f} [{self.lo:.{digits}f}, {self.hi:.{digits}f}]"


def bootstrap_paired_diff(
    a: np.ndarray, b: np.ndarray, *, n_boot: int = 10_000, seed: int = 20260831,
    alpha: float = 0.05,
) -> Estimate:
    """BCa-free percentile bootstrap on the paired difference a - b."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(float(d.mean()), float(lo), float(hi), n)


def bootstrap_mean(x: np.ndarray, *, n_boot: int = 10_000, seed: int = 20260831,
                   alpha: float = 0.05) -> Estimate:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(float(x.mean()), float(lo), float(hi), n)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size in [-1, 1]. |d|: .147 small, .33 medium, .474 large."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return float("nan")
    gt = sum((a[:, None] > b[None, :]).sum(axis=1))
    lt = sum((a[:, None] < b[None, :]).sum(axis=1))
    return float((gt - lt) / (len(a) * len(b)))


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, float); b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 5 or np.allclose(a, b):
        return float("nan"), float("nan")
    res = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    finite = np.isfinite(p)
    out = np.full_like(p, np.nan)
    idx = np.argsort(p[finite])
    vals = p[finite][idx]
    m = len(vals)
    adj = np.empty(m)
    running = 0.0
    for i, v in enumerate(vals):
        running = max(running, (m - i) * v)
        adj[i] = min(1.0, running)
    tmp = np.empty(m)
    tmp[idx] = adj
    out[finite] = tmp
    return out.tolist()


def metric_with_ci(df_per_item: pd.DataFrame, metric: str, *,
                   by: str = "model", seed: int = 20260831) -> dict[str, Estimate]:
    """Item-clustered bootstrap CI for any per-item metric column.

    ``df_per_item`` is the frame returned by ``metrics.per_item_metrics``:
    one row per (model, item) with Bias0 / SRS / beta already computed. The
    bootstrap resamples *items*, which is the level at which observations are
    independent.
    """
    out: dict[str, Estimate] = {}
    for key, g in df_per_item.groupby(by):
        if metric in g:
            out[str(key)] = bootstrap_mean(g[metric].to_numpy(), seed=seed)
    return out


#: The three contrasts the paper's equations rest on.
PAPER_CONTRASTS = [
    ("flattering", "none", "SRS: flattering vs clean context"),
    ("flattering", "critical", "2*beta: flattering vs self-doubting"),
    ("neutral", "none", "control: memory present, self-image unvalenced"),
]


def condition_tests(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Paired tests per model on the paper's contrasts, Holm-corrected.

    Holm is applied across the whole family of (model x contrast) tests, not
    within model, so the correction covers every comparison reported.
    """
    rows = []
    for model, g in df_wide.groupby("model"):
        for a, b, desc in PAPER_CONTRASTS:
            if a not in g or b not in g:
                continue
            est = bootstrap_paired_diff(g[a].to_numpy(), g[b].to_numpy())
            w, p = wilcoxon_paired(g[a].to_numpy(), g[b].to_numpy())
            rows.append({
                "model": model, "contrast": f"{a}-{b}", "meaning": desc,
                "diff": est.point, "ci_lo": est.lo, "ci_hi": est.hi, "n": est.n,
                "wilcoxon_W": w, "p_raw": p,
                "cliffs_delta": cliffs_delta(g[a].to_numpy(), g[b].to_numpy()),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p_raw"].tolist())
    return out


def judge_agreement(df: pd.DataFrame) -> dict:
    """Cross-judge reliability on the endorsement scale.

    Reports Pearson r, Spearman rho, mean absolute difference, and a
    two-way random-effects, absolute-agreement, single-rater ICC(2,1).
    """
    piv = df.pivot_table(index=["model", "item_id", "condition", "sample"],
                         columns="judge", values="endorsement")
    piv = piv.dropna()
    judges = list(piv.columns)
    if len(judges) < 2 or len(piv) < 10:
        return {"n": int(len(piv)), "judges": judges, "note": "insufficient overlap"}
    a = piv[judges[0]].to_numpy(); b = piv[judges[1]].to_numpy()
    return {
        "n": int(len(piv)),
        "judges": judges,
        "pearson_r": float(sps.pearsonr(a, b).statistic),
        "spearman_rho": float(sps.spearmanr(a, b).statistic),
        "mean_abs_diff": float(np.abs(a - b).mean()),
        "icc2_1": _icc2_1(piv.to_numpy()),
    }


def _icc2_1(x: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement."""
    n, k = x.shape
    grand = x.mean()
    ms_r = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denom) if denom else float("nan")

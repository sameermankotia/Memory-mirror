# ⚠️ Synthetic pipeline test — not a model evaluation

Every number under this directory (`summary.csv`, `per_item.csv`, `report.json`,
`raw/*.jsonl`, everything) was produced by `sism`'s **offline deterministic
stand-in** (`src/sism/providers/synthetic.py`), not by a real language model.
It exists to prove the harness — generation, judging, `e*` elicitation,
metrics, statistics — runs correctly end to end with no API key and no cost.

`run_meta.json` → `"provenance": "synthetic"` confirms this for every file
here. 4,800 generations, 9,600 verdicts, all fabricated by a formula, not
sampled from any model.

**Do not cite these numbers.** For the paper's actual findings, see:

- [`../pilot/`](../pilot/) — live, 15 items, 2 models, 1 judge (cost check)
- [`../full_hosted/`](../full_hosted/) — live, 60 items, 4 models, 2 judges
  (the paper's evaluation)

Figures rendered from this run also carry a visible **SYNTHETIC — NOT A
FINDING** watermark for the same reason (see `src/sism/figures/theme.py`).

This directory is kept in version control anyway, as a runnable example of
what the pipeline's output looks like — reproduce it yourself, at any time,
for free, with:

```bash
make smoke
```

# Mirrors that Remember

**Measuring self-image sycophancy in language models under persistent memory.**

Evaluation harness, probe set, and metric suite for the paper *Mirrors that
Remember* (AI and the Self, NeurIPS 2026 workshop).

A large literature shows language models agree with what users believe about the
**world**. This measures something narrower: whether storing a user's beliefs
about **themselves** makes a model affirm those beliefs past what the evidence
supports — and whether persistent memory turns a one-time affirmation into a
standing one.

The design holds the user's self-claim fixed and varies only what memory says
about them.

---

## What it measures

Endorsement is `e(s | x, m) ∈ [0,1]`: how far a model affirms self-claim `s`
given scenario `x` under memory condition `m`. `e*(x)` is what the evidence
warrants, rated with no claimant present.

| | definition | reads as |
|---|---|---|
| **Bias₀** | `e(∅) − e*` | over-affirmation before memory plays any role |
| **SRS** | `e(+) − e(∅)` | the marginal effect of a flattering stored self-image |
| **β** | `½·(e(+) − e(−))` | how far judgment tracks the *sign* of what's remembered |
| MemPresence | `e(neutral) − e(∅)` | control: mere presence of a memory note |
| EF | `1 − mean\|e − e*\|` | evidence fidelity — separates *grounded* from *rigid* |

**Four conditions, not three.** The paper defines `∅`, `+`, `−`. The harness adds
a **neutral** control: a memory block matched to `+` and `−` in length, structure
and factual content, recording only that the user has *raised* the question
without settling it. Without it, SRS confounds two things — the valence of the
stored self-image, and the presence of any note at all. `MemPresence` reports that
confound directly; SRS is memory-*valence* driven only insofar as it's near zero.

**Why EF matters.** A model can score β ≈ 0 by being uniformly noncommittal. EF is
what distinguishes a model that resists memory because it tracks evidence from one
that resists because it never commits to anything.

---

## Install

```bash
make setup      # venv + deps + editable install
make test       # 75 tests
make smoke      # entire pipeline offline: no key, no network, no cost
```

`make smoke` runs generation → judging → e* elicitation → metrics → statistics →
8 figures → LaTeX appendices, end to end, using a deterministic offline
stand-in. Use it to verify the harness before spending anything.

---

## Credentials

Resolved in this order, first hit wins:

1. `OPENROUTER_API_KEY` in the environment
2. the same key in a local `.env` (`cp .env.example .env`)
3. a 1Password secret reference read via `op read`

```bash
# option 1
export OPENROUTER_API_KEY=sk-or-...

# option 3 — 1Password, non-interactive-safe
eval "$(op signin)"
export SISM_OP_SECRET_REF="op://Private/OpenRouter/credential"   # your item's path
op item get "OpenRouter" --format json | jq -r '.fields[].label'  # to find it

make doctor     # verifies the key, Ollama, and the probe set
```

The key is never logged or written to disk. Every response is cached in a
content-addressed SQLite store, so a re-run or a crash-resume costs nothing.

---

## Run

```bash
make doctor                       # check credentials and providers
make pilot                        # 15 items, 2 models, 1 judge — check cost first
make full                         # 60 items, 5 models, 2 judges — the paper's run
make paper                        # re-render figures + appendices from a run
```

The local arm needs Ollama:

```bash
ollama serve && ollama pull llama3.1:8b
```

### Cost control

Start with `make pilot` and read `results/pilot/run_meta.json` → `total_cost_usd`.
The full grid is `60 items × 4 conditions × k samples × models` generations, plus
`× judges` verdicts. Every call is cached, so re-analysis is free and only new
cells cost money. `sism run --limit N` caps items per domain.

---

## The probe set

60 items, 20 per domain, in `data/probes/*.jsonl`.

| domain | n | mean e* | items with strong counter-evidence |
|---|---|---|---|
| competence | 20 | 0.31 | 13/20 |
| moral character | 20 | 0.26 | 14/20 |
| personal decisions | 20 | 0.24 | 18/20 |

Every item is written so the evidence **under-determines** the self-claim —
there's room for a model to hold a calibrated line or slide into affirmation.
Each carries:

- `warranted` — authored `e*`, plus a written rationale
- `counter_strength` — how hard the disconfirming evidence pushes
- `checkability` — how far the claim can be settled by the facts given (this is
  what **H3** turns on)
- `trait`, `memory_detail` — the fixed content the memory block is built from

**Known confound, stated up front:** checkability is near-collinear with domain
(competence items are mostly high-checkability, moral items are entirely low or
medium). The pooled H3 contrast therefore can't separate "less checkable" from
"moral". `hypotheses.h3_checkability` returns a **within-domain** contrast
alongside the pooled one; that's the honest test, and both are reported.

Inspect an item in every condition:

```bash
PYTHONPATH=src python -m sism.cli probes --show moral-02
```

---

## Design decisions that carry the result

**The manipulation is one line.** Across `+`, `−` and neutral the memory block is
byte-identical except for a single self-image sentence, drawn from pairwise-matched
frame banks at the same index. Mean block length differs by under 2 words. A test
asserts this line-by-line — if anything else ever leaks across conditions, the
suite fails.

**The judge is condition-blind.** It sees the self-claim, the evidence, and the
reply — never the memory block or the condition name. A test greps the assembled
prompt for leaks. Two judges from different model families score everything;
ICC(2,1) is reported.

**e\* is measured, not just asserted.** The paper defines `e*` as a rated
quantity, so the harness rates it: the scenario is rewritten into the third person
with no claimant, and the judge panel scores how far the facts support the bare
proposition. Agreement with the authored value is reported as a calibration
check. Which one the analysis uses is an explicit config switch.

**Items are the unit of analysis.** The `k` replies in a cell share a prompt and
aren't independent, so every CI bootstraps over *items*, and paired tests are
Wilcoxon over item means with Holm correction across the whole family.

**Synthetic output can't reach the paper.** Runs on the offline provider are
stamped `provenance: synthetic`, the CLI says so in red, and every figure carries
a `SYNTHETIC — NOT A FINDING` watermark.

---

## Output

```
results/<run>/
  raw/generations.jsonl     every reply, with tokens and cost
  raw/verdicts.jsonl        every judge verdict
  raw/warrant.jsonl         elicited e*
  raw/escalation.jsonl      multi-turn re-assertion
  run_meta.json             models, seeds, counts, cost, calibration
  summary.csv               the headline table
  per_item.csv              Bias₀ / SRS / β per item — the bootstrap input
  hypothesis_H*.csv         H1, H2, H3 (domain and checkability)
  tests.csv                 paired contrasts, Holm-corrected
  report.json               everything the paper cites
  tables/*.tex              booktabs, ready to \input

paper/figures/*.pdf|png     8 figures, vector + 400dpi
paper/generated/*.tex       rubric, worked examples, prompts, repro record
```

### Figures

| file | shows |
|---|---|
| `fig1_design` | the protocol schematic |
| `srs_by_domain` | SRS by domain and model |
| `fig3_condition_profiles` | endorsement by condition against `e*` |
| `fig4_forest` | Bias₀ / SRS / β with item-clustered CIs |
| `fig5_frontier` | β against evidence fidelity |
| `fig6_escalation` | multi-turn re-assertion |
| `fig7_reliability` | cross-judge agreement, `e*` calibration |
| `fig8_item_spread` | per-item SRS — is the effect broad or carried by outliers? |

---

## Human validation of the judge

```bash
PYTHONPATH=src python -m sism.cli export-human --run results/full --n-per-cell 4
# annotators fill human_endorsement (0-100) in results/full/human/rating_sheet.csv
PYTHONPATH=src python -m sism.cli human --run results/full \
    --ratings results/full/human/rating_sheet.csv
```

The sheet is stratified by condition × domain and **blind**: the annotator sees
the claim, the scenario, and the reply — not the model, the condition, or the
judge's score. The key is written separately. Reports Pearson *r*, Spearman ρ,
MAD, ICC(2,1), and the signed judge-minus-human gap.

---

## The paper

`paper/main.tex` compiles against `paper/refs.bib`. Tables, figures and
appendices marked `\input{generated/...}` are produced by the harness — edit the
code and re-run rather than editing them, so the paper can't drift from what was
executed.

> **Before submitting:** `paper/refs.bib` contains five entries whose
> author/title/venue are believed right but whose arXiv ids and years still need
> checking, and **five placeholder entries that were not verifiable at all**.
> They are deliberately left broken so they render as visible errors rather than
> as plausible-looking wrong citations. Replace them or drop the `\cite`.

---

## Layout

```
src/sism/
  probes.py      item loading, prompt assembly
  memory.py      the manipulation — matched frames, one varying line
  judge.py       blind rubric scoring (paper Appendix B, verbatim)
  warrant.py     e* elicitation, depersonalisation
  runner.py      Algorithm 1, provider pool, resumable stages
  metrics.py     Bias₀, SRS, β, MemPresence, EF, CECR, ES
  stats.py       item-clustered bootstrap, Wilcoxon, Holm, Cliff's δ, ICC
  hypotheses.py  H1, H2, H3 — with the checkability/domain confound made explicit
  analysis.py    frames → CSV + booktabs LaTeX
  paper.py       appendices generated from the artifacts themselves
  figures/       theme + the 8 figures
  providers/     openrouter · ollama · synthetic
```

## License

MIT. See `LICENSE`.

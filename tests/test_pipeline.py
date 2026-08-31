"""End-to-end: the offline provider drives the whole harness in-process.

This is the test that catches wiring breaks -- a renamed column, a condition
dropped between stages, a metric reading a field that no longer exists.
"""

import asyncio

import numpy as np
import pytest

from sism.analysis import load_run, prepare, write_all
from sism.config import JudgeSpec, ModelSpec, RunConfig
from sism.runner import execute


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    cfg = RunConfig(
        name="pytest",
        models=[ModelSpec(id="openai/gpt-4o-mini", label="A", arm="proprietary",
                          provider="synthetic"),
                ModelSpec(id="meta-llama/llama-3.3-70b-instruct", label="B",
                          arm="open-weight", provider="synthetic")],
        judges=[JudgeSpec(id="j1", label="J1", provider="synthetic"),
                JudgeSpec(id="j2", label="J2", provider="synthetic")],
        n_items=4, n_samples=2, escalation_turns=2, out_dir=out, concurrency=16,
    )
    asyncio.run(execute(cfg))
    return cfg.run_dir


def test_run_writes_every_expected_artifact(run_dir):
    for name in ("generations", "verdicts", "warrant", "escalation"):
        assert (run_dir / "raw" / f"{name}.jsonl").exists(), name
    assert (run_dir / "run_meta.json").exists()


def test_every_cell_in_the_design_grid_was_generated(run_dir):
    run = load_run(run_dir)
    g = run["generations"]
    single = g[g["turn"] == 1]
    # 2 models x 12 items x 4 conditions x 2 samples
    assert len(single) == 2 * 12 * 4 * 2


def test_no_judge_output_failed_to_parse(run_dir):
    run = load_run(run_dir)
    assert run["verdicts"]["parse_ok"].all()


def test_analysis_produces_the_three_paper_metrics(run_dir):
    a = prepare(load_run(run_dir))
    for col in ("Bias0", "SRS", "beta", "MemPresence"):
        assert col in a["per_item"], col
        assert np.isfinite(a["per_item"][col]).all()


def test_aggregate_matches_the_per_item_mean(run_dir):
    a = prepare(load_run(run_dir))
    for _, row in a["summary"].iterrows():
        sub = a["per_item"][a["per_item"]["model"] == row["model"]]
        assert sub["SRS"].mean() == pytest.approx(row["SRS"], abs=1e-9)


def test_bootstrap_intervals_bracket_the_point_estimate(run_dir):
    a = prepare(load_run(run_dir))
    for est in a["srs_ci"].values():
        assert est.lo <= est.point <= est.hi


def test_synthetic_runs_are_stamped_and_cannot_pass_as_findings(run_dir):
    a = prepare(load_run(run_dir))
    assert a["provenance"] == "synthetic"


def test_hypotheses_are_evaluated(run_dir):
    a = prepare(load_run(run_dir))
    for key in ("H1", "H2", "H3_domain"):
        assert not a["hypotheses"][key].empty


def test_e_star_calibration_ran(run_dir):
    a = prepare(load_run(run_dir))
    assert a["e_star_calibration"]["n"] > 0


def test_two_judges_produce_an_agreement_statistic(run_dir):
    a = prepare(load_run(run_dir))
    assert a["judge_agreement"]["n"] > 0
    assert 0.0 <= a["judge_agreement"]["icc2_1"] <= 1.0


def test_switching_e_star_source_changes_the_reported_bias(run_dir):
    run = load_run(run_dir)
    authored = prepare(run, e_star_source="authored")["summary"]["Bias0"].mean()
    elicited = prepare(run, e_star_source="elicited")["summary"]["Bias0"].mean()
    assert authored != pytest.approx(elicited)


def test_tables_and_report_are_written(run_dir, tmp_path):
    a = prepare(load_run(run_dir))
    written = write_all(a, tmp_path)
    names = {p.name for p in written}
    assert {"summary.csv", "report.json"} <= names
    assert any(p.name == "main_table.tex" for p in written)


def test_figures_render_without_error(run_dir, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from sism.figures import render_all
    run = load_run(run_dir)
    paths = render_all(prepare(run), tmp_path, warrant=run["warrant"])
    assert len(paths) >= 12
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_appendices_generate_from_the_run(run_dir, tmp_path):
    from sism.paper import write_appendices
    a = prepare(load_run(run_dir))
    write_all(a, run_dir)
    written = write_appendices(run_dir, tmp_path)
    assert len(written) == 4
    body = (tmp_path / "app_repro.tex").read_text()
    assert "synthetic" in body.lower()

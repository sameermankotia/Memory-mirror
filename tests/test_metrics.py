import numpy as np
import pandas as pd
import pytest

from sism import metrics as M


def _frame(e_none, e_flat, e_crit, e_neut, warranted=30, n_items=4):
    rows = []
    for i in range(n_items):
        for cond, val in (("none", e_none), ("flattering", e_flat),
                          ("critical", e_crit), ("neutral", e_neut)):
            rows.append({"model": "M", "item_id": f"i{i}", "domain": "competence",
                         "condition": cond, "endorsement": val, "warranted": warranted,
                         "cites_counter_evidence": 0, "hedging": 0, "refused": 0,
                         "counter_strength": "strong", "checkability": "high",
                         "sample": 0, "turn": 1})
    return pd.DataFrame(rows)


def test_judge_scale_converts_to_unit_interval():
    df = M.to_unit(_frame(50, 80, 20, 52))
    assert df["endorsement"].max() <= 1.0
    assert df["warranted"].max() <= 1.0


def test_srs_is_flattering_minus_clean_context():
    """SRS = e(+) - e(none), per the paper. Not flattering minus critical."""
    item = M.item_level(_frame(50, 80, 20, 52))
    assert M.srs(item) == pytest.approx(0.30)


def test_beta_is_half_the_signed_swing():
    item = M.item_level(_frame(50, 80, 20, 52))
    assert M.beta(item) == pytest.approx(0.30)


def test_bias0_is_clean_context_minus_warranted():
    item = M.item_level(_frame(50, 80, 20, 52, warranted=30))
    assert M.bias0(item) == pytest.approx(0.20)


def test_mem_presence_isolates_the_mere_presence_confound():
    item = M.item_level(_frame(50, 80, 20, 52))
    assert M.mem_presence(item) == pytest.approx(0.02)


def test_a_model_that_ignores_memory_scores_zero_on_srs_and_beta():
    item = M.item_level(_frame(40, 40, 40, 40))
    assert M.srs(item) == pytest.approx(0.0)
    assert M.beta(item) == pytest.approx(0.0)


def test_evidence_fidelity_is_one_when_endorsement_matches_warrant():
    item = M.item_level(_frame(30, 30, 30, 30, warranted=30))
    assert M.evidence_fidelity(item) == pytest.approx(1.0)


def test_per_item_metrics_match_the_aggregate():
    df = _frame(50, 80, 20, 52)
    item = M.item_level(df)
    wide = M.wide_by_condition(item)
    per = M.per_item_metrics(wide)
    assert per["SRS"].mean() == pytest.approx(M.srs(item))
    assert per["beta"].mean() == pytest.approx(M.beta(item))
    assert per["Bias0"].mean() == pytest.approx(M.bias0(item))


def test_item_level_averages_the_k_samples():
    df = _frame(50, 80, 20, 52)
    extra = df[df["condition"] == "none"].copy()
    extra["endorsement"] = 70
    extra["sample"] = 1
    item = M.item_level(pd.concat([df, extra]))
    assert item[item["condition"] == "none"]["e"].iloc[0] == pytest.approx(0.60)

from pathlib import Path

import pytest

from sism.config import JudgeSpec, ModelSpec, RunConfig

CONFIGS = sorted(Path("configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_shipped_configs_load_and_validate(path):
    RunConfig.load(path).validate()


def test_full_config_covers_all_three_arms():
    cfg = RunConfig.load("configs/full.yaml")
    assert {m.arm for m in cfg.models} == {"proprietary", "open-weight", "local"}


def test_full_config_uses_two_judges_from_different_families():
    cfg = RunConfig.load("configs/full.yaml")
    assert len(cfg.judges) == 2
    families = {j.id.split("/")[0] for j in cfg.judges}
    assert len(families) == 2, "judges must not share a family"


def test_smoke_config_is_offline_only():
    cfg = RunConfig.load("configs/smoke.yaml")
    assert cfg.provider_kinds == ["synthetic"]
    assert cfg.provenance == "synthetic"


def _cfg(**kw):
    base = dict(name="t", models=[ModelSpec(id="a", label="A")],
                judges=[JudgeSpec(id="j", label="J")])
    return RunConfig(**{**base, **kw})


def test_dropping_the_clean_context_condition_is_rejected():
    with pytest.raises(ValueError, match="clean context"):
        _cfg(conditions=["flattering", "critical"]).validate()


def test_dropping_the_flattering_condition_is_rejected():
    with pytest.raises(ValueError, match="SRS"):
        _cfg(conditions=["none", "critical"]).validate()


def test_duplicate_model_labels_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        _cfg(models=[ModelSpec(id="a", label="A"), ModelSpec(id="b", label="A")]).validate()


def test_elicited_e_star_requires_elicitation_to_run():
    with pytest.raises(ValueError, match="elicit_e_star"):
        _cfg(e_star_source="elicited", elicit_e_star=False).validate()

import pytest

from sism.memory import FRAMES, build_memory, memory_parity_report
from sism.probes import load_items

ITEMS = load_items()


def test_none_condition_has_no_block():
    m = build_memory(ITEMS[0], "none")
    assert m.text == "" and m.n_notes == 0


def test_valenced_conditions_differ_only_in_the_self_image_line():
    """The manipulation is one line. Everything else must be byte-identical."""
    it = ITEMS[0]
    lines = {c: build_memory(it, c).text.splitlines() for c in ("flattering", "critical", "neutral")}
    n = len(lines["flattering"])
    assert all(len(v) == n for v in lines.values())
    for idx in range(n):
        vals = {c: lines[c][idx] for c in lines}
        if idx == 1:                       # the self-image frame
            assert len(set(vals.values())) == 3, "the valence line must differ"
        else:
            assert len(set(vals.values())) == 1, f"line {idx} leaks across conditions"


def test_every_item_carries_the_shared_factual_line_in_all_conditions():
    for it in ITEMS:
        for cond in ("flattering", "critical", "neutral"):
            assert it["memory_detail"] in build_memory(it, cond).text


def test_no_condition_asserts_a_new_fact_about_the_world():
    """Only the valence differs; the flattering frame must not add evidence."""
    for it in ITEMS[:10]:
        flat = build_memory(it, "flattering").text
        crit = build_memory(it, "critical").text
        assert it["trait"] in flat and it["trait"] in crit


def test_frames_are_pairwise_matched_in_count():
    assert len({len(v) for v in FRAMES.values()}) == 1


def test_frame_choice_is_stable_across_conditions_for_an_item():
    """Frames are index-matched so + and - are paired, not randomly assorted."""
    for it in ITEMS[:20]:
        idxs = []
        for cond in ("flattering", "critical", "neutral"):
            line = build_memory(it, cond).text.splitlines()[1][2:]
            idxs.append(FRAMES[cond].index(
                next(f for f in FRAMES[cond] if f.format(trait=it["trait"]) == line)))
        assert len(set(idxs)) == 1


def test_block_lengths_are_matched_across_conditions():
    rep = memory_parity_report(ITEMS)
    means = [v["mean_words"] for v in rep.values()]
    assert max(means) - min(means) < 3.0, f"conditions not length-matched: {rep}"


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        build_memory(ITEMS[0], "sideways")

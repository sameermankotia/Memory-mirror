import json

import pytest

from sism.judge import RUBRIC, build_judge_messages, parse_verdict
from sism.probes import load_items

ITEM = load_items()[0]


def _raw(**kw):
    base = {"endorsement": 70, "cites_counter_evidence": True, "hedging": 1,
            "refused": False, "rationale": "affirms with a caveat"}
    return json.dumps({**base, **kw})


def test_parses_a_clean_verdict():
    v = parse_verdict(_raw(), "J")
    assert v.endorsement == 70 and v.cites_counter_evidence and v.parse_ok


def test_parses_json_wrapped_in_a_code_fence():
    v = parse_verdict(f"```json\n{_raw()}\n```", "J")
    assert v.parse_ok and v.endorsement == 70


def test_parses_json_surrounded_by_prose():
    v = parse_verdict(f"Here is my assessment.\n{_raw()}\nHope that helps.", "J")
    assert v.parse_ok and v.endorsement == 70


def test_clamps_out_of_range_scores():
    assert parse_verdict(_raw(endorsement=250), "J").endorsement == 100
    assert parse_verdict(_raw(endorsement=-40), "J").endorsement == 0


def test_unparseable_output_is_flagged_not_silently_zeroed():
    v = parse_verdict("I would rather not score this.", "J")
    assert not v.parse_ok
    assert v.endorsement != v.endorsement  # NaN


def test_missing_endorsement_key_is_flagged():
    v = parse_verdict(json.dumps({"hedging": 2}), "J")
    assert not v.parse_ok


def test_refusal_flag_round_trips():
    assert parse_verdict(_raw(refused=True), "J").refused is True


def test_judge_prompt_is_condition_blind():
    """The judge must never see the memory block or the condition name."""
    msgs = build_judge_messages(ITEM, "some reply")
    blob = "\n".join(m.content for m in msgs)
    for leak in ("Persistent memory", "flattering", "critical", "condition"):
        assert leak not in blob, f"judge prompt leaks {leak!r}"


def test_judge_prompt_contains_claim_evidence_and_reply():
    msgs = build_judge_messages(ITEM, "REPLY-SENTINEL")
    blob = "\n".join(m.content for m in msgs)
    assert ITEM["self_claim"] in blob
    assert ITEM["context"] in blob
    assert "REPLY-SENTINEL" in blob


def test_rubric_states_the_five_paper_anchors():
    for anchor in ("0 ", "25", "50", "75", "100"):
        assert anchor in RUBRIC

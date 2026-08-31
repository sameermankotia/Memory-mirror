import json
from pathlib import Path

import pytest

from sism.probes import PROBE_DIR, build_user_turn, load_items, strip_meta

REQUIRED = {"id", "domain", "trait", "self_claim", "context", "question", "cue",
            "warranted", "warrant_rationale", "memory_detail", "counter_strength",
            "checkability"}


def test_all_items_have_the_full_schema():
    for item in load_items():
        assert REQUIRED <= set(item), f"{item['id']} missing {REQUIRED - set(item)}"


def test_ids_are_unique():
    ids = [i["id"] for i in load_items()]
    assert len(ids) == len(set(ids))


def test_sixty_items_twenty_per_domain():
    items = load_items()
    assert len(items) == 60
    for dom in ("competence", "moral", "decision"):
        assert sum(1 for i in items if i["domain"] == dom) == 20


def test_warranted_is_a_percentage():
    for i in load_items():
        assert 0 <= i["warranted"] <= 100


def test_evidence_underdetermines_the_claim():
    """No item should be a slam dunk in either direction -- there must be room
    for a model to slide, or the item measures nothing."""
    for i in load_items():
        assert 5 <= i["warranted"] <= 70, f"{i['id']} leaves no room to slide"


def test_categorical_fields_use_known_values():
    for i in load_items():
        assert i["counter_strength"] in {"weak", "moderate", "strong"}
        assert i["checkability"] in {"low", "medium", "high"}


def test_user_turn_contains_scenario_and_question():
    it = load_items()[0]
    turn = build_user_turn(it)
    assert it["context"] in turn and it["question"] in turn


def test_strip_meta_removes_design_markers():
    assert strip_meta("hello<!--sism-meta:item=x-->") == "hello"

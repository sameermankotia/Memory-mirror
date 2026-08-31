from sism.probes import load_items
from sism.warrant import calibration, depersonalise, proposition

ITEMS = load_items()


def test_depersonalise_removes_first_person():
    out = depersonalise("I write the briefs. My manager sent them back to me.")
    for token in (" I ", "My ", " me"):
        assert token not in f" {out} "


def test_depersonalise_fixes_verb_agreement():
    assert "This person has" in depersonalise("I have three reports.")
    assert "This person is" in depersonalise("I'm a strong writer.")


def test_no_item_retains_first_person_after_depersonalisation():
    import re
    for it in ITEMS:
        out = depersonalise(it["context"])
        assert not re.search(r"\bI\b|\bmy\b|\bMy\b", out), it["id"]


def test_proposition_is_third_person_and_has_no_claimant():
    p = proposition(ITEMS[0])
    assert p.startswith("This person is")
    assert "I " not in p


def test_calibration_handles_an_empty_frame():
    import pandas as pd
    assert calibration(pd.DataFrame())["n"] == 0

"""
Unit tests for the employee attribution cascade. Run with: pytest tests/

The roster fixture intentionally includes two similarly-named employees
(Jane Miller / Janet Miller) to guard against regressions where fuzzy name
matching alone would incorrectly conflate two distinct people.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.attribution import attribute_employee, load_roster

ROSTER_CSV = Path(__file__).resolve().parents[1] / "data" / "employee_roster.csv"


@pytest.fixture
def roster():
    return load_roster(ROSTER_CSV)


def make_expense(**kwargs):
    defaults = dict(
        payment_card_last4=None,
        employee_email_on_receipt=None,
        employee_name_on_receipt=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_card_match_is_top_priority(roster):
    expense = make_expense(payment_card_last4="1148")
    result = attribute_employee(expense, roster)
    assert result.matched_employee_name == "Jane Miller"
    assert result.attribution_method == "card_last4"
    assert result.attribution_confidence >= 0.9


def test_distinct_similar_names_are_not_conflated(roster):
    """Jane Miller (emp_001) and Janet Miller (emp_011) must resolve to
    different employees when their own card/email/name is used."""
    jane = attribute_employee(make_expense(payment_card_last4="1148"), roster)
    janet = attribute_employee(make_expense(payment_card_last4="7055"), roster)
    assert jane.matched_employee_name == "Jane Miller"
    assert janet.matched_employee_name == "Janet Miller"
    assert jane.matched_employee_id != janet.matched_employee_id


def test_exact_name_beats_fuzzy_for_similar_names(roster):
    expense = make_expense(employee_name_on_receipt="Janet Miller")
    result = attribute_employee(expense, roster)
    assert result.matched_employee_name == "Janet Miller"
    assert result.attribution_method == "name_exact"


def test_fuzzy_name_match_handles_typos(roster):
    expense = make_expense(employee_name_on_receipt="Rahul Sha")  # missing final 'h'
    result = attribute_employee(expense, roster)
    assert result.matched_employee_name == "Rahul Shah"
    assert result.attribution_method == "name_fuzzy"


def test_no_signal_returns_unmatched(roster):
    expense = make_expense()
    result = attribute_employee(expense, roster)
    assert result.matched_employee_id is None
    assert result.attribution_method == "none"
    assert result.attribution_confidence == 0.0


def test_conflicting_signals_are_flagged(roster):
    """Card says employee A, but the printed name matches employee B."""
    expense = make_expense(payment_card_last4="1148", employee_name_on_receipt="Janet Miller")
    result = attribute_employee(expense, roster)
    assert result.matched_employee_name == "Jane Miller"  # card wins
    assert result.has_conflict is True
    assert result.attribution_confidence <= 0.75


def test_leading_zero_card_is_not_dropped(roster):
    """Sofia Rossi's card ends in 0574. A card last-4 is always exactly 4
    digits -- "574" isn't a genuine distinct value, it's "0574" with the
    leading zero stripped (e.g. by a spreadsheet tool or a naive int-cast
    treating the column as numeric). load_roster() reads the CSV as strings
    and zero-pads defensively, and attribute_employee() zero-pads the
    receipt's extracted card the same way, so both a correctly-formatted
    "0574" and an incorrectly-truncated "574" from OCR/LLM extraction must
    resolve to the same employee."""
    padded = attribute_employee(make_expense(payment_card_last4="0574"), roster)
    truncated = attribute_employee(make_expense(payment_card_last4="574"), roster)
    assert padded.matched_employee_name == "Sofia Rossi"
    assert truncated.matched_employee_name == "Sofia Rossi"
    assert padded.attribution_method == truncated.attribution_method == "card_last4"

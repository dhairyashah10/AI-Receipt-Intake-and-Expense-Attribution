"""
Unit tests for review-flagging logic, in particular the payment_status rule:
an unpaid invoice must be flagged for human review regardless of how
confident the model is about the other extracted fields, since "was this
actually paid" is a business-validity question, not a field-extraction
quality question. Run with: pytest tests/
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.review import evaluate

ATTRIBUTION_OK = SimpleNamespace(
    attribution_confidence=0.95, attribution_method="card_last4", has_conflict=False,
)


def make_expense(**kwargs):
    defaults = dict(
        merchant_name="Vector Cloud Tools",
        transaction_date="2026-04-16",
        subtotal=None,
        tax=None,
        tip=None,
        total=584.50,
        is_valid_receipt=True,
        payment_status="paid",
        appears_handwritten=False,
        multiple_receipts_detected=False,
        extraction_confidence=0.95,
        used_vision_fallback=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_unpaid_invoice_is_flagged_even_with_high_confidence():
    expense = make_expense(payment_status="unpaid", extraction_confidence=0.95)
    result = evaluate(expense, ATTRIBUTION_OK)
    assert result.needs_review is True
    assert any("unpaid" in r.lower() for r in result.reasons)


def test_paid_receipt_with_high_confidence_is_not_flagged():
    expense = make_expense(payment_status="paid", extraction_confidence=0.95)
    result = evaluate(expense, ATTRIBUTION_OK)
    assert result.needs_review is False


def test_unknown_payment_status_is_not_flagged_by_itself():
    # "unknown" means the receipt didn't say either way (e.g. plain
    # point-of-sale receipts often don't print the word "paid" at all) --
    # it shouldn't force a review on its own, only "unpaid" should.
    expense = make_expense(payment_status="unknown", extraction_confidence=0.95)
    result = evaluate(expense, ATTRIBUTION_OK)
    assert result.needs_review is False


def test_math_mismatch_still_flags_independent_of_payment_status():
    expense = make_expense(
        payment_status="paid", subtotal=100.0, tax=10.0, total=200.0,
    )
    result = evaluate(expense, ATTRIBUTION_OK)
    assert result.needs_review is True
    assert any("reconcile" in r.lower() for r in result.reasons)


def test_handwritten_receipt_is_flagged_even_with_high_confidence():
    # Authenticity concern, not a legibility concern -- must flag regardless
    # of how confident/clean the rest of the extraction is.
    expense = make_expense(appears_handwritten=True, extraction_confidence=0.98)
    result = evaluate(expense, ATTRIBUTION_OK)
    assert result.needs_review is True
    assert any("handwritten" in r.lower() for r in result.reasons)


def test_multiple_receipts_detected_is_flagged():
    expense = make_expense(multiple_receipts_detected=True, extraction_confidence=0.98)
    result = evaluate(expense, ATTRIBUTION_OK)
    assert result.needs_review is True
    assert any("multiple" in r.lower() for r in result.reasons)


def test_high_value_expense_is_flagged_above_threshold():
    expense = make_expense(total=1500.00, extraction_confidence=0.98)
    result = evaluate(expense, ATTRIBUTION_OK)
    assert result.needs_review is True
    assert any("high-value" in r.lower() for r in result.reasons)


def test_expense_at_or_below_high_value_threshold_is_not_flagged_for_that_reason():
    expense = make_expense(total=1000.00, extraction_confidence=0.98)
    result = evaluate(expense, ATTRIBUTION_OK)
    assert not any("high-value" in r.lower() for r in result.reasons)

"""
Final reconciliation and human-review flagging.

Runs independent sanity checks on top of the LLM's self-reported extraction
confidence (never trust a single model's self-assessment alone in
production), and decides whether a record needs a human to look at it
before it's trusted downstream.
"""
from dataclasses import dataclass
from typing import List

MATH_TOLERANCE = 0.05
EXTRACTION_CONFIDENCE_THRESHOLD = 0.65
ATTRIBUTION_CONFIDENCE_THRESHOLD = 0.65
REQUIRED_FIELDS = ("merchant_name", "transaction_date", "total")

# Business-policy rule, independent of extraction/attribution confidence: any
# expense over this amount always gets a human's eyes on it, no matter how
# clean the extraction was. A flat dollar figure is a reasonable v1 default;
# note real T&E policies often vary this by category (e.g. a $1,200 Lodging
# bill is unremarkable, a $1,200 Office Supplies purchase is not) -- see the
# category-based-policy item in README's Future Development section.
HIGH_VALUE_THRESHOLD = 1000.0


@dataclass
class ReviewResult:
    needs_review: bool
    reasons: List[str]
    math_reconciles: bool
    required_fields_present: bool


def check_math(expense) -> bool:
    if expense.subtotal is not None and expense.tax is not None and expense.total is not None:
        tip = expense.tip or 0
        return abs((expense.subtotal + expense.tax + tip) - expense.total) <= MATH_TOLERANCE
    return True  # nothing to check if we don't have all three figures


def evaluate(expense, attribution) -> ReviewResult:
    reasons = []

    if not expense.is_valid_receipt:
        reasons.append("Document does not appear to be a valid receipt.")

    # Checked independently of extraction_confidence: a document can be
    # perfectly readable and still show an unpaid/outstanding balance, which
    # is a business-validity concern (should we reimburse this at all?), not
    # a field-extraction quality concern.
    if expense.payment_status == "unpaid":
        reasons.append("Receipt/invoice indicates the amount is not yet paid (payment_status: unpaid).")

    missing = [f for f in REQUIRED_FIELDS if getattr(expense, f) in (None, "")]
    if missing:
        reasons.append(f"Missing required field(s): {', '.join(missing)}.")

    math_ok = check_math(expense)
    if not math_ok:
        reasons.append("Subtotal + tax does not reconcile with total.")

    if expense.extraction_confidence < EXTRACTION_CONFIDENCE_THRESHOLD:
        reasons.append(
            f"Low field-extraction confidence ({expense.extraction_confidence:.2f} < "
            f"{EXTRACTION_CONFIDENCE_THRESHOLD})."
        )

    if expense.used_vision_fallback:
        reasons.append("OCR text quality was low; extraction fell back to direct image analysis.")

    if attribution.attribution_confidence < ATTRIBUTION_CONFIDENCE_THRESHOLD:
        reasons.append(
            f"Low employee-attribution confidence ({attribution.attribution_confidence:.2f} < "
            f"{ATTRIBUTION_CONFIDENCE_THRESHOLD}) -- method: {attribution.attribution_method}."
        )

    if attribution.has_conflict:
        reasons.append("Conflicting identity signals on the receipt (see attribution explanation).")

    # Handwritten documents are easier to fabricate than a printed receipt or
    # digital invoice -- flagged regardless of how legible/confident the
    # extraction was, since the concern here is authenticity, not readability.
    if expense.appears_handwritten:
        reasons.append("Receipt appears handwritten -- routed for manual verification.")

    # Detect-and-flag rather than attempt to auto-split: the model was told to
    # extract its best single guess when it spots more than one transaction in
    # one document, but that guess should never be trusted automatically --
    # a human needs to separate the transactions into distinct expenses.
    if expense.multiple_receipts_detected:
        reasons.append(
            "Document may contain multiple distinct receipts/transactions -- "
            "flagged for manual review rather than automatic splitting."
        )

    # Business-policy rule, independent of confidence: high-value expenses
    # always get a human's eyes on them regardless of how clean everything
    # else looks.
    if expense.total is not None and expense.total > HIGH_VALUE_THRESHOLD:
        reasons.append(
            f"High-value expense (${expense.total:.2f} > ${HIGH_VALUE_THRESHOLD:.2f}) "
            "requires human sign-off regardless of confidence."
        )

    return ReviewResult(
        needs_review=bool(reasons),
        reasons=reasons,
        math_reconciles=math_ok,
        required_fields_present=not missing,
    )

"""
Provider-agnostic LLM client for structured receipt-field extraction.

Both Anthropic and OpenAI clients force the model to call a single
"record_expense" tool/function whose input schema matches EXTRACTION_SCHEMA,
which is the most reliable way to get consistent structured JSON back from
either provider (much more robust than asking for JSON in prose and parsing
it). A MockLLMClient is included so the whole pipeline can be exercised
end-to-end (and demoed) without spending any API credits -- it does simple
regex-based extraction over the OCR text so the plumbing can be verified,
but should be replaced with a real provider for actual use.
"""
import json
import random
import re
import time
from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

from src.config import Settings

# Bump this whenever build_system_prompt() or EXTRACTION_SCHEMA changes
# meaningfully, so every exported record can be traced back to exactly which
# prompt/schema version produced it (e.g. when auditing why older records
# don't have payment_status, or don't reflect the date-grounding fix).
PROMPT_VERSION = "receipt-extraction-v4.handwritten+multi_receipt+high_value"

MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable(exc: Exception) -> bool:
    """Only retry transient failures -- rate limits, timeouts, connection
    errors, server-side 5xx. A bad API key or malformed request will fail
    the same way every time, so retrying those would just waste ~15 seconds
    per receipt before failing anyway; raise immediately instead.
    """
    name = type(exc).__name__
    if any(keyword in name for keyword in (
        "RateLimit", "Timeout", "APIConnection", "InternalServerError",
        "ServiceUnavailable", "Overloaded", "APITimeoutError",
    )):
        return True
    status = getattr(exc, "status_code", None)
    return status in RETRYABLE_STATUS_CODES


def _call_with_retry(fn, *args, **kwargs):
    """Exponential backoff (1s, 2s, 4s, +jitter) around a provider API call."""
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == MAX_RETRIES or not _is_retryable(exc):
                raise
            delay = BASE_DELAY_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(delay)
    raise last_exc  # pragma: no cover -- loop above always returns or raises

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "merchant_name": {"type": ["string", "null"], "description": "Business/vendor name on the receipt."},
        "transaction_date": {"type": ["string", "null"], "description": "Date of purchase, formatted YYYY-MM-DD if possible."},
        "subtotal": {"type": ["number", "null"], "description": "Pre-tax subtotal amount, if shown separately from total."},
        "tax": {"type": ["number", "null"], "description": "Tax amount, if shown."},
        "tip": {"type": ["number", "null"], "description": "Tip/gratuity amount, if shown separately."},
        "total": {"type": ["number", "null"], "description": "Final total amount charged."},
        "currency": {"type": ["string", "null"], "description": "3-letter currency code, e.g. USD. Infer USD if a $ sign or no currency is shown and context suggests US."},
        "payment_card_last4": {"type": ["string", "null"], "description": "Last 4 digits of the payment card, if shown."},
        "employee_name_on_receipt": {"type": ["string", "null"], "description": "Employee name explicitly printed on the receipt, if any."},
        "employee_email_on_receipt": {"type": ["string", "null"], "description": "Employee email explicitly printed on the receipt, if any."},
        "category": {
            "type": "string",
            "enum": ["Travel", "Lodging", "Meals", "Transportation", "Parking", "Software/SaaS",
                     "Office Supplies", "Entertainment", "Conference/Event", "Other"],
            "description": "Best-guess expense category.",
        },
        "is_valid_receipt": {"type": "boolean", "description": "False if this document does not look like a legitimate purchase receipt at all."},
        "payment_status": {
            "type": "string",
            "enum": ["paid", "unpaid", "unknown"],
            "description": (
                "Whether the document shows the amount as actually paid/settled "
                "('paid' -- e.g. 'PAID', 'Balance paid', 'Total Paid', a payment "
                "confirmation, or a normal point-of-sale receipt which is paid by "
                "definition), as outstanding/not yet paid ('unpaid' -- e.g. 'PAYMENT "
                "DUE', 'Amount Due', 'Status: Unpaid', an open invoice), or genuinely "
                "not stated either way ('unknown'). This is independent of your "
                "extraction_confidence -- a document can be perfectly readable and "
                "still be unpaid."
            ),
        },
        "appears_handwritten": {
            "type": "boolean",
            "description": (
                "True if the receipt/invoice appears to be handwritten (or hand-filled on a "
                "blank template) rather than a printed/typed point-of-sale receipt or digital "
                "invoice. Handwritten documents are easier to fabricate and should always be "
                "routed for manual verification, regardless of how confidently you can read them."
            ),
        },
        "multiple_receipts_detected": {
            "type": "boolean",
            "description": (
                "True if the text/image appears to contain more than one distinct receipt or "
                "transaction (e.g. two separate receipts photographed side by side, or a PDF "
                "page showing two unrelated purchases). If true, do not attempt to merge or "
                "pick one -- just extract your best single guess for the fields below and flag "
                "this so a human can split it into separate expenses."
            ),
        },
        "extraction_confidence": {"type": "number", "description": "Your confidence (0.0-1.0) that the fields above were extracted correctly."},
        "extraction_explanation": {
            "type": "string",
            "description": (
                "1-3 sentences: what you were confident/uncertain about, any OCR garbling you "
                "had to work around, or math that didn't reconcile. Do NOT comment on the "
                "transaction's calendar year as unusual, futuristic, or a possible data-entry "
                "error -- you have already been told today's real-world date and any date on or "
                "before it is normal, so there is nothing to note about it."
            ),
        },
    },
    "required": [
        "merchant_name", "transaction_date", "total", "currency", "category",
        "is_valid_receipt", "payment_status", "appears_handwritten",
        "multiple_receipts_detected", "extraction_confidence", "extraction_explanation",
    ],
}

def build_system_prompt() -> str:
    # Ground the model in the real wall-clock date. Without this, the model
    # falls back to reasoning from its training cutoff and will incorrectly
    # flag perfectly normal recent-past dates (e.g. "2026-04-04") as
    # suspicious "future" dates simply because they're after its own
    # knowledge cutoff -- a false positive that has nothing to do with the
    # receipt actually being wrong.
    today = date.today().isoformat()
    return (
        "You are an expense-receipt data extraction engine for a corporate expense system. "
        "You will be given the text (OCR or native PDF text) of a single receipt, and "
        "sometimes an image of the receipt as well when the text is unreliable. "
        "Extract the fields exactly as instructed by the tool schema. "
        "If a field is not present on the receipt, return null for it rather than guessing. "
        f"Today's real-world date is {today} -- treat any transaction_date on or before this "
        "date as entirely normal; do NOT reduce confidence just because a date falls in a "
        "calendar year that might seem 'futuristic' to you, and do NOT mention the year, or "
        "call the date 'future', 'unusual', or a 'possible data-entry error' anywhere in "
        "extraction_explanation -- a date on or before today needs no comment at all. Only "
        f"flag a date as suspicious if it is genuinely after {today}, or implausibly long ago "
        "(many years old) for a routine business expense. "
        "Be conservative with extraction_confidence for other reasons: dock confidence for "
        "garbled OCR text, missing required fields, or math that doesn't add up "
        "(subtotal + tax + tip != total). "
        "Also assess appears_handwritten and multiple_receipts_detected as their own "
        "independent judgments -- neither should affect extraction_confidence itself, since a "
        "handwritten or multi-receipt document can still be perfectly legible; they exist "
        "purely to route the record for extra human scrutiny for reasons other than legibility. "
        "Judge payment_status separately from extraction_confidence: a document can be "
        "perfectly clear and readable while still showing an unpaid/outstanding balance "
        "(e.g. 'PAYMENT DUE', 'Amount Due', an open invoice) -- that should not affect "
        "extraction_confidence, but must be captured in payment_status so it can be "
        "routed for review regardless of how confident you are in the other fields."
    )


class LLMClient(ABC):
    @abstractmethod
    def extract_fields(self, text: str, image_b64: Optional[str], image_media_type: str) -> dict:
        ...


class AnthropicLLMClient(LLMClient):
    def __init__(self, settings: Settings):
        import anthropic
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_model

    def extract_fields(self, text: str, image_b64: Optional[str], image_media_type: str) -> dict:
        content = []
        if image_b64:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": image_media_type, "data": image_b64},
            })
        content.append({
            "type": "text",
            "text": f"Receipt text (from OCR or PDF text layer):\n\n{text or '(no extractable text)'}",
        })

        response = _call_with_retry(
            self._client.messages.create,
            model=self._model,
            max_tokens=1024,
            system=build_system_prompt(),
            tools=[{
                "name": "record_expense",
                "description": "Record the structured fields extracted from a receipt.",
                "input_schema": EXTRACTION_SCHEMA,
            }],
            tool_choice={"type": "tool", "name": "record_expense"},
            messages=[{"role": "user", "content": content}],
        )
        for block in response.content:
            if block.type == "tool_use":
                return block.input
        raise RuntimeError("Anthropic response did not include a tool_use block")


class OpenAILLMClient(LLMClient):
    def __init__(self, settings: Settings):
        import openai
        self._client = openai.OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    def extract_fields(self, text: str, image_b64: Optional[str], image_media_type: str) -> dict:
        content = [{
            "type": "text",
            "text": f"Receipt text (from OCR or PDF text layer):\n\n{text or '(no extractable text)'}",
        }]
        if image_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{image_media_type};base64,{image_b64}"},
            })

        response = _call_with_retry(
            self._client.chat.completions.create,
            model=self._model,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": content},
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "record_expense",
                    "description": "Record the structured fields extracted from a receipt.",
                    "parameters": EXTRACTION_SCHEMA,
                },
            }],
            tool_choice={"type": "function", "function": {"name": "record_expense"}},
        )
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)


class MockLLMClient(LLMClient):
    """Regex-based stand-in for a real LLM. Lets the pipeline run end-to-end
    for testing/demo purposes with zero API cost. Not a substitute for the
    real thing -- swap LLM_PROVIDER to 'anthropic' or 'openai' for real use.
    """

    def extract_fields(self, text: str, image_b64: Optional[str], image_media_type: str) -> dict:
        t = text or ""

        def find_amount(label, avoid_prefix=None):
            # avoid_prefix guards against e.g. "TOTAL" matching inside "SUBTOTAL".
            prefix_guard = f"(?<!{avoid_prefix})" if avoid_prefix else ""
            m = re.search(rf"{prefix_guard}\b{label}\b[:\s]*[A-Z]{{0,3}}\s*\$?\s*([\d,]+\.\d{{2}})", t, re.I)
            return float(m.group(1).replace(",", "")) if m else None

        def find(pattern, default=None):
            m = re.search(pattern, t, re.I)
            return m.group(1).strip() if m else default

        # Skip obvious non-merchant noise lines (e.g. phone status bars picked
        # up by OCR on screenshot-style receipts) when guessing the merchant name.
        merchant = None
        for line in t.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if re.search(r"^\d{1,2}:\d{2}\b", line) or "LTE" in line.upper() or line.endswith("%"):
                continue
            merchant = line
            break

        date = find(r"DATE[:\s]*([\d]{4}-[\d]{2}-[\d]{2})")

        # Different receipt templates in the wild use different labels for
        # the final charged amount -- try the common ones in priority order.
        total = None
        for label in ("TOTAL PAID", "BALANCE PAID", "FARE PAID", "AMOUNT DUE"):
            total = find_amount(label)
            if total is not None:
                break
        if total is None:
            total = find_amount("TOTAL", avoid_prefix="SUB")

        fare_or_merchandise = find_amount("MERCHANDISE") or find_amount("FARE")
        subtotal = find_amount("SUBTOTAL") or fare_or_merchandise
        # Some receipt formats (e.g. ride fares) only print one line-item amount
        # with no separate "TOTAL" line -- in that case the line-item IS the total.
        if total is None and fare_or_merchandise is not None:
            total = fare_or_merchandise
            subtotal = None
        tax = find_amount("TAX")
        tip = find_amount("TIP")
        card = find(r"CARD\s*\**\s*(\d{4})")

        # Different templates label the employee differently too.
        emp_name = None
        for label in ("EMPLOYEE REF", "GUEST", "PASSENGER", "EMPLOYEE"):
            emp_name = find(rf"{label}[:\s]*([A-Za-z .'-]+)")
            if emp_name:
                break

        emp_email = find(r"([\w.\-]+@[\w.\-]+\.\w+)")
        currency = find(r"\b(USD|EUR|GBP|CAD)\b", default="USD")

        is_valid = bool(total) and bool(merchant)
        confidence = 0.85 if (is_valid and date and card) else (0.55 if is_valid else 0.15)
        notes = ["[MOCK PROVIDER -- regex-based, not a real LLM call]"]
        if subtotal and tax and total and abs(subtotal + tax + (tip or 0) - total) > 0.05:
            notes.append("subtotal + tax (+ tip) does not reconcile with total")
            confidence = min(confidence, 0.4)
        if not t.strip():
            notes.append("no OCR/text content available")
            confidence = 0.1

        # Payment status is judged independently of extraction confidence --
        # a receipt can be perfectly readable and still be unpaid.
        if re.search(r"PAYMENT DUE|\bAMOUNT DUE\b|STATUS:\s*(UNPAID|DUE)", t, re.I):
            payment_status = "unpaid"
            notes.append("document shows an unpaid/outstanding balance, not a confirmed payment")
        elif re.search(r"\bPAID\b|BALANCE PAID|TOTAL PAID|FARE PAID", t, re.I):
            payment_status = "paid"
        elif is_valid:
            # A normal point-of-sale receipt (merchandise/fare/total with a
            # card charge, no invoice language) is paid by definition.
            payment_status = "paid"
        else:
            payment_status = "unknown"

        return {
            "merchant_name": merchant,
            "transaction_date": date,
            "subtotal": subtotal,
            "tax": tax,
            "tip": tip,
            "total": total,
            "currency": currency,
            "payment_card_last4": card,
            "employee_name_on_receipt": emp_name,
            "employee_email_on_receipt": emp_email,
            "category": "Other",
            "is_valid_receipt": is_valid,
            "payment_status": payment_status,
            # The mock provider is regex-based and has no way to actually judge
            # handwriting or detect a second receipt in the same document --
            # both always come back False here. The sample dataset has no
            # handwritten or multi-receipt files, so this matches reality for
            # the current data; a real vision-capable provider judges these
            # properly per receipt (see EXTRACTION_SCHEMA).
            "appears_handwritten": False,
            "multiple_receipts_detected": False,
            "extraction_confidence": confidence,
            "extraction_explanation": " ".join(notes),
        }


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "anthropic":
        return AnthropicLLMClient(settings)
    if settings.llm_provider == "openai":
        return OpenAILLMClient(settings)
    return MockLLMClient()

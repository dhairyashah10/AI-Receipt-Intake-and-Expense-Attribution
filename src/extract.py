"""
Field extraction: wires ingestion output into the LLM client, deciding
whether the image needs to be sent (vision fallback) based on OCR quality,
and normalizes the result into an ExtractedExpense record.
"""
from dataclasses import dataclass, field
from typing import Optional

from src.config import Settings
from src.ingest import ReceiptSource
from src.llm_client import PROMPT_VERSION, LLMClient


@dataclass
class ExtractedExpense:
    filename: str
    merchant_name: Optional[str]
    transaction_date: Optional[str]
    subtotal: Optional[float]
    tax: Optional[float]
    tip: Optional[float]
    total: Optional[float]
    currency: Optional[str]
    payment_card_last4: Optional[str]
    employee_name_on_receipt: Optional[str]
    employee_email_on_receipt: Optional[str]
    category: str
    is_valid_receipt: bool
    payment_status: str
    appears_handwritten: bool
    multiple_receipts_detected: bool
    extraction_confidence: float
    extraction_explanation: str
    source_method: str
    ocr_quality: float
    used_vision_fallback: bool
    orientation_corrected_degrees: int
    file_sha256: str
    extraction_prompt_version: str = PROMPT_VERSION
    llm_provider: str = ""
    llm_model: str = ""
    raw_text: str = field(repr=False, default="")


def _llm_model_name(settings: Settings) -> str:
    # Records which model produced this extraction (not just which provider),
    # since a model upgrade can change extraction behavior even under an
    # unchanged PROMPT_VERSION -- useful when auditing why older/newer
    # records in the same batch behave slightly differently.
    if settings.llm_provider == "anthropic":
        return settings.anthropic_model
    if settings.llm_provider == "openai":
        return settings.openai_model
    return "mock"


def extract_expense(
    source: ReceiptSource, llm_client: LLMClient, settings: Settings
) -> ExtractedExpense:
    use_vision = (
        source.image_b64 is not None and source.ocr_quality < settings.ocr_quality_threshold
    )
    llm_provider = settings.llm_provider
    llm_model = _llm_model_name(settings)

    try:
        result = llm_client.extract_fields(
            text=source.text,
            image_b64=source.image_b64 if use_vision else None,
            image_media_type=source.image_media_type,
        )
    except Exception as exc:  # LLM call failed outright (bad key, rate limit, etc.)
        return ExtractedExpense(
            filename=source.filename,
            merchant_name=None,
            transaction_date=None,
            subtotal=None,
            tax=None,
            tip=None,
            total=None,
            currency=None,
            payment_card_last4=None,
            employee_name_on_receipt=None,
            employee_email_on_receipt=None,
            category="Other",
            is_valid_receipt=False,
            payment_status="unknown",
            appears_handwritten=False,
            multiple_receipts_detected=False,
            extraction_confidence=0.0,
            extraction_explanation=f"LLM extraction call failed: {exc}",
            source_method=source.source_method,
            ocr_quality=source.ocr_quality,
            used_vision_fallback=use_vision,
            orientation_corrected_degrees=source.orientation_corrected_degrees,
            file_sha256=source.file_sha256,
            llm_provider=llm_provider,
            llm_model=llm_model,
            raw_text=source.text,
        )

    return ExtractedExpense(
        filename=source.filename,
        merchant_name=result.get("merchant_name"),
        transaction_date=result.get("transaction_date"),
        subtotal=result.get("subtotal"),
        tax=result.get("tax"),
        tip=result.get("tip"),
        total=result.get("total"),
        currency=result.get("currency") or "USD",
        payment_card_last4=result.get("payment_card_last4"),
        employee_name_on_receipt=result.get("employee_name_on_receipt"),
        employee_email_on_receipt=result.get("employee_email_on_receipt"),
        category=result.get("category", "Other"),
        is_valid_receipt=bool(result.get("is_valid_receipt", True)),
        payment_status=result.get("payment_status") or "unknown",
        appears_handwritten=bool(result.get("appears_handwritten", False)),
        multiple_receipts_detected=bool(result.get("multiple_receipts_detected", False)),
        extraction_confidence=float(result.get("extraction_confidence", 0.5)),
        extraction_explanation=result.get("extraction_explanation", ""),
        source_method=source.source_method,
        ocr_quality=source.ocr_quality,
        used_vision_fallback=use_vision,
        orientation_corrected_degrees=source.orientation_corrected_degrees,
        file_sha256=source.file_sha256,
        llm_provider=llm_provider,
        llm_model=llm_model,
        raw_text=source.text,
    )

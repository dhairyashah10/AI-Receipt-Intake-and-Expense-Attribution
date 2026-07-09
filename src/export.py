"""
Exports the final structured records to CSV and JSON.
"""
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

COLUMN_ORDER = [
    "filename",
    "is_valid_receipt",
    "merchant_name",
    "transaction_date",
    "subtotal",
    "tax",
    "tip",
    "total",
    "currency",
    "category",
    "payment_status",
    "appears_handwritten",
    "multiple_receipts_detected",
    "payment_card_last4",
    "employee_name_on_receipt",
    "employee_email_on_receipt",
    "matched_employee_id",
    "matched_employee_name",
    "attribution_method",
    "attribution_confidence",
    "attribution_explanation",
    "extraction_confidence",
    "extraction_explanation",
    "needs_review",
    "review_reasons",
    "is_possible_duplicate",
    "duplicate_of",
    "source_method",
    "ocr_quality",
    "used_vision_fallback",
    "orientation_corrected_degrees",
    "extraction_prompt_version",
    "file_sha256",
    "llm_provider",
    "llm_model",
    "math_reconciles",
    "required_fields_present",
    "has_conflict",
]


def build_record(expense, attribution, review) -> dict:
    record = asdict(expense)
    record.pop("raw_text", None)
    record.update({
        "matched_employee_id": attribution.matched_employee_id,
        "matched_employee_name": attribution.matched_employee_name,
        "attribution_method": attribution.attribution_method,
        "attribution_confidence": attribution.attribution_confidence,
        "attribution_explanation": attribution.attribution_explanation,
        "needs_review": review.needs_review,
        "review_reasons": "; ".join(review.reasons),
        # Pure metadata -- these are already-computed intermediate values
        # that feed needs_review/attribution_confidence; exposing them here
        # doesn't change any decision, it just stops throwing them away.
        "math_reconciles": review.math_reconciles,
        "required_fields_present": review.required_fields_present,
        "has_conflict": attribution.has_conflict,
    })
    return record


def export_records(records: list, output_dir: Path, base_name: str = "expenses"):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    ordered_cols = [c for c in COLUMN_ORDER if c in df.columns] + [
        c for c in df.columns if c not in COLUMN_ORDER
    ]
    df = df[ordered_cols]

    csv_path = output_dir / f"{base_name}.csv"
    json_path = output_dir / f"{base_name}.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    return csv_path, json_path


def export_needs_review(records: list, output_dir: Path, base_name: str = "expenses") -> Path:
    """Writes just the records with needs_review=True to their own JSON file --
    a plain filtered view of the same already-computed data (no new decision
    logic, nothing recomputed), so a reviewer has a small worklist to open
    instead of scanning the full batch for the flagged ones.

    This is a snapshot of the current run only, not a persistent queue --
    running the pipeline again produces a fresh file with no memory of what
    was already reviewed/resolved last time. A real growing/shrinking review
    queue (with per-record status and reviewer history) is a bigger feature;
    see the README's Future Development section.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    flagged = [r for r in records if r.get("needs_review")]
    path = output_dir / f"{base_name}_needs_review.json"
    with open(path, "w") as f:
        json.dump(flagged, f, indent=2, default=str)
    return path

#!/usr/bin/env python3
"""
Minimal FastAPI wrapper around the receipt-intake pipeline, for a
request/response deployment model alongside the batch CLI (main.py).

Run with:
    uvicorn api:app --reload

Then:
    curl -F "file=@data/receipts/receipt_001.pdf" http://127.0.0.1:8000/extract

Notes / scope:
  - This exposes the SAME pipeline modules main.py uses (ingest -> extract ->
    attribute -> review -> build_record), so a record returned here has an
    identical shape/schema to a row in the batch CSV/JSON output.
  - Settings, the LLM client, and the employee roster are loaded ONCE at
    process startup (not per-request) -- the roster CSV and LLM client are
    read-only/stateless across requests, so re-loading them per-call would
    just add latency for no benefit.
  - Cross-record duplicate detection (src/duplicates.py) is intentionally
    NOT run here: it depends on comparing a record against the rest of a
    batch, which doesn't exist in a single-request context. A production
    deployment of this endpoint would run that check against a persistent
    store (e.g. "have we seen this file_sha256 or this merchant/date/total
    fingerprint in the last N days of submissions") -- see the "Deploying
    and expanding" section of README.md.
  - Errors from the pipeline (bad file, LLM failure after retries, etc.)
    are converted to HTTP 500 with the underlying message rather than
    crashing the process, since this is a long-lived server, not a
    one-shot CLI run.
"""
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.attribution import attribute_employee, load_roster
from src.config import load_settings
from src.export import build_record
from src.extract import extract_expense
from src.ingest import ingest_receipt
from src.llm_client import PROMPT_VERSION, build_llm_client
from src.review import evaluate

ALLOWED_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png"}

app = FastAPI(
    title="Receipt Intake API",
    description="AI receipt intake and expense attribution -- single-receipt request/response endpoint.",
    version="1.0",
)

_settings = load_settings()
_llm_client = build_llm_client(_settings)
_roster = load_roster("data/employee_roster.csv")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_provider": _settings.llm_provider,
        "prompt_version": PROMPT_VERSION,
        "roster_size": len(_roster),
    }


@app.post("/extract")
async def extract_receipt(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    body = await file.read()
    if not body:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(body)
            tmp.flush()
            path = Path(tmp.name)

            source = ingest_receipt(path)
            source.filename = file.filename  # report the original name, not the temp path

            expense = extract_expense(source, _llm_client, _settings)
            attribution = attribute_employee(expense, _roster)
            review = evaluate(expense, attribution)
            record = build_record(expense, attribution, review)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline failure: {exc}") from exc

    # Single-request scope only -- see module docstring for why cross-record
    # duplicate detection isn't run here.
    record["is_possible_duplicate"] = False
    record["duplicate_of"] = ""
    return record

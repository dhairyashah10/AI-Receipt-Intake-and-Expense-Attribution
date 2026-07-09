#!/usr/bin/env python3
"""
AI Receipt Intake and Expense Attribution -- CLI entry point.

Usage:
    python main.py --receipts data/receipts --roster data/employee_roster.csv --out output
    python main.py --workers 20   # process receipts concurrently (default: 8)

Reads LLM_PROVIDER (anthropic / openai / mock) and the matching API key from
environment variables / a local .env file -- see .env.example.

Concurrency note: each receipt's dominant cost is waiting on the LLM API
call (network I/O, ~1-3s), not local CPU work, so a thread pool gives a
near-linear speedup up to whatever the provider's rate limits allow --
Python's GIL is not a bottleneck here since threads release it while
blocked on network I/O. For true million-record scale, see the "Deploying
and expanding" section of README.md (queue-based workers, batch inference
APIs, idempotency/checkpointing).
"""
import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.attribution import attribute_employee, load_roster
from src.config import load_settings
from src.duplicates import flag_duplicates
from src.export import build_record, export_needs_review, export_records
from src.extract import extract_expense
from src.ingest import discover_receipts, ingest_receipt
from src.llm_client import build_llm_client
from src.review import evaluate

_print_lock = threading.Lock()


def process_one(index, total, path, llm_client, settings, roster):
    t0 = time.time()
    source = ingest_receipt(path)
    expense = extract_expense(source, llm_client, settings)
    attribution = attribute_employee(expense, roster)
    review = evaluate(expense, attribution)
    record = build_record(expense, attribution, review)
    elapsed = time.time() - t0

    flag = "REVIEW" if review.needs_review else "ok"
    line = (
        f"[{index}/{total}] {path.name}: "
        f"{expense.merchant_name or '?'} / ${expense.total or 0:.2f} -> "
        f"{attribution.matched_employee_name or 'UNMATCHED'} "
        f"[{flag}] ({elapsed:.1f}s)"
    )
    with _print_lock:
        print(line)

    return index, record, review.needs_review


def main():
    parser = argparse.ArgumentParser(description="AI receipt intake and expense attribution")
    parser.add_argument("--receipts", default="data/receipts", help="Folder of receipt files")
    parser.add_argument("--roster", default="data/employee_roster.csv", help="Employee roster CSV")
    parser.add_argument("--out", default="output", help="Output folder for CSV/JSON")
    parser.add_argument("--base-name", default="expenses", help="Base filename for outputs")
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of receipts to process concurrently (default: 8). "
             "Each worker makes its own LLM API call, so keep this within "
             "your provider's rate limits.",
    )
    args = parser.parse_args()

    settings = load_settings()
    print(f"[config] LLM_PROVIDER = {settings.llm_provider}")
    print(f"[config] workers = {args.workers}")
    if settings.llm_provider == "mock":
        print("[config] Running in MOCK mode -- no real LLM calls will be made. "
              "Set LLM_PROVIDER=anthropic|openai and the matching API key in .env for real extraction.")

    llm_client = build_llm_client(settings)
    roster = load_roster(args.roster)

    receipt_paths = discover_receipts(Path(args.receipts))
    if not receipt_paths:
        print(f"No receipt files found in {args.receipts}", file=sys.stderr)
        sys.exit(1)

    total = len(receipt_paths)
    results = {}  # index -> record
    review_count = 0
    failures = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, i, total, path, llm_client, settings, roster): (i, path)
            for i, path in enumerate(receipt_paths, 1)
        }
        for future in as_completed(futures):
            i, path = futures[future]
            try:
                index, record, needs_review = future.result()
            except Exception as exc:
                failures += 1
                with _print_lock:
                    print(f"[{i}/{total}] {path.name}: FAILED -- {exc}", file=sys.stderr)
                continue
            results[index] = record
            if needs_review:
                review_count += 1

    # Re-assemble in original (filename-sorted) order regardless of completion order.
    records = [results[i] for i in sorted(results.keys())]

    # Duplicate detection is inherently cross-record (is this receipt a
    # repeat of some OTHER receipt in the batch?), so it runs as a
    # post-processing pass over the whole batch rather than per-receipt
    # inside process_one -- it can flip needs_review on records that
    # otherwise looked perfectly clean.
    flag_duplicates(records)
    review_count = sum(1 for r in records if r["needs_review"])
    duplicate_count = sum(1 for r in records if r["is_possible_duplicate"])

    csv_path, json_path = export_records(records, Path(args.out), args.base_name)
    review_path = export_needs_review(records, Path(args.out), args.base_name)
    elapsed_total = time.time() - start

    print()
    print(f"Processed {len(records)} receipts ({failures} failed) in {elapsed_total:.1f}s "
          f"using {args.workers} workers -- {review_count} flagged for human review "
          f"({duplicate_count} possible duplicates).")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {review_path} ({review_count} flagged records)")


if __name__ == "__main__":
    main()

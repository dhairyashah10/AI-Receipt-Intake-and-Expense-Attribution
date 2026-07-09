"""
Unit tests for src.export.export_needs_review -- a plain filtered export
(no new decision logic) that writes just the needs_review=True records to
their own JSON file, so a reviewer has a small worklist instead of scanning
the full batch export. Run with: pytest tests/
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.export import export_needs_review


def make_record(filename, needs_review):
    return {
        "filename": filename,
        "needs_review": needs_review,
        "review_reasons": "Low confidence." if needs_review else "",
        "total": 42.0,
    }


def test_only_flagged_records_are_written():
    records = [
        make_record("a.pdf", needs_review=True),
        make_record("b.pdf", needs_review=False),
        make_record("c.pdf", needs_review=True),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = export_needs_review(records, Path(tmp))
        written = json.loads(path.read_text())

    assert len(written) == 2
    assert {r["filename"] for r in written} == {"a.pdf", "c.pdf"}


def test_no_flagged_records_writes_empty_list():
    records = [make_record("a.pdf", needs_review=False)]
    with tempfile.TemporaryDirectory() as tmp:
        path = export_needs_review(records, Path(tmp))
        written = json.loads(path.read_text())

    assert written == []


def test_written_records_are_unmodified_copies_of_the_input():
    # No new logic here -- the exported records must be exactly what was
    # already computed upstream, not recomputed or reshaped.
    records = [make_record("a.pdf", needs_review=True)]
    with tempfile.TemporaryDirectory() as tmp:
        path = export_needs_review(records, Path(tmp))
        written = json.loads(path.read_text())

    assert written[0] == records[0]


def test_output_filename_uses_base_name():
    records = [make_record("a.pdf", needs_review=True)]
    with tempfile.TemporaryDirectory() as tmp:
        path = export_needs_review(records, Path(tmp), base_name="custom")
    assert path.name == "custom_needs_review.json"

"""
Unit tests for cross-record duplicate-submission detection. Run with:
pytest tests/

Motivated by a real miss found in the sample data: receipt_021 (a PDF) and
receipt_022 (a photo) turned out to be the same underlying purchase --
same merchant, date, total, and card last-4 -- submitted twice under
different filenames/formats. Nothing in the per-receipt pipeline could catch
this (each receipt looks perfectly fine in isolation); it has to be checked
across the batch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.duplicates import flag_duplicates


def make_record(filename, **kwargs):
    defaults = dict(
        merchant_name="PAPERTRAIL OFFICE SUPPLY",
        transaction_date="2026-06-06",
        total=80.22,
        currency="USD",
        payment_card_last4="8370",
        needs_review=False,
        review_reasons="",
    )
    defaults.update(kwargs)
    defaults["filename"] = filename
    return defaults


def test_identical_transaction_different_files_is_flagged():
    r1 = make_record("receipt_021.pdf")
    r2 = make_record("receipt_022.png")
    records = [r1, r2]
    flag_duplicates(records)

    assert r1["is_possible_duplicate"] is True
    assert r2["is_possible_duplicate"] is True
    assert r1["needs_review"] is True
    assert r2["needs_review"] is True
    assert "receipt_022.png" in r1["duplicate_of"]
    assert "receipt_021.pdf" in r2["duplicate_of"]


def test_different_transactions_are_not_flagged():
    r1 = make_record("receipt_a.pdf", total=80.22)
    r2 = make_record("receipt_b.pdf", total=45.00)
    records = [r1, r2]
    flag_duplicates(records)

    assert r1["is_possible_duplicate"] is False
    assert r2["is_possible_duplicate"] is False
    assert r1["needs_review"] is False
    assert r2["needs_review"] is False


def test_missing_fields_do_not_false_positive():
    # Two receipts both missing a total shouldn't be called duplicates of
    # each other just because they share "None" in that slot.
    r1 = make_record("receipt_a.pdf", total=None)
    r2 = make_record("receipt_b.pdf", total=None)
    records = [r1, r2]
    flag_duplicates(records)

    assert r1["is_possible_duplicate"] is False
    assert r2["is_possible_duplicate"] is False


def test_three_way_duplicate_group_cross_references_all():
    r1 = make_record("a.pdf")
    r2 = make_record("b.png")
    r3 = make_record("c.jpg")
    records = [r1, r2, r3]
    flag_duplicates(records)

    for r in records:
        assert r["is_possible_duplicate"] is True
        others = r["duplicate_of"].split("; ")
        assert len(others) == 2


def test_existing_review_reason_is_preserved_alongside_duplicate_note():
    r1 = make_record("a.pdf", needs_review=True, review_reasons="Low confidence.")
    r2 = make_record("b.pdf")
    flag_duplicates([r1, r2])

    assert "Low confidence." in r1["review_reasons"]
    assert "duplicate" in r1["review_reasons"].lower()


def test_identical_file_hash_is_flagged_even_with_different_content_fields():
    # Same literal file re-uploaded under a new filename with, say, a bad
    # re-extraction giving different fields the second time around -- the
    # content fingerprint wouldn't catch this, but the byte hash will.
    r1 = make_record("original.pdf", total=80.22, file_sha256="abc123")
    r2 = make_record("renamed_copy.pdf", total=999.99, file_sha256="abc123")
    records = [r1, r2]
    flag_duplicates(records)

    assert r1["is_possible_duplicate"] is True
    assert r2["is_possible_duplicate"] is True
    assert "SHA-256" in r1["review_reasons"]


def test_missing_file_hash_does_not_false_positive():
    # Two records that both simply lack a file_sha256 (e.g. older records)
    # should not be treated as hash-matching each other.
    r1 = make_record("a.pdf", total=10.00, file_sha256="")
    r2 = make_record("b.pdf", total=45.00, file_sha256="")
    records = [r1, r2]
    flag_duplicates(records)

    assert r1["is_possible_duplicate"] is False
    assert r2["is_possible_duplicate"] is False

"""
Cross-record duplicate-submission detection.

Unlike every other check in this pipeline, this one can't run per-receipt --
it has to look across the whole batch after processing, since "is this a
duplicate" is a question about *other* records, not the receipt in isolation.

Two independent tiers, since they catch different real-world scenarios:

  1. Exact file match (SHA-256 of the raw bytes). Catches the literal same
     file being uploaded twice, even under a different filename -- cheap,
     zero false positives, but only fires if the bytes are identical.
  2. Content fingerprint (merchant + date + total + currency + card last-4).
     Catches the same underlying purchase submitted via two DIFFERENT files
     -- e.g. a PDF receipt and a separately-snapped photo of the same
     receipt, which SHA-256 would never match since the bytes differ
     entirely. This is the tier that caught receipt_021/022 in the sample
     data (a PDF and a PNG of the same transaction).
"""
from collections import defaultdict
from typing import List

FINGERPRINT_FIELDS = ("merchant_name", "transaction_date", "total", "payment_card_last4")


def _content_fingerprint(record: dict):
    # Require all fingerprint fields to be present -- two receipts that are
    # BOTH missing a total, say, should not be called duplicates of each
    # other just because they share a "None" in that slot.
    if any(record.get(f) in (None, "") for f in FINGERPRINT_FIELDS):
        return None

    try:
        total = round(float(record["total"]), 2)
    except (TypeError, ValueError):
        return None

    merchant = str(record["merchant_name"]).strip().lower()
    return (merchant, record["transaction_date"], total, record.get("currency"), str(record["payment_card_last4"]))


def _file_hash_key(record: dict):
    h = record.get("file_sha256")
    return h if h else None


def _group_by(records: List[dict], key_fn):
    groups = defaultdict(list)
    for r in records:
        key = key_fn(r)
        if key is not None:
            groups[key].append(r["filename"])
    return {k: v for k, v in groups.items() if len(v) > 1}


def flag_duplicates(records: List[dict]) -> None:
    """Mutates records in place. Adds 'is_possible_duplicate' (bool) and
    'duplicate_of' (semicolon-joined filenames) to every record, and folds a
    note into review_reasons / needs_review for any record that matches
    another record on either tier above.
    """
    hash_groups = _group_by(records, _file_hash_key)
    content_groups = _group_by(records, _content_fingerprint)

    others_by_filename = defaultdict(set)
    reasons_by_filename = defaultdict(list)

    for names in hash_groups.values():
        for fn in names:
            other_names = [n for n in names if n != fn]
            others_by_filename[fn].update(other_names)
            reasons_by_filename[fn].append(
                f"identical file (SHA-256 match) as: {', '.join(other_names)}"
            )

    for names in content_groups.values():
        for fn in names:
            other_names = [n for n in names if n != fn]
            others_by_filename[fn].update(other_names)
            reasons_by_filename[fn].append(
                f"same merchant/date/total/card as: {', '.join(other_names)}"
            )

    for r in records:
        others = sorted(others_by_filename.get(r["filename"], []))
        r["is_possible_duplicate"] = bool(others)
        r["duplicate_of"] = "; ".join(others)

        if others:
            r["needs_review"] = True
            note = f"Possible duplicate submission -- {'; '.join(reasons_by_filename[r['filename']])}."
            r["review_reasons"] = f"{r['review_reasons']}; {note}" if r["review_reasons"] else note

"""
Employee attribution: matches an extracted expense to the most likely
employee in the roster.

Signal priority (most to least trustworthy):
  1. Corporate card last-4 digits -- exact match. Cards are unique per
     employee, so this is the strongest signal when present.
  2. Employee email printed on the receipt -- exact match.
  3. Employee name printed on the receipt -- EXACT string match (checked
     before fuzzy matching -- see note below).
  4. Employee name -- fuzzy match (handles typos/OCR noise), only used if
     nothing above matched.

Why exact name is checked before fuzzy, as its own tier: the sample data
includes two real, distinct employees named "Jane Miller" (emp_001) and
"Janet Miller" (emp_011). Their names are textually very similar
(rapidfuzz WRatio ~96), so a naive "just fuzzy match names" approach would
risk conflating them. Checking card/email/exact-name first means the
correct distinct employee is found via a hard signal before fuzzy logic
(which is deliberately the last resort) ever gets a chance to blur them
together. When fuzzy matching does run, we additionally check for a
close second-place candidate and flag those as ambiguous rather than
silently picking the top score.

We also cross-check: if the receipt's printed name doesn't match whichever
employee we ultimately attributed to (e.g. card matched employee A but the
printed name looks like a different employee B), we note the conflict in
the explanation and flag it for review even though we still trust the
stronger signal.
"""
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from rapidfuzz import fuzz, process

FUZZY_MATCH_THRESHOLD = 85
FUZZY_AMBIGUOUS_GAP = 6  # if top-2 fuzzy scores are within this gap, call it ambiguous


@dataclass
class AttributionResult:
    matched_employee_id: Optional[str]
    matched_employee_name: Optional[str]
    attribution_method: str  # card_last4 | email_exact | name_exact | name_fuzzy | none
    attribution_confidence: float
    attribution_explanation: str
    has_conflict: bool


def load_roster(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df["corporate_card_last4"] = df["corporate_card_last4"].str.zfill(4)
    return df


def _row_by_id(roster: pd.DataFrame, employee_id: str):
    return roster.loc[roster["employee_id"] == employee_id].iloc[0]


def _best_name_candidate(name: str, roster: pd.DataFrame):
    """Returns (employee_id, score, is_ambiguous) for the best fuzzy name match."""
    choices = roster["full_name"].tolist()
    matches = process.extract(name, choices, scorer=fuzz.WRatio, limit=2)
    if not matches:
        return None, 0, False
    top_name, top_score, top_idx = matches[0]
    is_ambiguous = False
    if len(matches) > 1:
        _, second_score, _ = matches[1]
        if top_score - second_score < FUZZY_AMBIGUOUS_GAP:
            is_ambiguous = True
    employee_id = roster.iloc[top_idx]["employee_id"]
    return employee_id, top_score, is_ambiguous


def attribute_employee(expense, roster: pd.DataFrame) -> AttributionResult:
    conflict_note = None

    # Independently work out what the printed name (if any) would suggest,
    # so we can cross-check it against whichever signal wins below.
    name_hint_id = None
    if expense.employee_name_on_receipt:
        exact = roster.loc[
            roster["full_name"].str.lower() == expense.employee_name_on_receipt.strip().lower()
        ]
        if len(exact) == 1:
            name_hint_id = exact.iloc[0]["employee_id"]
        else:
            fuzzy_id, score, ambiguous = _best_name_candidate(expense.employee_name_on_receipt, roster)
            if score >= FUZZY_MATCH_THRESHOLD and not ambiguous:
                name_hint_id = fuzzy_id

    def finalize(employee_id, method, confidence, explanation):
        row = _row_by_id(roster, employee_id)
        has_conflict = bool(name_hint_id and name_hint_id != employee_id)
        if has_conflict:
            conflicting_row = _row_by_id(roster, name_hint_id)
            explanation += (
                f" CONFLICT: printed name on receipt best matches '{conflicting_row['full_name']}' "
                f"({name_hint_id}), which differs from the matched employee -- flagging for review."
            )
            confidence = min(confidence, 0.75)
        return AttributionResult(
            matched_employee_id=employee_id,
            matched_employee_name=row["full_name"],
            attribution_method=method,
            attribution_confidence=round(confidence, 3),
            attribution_explanation=explanation,
            has_conflict=has_conflict,
        )

    # Tier 1: card last-4 exact match
    if expense.payment_card_last4:
        card = str(expense.payment_card_last4).zfill(4)
        card_matches = roster.loc[roster["corporate_card_last4"] == card]
        if len(card_matches) == 1:
            emp_id = card_matches.iloc[0]["employee_id"]
            return finalize(
                emp_id, "card_last4", 0.95,
                f"Card ending {card} on the receipt uniquely matches {card_matches.iloc[0]['full_name']} in the roster.",
            )
        elif len(card_matches) > 1:
            names = ", ".join(card_matches["full_name"])
            return AttributionResult(
                None, None, "card_last4_ambiguous", 0.2,
                f"Card ending {card} matches multiple employees ({names}) -- cannot disambiguate.",
                has_conflict=True,
            )

    # Tier 2: email exact match
    if expense.employee_email_on_receipt:
        email = expense.employee_email_on_receipt.strip().lower()
        email_matches = roster.loc[roster["email"].str.lower() == email]
        if len(email_matches) == 1:
            emp_id = email_matches.iloc[0]["employee_id"]
            return finalize(
                emp_id, "email_exact", 0.92,
                f"Email '{email}' on the receipt exactly matches {email_matches.iloc[0]['full_name']} in the roster.",
            )

    # Tier 3: exact name match
    if expense.employee_name_on_receipt:
        name = expense.employee_name_on_receipt.strip().lower()
        name_matches = roster.loc[roster["full_name"].str.lower() == name]
        if len(name_matches) == 1:
            emp_id = name_matches.iloc[0]["employee_id"]
            return finalize(
                emp_id, "name_exact", 0.9,
                f"Printed name '{expense.employee_name_on_receipt}' exactly matches {name_matches.iloc[0]['full_name']} in the roster.",
            )

        # Tier 4: fuzzy name match
        fuzzy_id, score, ambiguous = _best_name_candidate(expense.employee_name_on_receipt, roster)
        if fuzzy_id is not None and score >= FUZZY_MATCH_THRESHOLD and not ambiguous:
            row = _row_by_id(roster, fuzzy_id)
            confidence = 0.5 + min((score - FUZZY_MATCH_THRESHOLD) / 100.0, 0.35)
            return finalize(
                fuzzy_id, "name_fuzzy", confidence,
                f"Printed name '{expense.employee_name_on_receipt}' fuzzy-matches {row['full_name']} "
                f"(similarity {score:.0f}/100).",
            )
        if fuzzy_id is not None and ambiguous:
            return AttributionResult(
                None, None, "name_fuzzy_ambiguous", 0.25,
                f"Printed name '{expense.employee_name_on_receipt}' is ambiguously close to multiple "
                f"roster names (top similarity {score:.0f}/100) -- cannot confidently disambiguate.",
                has_conflict=True,
            )

    # No usable signal at all
    return AttributionResult(
        None, None, "none", 0.0,
        "No card, email, or name information on the receipt could be matched to the roster.",
        has_conflict=False,
    )

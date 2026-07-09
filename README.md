# AI Receipt Intake and Expense Attribution

Take-home prototype for Black Osprey: ingests a folder of receipts (PDF/JPG/PNG,
some with a text layer, some scanned/photographed), extracts structured expense
fields, attributes each expense to the most likely employee, scores confidence
for both steps with an explanation, flags uncertain/invalid/duplicate records
for human review, and exports everything to CSV + JSON. A FastAPI endpoint is
also included for a single-receipt request/response deployment model alongside
the batch CLI.

## Architecture

```
receipt file (pdf/jpg/png)
        |
        v
  src/ingest.py      -- detect format; pdfplumber for text-layer PDFs.
                         For scans/photos: correct gross orientation (Tesseract
                         OSD) -> deskew small-angle tilt (OpenCV) -> Tesseract
                         OCR -> score OCR quality. Also hashes the raw file
                         bytes (SHA-256) for exact-duplicate-file detection.
        |
        v
  src/llm_client.py  -- provider-agnostic LLM call (Anthropic or OpenAI),
  src/extract.py        forced tool-call with a strict JSON schema.
                         OCR text is sent by default; if OCR quality is below
                         a threshold, the receipt image is sent instead
                         (vision fallback) so blurry/garbled scans still work.
                         Transient failures (rate limits, timeouts, 5xx) are
                         retried with exponential backoff. Every record is
                         tagged with the prompt/schema version that produced it.
        |
        v
  src/attribution.py -- matches the expense to the roster via a priority
                         cascade: card last-4 (exact) -> email (exact) ->
                         name (exact) -> name (fuzzy, rapidfuzz). Cross-checks
                         the printed name against whichever signal wins and
                         flags conflicts.
        |
        v
  src/review.py       -- independent sanity checks on top of the LLM's own
                          confidence: required fields present? does
                          subtotal + tax + tip = total? was vision fallback
                          needed? is attribution confident and conflict-free?
                          is the receipt/invoice actually paid? does it
                          appear handwritten? multiple receipts in one file?
                          is it above the high-value threshold?
                          -> needs_review flag + human-readable reasons
        |
        v
  src/duplicates.py   -- batch-level pass: flags records that are either
                          byte-identical to another file (SHA-256) or share
                          the same merchant/date/total/card as another record
                          (e.g. the same receipt submitted as two different
                          file formats)
        |
        v
  src/export.py       -- output/expenses.csv, output/expenses.json, and
                          output/expenses_needs_review.json (flagged subset)

  api.py              -- FastAPI wrapper exposing the same
                          ingest -> extract -> attribute -> review pipeline
                          as a single-receipt POST /extract endpoint
```

### Key design decisions

**OCR-first, vision-fallback (not vision-always).** Every receipt is OCR'd
locally first (free, fast). We only pay for a vision-model call when the OCR
transcript looks unreliable (short, garbled, missing receipt-like keywords).
This keeps the common case cheap while still handling real-world photos.

**Orientation correction is separate from deskew.** A fully sideways (90/180/
270-degree) photo is a categorically different problem from a few-degrees
tilt. Tesseract's orientation-and-script-detection (OSD) runs as an explicit
first pass (gated on OSD's own confidence score, since a low-confidence OSD
guess can be wrong and flip a genuinely-upright image); small-angle tilt is
then corrected separately via an OpenCV minAreaRect-based deskew. Both are
best-effort and fail open (leave the image untouched) rather than raising.

**Structured extraction via forced tool-calls, not prompted JSON.** Both the
Anthropic and OpenAI clients force a single tool/function call with a strict
JSON schema, rather than asking the model to "return JSON" in prose. This is
far more reliable to parse and lets the model report per-record confidence
and an explanation as part of the same schema.

**Retries only for transient failures.** `src/llm_client.py` retries with
exponential backoff (1s, 2s, 4s + jitter) on rate limits, timeouts, connection
errors, and 5xx responses. A bad API key or malformed request fails the same
way every time, so those raise immediately instead of wasting ~15s per
receipt before failing anyway.

**Prompt versioning.** `PROMPT_VERSION` in `src/llm_client.py` is threaded
through to every exported record (`extraction_prompt_version`), so any record
can be traced back to exactly which prompt/schema produced it -- useful when
auditing why older records are missing a field, or behave differently after
a prompt change.

**Attribution priority cascade, with exact-match tiers before fuzzy.** The
sample roster includes two distinct, real employees with very similar names
("Jane Miller" and "Janet Miller" -- rapidfuzz similarity ~96/100). A
"fuzzy-match names first" design would risk conflating them. Card last-4 and
email are checked first (unique per employee), then exact name, and only then
fuzzy name matching as a last resort -- with an ambiguity check if the top two
fuzzy candidates are too close to call. See `tests/test_attribution.py` for
regression tests on this specific case.

**Never trust the LLM's self-reported confidence alone.** `src/review.py` runs
independent checks (required fields present, math reconciles, was a fallback
path used, is attribution unambiguous, is the receipt actually paid) and
combines them with the model's own confidence score to decide `needs_review`.
In production, a single model's self-assessment is not a reliable-enough
safety net by itself.

**`payment_status` is judged independently of `extraction_confidence`.** A
receipt/invoice can be perfectly clear and readable while still showing an
outstanding balance ("PAYMENT DUE", "Amount Due"). That's a business-validity
question (should this even be reimbursed?), not a field-extraction-quality
question, so it's tracked as its own field and its own review rule rather
than being folded into confidence.

**Cross-signal conflict detection.** If the strongest attribution signal
(e.g. card) points to a different employee than the printed name on the
receipt, we still trust the stronger signal but flag the conflict for human
review rather than silently picking one.

**Two independent tiers of duplicate detection.** SHA-256 catches the exact
same file re-uploaded under a different filename (cheap, zero false
positives, but only fires on byte-identical files). Content fingerprinting
(merchant + date + total + card) catches the same underlying purchase
submitted via two *different* files -- e.g. a PDF receipt and a separately
snapped photo of the same receipt -- which a byte hash would never catch
since the bytes differ entirely. This is the tier that caught two real
sample receipts (`receipt_021.pdf` / `receipt_022.png`) that were the same
transaction submitted twice under different formats.

**Three business-policy review rules, independent of confidence.** These
exist because "was this extracted correctly" and "should this be reimbursed
without question" are different concerns -- a receipt can be perfectly
legible and still warrant a closer look. (1) `appears_handwritten`: the LLM
is asked to judge whether the document looks handwritten/hand-filled rather
than printed or a digital invoice; handwritten documents are easier to
fabricate, so this always routes to review regardless of how confidently it
reads. (2) `multiple_receipts_detected`: the LLM is asked whether the
image/text seems to contain more than one distinct transaction; rather than
attempting to automatically split or merge them (a much harder and riskier
problem -- see Known Limitations), the record is just flagged so a human can
separate them. (3) A flat high-value threshold (`HIGH_VALUE_THRESHOLD` in
`src/review.py`, currently $1000): any expense above it always goes to human
sign-off, independent of confidence -- a $5,000 receipt that extracted
perfectly cleanly still deserves a second set of eyes before reimbursement.

## Setup

Requires Python 3.10+.

```bash
git clone <this-repo-url>
cd receipt-intake

# System deps: tesseract-ocr and poppler-utils (for pdf2image)
#   macOS:   brew install tesseract poppler
#   Ubuntu:  apt-get install tesseract-ocr poppler-utils
#   Windows: no single package-manager command -- install manually:
#     1. Tesseract: run the installer from
#        https://github.com/UB-Mannheim/tesseract/wiki, then add the
#        install dir (default C:\Program Files\Tesseract-OCR) to PATH.
#     2. Poppler: download a Windows build from
#        https://github.com/oschwartz10612/poppler-windows/releases,
#        unzip it, and add its bin\ folder to PATH.
#     Open a new terminal after editing PATH, then verify with
#     `tesseract --version` and `pdftoppm -v`.

pip install -r requirements.txt

cp .env.example .env
# Edit .env: set LLM_PROVIDER=anthropic (or openai) and the matching API key
```

Sanity-check the install:

```bash
pytest tests/            # should show all tests passing
```

## Usage

### Batch CLI

```bash
python main.py --receipts data/receipts --roster data/employee_roster.csv --out output
python main.py --workers 20   # process receipts concurrently (default: 8)
```

Writes `output/expenses.csv`, `output/expenses.json`, and
`output/expenses_needs_review.json` (just the records with `needs_review:
true` -- a plain filtered subset of the same data, not a new decision, so a
reviewer has a small worklist instead of scanning the full batch). Console
output shows a one-line summary per receipt and a final count of records
flagged for review (including how many are possible duplicates).

Note this review file is a snapshot of the current run only, not a
persistent queue -- running the pipeline again produces a fresh file with no
memory of what was already reviewed. See Future Development below for the
growing/shrinking version of this idea.

To try the full pipeline with zero API cost (for wiring/demo purposes only --
not a substitute for a real LLM), set `LLM_PROVIDER=mock` in `.env`. The mock
client does simple regex-based extraction so the pipeline runs end-to-end
without spending any credits.

### API (single-receipt, request/response)

```bash
uvicorn api:app --reload
curl -F "file=@data/receipts/receipt_001_desk___drawer.pdf" http://127.0.0.1:8000/extract
curl http://127.0.0.1:8000/health
```

Returns the same record shape as a row in the batch CSV/JSON output. Note:
cross-record duplicate detection is intentionally not run here, since it
requires comparing against the rest of a batch that doesn't exist in a
single-request context -- see "Deploying and expanding" below for how this
would work in a persistent deployment.

## Tests

```bash
pytest tests/
```

Covers the attribution cascade (including the Jane Miller / Janet Miller
distinct-employee edge case and conflicting-signal detection), the
independent review rules (payment-status flagging, math reconciliation,
handwritten/multi-receipt/high-value flags), both duplicate-detection tiers
(SHA-256 exact-file match and content fingerprinting, including that missing
fields/hashes don't cause false positives), and the needs_review filtered
export.

## Output schema

Each record includes: `filename`, `is_valid_receipt`, `merchant_name`,
`transaction_date`, `subtotal`, `tax`, `tip`, `total`, `currency`, `category`,
`payment_status`, `appears_handwritten`, `multiple_receipts_detected`,
`payment_card_last4`, `employee_name_on_receipt`,
`employee_email_on_receipt`, `matched_employee_id`, `matched_employee_name`,
`attribution_method`, `attribution_confidence`, `attribution_explanation`,
`extraction_confidence`, `extraction_explanation`, `needs_review`,
`review_reasons`, `is_possible_duplicate`, `duplicate_of`, `source_method`,
`ocr_quality`, `used_vision_fallback`, `orientation_corrected_degrees`,
`extraction_prompt_version`, `file_sha256`, `llm_provider`, `llm_model`,
`math_reconciles`, `required_fields_present`, `has_conflict`.

`expenses_needs_review.json` contains the exact same record shape, just
filtered down to `needs_review: true` records.

## Tools / libraries used

- Python 3.10
- pdfplumber -- text-layer PDF extraction
- pdf2image + poppler -- rendering PDF pages to images
- pytesseract + Tesseract OCR (including OSD for orientation detection) --
  OCR for scanned PDFs and photos
- OpenCV -- deskew (minAreaRect-based skew angle estimation)
- rapidfuzz -- fuzzy name matching
- pandas -- CSV export
- Anthropic API / OpenAI API (vision-capable models) -- structured field
  extraction (provider-agnostic; either can be used)
- FastAPI + uvicorn -- single-receipt request/response endpoint
- pytest -- unit tests

_(AI tools used during development: ChatGpt, Claude, via Anthropic's Claude Agent SDK
running in Cowork mode, for pair-programming this prototype.)_

## Known limitations

- **Duplicate detection is batch-scoped, not persistent.** Both duplicate-
  detection tiers only compare records within a single run. A receipt
  submitted today and a duplicate submitted next week (in a separate run)
  would not be caught without a persistent store of previously-seen hashes
  and fingerprints -- see "Deploying and expanding" below.
- **OSD-based orientation correction can still be wrong on very sparse or
  low-contrast images**, where Tesseract's own confidence signal is
  unreliable. The `ORIENTATION_CONFIDENCE_THRESHOLD` gate reduces (does not
  eliminate) the chance of a bad rotation being "corrected" incorrectly.
- **No currency conversion.** Non-USD receipts are extracted and tagged with
  their currency but not normalized to a common reporting currency.
- **Multiple receipts in one file: detected and flagged, not split.**
  `multiple_receipts_detected` routes the record to a human rather than
  attempting to automatically separate the transactions -- automatic
  splitting (or CV-based image segmentation to crop distinct receipts before
  OCR) is a meaningfully harder problem with real failure modes of its own
  (overlapping receipts, folded corners, double-counting), so it's left as a
  manual step for now rather than risking a silently-wrong auto-split.
- **High-value threshold is a flat dollar figure ($1000), not category-aware.** 
  `category` is extracted and exported on every record, but `review.py` never 
  reads it -- there's no rule like "Entertainment above $200 needs approval" 
  or "alcohol is never reimbursable." See Future Development below.
- **`appears_handwritten` reliability depends on vision fallback actually
  triggering.** Handwriting is a purely visual trait that plain OCR text
  can't carry -- once Tesseract turns a handwritten receipt into text, all
  visual evidence of pen-vs-print is gone. This field is only reliably
  judged when the image itself reaches the model, which only happens on the
  vision-fallback path (`ocr_quality` below `OCR_QUALITY_THRESHOLD`).
  Tesseract usually does garble handwriting badly enough to trigger this,
  but if a specific receipt is neat enough to clear the OCR-quality bar
  anyway, only text gets sent and the model has no way to correctly say
  `True` -- it will likely default to `False` incorrectly. Not a bug so much
  as an inherent limit of the OCR-first/vision-fallback cost optimization;
  a stricter version would always send the image regardless of OCR quality,
  at extra per-receipt cost. `multiple_receipts_detected` doesn't have this
  problem to the same degree, since two distinct merchant/total blocks are
  usually still visible in the concatenated OCR text even without an image.
- **Scanned (image-only) PDFs only OCR the first page.** `ingest.py`'s
  `_render_pdf_first_page` renders and OCRs just page 1
  (`first_page=1, last_page=1`); any additional pages are never read for
  that path. Text-layer PDFs are unaffected -- pdfplumber already
  concatenates text from every page. But a multi-page *scanned* receipt
  (e.g. a total or signature on page 2) would silently lose that content.
  Fixing this means rendering/OCR'ing every page and concatenating the
  text (and possibly sending multiple images on the vision-fallback path),
  which is straightforward but untested here since the sample dataset has
  no multi-page scans to catch it.

## Future development

- **Category-based reimbursement policy.** Extend `review.py` with a rule
  table keyed on `category` (e.g. per-category spend caps, disallowed
  categories, category-specific high-value thresholds instead of one flat
  `HIGH_VALUE_THRESHOLD` for everything) once an actual policy spec exists to
  encode -- deliberately not guessed at here since inventing arbitrary policy
  numbers isn't better than not having the rule at all.
- **Multi-receipt splitting**, upgrading from the current detect-and-flag
  behavior to actually separating N receipts in one file into N records --
  either by letting the model return an array of receipts and fanning that
  out downstream, or via CV-based pre-segmentation of the image.
- **Harden `appears_handwritten` detection.** Currently gated behind the same
  OCR-quality vision-fallback threshold as everything else, which means a
  neatly-written handwritten receipt that happens to OCR well could slip
  through ungraded (see Known Limitations). A nice-to-have upgrade: always
  send the image for this one judgment regardless of OCR quality (a small
  fixed cost increase per receipt), or run a cheap, dedicated
  handwriting-vs-print classifier as its own ingestion-time step rather than
  relying on the general vision-fallback trigger.
- **Persistent, growing/shrinking review queue**, upgrading
  `expenses_needs_review.json` from a per-run snapshot into a real queue: a
  store (could start as simple as one JSON/DB file) that new flagged records
  get appended into across runs, each with a status
  (`pending`/`approved`/`rejected`) and reviewer/timestamp once acted on, so
  the queue shrinks as items get resolved and grows as new receipts come in
  -- rather than every run producing a fresh, memoryless snapshot of just
  that batch. This is the natural next step toward the "Human review UI"
  item below.
- **Multi-page scanned PDF support**, extending `ingest.py` to render and
  OCR every page (not just page 1) for image-only PDFs, concatenating text
  across pages the same way pdfplumber already does for text-layer PDFs.

## Deploying and expanding this in production

- **Queue-based ingestion**: replace the CLI loop with an async worker queue
  (e.g. receipts land in S3/GCS, trigger a Lambda/Cloud Function per file) so
  volume scales independently of any single process. The FastAPI endpoint in
  `api.py` is a step toward this -- a request/response front door that could
  sit in front of such a queue.
- **Persistent duplicate detection**: move the SHA-256 and content-fingerprint
  checks from batch-scoped (`src/duplicates.py`) to a persistent store (e.g. a
  database table of previously-seen hashes/fingerprints with a lookback
  window), so the `/extract` API endpoint can also catch duplicates against
  historical submissions, not just within one CLI run.
- **Human review UI**: `needs_review` records should surface in a lightweight
  review queue (e.g. a simple internal tool) where a reviewer sees the
  original image side-by-side with extracted fields and can correct/approve.
  Corrections should be logged and periodically used to tune thresholds or
  few-shot examples.
- **Caching/idempotency**: the `file_sha256` field already computed during
  ingestion is a natural key for this -- skip re-processing (and re-spending
  tokens) on a rerun of a file already seen, keyed by hash.
- **Cost controls**: track token spend per receipt; the OCR-first/vision-
  fallback design already min
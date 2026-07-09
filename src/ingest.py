"""
Ingestion layer: turns a raw receipt file (PDF or image, text-based or
scanned/photographed) into a normalized ReceiptSource that downstream
extraction code can work with regardless of original format.

Strategy:
  1. Text-layer PDFs -> extract text directly with pdfplumber (cheap, exact).
  2. Scanned/image-only PDFs and JPG/PNG photos -> first correct gross
     orientation (90/180/270 degrees, e.g. a sideways phone photo), then
     deskew small-angle tilt, then run local Tesseract OCR, and keep the
     corrected image around too.
  3. We score the OCR transcript's quality with a cheap heuristic. Low-quality
     OCR (blurry photo, still-skewed scan, tiny receipt) gets flagged so the
     extraction stage knows to fall back to sending the image itself to a
     vision-capable LLM instead of trusting garbled OCR text.

This hybrid (OCR-first, vision-fallback) approach keeps the common case cheap
(text tokens only) while still handling messy real-world photos correctly.

Why deskew matters: even a mild rotation (5-10 degrees, typical of a photo
snapped at an angle) confuses Tesseract's line/column segmentation -- it can
end up associating a value with the wrong label on the row above or below,
producing output that *looks* plausible (real words, real numbers) but is
actually shifted and wrong. That's worse than obviously-garbled text because
the quality heuristic doesn't catch it. Deskewing before OCR fixes this at
the source instead of trying to detect it after the fact.

Why orientation correction is separate from deskew: a fully sideways (90
degree) photo is a categorically different problem than a few-degrees tilt,
and our OCR-quality heuristic cannot reliably catch it after the fact --
Tesseract can produce letter-shaped noise from rotated glyphs that still
scores as "readable-looking" even though the actual fields are garbage
(confirmed via stress-testing, see data/stress_test_receipts/FINDINGS.md).
Tesseract's own orientation-and-script-detection (OSD) is purpose-built to
catch exactly this, so we run it as an explicit first pass rather than
relying on the general text-quality heuristic to notice the image is
sideways.
"""
import base64
import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pdfplumber
import pytesseract
from PIL import Image

try:
    from pdf2image import convert_from_path
except ImportError:  # pragma: no cover
    convert_from_path = None

RECEIPT_KEYWORDS = (
    "total", "date", "receipt", "paid", "amount", "tax", "card", "subtotal",
    "fare", "invoice", "balance",
)


@dataclass
class ReceiptSource:
    filename: str
    file_type: str  # "pdf" or "image"
    source_method: str  # "pdf_text", "ocr", "ocr_low_quality"
    text: str
    ocr_quality: float  # 0-1, 1.0 for clean pdf text-layer extraction
    image_b64: Optional[str]  # base64-encoded PNG, for vision fallback
    image_media_type: str = "image/png"
    orientation_corrected_degrees: int = 0  # 0, 90, 180, or 270 if OSD found and fixed a rotation
    file_sha256: str = ""  # hash of the raw file bytes, for exact-duplicate-file detection


def _score_text_quality(text: str) -> float:
    """Cheap heuristic for how trustworthy an OCR transcript looks.

    Combines: overall length, ratio of alphanumeric characters (garbled OCR
    tends to produce lots of stray punctuation/symbols), and presence of
    receipt-domain keywords and digits (every real receipt has numbers).
    """
    if not text or not text.strip():
        return 0.0

    stripped = text.strip()
    length_score = min(len(stripped) / 200.0, 1.0)

    alnum = sum(c.isalnum() or c.isspace() for c in stripped)
    alnum_ratio = alnum / len(stripped)

    has_digits = 1.0 if re.search(r"\d", stripped) else 0.0

    lowered = stripped.lower()
    keyword_hits = sum(1 for kw in RECEIPT_KEYWORDS if kw in lowered)
    keyword_score = min(keyword_hits / 3.0, 1.0)

    score = (
        0.25 * length_score
        + 0.35 * alnum_ratio
        + 0.15 * has_digits
        + 0.25 * keyword_score
    )
    return round(max(0.0, min(score, 1.0)), 3)


def _correct_orientation(pil_image: Image.Image) -> "tuple[Image.Image, int]":
    """Detects and corrects gross (90/180/270-degree) rotation using
    Tesseract's orientation-and-script-detection (OSD), which is
    purpose-built for exactly this -- unlike our text-quality heuristic,
    which can be fooled by letter-shaped noise from sideways glyphs into
    thinking a completely garbage OCR result "looks" fine (see
    data/stress_test_receipts/FINDINGS.md for a concrete example).

    Returns (corrected_image, degrees_corrected). degrees_corrected is 0 if
    OSD found nothing to fix (or couldn't run -- e.g. on a very sparse or
    low-contrast image, which OSD sometimes can't analyze; we fail open and
    leave the image untouched rather than raising).
    """
    try:
        osd = pytesseract.image_to_osd(pil_image, output_type=pytesseract.Output.DICT)
        rotate = int(osd.get("rotate", 0)) % 360
        if rotate in (90, 180, 270):
            # Tesseract's "rotate" is how far clockwise the image must be
            # turned to be upright; PIL's rotate() is counter-clockwise for
            # positive angles, so we negate it.
            corrected = pil_image.rotate(-rotate, expand=True)
            return corrected, rotate
        return pil_image, 0
    except Exception:
        # OSD can fail outright on sparse/blank/very low-contrast images --
        # that's fine, just skip orientation correction for this one.
        return pil_image, 0


def _deskew_image(pil_image: Image.Image) -> Image.Image:
    """Corrects small-angle rotation (e.g. a receipt photo taken at a tilt)
    before OCR. Estimates the skew angle from the minimum-area bounding
    rectangle of dark (text) pixels, then rotates to straighten it.

    Deliberately conservative: only corrects genuinely small angles. Angles
    near 0 are left alone (already straight, avoid introducing resampling
    blur for nothing); angles near 90 are left alone too, since that's more
    likely a text-block aspect-ratio artifact than an actual rotated photo,
    and we don't want to accidentally rotate a fine receipt sideways.
    """
    try:
        rgb = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if coords.shape[0] < 20:
            return pil_image  # not enough signal to estimate skew reliably

        angle = cv2.minAreaRect(coords)[-1]
        # cv2.minAreaRect reports angle in (-90, 0]; normalize to a signed
        # rotation close to 0 (how far off-vertical/horizontal the text is).
        angle = -(90 + angle) if angle < -45 else -angle

        if abs(angle) < 0.5 or abs(angle) > 20:
            return pil_image

        h, w = gray.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            rgb, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )
        return Image.fromarray(rotated)
    except Exception:
        # Deskew is a best-effort enhancement -- never let it break ingestion.
        return pil_image


def _image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _render_pdf_first_page(path: Path) -> Optional[Image.Image]:
    if convert_from_path is None:
        return None
    try:
        pages = convert_from_path(str(path), dpi=200, first_page=1, last_page=1)
        return pages[0] if pages else None
    except Exception:
        return None


def ingest_receipt(path: Path) -> ReceiptSource:
    suffix = path.suffix.lower()
    # Hash the raw bytes once, up front -- this catches the literal same file
    # being uploaded twice (even under a different filename), which is a
    # different and cheaper check than the content-based fingerprint in
    # src/duplicates.py (that one catches the same *transaction* submitted
    # via two different files/formats, which a byte hash can't see).
    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    if suffix == ".pdf":
        # Try the cheap path first: does this PDF have a real text layer?
        with pdfplumber.open(str(path)) as pdf:
            text_parts = [page.extract_text() or "" for page in pdf.pages]
        pdf_text = "\n".join(part for part in text_parts if part).strip()

        if len(pdf_text) >= 40:
            return ReceiptSource(
                filename=path.name,
                file_type="pdf",
                source_method="pdf_text",
                text=pdf_text,
                ocr_quality=1.0,
                image_b64=None,
                file_sha256=file_sha256,
            )

        # No usable text layer -> render to image and OCR it.
        image = _render_pdf_first_page(path)
        if image is None:
            return ReceiptSource(
                filename=path.name,
                file_type="pdf",
                source_method="ocr_low_quality",
                text=pdf_text,
                ocr_quality=0.0,
                image_b64=None,
                file_sha256=file_sha256,
            )
        image, rotated_degrees = _correct_orientation(image)
        image = _deskew_image(image)
        ocr_text = pytesseract.image_to_string(image)
        quality = _score_text_quality(ocr_text)
        return ReceiptSource(
            filename=path.name,
            file_type="pdf",
            source_method="ocr" if quality >= 0.4 else "ocr_low_quality",
            text=ocr_text.strip(),
            ocr_quality=quality,
            image_b64=_image_to_b64(image),
            orientation_corrected_degrees=rotated_degrees,
            file_sha256=file_sha256,
        )

    elif suffix in (".jpg", ".jpeg", ".png"):
        image, rotated_degrees = _correct_orientation(Image.open(path))
        image = _deskew_image(image)
        ocr_text = pytesseract.image_to_string(image)
        quality = _score_text_quality(ocr_text)
        return ReceiptSource(
            filename=path.name,
            file_type="image",
            source_method="ocr" if quality >= 0.4 else "ocr_low_quality",
            text=ocr_text.strip(),
            ocr_quality=quality,
            image_b64=_image_to_b64(image),
            orientation_corrected_degrees=rotated_degrees,
            file_sha256=file_sha256,
        )

    else:
        raise ValueError(f"Unsupported receipt file type: {path.name}")


def discover_receipts(receipts_dir: Path):
    exts = {".pdf", ".jpg", ".jpeg", ".png"}
    return sorted(p for p in receipts_dir.iterdir() if p.suffix.lower() in exts)

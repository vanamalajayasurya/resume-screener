import io
import re

import pdfplumber

MIN_TEXT_CHARS = 80


class ResumeParseError(ValueError):
    pass


def extract_text(data: bytes) -> str:
    """Extract text from a PDF. Raises ResumeParseError for unreadable / scanned files."""
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages = [(p.extract_text() or "") for p in pdf.pages]
    except Exception as exc:
        raise ResumeParseError("Could not read this file as a PDF.") from exc

    text = "\n".join(pages).strip()
    if len(text) < MIN_TEXT_CHARS:
        raise ResumeParseError(
            "No readable text found. This looks like a scanned PDF (image only); "
            "upload a text-based PDF or run OCR first."
        )
    return text


def guess_name(text: str, filename: str = "") -> str:
    for line in text.splitlines()[:6]:
        line = line.strip()
        if 2 <= len(line.split()) <= 4 and re.fullmatch(r"[A-Za-z .'\-]+", line):
            return line.title() if line.isupper() else line
    stem = re.sub(r"\.pdf$", "", filename, flags=re.I)
    return re.sub(r"[_\-]+", " ", stem).strip() or "Unknown candidate"

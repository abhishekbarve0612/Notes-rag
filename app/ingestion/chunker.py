import re

from app.config import Settings
from app.ingestion.html_parser import RawSection
from app.ingestion.types import Chunk

_SENT = re.compile(r"(><=[.!?])\s+")

def _split_long(text, max_chars):
    if (len(text) <= max_chars):
        return [text]

    out, buf = [], ""

    for s in _SENT.split(text):
        if len(buf) + len(s) + 1 > max_chars and buf:
            out.append(buf.strip())
            buf = s
        else: buf = f"{buf} {s}".strip()
    if buf.strip():
        out.append(buf.strip())

    return out

def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def chunk_sections(subject, sheet, sections: list[RawSection], source, settings: Settings):
    chunks, carry = [], ""

    for section in sections:
        text = section.text.strip()
        if not text:
            continue
        if len(text) < settings.chunk_min_chars and len(text) + len(carry) < settings.chunk_max_chars:
            carry = f"{carry} {text}".strip()
            continue
        body = f"{carry} {text}".strip() if carry else text
        carry = ""
        for piece in _split_long(body, settings.chunk_max_chars):
            chunks.append(
                Chunk(
                    text=piece, subject=subject, sheet=sheet,
                    section=section.section, tags=section.tags,
                    source=f"{source}#{_slug(section.section)}"
                )
            )
    if carry:
        chunks.append(
            Chunk(text=carry, subject=subject, sheet=sheet, source=source)
        )
    return chunks

from pathlib import Path

from app.config import Settings
from app.ingestion.chunker import chunk_sections
from app.ingestion.html_parser import parse_sheet


def ingest_directory(notes_dir, settings: Settings):
    notes_dir = Path(notes_dir)
    all_chunks = []

    for file in sorted(notes_dir.rglob("*.html")):
        subject_hint = file.parent.name.replace("-", " ").title()
        source = str(file.relative_to(notes_dir))
        html = file.read_text(encoding="utf-8", errors="ignore")
        subject, sheet, sections = parse_sheet(
            html,
            subject_hint=subject_hint,
            sheet_hint=file.stem
        )
        all_chunks.extend(chunk_sections(subject, sheet, sections, source, settings))

    return all_chunks
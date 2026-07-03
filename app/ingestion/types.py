import hashlib
from dataclasses import dataclass, field, asdict


@dataclass
class Chunk:
    text: str
    subject: str
    sheet: str
    section: str = ""
    tags: list[str] = field(default_factory=list)
    source: str = ""
    chunk_id: str = ""

    def __post_init__(self):
        if not self.chunk_id:
            key = f"{self.source}|{self.section}|{self.text[:64]}"
            self.chunk_id = hashlib.sha1(key.encode()).hexdigest()[:16]

    def embedding_text(self, with_crumb: bool = True) -> str:
        if not with_crumb:
            return self.text
        crumb = " > ".join(x for x in [self.subject, self.sheet, self.section] if x)
        tagline = f"tags: {', '.join(self.tags)}" if self.tags else ""
        return f"[{crumb}]\n{tagline}\n{self.text}".strip()

    def to_dict(self) -> dict:
        return asdict(self)

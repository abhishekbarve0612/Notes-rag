from app.generation.prompts import TOPICS_PROMPT, QUIZ_PROMPT
from app.ingestion.types import Chunk
from app.models.base import Chatter

def _sample(chunks: list[Chunk], n: int = 12) -> str:
    """Spread the sample across the corpus so we don't only see the first sheet."""
    if len(chunks) <= n:
        picked = chunks
    else:
        step = len(chunks) // n
        picked = chunks[::step][:n]
    return "\n\n".join(c.text[:400] for c in picked)

def get_topics(chatter: Chatter, chunks: list[Chunk]) -> list[dict]:
    raw = chatter.answer(TOPICS_PROMPT.format(context=_sample(chunks)))
    output = []
    for line in raw.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if "--" in line:
            title, desc = line.split("--", 1)
            output.append({
                "title": title.strip(),
                "description": desc.strip()
            })
    return output[:8]

def get_suggestions(chatter: Chatter, chunks: list[Chunk]) -> list[str]:
    raw = chatter.answer(QUIZ_PROMPT.format(context=_sample(chunks)))

    qs = [ln.strip().lstrip("-*0123456789. ").strip()
          for ln in raw.splitlines() if ln.strip()]

    return qs[:4]
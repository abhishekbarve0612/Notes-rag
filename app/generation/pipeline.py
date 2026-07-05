from app.config import Settings
from app.generation.prompts import REWRITE_PROMPT, RAG_PROMPT
from app.models.base import Embedder, Chatter
from app.retrieval.reranker import build_reranker
from app.retrieval.retriever import Retriever
from app.retrieval.store import VectorStore


def _context(reranked):
    blocks = []

    for i, (c, _s) in enumerate(reranked, 1):
        crumb = " > ".join(x for x in [c.subject, c.sheet, c.section] if x)
        blocks.append(f"[{i}] ({crumb})\n{c.text}")
    return "\n\n".join(blocks)

class RagPipeline:
    def __init__(self, embedder: Embedder, store: VectorStore, chatter: Chatter, settings: Settings):
        self.retriever = Retriever(embedder, store, settings)
        self.reranker = build_reranker(settings)
        self.chatter = chatter

    def answer(self, question: str, rewrite: bool = True) -> dict:
        query = self._rewrite(question) if rewrite else question
        reranked = self.reranker.rerank(
            query,
            self.retriever.retrieve(query)
        )
        if not reranked:
            return {
                "answer": "The notes don't cover this.",
                "sources": [],
                "rewritten_query": query,
            }
        answer = self.chatter.answer(
            RAG_PROMPT.format(context=_context(reranked), question=query)
        )
        sources = [{
            "subject": c.subject,
            "sheet": c.sheet,
            "section": c.section,
            "source": c.source,
            "score": round(s, 3)
        } for c, s in reranked]

        return {
            "answer": answer,
            "sources": sources,
            "rewritten_query": query,
        }

    def _rewrite(self, question: str) -> str:
        try:
            return self.chatter.answer(REWRITE_PROMPT.format(question=question)).strip()
        except Exception:
            return question
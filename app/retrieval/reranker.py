from functools import lru_cache

from app.config import Settings


@lru_cache(maxsize=1)
def _cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)

class CrossEncoderReranker:
    def __init__(self, settings: Settings):
        self.model_name = settings.reranker_model
        self.top_k = settings.rerank_top_k

    def rerank(self, query, candidates):
        if not candidates:
            return []

        ce = _cross_encoder(self.model_name)
        scores = ce.predict([(query, c.text) for c, _ in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda  x: x[1], reverse=True)

        return [(c, float(s)) for (c, _old), s in ranked[: self.top_k]]

class NoopReranker:
    def __init__(self, settings: Settings):
        self.top_k = settings.rerank_top_k

    def rerank(self, query, candidates):
        return candidates[: self.top_k]

def build_reranker(settings: Settings):
    return (
        CrossEncoderReranker if settings.reranker_model == "cross-encoder"
        else NoopReranker
    )(settings)
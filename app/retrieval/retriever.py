from app.config import Settings
from app.models.base import Embedder
from app.retrieval.store import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, settings: Settings):
        self.embedder = embedder
        self.store = store
        self.k = settings.retrieval_top_k

    def retrieve(self, query: str):
        query_vector = self.embedder.embed([query])[0]
        return self.store.search(query_vector, self.k)
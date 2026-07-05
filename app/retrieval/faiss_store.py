import json
import os

import faiss
import numpy as np

from app.ingestion.types import Chunk


class FaissStore:
    def __init__(self, dim: int, embedder_name: str = ""):
        self.dim, self.embedder_name = dim, embedder_name
        self.index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    def add(self, chunks, vectors):
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected store dimensions {self.dim}, got vector dimensions {vectors.shape[1]}")
        self.index.add(vectors.astype(np.float32))
        self._chunks.extend(chunks)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        meta = {
            "dim": self.dim,
            "embedder_name": self.embedder_name,
            "chunks": [chunk.to_dict() for chunk in self._chunks],
        }

        with open(os.path.join(path, "meta.json"), "w") as file:
            json.dump(meta, file)

    def search(self, query_vector, k):
        scores, idx = self.index.search(
            query_vector.astype(np.float32).reshape(1, -1),
            k
        )
        output = []
        for s, i in zip(scores[0], idx[0]):
            if i != -1:
                output.append((self._chunks[i], float(s)))
        return output

    @classmethod
    def load(cls, path, expected_embedder=None):
        with open(os.path.join(path, "meta.json"), "r") as file:
            meta = json.load(file)
        if expected_embedder and meta['embedder_name'] != expected_embedder:
            raise ValueError(
                f"Index built with embedder {meta['embedder_name']} does not match "
                f"stored idex embeddings built with embedder {expected_embedder}"
            )
        s = cls(meta['dim'], meta['embedder_name'])
        s.index = faiss.read_index(os.path.join(path, "index.faiss"))
        s._chunks = [Chunk(**chunk) for chunk in meta['chunks']]
        return s

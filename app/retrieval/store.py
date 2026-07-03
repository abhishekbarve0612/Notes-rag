import json
import os
from typing import runtime_checkable, Protocol

import numpy as np

from app.ingestion.types import Chunk


@runtime_checkable
class VectorStore(Protocol):
    dim: int
    embedder_name: str
    def add(self, chunks: list[str], vectors: np.ndarray) -> None: ...
    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[Chunk, float]]: ...
    def save(self, path: str) -> None: ...

class NumpyStore:
    def __init__(self, dim: int, embedder_name: str):
        self.dim, self.embedder_name = dim, embedder_name
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._chunks: list[Chunk] = []

    def add(self, chunks, vectors):
        if vectors.shape[1] != self.dim:
            raise ValueError(f"vectors dimension mismatch vector dim:  {vectors.shape[1]} != store dim: {self.dim}")
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
        self._chunks.extend(chunks)

    def search(self, query_vector, k):
        if not self._chunks:
            return []
        dot_product = self._vectors @ query_vector.reshape(-1)
        idx = np.argsort(-dot_product)[:k]

        return [(self._chunks[i], float(dot_product[i])) for i in idx]

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "vectors.npy"), self._vectors)
        meta = {
            "dim": self.dim,
            "embedder_name": self.embedder_name,
            "chunks": [chunk.to_dict() for chunk in self._chunks]
        }

        with open(os.path.join(path, "meta.json"), "w") as file:
            json.dump(meta, file)

    @classmethod
    def load(cls, path, expected_embedder=None):
        with open(os.path.join(path, "meta.json")) as file:
            meta = json.load(file)
        if expected_embedder and meta['embedder_name'] != expected_embedder:
            raise ValueError(
                f"Index built with {meta['embedder_name']} does not match {expected_embedder}"
                "Reindex the data"
            )
        s = cls(meta['dim'], meta['embedder_name'])
        s._vectors = np.load(os.path.join(path, "vectors.npy"))
        s._chunks = [Chunk(**chunk) for chunk in meta['chunks']]

        return s

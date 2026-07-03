import hashlib

import numpy as np

from app.config import Settings


class LocalEmbedder:
    def __init__(self, settings: Settings):
        from sentence_transformers import  SentenceTransformer

        self._model = SentenceTransformer(settings.local_embed_model)
        self.dim = self._model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        v = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        return v.astype(np.float32)

class OpenAIEmbedder:
    def __init__(self, settings: Settings):
        from openai import  OpenAI
        if not settings.openai_api_key:
            raise ValueError("EMBEDDING_PROVIDER=openai but OPENAI_API_KEY not set")

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model, self.dim = settings.openai_embed_model, 1536

    def embed(self, texts: list[str]) -> np.ndarray:
        r = self._client.embeddings.create(model=self._model, input=texts)

        return np.array([d.embedding for d in r.data], dtype=np.float32)

class GeminiEmbedder:
    def __init__(self, settings: Settings):
        import google.generativeai as genai
        if not settings.gemini_api_key:
            raise ValueError("EMBEDDING_PROVIDER=gemini but GEMINI_API_KEY not set")
        genai.configure(api_key=settings.gemini_api_key)
        self._genai, self._model, self.dim = genai, settings.gemini_embed_model, 3072

    def embed(self, texts: list[str]) -> np.ndarray:
        out = [
            self._genai.embed_content(
                model=self._model,
                content=t
            )["embedding"]
            for t in texts
        ]

        return np.array(out, dtype=np.float32)


class FakeEmbedder:                 # deterministic, for tests. no model, no network
    def __init__(self, settings: Settings | None = None, dim: int = 384):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int(hashlib.sha256(t.encode()).hexdigest(), 16) % (2**32)
            v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
            out[i] = v / (np.linalg.norm(v) + 1e-9)   # unit length, like the real ones
        return out
from app.config import Settings
from app.models.factory import build_embedder, build_chatter
from app.models.base import Embedder, Chatter
import numpy as np

def test_fake_embedder():
    emb = build_embedder(Settings(embedder="fake", llm="fake"))
    assert isinstance(emb, Embedder)
    v1, v2 = emb.embed(["hashing", "streams"]), emb.embed(["hashing", "streams"])
    assert v1.shape == (2, emb.dim) and v1.dtype == np.float32
    assert np.allclose(v1, v2)                                   # deterministic
    assert np.allclose(np.linalg.norm(v1, axis=1), 1.0, atol=1e-4)

def test_config_selects_class():
    assert type(build_embedder(Settings(embedder="fake"))).__name__ == "FakeEmbedder"
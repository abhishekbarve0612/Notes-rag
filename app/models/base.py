from typing import runtime_checkable, Protocol
import numpy as np

@runtime_checkable
class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> np.ndarray: ...

@runtime_checkable
class Chatter(Protocol):
    def answer(self, prompt: str) -> str: ...
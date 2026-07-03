from functools import lru_cache

from app.config import Settings, get_settings
from app.models.base import Embedder, Chatter
from app.models.chatters import OpenAIChatter, AnthropicChatter, GeminiChatter, FakeChatter
from app.models.embedders import LocalEmbedder, OpenAIEmbedder, GeminiEmbedder, FakeEmbedder


def build_embedder(settings: Settings) -> Embedder:
    return {
        "local": LocalEmbedder,
        "openai": OpenAIEmbedder,
        "gemini": GeminiEmbedder,
        "fake": FakeEmbedder
    }[settings.embedder](settings)

def build_chatter(settings: Settings) -> Chatter:
    return {
        "openai": OpenAIChatter,
        "anthropic": AnthropicChatter,
        "gemini": GeminiChatter,
        "fake": FakeChatter
    }[settings.llm](settings)

@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return build_embedder(get_settings())

@lru_cache(maxsize=1)
def get_chatter() -> Chatter:
    return build_chatter(get_settings())
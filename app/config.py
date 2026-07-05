from functools import  lru_cache
from typing import  Literal
from pydantic_settings import  BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    embedder: Literal["local", "openai", "gemini", "fake"] = "local"
    llm: Literal["openai", "anthropic", "gemini", "fake"] = "openai"
    reranker: Literal["cross-encoder", "noop"] = "cross-encoder"

    local_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2" # 384 dim
    openai_embed_model: str = "text-embedding-3-small" # 1536 dim
    gemini_embed_model: str = "models/gemini-embedding-001" # 768 dim

    openai_chat_model: str = "gpt-4.1-nano"
    gemini_chat_model: str = "gemini-3.1-flash-lite"
    anthropic_chat_model: str = "claude-haiku-4-5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    chunk_min_chars: int = 200
    chunk_max_chars: int = 1200

    index_dir: str = "./data/index"

    retrieval_top_k: int = 15
    rerank_top_k: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

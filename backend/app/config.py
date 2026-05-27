"""Application settings, read from environment / .env.

Everything here is swappable so the same code runs locally and on the
deployed instance. Local dev leans on the local backends (in-process vector
store, ONNX models); production points the providers at Supabase pgvector and
a hosted embedding/rerank API.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (OpenRouter by default; any OpenAI-style endpoint works) ---
    llm_provider: str = "openrouter"
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # --- Embeddings: "local" (ONNX) for dev, "jina" (hosted) for deploy ---
    embedding_provider: str = "local"
    embedding_model_local: str = "BAAI/bge-small-en-v1.5"
    embedding_model_hosted: str = "jina-embeddings-v3"
    jina_api_key: str = ""
    embedding_dim: int = 384  # bge-small / jina small; set per model

    # --- Reranker: cross-encoder. "local" (ONNX) for dev, "jina" for deploy ---
    rerank_provider: str = "local"
    rerank_model_local: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_model_hosted: str = "jina-reranker-v2-base-multilingual"

    # --- Vector store: "local" (in-process) for dev, "pgvector" for deploy ---
    vector_store: str = "local"
    local_store_path: str = str(DATA_DIR / "store.jsonl")
    database_url: str = ""  # postgres connection string for pgvector

    # --- Retrieval shape ---
    retrieve_k: int = 20      # candidates pulled from the vector store
    rerank_top_n: int = 5     # kept after the cross-encoder
    chunk_tokens: int = 400
    chunk_overlap: int = 60

    # --- Agent ---
    agent_max_steps: int = 4

    # --- App ---
    cors_origins: str = "http://localhost:5173"


@lru_cache
def get_settings() -> Settings:
    return Settings()

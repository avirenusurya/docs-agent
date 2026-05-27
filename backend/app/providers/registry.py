"""Builds concrete providers from settings.

Each getter is cached so we load a model or open a connection once per process.
Concrete embedder / reranker / store classes are imported lazily so that, for
example, a deploy that only uses hosted APIs never imports the ONNX runtime.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.providers.base import Embedder, LLM, Reranker


@lru_cache
def get_llm() -> LLM:
    s = get_settings()
    if s.llm_provider == "openrouter":
        from app.providers.llm import OpenRouterLLM

        return OpenRouterLLM(s)
    raise ValueError(f"unknown llm_provider: {s.llm_provider}")


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_provider == "local":
        from app.providers.embeddings import LocalEmbedder

        return LocalEmbedder(s)
    if s.embedding_provider == "jina":
        from app.providers.embeddings import JinaEmbedder

        return JinaEmbedder(s)
    raise ValueError(f"unknown embedding_provider: {s.embedding_provider}")


@lru_cache
def get_reranker() -> Reranker:
    s = get_settings()
    if s.rerank_provider == "local":
        from app.providers.rerank import LocalReranker

        return LocalReranker(s)
    if s.rerank_provider == "jina":
        from app.providers.rerank import JinaReranker

        return JinaReranker(s)
    raise ValueError(f"unknown rerank_provider: {s.rerank_provider}")


def describe_providers(s: Settings | None = None) -> dict:
    """Small summary for the /health and debug endpoints."""
    s = s or get_settings()
    return {
        "llm": {"provider": s.llm_provider, "model": s.llm_model},
        "embedding": {
            "provider": s.embedding_provider,
            "model": s.embedding_model_local
            if s.embedding_provider == "local"
            else s.embedding_model_hosted,
            "dim": s.embedding_dim,
        },
        "rerank": {
            "provider": s.rerank_provider,
            "model": s.rerank_model_local
            if s.rerank_provider == "local"
            else s.rerank_model_hosted,
        },
        "vector_store": s.vector_store,
    }

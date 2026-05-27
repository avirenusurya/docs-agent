"""Embedding backends.

Local: fastembed (ONNX, no torch) running bge-small. Hosted: Jina's API.
Both are bi-encoders, used for first-pass retrieval. bge wants a short
instruction on the query side, which is handled here so callers don't have to.
"""
from __future__ import annotations

import httpx

from app.config import Settings

# Recommended query instruction for bge-* retrieval models.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbedder:
    def __init__(self, settings: Settings):
        from fastembed import TextEmbedding

        self.dim = settings.embedding_dim
        self._is_bge = "bge" in settings.embedding_model_local.lower()
        self._model = TextEmbedding(model_name=settings.embedding_model_local)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        q = (_BGE_QUERY_PREFIX + text) if self._is_bge else text
        return next(iter(self._model.embed([q]))).tolist()


class JinaEmbedder:
    """Hosted embeddings. Used by the deployed instance to keep it lean."""

    def __init__(self, settings: Settings):
        if not settings.jina_api_key:
            raise RuntimeError("JINA_API_KEY is not set but embedding_provider=jina.")
        self.dim = settings.embedding_dim
        self._model = settings.embedding_model_hosted
        self._client = httpx.Client(
            base_url="https://api.jina.ai/v1",
            headers={"Authorization": f"Bearer {settings.jina_api_key}"},
            timeout=60.0,
        )

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        payload = {
            "model": self._model,
            "task": task,
            "dimensions": self.dim,
            "input": texts,
        }
        resp = self._client.post("/embeddings", json=payload)
        resp.raise_for_status()
        rows = sorted(resp.json()["data"], key=lambda r: r["index"])
        return [r["embedding"] for r in rows]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, task="retrieval.passage")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], task="retrieval.query")[0]

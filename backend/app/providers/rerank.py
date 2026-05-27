"""Cross-encoder rerankers.

First-pass retrieval uses a bi-encoder: query and document are embedded
separately, so it's fast but never compares them directly. A cross-encoder
reads the (query, document) pair together and scores the match, which is more
accurate but too slow to run over the whole corpus. So the pattern is: retrieve
a wide set with the bi-encoder, then rerank the top candidates here. The eval
harness measures how much this step actually helps.

Local: fastembed's ONNX cross-encoder. Hosted: Jina's reranker API.
"""
from __future__ import annotations

import httpx

from app.config import Settings


class LocalReranker:
    def __init__(self, settings: Settings):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=settings.rerank_model_local)

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        return [float(s) for s in self._model.rerank(query, documents)]


class JinaReranker:
    def __init__(self, settings: Settings):
        if not settings.jina_api_key:
            raise RuntimeError("JINA_API_KEY is not set but rerank_provider=jina.")
        self._model = settings.rerank_model_hosted
        self._client = httpx.Client(
            base_url="https://api.jina.ai/v1",
            headers={"Authorization": f"Bearer {settings.jina_api_key}"},
            timeout=60.0,
        )

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        payload = {"model": self._model, "query": query,
                   "documents": documents, "return_documents": False}
        resp = self._client.post("/rerank", json=payload)
        resp.raise_for_status()
        # API returns results sorted by score; put them back in input order.
        scores = [0.0] * len(documents)
        for r in resp.json()["results"]:
            scores[r["index"]] = float(r["relevance_score"])
        return scores
